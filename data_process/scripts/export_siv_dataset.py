"""把 Contact→SIV 结果导出成一个**独立的新数据集**，放在 SIV_WAM 下，不动原数据集。

产出 <export_root>/<task_name>/，结构与原 LeRobot task 目录一致，可直接喂
siv_wam/dataset/lerobot_latent_dataset.py（把 config.dataset_path 指到 export_root）：

  <task_name>/
    data      -> symlink 原数据集（parquet）
    meta      -> symlink 原数据集（info.json / episodes.jsonl）
    videos    -> symlink 原数据集（mp4）
    latents/chunk-000/observation.images.<cam>      -> symlink 原数据集（video latent）
    latents/chunk-000/observation.masks.<cam>/...   <- 本工具新生成的 SIV latent
    siv_raw/<cam>_siv.npz, siv_stats.json           <- 中间产物
    contacts/ep<id>_contacts.jsonl, cameras.json, cameras.jsonl

用法（仿真真实接触，推荐；需 GPU 节点跑物理仿真）：
  python scripts/export_siv_dataset.py \
    --task_dir <原 LeRobot task 目录> \
    --sim_dir  <build_contacts_sim.py 的输出目录> \
    --export_root ./outputs/contact_siv_dataset \
    --episodes 0 1 2 --cams cam_high cam_left_wrist cam_right_wrist \
    --encode

离线兜底（无仿真，纯 EEF 接触代理，只有 cam_high 有标定）：
  python scripts/export_siv_dataset.py --task_dir <TASK> --offline \
    --export_root ... --episodes 0 --cams cam_high --encode
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2s.contacts import EEFContactEstimator, read_jsonl
from c2s.geometry import Camera
from c2s.lerobot_io import read_episode, read_video_latent
from c2s.siv import SIVBuilder, SIVConfig, build_sequence
from c2s.spec import CAM_KEYS, MASK_KEYS

# LeRobot 相机名 <-> RoboTwin 仿真相机名
CAM_TO_SIM = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:  # 不支持 symlink 的文件系统
        shutil.copytree(src, dst, dirs_exist_ok=True) if os.path.isdir(
            src
        ) else shutil.copy2(src, dst)


def make_camera(name: str, item: dict, image_size):
    from c2s.contacts import ContactFrame  #  保持 import 完整性

    K = np.asarray(item["K"], dtype=np.float64)
    T = np.asarray(item["world_to_camera"], dtype=np.float64)
    return Camera.from_opencv(
        name, K, T, tuple(item.get("image_size")) or None, image_size
    )


def prepare_task_dir(task_dir: str, export_root: str, cams: list) -> str:
    """搭好骨架：data/meta/videos/video latent 全部软链到原数据集。"""
    task = os.path.basename(task_dir.rstrip("/"))
    dst = os.path.join(export_root, task)
    os.makedirs(dst, exist_ok=True)
    for sub in ("data", "meta", "videos"):
        src = os.path.join(task_dir, sub)
        if os.path.isdir(src):
            link_or_copy(src, os.path.join(dst, sub))
    lat_dir = os.path.join(dst, "latents", "chunk-000")
    os.makedirs(lat_dir, exist_ok=True)
    for cam in CAM_KEYS:
        src = os.path.join(task_dir, "latents", "chunk-000", CAM_KEYS[cam])
        if os.path.isdir(src):
            link_or_copy(src, os.path.join(lat_dir, CAM_KEYS[cam]))
    return dst


def build_maps_for(frames, cams, cams_meta, per_frame, image_sizes, cfg, depth_loader):
    """image_sizes: {cam: (w, h)}，必须各自对齐该相机 video latent 的 video_width/height，
    否则 VAE 编码出的 latent 高度/宽度会和 video latent 不一致（训练时空间错位）。"""
    out = {}
    for cam in cams:
        sim_name = CAM_TO_SIM.get(cam, cam)
        size = image_sizes[cam]
        if per_frame and any(sim_name in c for c in per_frame.values()):

            def cam_at(i, _n=sim_name, _s=size):
                item = per_frame.get(i, {}).get(_n) or cams_meta[_n]
                return make_camera(_n, item, _s)

            def depth_at(f, _n=sim_name):
                if isinstance(f.depth_path, dict) and f.depth_path.get(_n):
                    return depth_loader(f.depth_path[_n])
                return None

            out[cam] = build_sequence(frames, cam_at, cfg, depth_at)
        else:
            item = cams_meta.get(sim_name) or cams_meta.get(cam)
            camera = make_camera(sim_name, item, size)
            out[cam] = SIVBuilder(camera, cfg).build(
                frames, depth_loader=depth_loader, cam_name=sim_name
            )
        print(
            f"  [{cam}] maps={out[cam].maps.shape} (target {size[1]}x{size[0]}) "
            f"stats={out[cam].stats}"
        )
    return out


def encode(results, task_dir, out_task_dir, episode, device):
    from c2s.latents import WanVAEEncoder, load_vae, save_mask_latent, siv_to_rgb
    from c2s.spec import WAN_VAE_PATH

    enc = WanVAEEncoder(load_vae(WAN_VAE_PATH, device=device))
    saved = {}
    for cam, res in results.items():
        vd = read_video_latent(task_dir, episode, CAM_KEYS[cam])
        if vd is None:
            print(f"  [{cam}] 无 video latent，跳过")
            continue
        assert res.maps.shape[1:] == (
            int(vd["video_height"]),
            int(vd["video_width"]),
        ), (
            f"[{cam}] SIV map {res.maps.shape[1:]} != video {(vd['video_height'], vd['video_width'])}"
        )
        fid = np.clip(np.asarray(list(vd["frame_ids"])), 0, res.maps.shape[0] - 1)
        lat = enc.encode(siv_to_rgb(res.maps[fid]), device=device)
        assert (lat.shape[0], lat.shape[1], lat.shape[2]) == (
            vd["latent_num_frames"],
            vd["latent_height"],
            vd["latent_width"],
        ), (
            f"[{cam}] SIV latent {lat.shape[:3]} != video "
            f"{(vd['latent_num_frames'], vd['latent_height'], vd['latent_width'])}"
        )
        base = os.path.basename(
            min(
                glob.glob(
                    os.path.join(
                        task_dir,
                        "latents",
                        "chunk-000",
                        CAM_KEYS[cam],
                        f"episode_{episode:06d}_*.pth",
                    )
                )
            )
        )
        dst = os.path.join(out_task_dir, "latents", "chunk-000", MASK_KEYS[cam], base)
        save_mask_latent(dst, lat, vd)
        saved[cam] = dst
        print(f"  [{cam}] -> {dst} latent={tuple(lat.shape)}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_dir", required=True, help="原 LeRobot task 目录（只读）")
    ap.add_argument(
        "--sim_dir",
        default=None,
        help="build_contacts_sim.py 输出目录（含 ep*_contacts.jsonl / cameras.json）",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="不用仿真，直接对 task_dir 用 EEF 接触代理（仅 cam_high 有效）",
    )
    ap.add_argument("--export_root", required=True, help="新数据集根目录")
    ap.add_argument("--episodes", type=int, nargs="*", default=[0])
    ap.add_argument("--cams", nargs="*", default=["cam_high"])
    ap.add_argument(
        "--image_size", default=None, help="HxW；不给则用 video latent 分辨率"
    )
    ap.add_argument("--contact_radius", type=float, default=0.02)
    ap.add_argument("--depth_tolerance", type=float, default=0.05)
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.sim_dir and not args.offline:
        raise SystemExit("需要 --sim_dir（仿真真实接触）或 --offline（EEF 接触代理）")

    cfg = SIVConfig(
        contact_radius=args.contact_radius, depth_tolerance=args.depth_tolerance
    )

    cams_meta, per_frame = {}, None
    if args.sim_dir:
        with open(os.path.join(args.sim_dir, "cameras.json"), encoding="utf-8") as f:
            cams_meta = json.load(f)
    pf_cache = {}
    if args.sim_dir:
        pf_cache = {
            os.path.basename(p): p
            for p in glob.glob(os.path.join(args.sim_dir, "cameras_ep*.jsonl"))
        }

    out_task_dir = prepare_task_dir(args.task_dir, args.export_root, args.cams)
    raw_root = os.path.join(out_task_dir, "siv_raw")
    os.makedirs(raw_root, exist_ok=True)
    if args.sim_dir:
        link_or_copy(
            os.path.join(args.sim_dir, "cameras.json"),
            os.path.join(out_task_dir, "contacts", "cameras.json"),
        )
        for name, path in pf_cache.items():
            link_or_copy(path, os.path.join(out_task_dir, "contacts", name))

    def depth_loader(p):
        return np.load(p)

    for ep in args.episodes:
        print(f"=== episode {ep} ===")
        if args.sim_dir:
            path = os.path.join(args.sim_dir, f"ep{ep:06d}_contacts.jsonl")
            frames = read_jsonl(path)
            # 逐帧标定按 episode 单独存（wrist 相机外参逐帧变）
            per_frame = None
            pf_path = pf_cache.get(f"cameras_ep{ep:06d}.jsonl")
            if pf_path:
                per_frame = {}
                with open(pf_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            per_frame[int(d["frame"])] = d["cameras"]
                print(f"  [cams] 逐帧标定 {len(per_frame)} 帧")
        else:
            frames = EEFContactEstimator()(read_episode(args.task_dir, ep))
            # 离线兜底：相机用本地已验证的 head_camera 标定
            cam = Camera.head_camera(image_size=(320, 256))
            cams_meta.setdefault(
                "head_camera",
                {
                    "K": cam.K.tolist(),
                    "world_to_camera": np.hstack(
                        [cam.R_cw, cam.t_cw[:, None]]
                    ).tolist(),
                    "image_size": [cam.calib_size[0], cam.calib_size[1]],
                },
            )
            print(f"  [offline] 接触代理 {len(frames)} 帧")

        # 每个相机的 map 分辨率 = 该相机 video latent 的 video_width/height
        image_sizes = {}
        for cam in args.cams:
            if args.image_size:
                h, w = (int(x) for x in args.image_size.lower().split("x"))
                image_sizes[cam] = (w, h)
            else:
                vd = read_video_latent(args.task_dir, ep, CAM_KEYS[cam])
                image_sizes[cam] = (
                    (int(vd["video_width"]), int(vd["video_height"]))
                    if vd
                    else (320, 256)
                )
        results = build_maps_for(
            frames, args.cams, cams_meta, per_frame, image_sizes, cfg, depth_loader
        )
        for cam, res in results.items():
            np.savez_compressed(
                os.path.join(raw_root, f"ep{ep:06d}_{cam}_siv.npz"),
                maps=res.maps,
                valid=res.valid,
                depths=res.depths,
            )
            with open(
                os.path.join(raw_root, f"ep{ep:06d}_{cam}_stats.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(res.stats, f, indent=2)
        if args.encode:
            encode(results, args.task_dir, out_task_dir, ep, args.device)
    print(f"[done] 新数据集 -> {out_task_dir}")


if __name__ == "__main__":
    main()
