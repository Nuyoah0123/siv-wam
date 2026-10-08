"""离线端到端：本机 LeRobot 数据集 -> 接触点 -> SIV map -> (可选) latent。

不依赖仿真器，直接跑在本机已下载的 RoboTwin 数据集上：

  python scripts/build_siv_lerobot.py \
      --task_dir <.../lerobot_robotwin_eef_aug_500/lift_pot-aloha-agilex_randomized_500-1000> \
      --episodes 0 1 --cams cam_high --out_dir ./_c2s_out --viz

  # 编成 observation.masks.* latent（需要 GPU + Wan VAE）
  python scripts/build_siv_lerobot.py --task_dir <TASK> --episodes 0 --encode \
      --vae_path <vae_dir>

流程：
  16D 末端位姿 -> EEFContactEstimator（接触代理） -> ContactFrame 序列
  -> 相机投影 + 深度自适应 Gaussian splatting -> (T,H,W) SIV map
  -> [可选] VAE 编码 -> latents/chunk-000/observation.masks.<cam>/*.pth
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2s.contacts import EEFContactConfig, EEFContactEstimator, write_jsonl
from c2s.geometry import Camera
from c2s.lerobot_io import (
    build_wrist_camera,
    list_episodes,
    map_size_for,
    read_episode,
    read_frames,
    read_video_latent,
    save_maps_npz,
)
from c2s.siv import SIVBuilder, SIVConfig, SIVResult
from c2s.spec import CAM_KEYS, DEFAULT_MAP_SIZE, MASK_KEYS
from c2s.viz import overlay, write_video


def process(task_dir, episode, cams, out_dir, cfg, ecfg, do_viz, state_key):
    eef = read_episode(task_dir, episode, state_key)
    frames_out = EEFContactEstimator(ecfg)(eef)
    task = os.path.basename(task_dir.rstrip("/"))
    result = {}
    for cam in cams:
        size = map_size_for(task_dir, episode, cam, DEFAULT_MAP_SIZE[cam])  # (H,W)
        image_size = (size[1], size[0])
        if cam == "cam_high":
            camera = Camera.head_camera(image_size=image_size)
            res = SIVBuilder(camera, cfg).build(frames_out)
        else:
            # wrist 相机装在臂上，外参逐帧变化
            res = build_with_dynamic_camera(frames_out, eef, cam, size, cfg)
            camera = build_wrist_camera(cam, eef[len(eef) // 2], size)
        npz = os.path.join(out_dir, f"{task}_ep{episode:06d}_{cam}_siv.npz")
        save_maps_npz(npz, res.maps, res.valid, res.depths, res.meta, res.stats)
        result[cam] = (npz, res, camera)
        print(f"[{task} ep{episode} {cam}] maps={res.maps.shape} stats={res.stats}")

        if do_viz:
            frames = read_frames(task_dir, episode, CAM_KEYS[cam])
            if frames is None:
                print("  [warn] 无视频，跳过可视化")
                continue
            n = min(len(frames), res.maps.shape[0])
            blended = [overlay(frames[i], res.maps[i]) for i in range(n)]
            out_mp4 = npz.replace(".npz", "_overlay.mp4")
            write_video(blended, out_mp4, fps=20)
            print(f"  [viz] {out_mp4}")
    return frames_out, result


def build_with_dynamic_camera(frames_out, eef, cam, size, cfg):
    """wrist 相机外参逐帧变化时的 SIV 构建。"""
    maps, valids, depths, meta = [], [], [], []
    stats = {
        "frames": len(frames_out),
        "contacts": 0,
        "projected": 0,
        "depth_rejected": 0,
        "empty_frames": 0,
    }
    for i, f in enumerate(frames_out):
        camera = build_wrist_camera(cam, eef[i], size)
        v, vm, db, m = SIVBuilder(camera, cfg).build_frame(f)
        stats["contacts"] += len(f.contacts)
        stats["projected"] += m["accepted"]
        stats["depth_rejected"] += m["depth_rejected"]
        stats["empty_frames"] += int(m["accepted"] == 0)
        maps.append(v)
        valids.append(vm)
        depths.append(db)
        meta.append(m)
    return SIVResult(
        np.asarray(maps, np.float32),
        np.asarray(valids, np.uint8),
        np.asarray(depths, np.float32),
        meta,
        stats,
    )


def encode_latents(task_dir, episode, cams, result, vae_path, device, out_root=None):
    from c2s.latents import WanVAEEncoder, load_vae, save_mask_latent, siv_to_rgb

    vae = load_vae(vae_path, device=device)
    enc = WanVAEEncoder(vae)
    out = {}
    for cam in cams:
        npz, res, _ = result[cam]
        vd = read_video_latent(task_dir, episode, CAM_KEYS[cam])
        if vd is None:
            print(f"[{cam}] 无 video latent，跳过")
            continue
        fid = np.clip(np.asarray(list(vd["frame_ids"])), 0, res.maps.shape[0] - 1)
        rgb = siv_to_rgb(res.maps[fid])
        lat = enc.encode(rgb, device=device)
        Fl, Hl, Wl, _ = lat.shape
        assert (Fl, Hl, Wl) == (
            vd["latent_num_frames"],
            vd["latent_height"],
            vd["latent_width"],
        ), (
            f"[{cam}] SIV latent {(Fl, Hl, Wl)} != video {(vd['latent_num_frames'], vd['latent_height'], vd['latent_width'])}"
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
        # 默认写回数据集同目录（覆盖旧 mask）；--latent_out_dir 可改写到别处先验证
        root = out_root if out_root else task_dir
        dst = os.path.join(root, "latents", "chunk-000", MASK_KEYS[cam], base)
        save_mask_latent(dst, lat, vd)
        out[cam] = dst
        print(f"[{cam}] -> {dst} latent={tuple(lat.shape)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_dir", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--cams", nargs="*", default=["cam_high"])
    ap.add_argument("--out_dir", default="./_c2s_out")
    ap.add_argument("--state_key", default="action")
    ap.add_argument(
        "--contact_radius",
        type=float,
        default=0.02,
        help="真实接触半径 r_world (m)，深度自适应 sigma 用",
    )
    ap.add_argument(
        "--fixed_sigma",
        type=float,
        default=None,
        help="非 None 时关闭深度自适应（消融）",
    )
    ap.add_argument("--blend", choices=("max", "sum"), default="max")
    ap.add_argument("--depth_tolerance", type=float, default=None)
    ap.add_argument("--close_thresh", type=float, default=0.5)
    ap.add_argument("--release_window", type=int, default=5)
    ap.add_argument(
        "--dump_contacts",
        action="store_true",
        help="同时把接触点导出为 JSONL（供检查/复用）",
    )
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--encode", action="store_true", help="编码为 mask latent")
    ap.add_argument("--vae_path", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--latent_out_dir",
        default=None,
        help="mask latent 写入根目录；不给则写回 task_dir（覆盖已有 mask）",
    )
    args = ap.parse_args()

    cfg = SIVConfig(
        contact_radius=args.contact_radius,
        fixed_sigma=args.fixed_sigma,
        blend=args.blend,
        depth_tolerance=args.depth_tolerance,
    )
    ecfg = EEFContactConfig(
        close_thresh=args.close_thresh, release_window=args.release_window
    )
    os.makedirs(args.out_dir, exist_ok=True)

    eps = (
        args.episodes
        if args.episodes is not None
        else list_episodes(args.task_dir)[: args.limit]
    )
    for ep in eps:
        frames_out, result = process(
            args.task_dir,
            ep,
            args.cams,
            args.out_dir,
            cfg,
            ecfg,
            args.viz,
            args.state_key,
        )
        if args.dump_contacts:
            p = os.path.join(args.out_dir, f"ep{ep:06d}_contacts.jsonl")
            write_jsonl(frames_out, p)
            print(f"  [contacts] {p}")
        if args.encode:
            if not args.vae_path:
                from c2s.spec import WAN_VAE_PATH

                args.vae_path = WAN_VAE_PATH
            encode_latents(
                args.task_dir,
                ep,
                args.cams,
                result,
                args.vae_path,
                args.device,
                args.latent_out_dir,
            )


if __name__ == "__main__":
    main()
