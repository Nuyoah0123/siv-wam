#!/usr/bin/env python3
"""通用入口：contacts.jsonl + cameras.json -> 各视角 SIV map。

既吃仿真器导出的真实接触（build_contacts_sim.py 的产物），也吃离线 EEF 接触，
两者 JSONL 结构相同。

  python scripts/build_siv_from_contacts.py \
      --contacts ./_c2s_contacts/ep000000_contacts.jsonl \
      --cameras  ./_c2s_contacts/cameras.json \
      --cams head_camera --image_size 256x320 \
      --out_dir ./_c2s_out --viz_frames ./frames_dir
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2s.contacts import read_jsonl
from c2s.geometry import Camera
from c2s.lerobot_io import save_maps_npz
from c2s.siv import SIVBuilder, SIVConfig, build_sequence
from c2s.viz import overlay, write_video


def load_cameras(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_per_frame_cameras(path: str) -> dict:
    """cameras.jsonl（build_contacts_sim.py --dump_cameras_every）-> {frame: {cam: cfg}}。

    wrist 相机装在臂上，外参逐帧变化，静态 cameras.json 只能覆盖 head/front。
    """
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[int(d["frame"])] = d["cameras"]
    return out


def make_camera(name: str, item: dict, image_size=None) -> Camera:
    K = np.asarray(item["K"], dtype=np.float64)
    T = np.asarray(item["world_to_camera"], dtype=np.float64)
    calib = tuple(item.get("image_size", [int(K[0, 2] * 2), int(K[1, 2] * 2)]))
    return Camera.from_opencv(name, K, T, calib, image_size or calib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contacts", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--cams", nargs="*", default=None, help="默认处理 cameras.json 全部")
    ap.add_argument("--image_size", default=None, help="HxW，输出 map 分辨率")
    ap.add_argument("--out_dir", default="./_c2s_out")
    ap.add_argument("--contact_radius", type=float, default=0.02)
    ap.add_argument("--fixed_sigma", type=float, default=None)
    ap.add_argument("--blend", choices=("max", "sum"), default="max")
    ap.add_argument("--depth_tolerance", type=float, default=None,
                    help="给 npy 深度时启用遮挡过滤（米）")
    ap.add_argument("--per_frame_cameras", default=None,
                    help="cameras.jsonl；wrist 相机外参逐帧变，必须给")
    ap.add_argument("--viz_frames_dir", default=None, help="叠加可视化的 RGB 帧目录")
    ap.add_argument("--encode_task_dir", default=None,
                    help="给 LeRobot task 目录 + --encode_episode 时，把 SIV map 编成 observation.masks.* latent")
    ap.add_argument("--encode_episode", type=int, default=None)
    ap.add_argument("--encode_cams", nargs="*", default=None,
                    help="cameras.json 相机名 -> LeRobot cam 名映射之外的手动指定，默认 head_camera=cam_high")
    ap.add_argument("--latent_out_dir", default=None, help="latent 写出根目录，默认写回 task_dir")
    args = ap.parse_args()

    frames = read_jsonl(args.contacts)
    cams_meta = load_cameras(args.cameras)
    names = args.cams or list(cams_meta)
    os.makedirs(args.out_dir, exist_ok=True)

    image_size = None
    if args.image_size:
        h, w = (int(x) for x in args.image_size.lower().split("x"))
        image_size = (w, h)

    def depth_loader(path):
        return np.load(path)

    cfg = SIVConfig(contact_radius=args.contact_radius,
                    fixed_sigma=args.fixed_sigma, blend=args.blend,
                    depth_tolerance=args.depth_tolerance)
    per_frame = None
    if args.per_frame_cameras:
        per_frame = load_per_frame_cameras(args.per_frame_cameras)
        print(f"[cams] 逐帧标定 {len(per_frame)} 帧：{args.per_frame_cameras}")

    report = {}
    for name in names:
        if per_frame is not None and any(name in c for c in per_frame.values()):
            # 逐帧外参（wrist）：每帧单独构造相机
            def cam_at(i, name=name):
                cfg_cam = per_frame.get(i, {}).get(name) or cams_meta[name]
                return make_camera(name, cfg_cam, image_size)

            def depth_at(f, name=name):
                if isinstance(f.depth_path, dict) and f.depth_path.get(name):
                    return depth_loader(f.depth_path[name])
                return None

            res = build_sequence(frames, cam_at, cfg, depth_at)
        else:
            camera = make_camera(name, cams_meta[name], image_size)
            res = SIVBuilder(camera, cfg).build(frames, depth_loader=depth_loader,
                                                cam_name=name)
        npz = os.path.join(args.out_dir, f"{name}_siv.npz")
        save_maps_npz(npz, res.maps, res.valid, res.depths, res.meta, res.stats)
        report[name] = res.stats
        print(f"[{name}] {npz} maps={res.maps.shape} stats={res.stats}")
        if args.viz_frames_dir:
            import glob
            from PIL import Image
            files = sorted(glob.glob(os.path.join(args.viz_frames_dir, "*.png")))
            n = min(len(files), res.maps.shape[0])
            if n:
                imgs = [overlay(np.asarray(Image.open(f).convert("RGB")), res.maps[i])
                        for i, f in enumerate(files[:n])]
                write_video(imgs, npz.replace(".npz", "_overlay.mp4"))
    with open(os.path.join(args.out_dir, "siv_stats.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.encode_task_dir and args.encode_episode is not None:
        from c2s.latents import WanVAEEncoder, load_vae, save_mask_latent, siv_to_rgb
        from c2s.lerobot_io import read_video_latent
        from c2s.spec import CAM_KEYS, MASK_KEYS, WAN_VAE_PATH
        import glob
        import torch

        name_to_cam = {"head_camera": "cam_high", "left_camera": "cam_left_wrist",
                       "right_camera": "cam_right_wrist"}
        cams = args.encode_cams or ["head_camera"]
        enc = WanVAEEncoder(load_vae(WAN_VAE_PATH))
        for name in cams:
            npz = os.path.join(args.out_dir, f"{name}_siv.npz")
            maps = np.load(npz, allow_pickle=True)["maps"]
            cam = name_to_cam.get(name, name)
            vd = read_video_latent(args.encode_task_dir, args.encode_episode, CAM_KEYS[cam])
            if vd is None:
                print(f"[{cam}] 无 video latent，跳过")
                continue
            fid = np.clip(np.asarray(list(vd["frame_ids"])), 0, maps.shape[0] - 1)
            lat = enc.encode(siv_to_rgb(maps[fid]))
            base = os.path.basename(sorted(glob.glob(os.path.join(
                args.encode_task_dir, "latents", "chunk-000", CAM_KEYS[cam],
                f"episode_{args.encode_episode:06d}_*.pth")))[0])
            root = args.latent_out_dir or args.encode_task_dir
            dst = os.path.join(root, "latents", "chunk-000", MASK_KEYS[cam], base)
            save_mask_latent(dst, lat, vd)
            print(f"[{cam}] -> {dst} latent={tuple(lat.shape)}")


if __name__ == "__main__":
    main()
