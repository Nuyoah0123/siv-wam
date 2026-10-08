#!/usr/bin/env python3
"""投影 sanity check（报告附录 C 强制要求）。

把末端执行器关键点投影到真实视频帧上，确认几何没有写反：
  - cam_high：应落在可见的夹爪根部；
  - wrist：若外参(URDF 安装偏移)正确，夹爪应出现在画面中下方。

用法：
  python scripts/verify_projection.py --task_dir <TASK> --episode 0 \
      --out_dir ./_c2s_out --cams cam_high cam_left_wrist --num 8
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2s.geometry import Camera, sanity_check_eef
from c2s.lerobot_io import build_wrist_camera, read_episode, read_frames
from c2s.spec import CAM_KEYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_dir", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--cams", nargs="*", default=["cam_high"])
    ap.add_argument("--out_dir", default="./_c2s_out/verify")
    ap.add_argument("--num", type=int, default=8, help="均匀采样多少帧")
    ap.add_argument("--state_key", default="action")
    args = ap.parse_args()

    eef = read_episode(args.task_dir, args.episode, args.state_key)
    T = eef.shape[0]
    idxs = np.linspace(0, T - 1, min(args.num, T)).astype(int) # 均匀采样8帧
    os.makedirs(args.out_dir, exist_ok=True)

    for cam in args.cams:
        frames = read_frames(args.task_dir, args.episode, CAM_KEYS[cam])
        if frames is None:
            print(f"[{cam}] 无视频，跳过")
            continue
        H, W = frames[0].shape[:2]
        out_rows = []
        for t in idxs:
            if cam == "cam_high":
                camera = Camera.head_camera(image_size=(W, H))
            else:
                camera = build_wrist_camera(cam, eef[t], (H, W))
            res = sanity_check_eef(camera, eef[t])
            img = frames[t][:, :, ::-1].copy()   # RGB -> BGR
            for arm, p in res.items():
                if p is None:
                    out_rows.append((t, arm, None, None, None))
                    continue
                u, v, z = p
                col = (0, 255, 255) if arm == "left" else (0, 165, 255)
                cv2.circle(img, (int(round(u)), int(round(v))), 5, col, -1)  # noqa: RUF046
                out_rows.append((t, arm, round(u, 1), round(v, 1), round(z, 3)))
            cv2.imwrite(os.path.join(args.out_dir, f"{cam}_f{t:04d}.png"), img)
        print(f"[{cam}] 图像 {W}x{H}，投影结果 (frame, arm, u, v, depth_m):")
        for r in out_rows:
            print("   ", r)


if __name__ == "__main__":
    main()
