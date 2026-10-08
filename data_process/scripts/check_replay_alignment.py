#!/usr/bin/env python3
"""校验 replay 出来的场景和原数据集视频是不是同一集（seed 保真度检查）。

LeRobot 导出里没有 RoboTwin 的 seed.txt，batch_export.py 用 seed=episode_index 重建场景。
本脚本渲染 replay 的首帧 head_camera，与原视频首帧比 MSE / 直方图相关，判断是否同一场景。

用法（GPU 节点）：
  cd <RoboTwin 仓库>
  python scripts/check_replay_alignment.py \
    --robotwin_root . --task_dir <TASK> --episodes 0 72 96 150
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2s.lerobot_io import read_frames
from scripts.batch_export import build_env
from c2s.seeds import load_seed_map, seed_for_episode


def compare(img_a: np.ndarray, img_b: np.ndarray) -> dict:
    """注意：仿真渲染是 320x240，数据集视频是 640x480，必须 resize 而不是裁剪，
    否则比的是真实帧的左上角，相关性必然接近 0。"""
    import cv2
    a = np.asarray(img_a, dtype=np.float32)[:, :, :3]
    b = np.asarray(img_b, dtype=np.float32)[:, :, :3]
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    a = a.reshape(-1, 3)
    b = b.reshape(-1, 3)
    mse = float(np.mean((a - b) ** 2))
    corr = float(np.corrcoef(a.mean(1), b.mean(1))[0, 1])
    return dict(mse=round(mse, 2), gray_corr=round(corr, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robotwin_root", required=True)
    ap.add_argument("--task_dir", required=True)
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--cross", type=int, nargs="*", default=None,
                    help="对照：把第 1 个 episode 的 replay 首帧与这些 episode 的真实首帧"
                         "逐一比较（本集应明显优于其它集）")
    ap.add_argument("--seed_file", default=None,
                    help="seed.txt or JSON episode->seed mapping")
    ap.add_argument("--allow_episode_seed", action="store_true",
                    help="Explicitly allow the unsafe seed=episode fallback")
    ap.add_argument("--denoiser", choices=("none", "oidn", "optix"), default="none")
    args = ap.parse_args()

    seed_map = load_seed_map(args.seed_file) if args.seed_file else None
    if seed_map is None and not args.allow_episode_seed:
        ap.error("Replay alignment requires --seed_file")
    import sapien
    sapien.render.set_ray_tracing_denoiser(args.denoiser)

    import re
    task_name = re.sub(r"-(aloha-agilex|aloha|piper|franka-panda|ur5-wsg|ARX-X5).*$", "",
                       os.path.basename(args.task_dir.rstrip("/")))
    for ep in args.episodes:
        task = None
        try:
            seed = seed_for_episode(seed_map, ep) if seed_map is not None else ep
            task = build_env(args.robotwin_root, task_name, args.task_config,
                             seed=seed, ep_num=ep)
            import sapien
            sapien.render.set_ray_tracing_denoiser(args.denoiser)
            # Use RoboTwin's canonical observation path so renderer and camera
            # poses are refreshed before reading the head RGB frame.
            obs = task.get_obs()
            sim = np.asarray(obs["observation"]["head_camera"]["rgb"])[:, :, :3]
            real = read_frames(args.task_dir, ep,
                               "observation.images.cam_high")[0]
            m = compare(sim, real)
            verdict = "同一场景" if m["gray_corr"] > 0.6 and m["mse"] < 2000 else "**可能不同**"
            print(f"ep{ep} seed={seed}: mse={m['mse']} gray_corr={m['gray_corr']} -> {verdict}")
            if args.cross and ep == args.episodes[0]:
                print("  --cross 对照（同一 replay 帧 vs 各集真实首帧）--")
                for other in args.cross:
                    try:
                        r = read_frames(args.task_dir, other,
                                        "observation.images.cam_high")[0]
                        mm = compare(sim, r)
                        print(f"  vs ep{other}: mse={mm['mse']} corr={mm['gray_corr']}")
                    except Exception as e:               # noqa: BLE001
                        print(f"  vs ep{other}: ERR {e}")
        except Exception as e:                       # noqa: BLE001
            print(f"ep{ep}: ENV FAIL {type(e).__name__}: {e}")
        finally:
            del task
    return


if __name__ == "__main__":
    main()
