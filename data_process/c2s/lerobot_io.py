"""读写本机 RoboTwin LeRobot 数据集（lerobot v2.1 结构）。

task_dir/
  data/chunk-000/episode_XXXXXX.parquet      # 16D action / observation.state
  videos/chunk-000/<video_key>/episode_XXXXXX.mp4
  latents/chunk-000/observation.images.<cam>/episode_XXXXXX_<s>_<e>.pth
  meta/{info.json,episodes.jsonl}
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Optional, Tuple  # noqa: UP035

import numpy as np
import pandas as pd
import torch

from .geometry import Camera
from .spec import ARM_SLICES, CAM_KEYS, WRIST_CAMERA_MOUNT, eef_to_tip_pose


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def list_episodes(task_dir: str) -> list[int]:
    files = sorted(
        glob.glob(os.path.join(task_dir, "data", "chunk-*", "episode_*.parquet"))
    )
    return [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files]


def read_episode(task_dir: str, episode: int, state_key: str = "action") -> np.ndarray:
    """
    返回 (T,16) 末端绝对位姿。默认用 action（与 extract_mask_latents 一致）。
    T = episode 的时间帧数，16 = 双臂末端执行器状态维度
    Parquet 是高效的列式数据存储格式，专为大数据分析和快速读写场景设计。相比 CSV，它们具有更好的压缩比和查询性能。
    [左臂 xyz, 左臂 quaternion, 左夹爪,
     右臂 xyz, 右臂 quaternion, 右夹爪]
    df: 143 rows x 7 columns
    ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
    np.stack 按行堆叠 (T,16)，原始的df[state_key]-->143 rows x 1 columns，一列的list长16，16个动作维度
    """
    pq = os.path.join(task_dir, "data", "chunk-000", f"episode_{episode:06d}.parquet")
    df = pd.read_parquet(pq)
    return np.stack(df[state_key].to_numpy()).astype(np.float64)


def read_frames(task_dir: str, episode: int, video_key: str) -> np.ndarray | None:
    import av

    vid = os.path.join(
        task_dir, "videos", "chunk-000", video_key, f"episode_{episode:06d}.mp4"
    )
    if not os.path.exists(vid):
        return None
    frames = [f.to_ndarray(format="rgb24") for f in av.open(vid).decode(video=0)]
    return np.asarray(frames)


def read_video_latent(task_dir: str, episode: int, video_key: str) -> Optional[Dict]:
    d = os.path.join(task_dir, "latents", "chunk-000", video_key)
    hits = sorted(glob.glob(os.path.join(d, f"episode_{episode:06d}_*.pth")))
    if not hits:
        return None
    return torch.load(hits[0], weights_only=False)


def map_size_for(
    task_dir: str, episode: int, cam: str, fallback: Tuple[int, int]
) -> tuple[int, int]:
    """优先用 video latent 记录的渲染分辨率 (height, width)。"""
    vd = read_video_latent(task_dir, episode, CAM_KEYS[cam])
    if vd is None:
        return fallback
    return int(vd["video_height"]), int(vd["video_width"])


# --------------------------------------------------------------------------- #
# 相机
# --------------------------------------------------------------------------- #
def build_cameras(image_sizes: Dict[str, Tuple[int, int]]) -> Dict[str, Camera]:
    """cam_high 用固定标定；wrist 相机需要逐帧位姿，见 build_wrist_camera。"""
    cams = {
        "cam_high": Camera.head_camera(
            image_size=(image_sizes["cam_high"][1], image_sizes["cam_high"][0])
        )
    }
    return cams


def build_wrist_camera(
    name: str, eef_row: np.ndarray, image_size: Tuple[int, int]
) -> Camera:
    """由末端位姿 + URDF 安装偏移构造 wrist 相机（外参未经渲染验证）。"""
    arm = "left" if "left" in name else "right"
    sl = ARM_SLICES[arm]
    xyz = eef_row[sl["xyz"][0] : sl["xyz"][1]]
    quat = eef_row[sl["quat"][0] : sl["quat"][1]]
    # The wrist camera joint is attached to fl_link6/fr_link6. The EEF tip
    # offset is only for contact/landmark geometry and must not move the camera.
    _, R = eef_to_tip_pose(xyz, quat)
    return Camera.from_robotwin_wrist(
        name, xyz, R, (image_size[1], image_size[0]), WRIST_CAMERA_MOUNT
    )


# --------------------------------------------------------------------------- #
# 写出
# --------------------------------------------------------------------------- #
def save_maps_npz(path: str, maps, valid, depths, meta, stats) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path, maps=maps, valid=valid, depths=depths, meta=np.asarray(meta, dtype=object)
    )
    with open(str(path).replace(".npz", ".stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
