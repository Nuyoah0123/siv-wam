"""可视化：把 SIV map 叠加到真实视频帧上（报告附录 D 要求的人工检查）。"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np


def overlay(frame_rgb: np.ndarray, siv: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    heat = cv2.resize(siv, (W, H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    cm = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8),
                           cv2.COLORMAP_JET).astype(np.float32)
    m = np.clip(heat, 0, 1)[..., None]
    return (img * (1 - m * alpha) + cm * (m * alpha)).clip(0, 255).astype(np.uint8)


def overlay_points(img_bgr: np.ndarray, pixels: Sequence, color=(0, 255, 255),
                   radius: int = 4, scale: tuple = (1.0, 1.0)) -> np.ndarray:
    """pixels 为 map 网格坐标，scale=(sx, sy) 把它放大到图像分辨率。"""
    sx, sy = scale
    for p in pixels:
        u, v = float(p[0]) * sx, float(p[1]) * sy
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        cv2.circle(img_bgr, (int(round(u)), int(round(v))), radius, color, -1)
    return img_bgr


def write_video(frames_bgr: Sequence[np.ndarray], out_path: str, fps: int = 20) -> str:
    H, W = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames_bgr:
        vw.write(f)
    vw.release()
    return out_path


def peak_pixel(siv: np.ndarray) -> Optional[tuple]:
    if siv.max() <= 0:
        return None
    v, u = np.unravel_index(np.argmax(siv), siv.shape)
    return float(u), float(v)
