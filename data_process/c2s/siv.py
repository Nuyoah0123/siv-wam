"""Contact -> SIV map。

链路（SIV-WAM 报告第 27 节 / 第 4 节）：
  接触点(world) -> world->camera -> 透视投影 -> 可见性/深度检查
  -> 深度自适应 Gaussian splatting -> [0,1] 归一化 -> 单视角 SIV map
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence  # noqa: UP035

import numpy as np

from .contacts import ContactFrame
from .geometry import Camera


@dataclass
class SIVConfig:
    contact_radius: float = 0.02  # r_world (m)，真实接触区域半径
    sigma_min: float = 2.0  # 输出网格上的 sigma 下限(px)
    sigma_max: float = 40.0  # 上限，防止近距离爆掉
    fixed_sigma: Optional[float] = None  # 非 None -> 固定 sigma（消融用）
    blend: str = "max"  # max | sum
    min_weight: float = 0.05
    depth_tolerance: Optional[float] = None  # 米；给深度图时做一致性过滤
    normalize: str = "max"  # max | none


def build_sequence(
    frames, camera_for_frame, cfg: SIVConfig = None, depth_for_frame=None) -> SIVResult:
    """逐帧外参会变时（wrist 相机）的 SIV 构建。

    camera_for_frame(frame_index) -> Camera
    depth_for_frame(frame) -> (H,W) 深度图或 None
    """
    builder = SIVBuilder(camera_for_frame(0), cfg or SIVConfig())
    maps, valids, depths, meta = [], [], [], []
    stats = {
        "frames": len(frames), "contacts": 0, "projected": 0, "depth_rejected": 0, "empty_frames": 0
    }
    for f in frames:
        builder.camera = camera_for_frame(f.frame)
        dm = depth_for_frame(f) if depth_for_frame else None
        value, vm, db, m = builder.build_frame(f, dm)
        stats["contacts"] += len(f.contacts)
        stats["projected"] += m["accepted"]
        stats["depth_rejected"] += m["depth_rejected"]
        stats["empty_frames"] += int(m["accepted"] == 0)
        maps.append(value)
        valids.append(vm)
        depths.append(db)
        meta.append(m)
    return SIVResult(
        np.asarray(maps, dtype=np.float32),
        np.asarray(valids, dtype=np.uint8),
        np.asarray(depths, dtype=np.float32),
        meta,
        stats,
    )


@dataclass
class SIVResult:
    maps: np.ndarray  # (T,H,W) float32, [0,1]
    valid: np.ndarray  # (T,H,W) uint8，有效投影点
    depths: np.ndarray  # (T,H,W) float32，稀疏深度缓冲（米）
    meta: List[dict]
    stats: dict


def _gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    """radius: 表示从中心向上下左右扩散多少个像素。
    sigma: 表示 Gaussian 的扩散程度：
        sigma 越小 → 热图越尖
        sigma 越大 → 热图越宽
    
    contact 3D point
        → project 到像素 (u,v)
        → 计算 sigma
        → 生成 Gaussian patch
        → 乘 contact weight
        → 放入整张 SIV map"""
    ax = np.arange(2 * radius + 1) - radius # 创建坐标轴
    xx, yy = np.meshgrid(ax, ax) # 创建二维坐标网格
    g = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma)) # 计算二维 Gaussian
    return (g / g.max()).astype(np.float32)


class SIVBuilder:
    def __init__(self, camera: Camera, config: Optional[SIVConfig] = None):
        self.camera = camera
        self.cfg = config or SIVConfig()

    # ---------------- 单帧 ---------------- #
    def build_frame(self, frame: ContactFrame, depth_map=None) -> tuple:
        """把一帧世界坐标下的 contact 点，投影到图像上，
        过滤掉无效或被遮挡的点，
        再把每个有效点扩散成 Gaussian SIV 热区。"""
        cfg = self.cfg
        cam = self.camera
        h, w = cam.image_size[1], cam.image_size[0] # NumPy 图像数组需要：(height, width)

        contacts = [c for c in frame.contacts if c.weight >= cfg.min_weight] # 过滤低权重 contact
        value = np.zeros((h, w), dtype=np.float32) # 最终的 SIV 热图。
        valid_map = np.zeros((h, w), dtype=np.uint8) # 离散标记图，表示：这个像素上存在一个有效的 3D contact 投影点。  
        depth_buf = np.zeros((h, w), dtype=np.float32) # 保存投影点的深度。
        if not contacts: # 没有 contact 的帧不能丢掉，必须保留为空图，这样时间序列才能和 RGB、动作对齐。
            return (
                value,
                valid_map,
                depth_buf,
                {
                    "frame": frame.frame,
                    "accepted": 0,
                    "depth_rejected": 0,
                    "pixels": [],
                    "depth": [],
                    "pairs": [],
                },
            )

        P = np.asarray([c.position for c in contacts], dtype=np.float64)
        weights = np.clip(
            np.asarray([c.weight for c in contacts], dtype=np.float64), 0, 1
        )
        pixels, depth, valid = cam.project(P) # 3D world point 投影到图像

        # 深度一致性：被其它表面遮挡的接触点投影剔除
        n_depth_rejected = 0
        if depth_map is not None and cfg.depth_tolerance is not None:
            dm = np.asarray(depth_map, dtype=np.float64)
            if dm.shape != (h, w):
                # 深度图是原始渲染分辨率（如 320x240），按最近邻重采样到 map 网格
                # 尺寸不一致，需要将 depth map 调整到 SIV map 的尺寸。
                # 为什么用最近邻？因为 depth 是几何测量值，不能像 RGB 一样随意进行颜色式插值，否则可能产生不真实的深度。
                ys = np.clip(
                    (np.arange(h) * dm.shape[0] / h).astype(int), 0, dm.shape[0] - 1
                )
                xs = np.clip(
                    (np.arange(w) * dm.shape[1] / w).astype(int), 0, dm.shape[1] - 1
                )
                dm = dm[np.ix_(ys, xs)]
            xi = np.clip(np.rint(pixels[valid, 0]).astype(int), 0, dm.shape[1] - 1)
            yi = np.clip(np.rint(pixels[valid, 1]).astype(int), 0, dm.shape[0] - 1)
            obs = dm[yi, xi]
            ok = (
                np.isfinite(obs)
                & (obs > 0)
                & (np.abs(obs - depth[valid]) <= cfg.depth_tolerance)
            )
            n_depth_rejected = int((~ok).sum())
            idx = np.flatnonzero(valid)
            valid[idx[~ok]] = False

        acc = int(valid.sum())
        sigmas = (
            np.full(acc, cfg.fixed_sigma)
            if cfg.fixed_sigma
            else cam.sigma_px(
                depth[valid], cfg.contact_radius, cfg.sigma_min, cfg.sigma_max
            )
        )
        for (u, v), s, wt in zip(pixels[valid], sigmas, weights[valid]):
            radius = max(1, int(np.ceil(3.0 * s))) # 确定 Gaussian patch 半径
            g = _gaussian_kernel(radius, float(s)) * float(wt) # 生成 Gaussian 核
            # 计算 patch 在图像中的范围
            x0, x1 = max(0, int(u) - radius), min(w, int(u) + radius + 1)
            y0, y1 = max(0, int(v) - radius), min(h, int(v) + radius + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            # 处理边界裁剪后的 Gaussian 偏移
            gx0, gy0 = x0 - (int(u) - radius), y0 - (int(v) - radius)
            patch = g[gy0 : gy0 + (y1 - y0), gx0 : gx0 + (x1 - x0)]
            if cfg.blend == "sum":
                # 融合 Gaussian patch。同一位置有多个接触贡献但数值可能变得很大，后面需要归一化。
                value[y0:y1, x0:x1] += patch
            else:
                # blend = max 表示多个 contact 重叠时，保留更高的交互价值
                value[y0:y1, x0:x1] = np.maximum(value[y0:y1, x0:x1], patch)

        if cfg.normalize == "max" and value.max() > 0:
            value /= value.max()
        value = np.clip(value, 0.0, 1.0)

        for (u, v), z in zip(pixels[valid], depth[valid]):
            yi, xi = int(round(v)), int(round(u))  # noqa: RUF046
            if 0 <= yi < h and 0 <= xi < w:
                valid_map[yi, xi] = 1
                depth_buf[yi, xi] = (
                    z if depth_buf[yi, xi] == 0 else min(depth_buf[yi, xi], z)
                )

        meta = {
            "frame": frame.frame,
            "accepted": acc,
            "depth_rejected": n_depth_rejected,
            "pixels": pixels[valid].tolist(),
            "depth": depth[valid].tolist(),
            "pairs": [contacts[i].pair for i in np.flatnonzero(valid)],
        }
        return value, valid_map, depth_buf, meta

    # ---------------- 序列 ---------------- #
    def build(
        self,
        frames: Sequence[ContactFrame],
        depth_loader=None,
        cam_name: Optional[str] = None,
    ) -> SIVResult:
        """cam_name 用于 f.depth_path 为 {cam: path} 字典时选取对应深度图。"""
        maps, valids, depths, meta = [], [], [], []
        stats = {
            "frames": len(frames),
            "contacts": 0,
            "projected": 0,
            "depth_rejected": 0,
            "empty_frames": 0,
        }
        for f in frames:
            stats["contacts"] += len(f.contacts)
            dm = None
            ref = f.depth_path
            if isinstance(ref, dict) and cam_name is not None:
                ref = ref.get(cam_name)
            if ref and depth_loader is not None:
                dm = depth_loader(ref)
            value, vm, db, m = self.build_frame(f, dm)
            if m["accepted"] == 0:
                stats["empty_frames"] += 1
            stats["projected"] += m["accepted"]
            stats["depth_rejected"] += m["depth_rejected"]
            maps.append(value)
            valids.append(vm)
            depths.append(db)
            meta.append(m)
        return SIVResult(
            maps=np.asarray(maps, dtype=np.float32),
            valid=np.asarray(valids, dtype=np.uint8),
            depths=np.asarray(depths, dtype=np.float32),
            meta=meta,
            stats=stats,
        )
