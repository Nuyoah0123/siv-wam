"""相机几何：world -> camera -> pixel，以及深度自适应 sigma。

约定（与本地已验证的 wan_va/modules/spatial_action.py 一致）：
  * 世界 -> SAPIEN 相机：Rw2c = [forward, left, up]^T（列为相机轴在世界系下的方向）
  * SAPIEN 相机 -> OpenCV：S = [[0,-1,0],[0,0,-1],[1,0,0]]（x 右、y 下、z 前）
  * 因此 R_cw = S @ Rw2c，t_cw = -R_cw @ cam_pos
  * 若直接能拿到 SAPIEN 相机对象，则用 get_intrinsic_matrix()（OpenCV K）与
    get_extrinsic_matrix()（OpenCV 3x4），跳过手工推导，避免坐标系写反。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec import CALIB_HEIGHT, CALIB_WIDTH, D435_FOVY_DEG, HEAD_CAMERA

# SAPIEN 相机轴（x=forward, y=left, z=up） -> OpenCV（x=right, y=down, z=forward）
S_SAPIEN_TO_CV = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


@dataclass
class Camera:
    """针孔相机，OpenCV 约定。内参定义在 calib_size，投影结果缩放到 image_size。"""

    name: str
    K: np.ndarray  # (3,3) OpenCV 内参，标定分辨率下
    R_cw: np.ndarray  # (3,3) world -> camera 旋转
    t_cw: np.ndarray  # (3,)  world -> camera 平移
    calib_size: tuple  # (width, height) 内参所在分辨率
    image_size: tuple  # (width, height) 输出 map 分辨率

    # ---------------- 构造 ---------------- #
    @classmethod
    def from_opencv(
        cls, name, K, T_cw, calib_size=(CALIB_WIDTH, CALIB_HEIGHT), image_size=None
    ):
        K = np.asarray(K, dtype=np.float64)
        if calib_size is None:
            # 从主点反推标定分辨率（SAPIEN D435 是 320x240，不是 640x480，写错会差一倍）
            calib_size = (round(K[0, 2] * 2), round(K[1, 2] * 2))
        T = np.asarray(T_cw, dtype=np.float64)
        if T.shape == (4, 4) or T.shape == (3, 4):
            R, t = T[:3, :3], T[:3, 3]
        elif T.shape == (3, 3):
            R, t = T, np.zeros(3)
        else:
            raise ValueError(f"T_cw must be 4x4 / 3x4 / 3x3, got {T.shape}")
        image_size = tuple(image_size) if image_size else tuple(calib_size)
        return cls(name, K, R, t, tuple(calib_size), image_size)

    @classmethod
    def from_sapien(cls, name, camera, image_size, calib_size=None):
        """用 SAPIEN 相机对象构造（camera: sapien.render.RenderCameraComponent）。"""
        K = np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64)
        T34 = np.asarray(camera.get_extrinsic_matrix(), dtype=np.float64)
        T = np.eye(4)
        T[:3, :4] = T34
        # SAPIEN 返回的内参基于自身渲染分辨率
        if calib_size is None:
            calib_size = (round(K[0, 2] * 2), round(K[1, 2] * 2))
        return cls.from_opencv(name, K, T, calib_size, image_size)

    @classmethod
    def from_robotwin_static(
        cls,
        name,
        position,
        forward,
        left,
        width=CALIB_WIDTH,
        height=CALIB_HEIGHT,
        fovy_deg=D435_FOVY_DEG,
        image_size=None,
    ):
        """用 RoboTwin config.yml 的 position/forward/left 构造静态相机。"""
        pos = np.asarray(position, dtype=np.float64)
        fwd = np.asarray(forward, dtype=np.float64)
        fwd = fwd / np.linalg.norm(fwd)
        lft = np.asarray(left, dtype=np.float64)
        lft = lft / np.linalg.norm(lft)
        up = np.cross(fwd, lft)
        Rw2c = np.stack([fwd, lft, up], axis=1).T
        R_cw = S_SAPIEN_TO_CV @ Rw2c
        t_cw = -R_cw @ pos
        fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
        K = np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
        return cls.from_opencv(
            name, K, np.hstack([R_cw, t_cw[:, None]]), (width, height), image_size
        )

    @classmethod
    def head_camera(cls, image_size=(CALIB_WIDTH, CALIB_HEIGHT)):
        return cls.from_robotwin_static(
            "cam_high", image_size=image_size, **HEAD_CAMERA
        )

    @classmethod
    def from_robotwin_wrist(cls, name, link_xyz, R_link, image_size, mount):
        """wrist 相机：由 fl_link6/fr_link6 位姿 + URDF 安装偏移得到。

        SAPIEN 相机朝向 = 自身帧 +x（forward），+y（left），+z（up）。

        ``link_xyz`` is the parent link origin. The camera joint is attached
        to fl_link6/fr_link6, not to the gripper tip.
        """
        from .spec import rpy_to_R  # 局部导入避免循环

        R_cam = R_link @ rpy_to_R(mount["rpy"])
        p_cam = np.asarray(link_xyz, dtype=np.float64) + R_link @ mount["xyz"]
        Rw2c = R_cam.T
        R_cw = S_SAPIEN_TO_CV @ Rw2c
        t_cw = -R_cw @ p_cam
        fy = (CALIB_HEIGHT / 2.0) / np.tan(np.deg2rad(D435_FOVY_DEG) / 2.0)
        K = np.array(
            [
                [fy, 0.0, CALIB_WIDTH / 2.0],
                [0.0, fy, CALIB_HEIGHT / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return cls.from_opencv(
            name,
            K,
            np.hstack([R_cw, t_cw[:, None]]),
            (CALIB_WIDTH, CALIB_HEIGHT),
            image_size,
        )

    # ---------------- 投影 ---------------- #
    @property
    def scale(self) -> tuple:
        return (
            self.image_size[0] / self.calib_size[0],
            self.image_size[1] / self.calib_size[1],
        )

    def to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """
        P_camera = R_cw P_world + t_cw
        R_cw: world → camera 的旋转矩阵
        [ 1. ,  0. ,  0. ],
        [ 0. , -0.8, -0.6],
        [ 0. ,  0.6, -0.8]
        t_cw: world → camera 的平移向量
        [0.032, 0.45 , 1.35 ]
        """
        P = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        return (self.R_cw @ P.T).T + self.t_cw

    def project(self, points_world: np.ndarray, check_bounds: bool = True):
        """
        check_bounds=True 表示检查投影点是否落在图像范围内。
        世界点 -> (像素[image_size 尺度], 深度, valid)。

        valid = 有限 && z>0 && (可选) 落在图像范围内。
        """
        P = np.asarray(points_world, dtype=np.float64)
        # 没点，填充，有点，转为[N,3]形状
        P = np.empty((0, 3)) if P.size == 0 else P.reshape(-1, 3)
        Xc = self.to_camera(P) # 世界坐标转相机坐标
        depth = Xc[:, 2] # 提取深度 --> z
        valid = np.isfinite(depth) & (depth > 1e-6) # 判断点是否有效：深度必须是有限值并且点必须在相机前方  
        pixels = np.full((len(P), 2), np.nan) # 初始化像素，先填 NaN,因为无效点不应该伪造一个像素坐标。后面只有有效点才会被写入真正的投影结果。
        if valid.any(): # 只要 valid 中至少有一个 True，就进入下面的投影计算
            """
            相机内参通常是
            K =
                [
                    [fx,  0, cx],
                    [ 0, fy, cy],
                    [ 0,  0,  1]
                ]
            Xc[valid].T =
                [
                 x,
                 y,
                 z
                ]
            3x3 @ 3x1 矩阵乘法之后得到
            [
                [fx*X + cx*Z],
                [fy*Y + cy*Z],
                [Z]
            ]
            u = (fx * X + cx * Z) / Z
              = fx * X / Z + cx
            v = (fy * Y + cy * Z) / Z
              = fy * Y / Z + cy
            """
            uv = (self.K @ Xc[valid].T).T[:, :2] / depth[valid, None] # 应用相机内参进行投影
            sx, sy = self.scale # 进行分辨率缩放
            uv[:, 0] *= sx
            uv[:, 1] *= sy
            pixels[valid] = uv
        valid &= np.isfinite(pixels).all(axis=1) # .all(axis=1) 表示每一行的两个坐标都必须有效
        if check_bounds:
            w, h = self.image_size
            valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
            valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        return pixels, depth, valid

    def focal_at_image_size(self) -> float:
        """输出分辨率下的等效焦距（用于深度自适应 sigma）。
        不是世界单位中的焦距，而是当前输出图像分辨率下的像素焦距。
        例如原始相机标定尺寸可能是：640 × 480
        但是 SIV map 输出尺寸是：320 × 256
        那么焦距也必须相应缩放。"""
        sx, sy = self.scale
        return float(self.K[0, 0]) * (sx + sy) / 2.0

    def sigma_px(
        self,
        depth,
        contact_radius: float,
        sigma_min: float = 2.0,
        sigma_max: float = 40.0,
    ):
        """sigma_px ∝ f * r_world / z（报告附录 D），带上下限。
        sigma_px = 当前图像焦距 × 现实接触半径 ÷ 相机深度
        根据接触点到相机的距离，计算这个接触点在图像上应该使用多大的 Gaussian 热图半径 sigma。
        核心公式：sigma_px = f · r_world / z
        - f：当前图像分辨率下的焦距，单位是像素
        - r_world：真实接触区域半径，单位是米
        - z：接触点到相机的深度，单位是米
        - sigma_px：Gaussian 核宽，单位是像素"""
        f = self.focal_at_image_size()
        z = np.asarray(depth, dtype=np.float64)
        """接触区域在现实世界中有一个固定大小：r_world
        但它投影到图像上之后，大小取决于距离：
        离相机越近 → 投影越大
        离相机越远 → 投影越小
        所以：
        z 越小 → sigma 越大
        z 越大 → sigma 越小"""
        sigma = f * contact_radius / np.maximum(z, 1e-6) # 针孔相机模型 sigma = f · r / z
        """裁剪避免热图太尖，几乎只有一个像素
        热图太大，覆盖整片区域
        极近或极远深度导致训练不稳定"""
        return np.clip(sigma, sigma_min, sigma_max)


def sanity_check_eef(camera: Camera, eef_row: np.ndarray, tip_len: float = 0.12):
    """报告附录 C 要求的 landmark sanity check：把夹爪尖投影到图像上。

    返回 (u, v, depth) 或 None（在相机后方/出画）。
    """
    from .spec import ARM_SLICES, eef_to_tip_pose

    out = {}
    for arm, sl in ARM_SLICES.items():
        xyz = eef_row[sl["xyz"][0] : sl["xyz"][1]] # eef_row[0:3]
        quat = eef_row[sl["quat"][0] : sl["quat"][1]] # eef_row[3,7]
        tip, _ = eef_to_tip_pose(xyz, quat, tip_len)
        px, depth, valid = camera.project(tip[None, :])
        out[arm] = (
            (float(px[0, 0]), float(px[0, 1]), float(depth[0])) if valid[0] else None
        )
    return out
