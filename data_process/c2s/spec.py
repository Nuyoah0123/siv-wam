"""本地环境常量：RoboTwin 2.0 / aloha-agilex / LeRobot 数据集约定。

这里的所有数值都来自本机已跑通的 WAM 代码与数据，不要凭文档猜：
  - 相机内参/外参：the RoboTwin aloha-agilex camera configuration
    （static_camera_list.head_camera, type=D435）
  - D435 fovy：RoboTwin/task_config/_camera_config.yml
  - 夹爪几何：同 config.yml（gripper_bias / gripper_scale）+ URDF finger joint limit，
    与 the corresponding spatial-action calibration implementation 完全一致（已多任务可视化验证）
  - 16D 动作布局：LeRobot meta/info.json features
  - wrist 相机安装位姿：URDF arx5_description_isaac.urdf 的 left/right_camera_joint
"""

from __future__ import annotations

import os

import numpy as np

# --------------------------------------------------------------------------- #
# 路径（本机实际位置）
# --------------------------------------------------------------------------- #
ROBOTWIN_LEROBOT_ROOT = os.environ.get(
    "SIV_WAM_DATASET_PATH", "./data/robotwin_lerobot"
)
ROBOTWIN_REPO = os.environ.get(
    "SIV_WAM_ROBOTWIN_ROOT", "./external/RoboTwin"
)
WAN_VAE_PATH = os.environ.get(
    "SIV_WAM_VAE_PATH", "./assets/pretrained/vae"
)

# --------------------------------------------------------------------------- #
# 相机
# --------------------------------------------------------------------------- #
# D435: fovy=37deg, 320x240; Large_D435: fovy=37deg, 640x480。
# 数据集里实际存的是 640x480（videos 与 latent 的 video_height/width 均按此缩放），
# 因此标定统一表达在 640x480 上，投影后按目标分辨率缩放像素坐标。
CALIB_WIDTH, CALIB_HEIGHT = 640, 480
D435_FOVY_DEG = 37.0

# head_camera（config.yml -> static_camera_list[0]）
HEAD_CAMERA = dict(
    position=np.array([-0.032, -0.45, 1.35], dtype=np.float64),
    forward=np.array([0.0, 0.6, -0.8], dtype=np.float64),
    left=np.array([-1.0, 0.0, 0.0], dtype=np.float64),
)

# wrist 相机：URDF left/right_camera_joint（parent fl_link6 / fr_link6）
#   origin xyz="0.07 0.032 0.065" rpy="0 0.4 0"
# 该外参**未经渲染验证**（本机 SAPIEN 无法渲染），使用 scripts/verify_projection.py
# 叠加真实 wrist 视频确认后再用于训练。
WRIST_CAMERA_MOUNT = dict(
    xyz=np.array([0.07, 0.032, 0.065], dtype=np.float64),
    rpy=np.array([0.0, 0.4, 0.0], dtype=np.float64),  # roll-pitch-yaw, URDF 顺序
)

# LeRobot 观测键 -> 相机名
CAM_KEYS = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}
MASK_KEYS = {name: f"observation.masks.{name}" for name in CAM_KEYS}

# 数据集中 latent 对应的渲染分辨率（从 video latent 的 video_height/width 读，
# 这里只作为无 latent 时的兜底）
DEFAULT_MAP_SIZE = {"cam_high": (256, 320), "cam_left_wrist": (128, 160),
                    "cam_right_wrist": (128, 160)}

# --------------------------------------------------------------------------- #
# 动作 / 末端位姿（16D）
# --------------------------------------------------------------------------- #
ARM_SLICES = {
    "left": dict(xyz=(0, 3), quat=(3, 7), gripper=7),
    "right": dict(xyz=(8, 11), quat=(11, 15), gripper=15),
}
# RoboTwin config.yml: action quaternion -> link rotation.
# RoboTwin exports R_action = R_link @ global_trans_matrix, so recover the
# link frame with R_action @ global_trans_matrix (right multiplication).
GLOBAL_TRANS = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
# gripper_bias：endpose 参考点相对 fl_link6 沿工具 +x 回退 0.12m
GRIPPER_TIP_LEN = 0.12
# gripper=0/1 时单指相对中心的横向偏移（URDF fl_joint7 limit 0.04765）
FINGER_BASE_OFF = 0.024
FINGER_OPEN_GAIN = 0.0476
# 接触垫位于指尖稍后方
FINGER_PAD_BACK = 0.02
GRIPPER_NAME_LINKS = ("fl_link7", "fl_link8", "fr_link7", "fr_link8")


def quat_wxyz_to_R(q) -> np.ndarray:
    """四元数(wxyz) -> 3x3 旋转矩阵，右乘 global_trans_matrix。纯 numpy。"""
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(-1)[:4]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        R = np.eye(3)
    else:
        s = 2.0 / n
        R = np.array([
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ])
    return R @ GLOBAL_TRANS


def eef_to_tip_pose(xyz, quat_wxyz, tip_len: float = GRIPPER_TIP_LEN):
    """endpose（腕部参考点）-> fl_link6 位姿。返回 (tip_xyz(3,), R(3,3))。
    根据机器人末端的当前位置和姿态，计算工具坐标系沿自身 +x 方向前进 tip_len 米之后的位置。
    p_target = p_eef + R · [tip_len, 0, 0]^T
    其中：
    - p_eef：末端参考点的位置
    - R：末端姿态对应的旋转矩阵
    - [tip_len, 0, 0]：工具坐标系中的局部偏移
    - p_target：世界坐标系中的目标点
    四元数 [w,x,y,z] → 3×3 旋转矩阵 R旋转矩阵可以表示：工具坐标系相对于世界坐标系的朝向。
    np.array([tip_len, 0.0, 0.0])
    np.asarray(xyz, dtype=np.float64)它表示工具坐标系中的一个局部向量：沿工具局部 +x 方向前进 tip_len
    但是这个向量还处在工具坐标系中，不能直接和世界坐标下的 xyz 相加。
    所以要先旋转：R @ np.array([tip_len, 0.0, 0.0])"""
    R = quat_wxyz_to_R(quat_wxyz)
    tip = np.asarray(xyz, dtype=np.float64) + R @ np.array([tip_len, 0.0, 0.0])
    return tip, R


def finger_half_open(gripper: float) -> float:
    # 根据夹爪开合值，计算每根手指相对中心的横向距离。
    return FINGER_BASE_OFF + float(np.clip(gripper, 0.0, 1.0)) * FINGER_OPEN_GAIN


def rpy_to_R(rpy) -> np.ndarray:
    """URDF rpy (roll=x, pitch=y, yaw=z) -> 3x3，右手系。"""
    rx, ry, rz = np.asarray(rpy, dtype=np.float64)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx
