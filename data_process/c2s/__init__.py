"""Contact2SIV：RoboTwin 接触点 -> 空间交互价值图(SIV map) -> 训练用 latent。

接地于本机环境（RoboTwin 2.0 / aloha-agilex / LeRobot 数据集 / Wan2.2 VAE）。
"""

from .geometry import Camera
from .contacts import (
    ContactFrame,
    ContactPoint,
    EEFContactEstimator,
    EEFContactConfig,
    RobotwinSimContactSource,
    read_jsonl,
    write_jsonl,
)
from .siv import SIVBuilder, SIVConfig, SIVResult

__all__ = [
    "Camera", "ContactFrame", "ContactPoint", "EEFContactEstimator",
    "EEFContactConfig", "RobotwinSimContactSource", "read_jsonl", "write_jsonl",
    "SIVBuilder", "SIVConfig", "SIVResult",
]
