"""接触点来源。

两条路：
  1) RobotwinSimContactSource —— 接真实仿真器（RoboTwin/SAPIEN），
     用 scene.get_contacts() 取接触点与接触对，是报告里"Contact -> SIV"的正规来源。
     需要能渲染的 GPU 节点（本机 SAPIEN 起不来 Vulkan，见 README）。
  2) EEFContactEstimator —— 离线兜底：只用 LeRobot 里的 16D 末端位姿重建夹爪几何，
     按"夹爪闭合/释放"判定接触事件。无需仿真器，可在本机直接跑，
     但它是接触的*代理*，不是物理接触点。

两种产出的 ContactFrame 结构完全相同，下游 SIV 构建不区分来源。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence  # noqa: UP035

import numpy as np

from .spec import (
    ARM_SLICES,
    FINGER_PAD_BACK,
    GRIPPER_NAME_LINKS,
    eef_to_tip_pose,
    finger_half_open,
)

#: 机器人本体中"非夹爪"的连杆（含腕相机），出现在接触对任一侧即剔除。
#: 名字来自 aloha-agilex URDF：fl_link1..8 / fr_link1..8 / *_camera / camera_*
DEFAULT_ROBOT_EXCLUDE = (
    "left_camera", "right_camera", "camera_base_link", "camera_link1", "camera_link2",
    "base_link", "base", "fl_base_link", "fr_base_link",
    "fl_link1", "fl_link2", "fl_link3", "fl_link4", "fl_link5", "fl_link6",
    "fr_link1", "fr_link2", "fr_link3", "fr_link4", "fr_link5", "fr_link6",
)


@dataclass
class ContactPoint:
    position: List[float]          # world frame, 米  
    weight: float = 1.0            # [0,1] 接触强度
    pair: Optional[List[str]] = None   # 接触对，如 ["fl_link7", "object_0"] 
    kind: str = "grasp"            # grasp / place / support


@dataclass
class ContactFrame:
    frame: int
    contacts: List[ContactPoint] = field(default_factory=list)
    depth_path: Optional[str] = None   # 可选 .npy 深度图（仿真器渲染），米


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #
def frame_to_dict(f: ContactFrame) -> dict:
    return {
        "frame": f.frame,
        "contacts": [
            {k: v for k, v in asdict(c).items() if v is not None} for c in f.contacts
        ],
        **({"depth_path": f.depth_path} if f.depth_path else {}),
    }


def write_jsonl(frames: Iterable[ContactFrame], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for f in frames:
            fh.write(json.dumps(frame_to_dict(f), ensure_ascii=False) + "\n")


def read_jsonl(path) -> List[ContactFrame]:
    frames = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            frames.append(
                ContactFrame(
                    frame=int(d["frame"]),
                    contacts=[ContactPoint(**c) for c in d.get("contacts", [])],
                    depth_path=d.get("depth_path"),
                )
            )
    return frames


# --------------------------------------------------------------------------- #
# 1) 仿真器接触源
# --------------------------------------------------------------------------- #
class RobotwinSimContactSource:
    """从 RoboTwin/SAPIEN 场景抽取接触点（world frame）。

    RoboTwin 的 envs/_base_task.py::get_gripper_actor_contact_position 已经示范了
    正确的取法：scene.get_contacts() -> contact.bodies[i].entity.name / contact.points。
    这里把同样的逻辑泛化成"逐帧导出 JSONL"。

    实测（lift_pot）发现三类必须剔除的"假接触"：
      - 两指互碰        fl_link7 ↔ fl_link8
      - 腕相机刚性安装   fl_link7 ↔ left_camera（每帧都在，113/113）
      - 其它本体连杆     fl_link1..6 / base 等
    规则：只有"夹爪 ↔ 非本体"才是任务接触。`include_object_pairs=True` 时另外保留
    不涉及夹爪的物体间接触（place 可行性），但会引入物体自接触噪声，默认关闭。

    Args:
        gripper_names: 夹爪 entity 名集合。
        exclude_names: 机器人本体其它连杆名，出现即剔除（含腕相机）。
        max_points: 单个 contact 最多取多少个点（接触流形可能上百点）。
        sample: "random" 或 "all"。
        include_object_pairs: 是否保留不涉及夹爪的物体间接触。
    """

    def __init__(self, gripper_names: Sequence[str] = GRIPPER_NAME_LINKS,
                 exclude_names: Optional[Sequence[str]] = None,
                 max_points: int = 32, sample: str = "random",
                 include_object_pairs: bool = False):
        self.gripper_names = set(gripper_names)
        self.exclude = set(exclude_names) if exclude_names is not None else set(DEFAULT_ROBOT_EXCLUDE)
        self.max_points = int(max_points)
        self.sample = sample
        self.include_object_pairs = include_object_pairs
        self.rng = np.random.default_rng(0)

    @staticmethod
    def _entity(collision_body) -> str:
        entity = getattr(collision_body, "entity", None)
        return getattr(entity, "name", str(collision_body))

    def extract(self, scene, weight_by_count: bool = True) -> List[ContactPoint]:
        contacts = scene.get_contacts()
        out: List[ContactPoint] = []
        for c in contacts:
            bodies = list(c.bodies)
            if len(bodies) < 2:
                continue
            names = [self._entity(b) for b in bodies]
            gripper_side = [i for i, n in enumerate(names) if n in self.gripper_names]
            if gripper_side:
                # 夹爪参与：另一侧必须不是本体连杆，也不是另一个夹爪（两指互碰）
                other = names[1 - gripper_side[0]]
                if other in self.exclude or other in self.gripper_names:
                    continue
                kind = "grasp"
            elif self.include_object_pairs:
                if any(n in self.exclude or n in self.gripper_names for n in names):
                    continue
                kind = "support"
            else:
                continue
            pts = np.asarray([np.asarray(p.position, dtype=np.float64)
                              for p in c.points])
            if pts.size == 0:
                continue
            if len(pts) > self.max_points:
                if self.sample == "random":
                    idx = self.rng.choice(len(pts), self.max_points, replace=False)
                else:
                    idx = np.linspace(0, len(pts) - 1, self.max_points).astype(int)
                pts = pts[idx]
            # 接触点数越多 -> 接触面积越大 -> 权重越高（粗粒度力代理）
            w = 1.0
            if weight_by_count:
                w = float(np.clip(len(pts) / self.max_points, 0.2, 1.0))
            pair = names
            for p in pts:
                out.append(ContactPoint(position=p.tolist(), weight=w,
                                        pair=pair, kind=kind))
        return out

    def frame(self, scene, index: int, depth_path: Optional[str] = None) -> ContactFrame:
        return ContactFrame(frame=index, contacts=self.extract(scene),
                            depth_path=depth_path)


# --------------------------------------------------------------------------- #
# 2) 离线兜底：末端位姿 -> 接触事件
# --------------------------------------------------------------------------- #
@dataclass
class EEFContactConfig:
    close_thresh: float = 0.5      # gripper < 该值视为闭合（接触中）
    release_window: int = 5        # 释放后仍标记 placement 接触的帧数
    place_weight: float = 1.0
    min_weight: float = 0.05
    tip_len: float = 0.12
    pad_back: float = FINGER_PAD_BACK


class EEFContactEstimator:
    """从 (T,16) 末端绝对位姿推断接触事件。

    - grasp：gripper 闭合期间，接触点取两指垫（finger_L/R 往回 pad_back）。
      权重 = (close_thresh - g) / close_thresh，夹得越紧权重越高。
    - place：闭合 -> 张开的那一帧起 release_window 帧内，在指尖（物体落点附近）
      标记一个接触点，权重线性衰减。

    注意：这是**接触代理**，不是物理接触点。真实接触请用 RobotwinSimContactSource。
    """

    def __init__(self, config: Optional[EEFContactConfig] = None):
        self.cfg = config or EEFContactConfig()

    def __call__(self, eef: np.ndarray) -> List[ContactFrame]:
        eef = np.asarray(eef, dtype=np.float64)
        T = eef.shape[0]
        cfg = self.cfg
        closed = {}
        for arm, sl in ARM_SLICES.items():
            g = np.clip(eef[:, sl["gripper"]], 0.0, 1.0) # 0 -> 完全闭合， 1 -> 完全打开
            closed[arm] = g < cfg.close_thresh

        # 释放帧：上一帧闭合、本帧张开
        release_at = {arm: {} for arm in ARM_SLICES}
        for arm in ARM_SLICES:
            for t in range(1, T):
                if closed[arm][t - 1] and not closed[arm][t]: # 上一帧闭合，且当前帧打开
                    release_at[arm][t] = 1.0 # 记录：哪些帧发生了“上一帧闭合、当前帧打开”。

        frames: List[ContactFrame] = [] # 逐帧构造 contact
        for t in range(T):
            pts: List[ContactPoint] = [] # 当前帧所有接触点都会放到这里。
            for arm, sl in ARM_SLICES.items():
                g = float(np.clip(eef[t, sl["gripper"]], 0.0, 1.0))
                xyz = eef[t, sl["xyz"][0]:sl["xyz"][1]]
                quat = eef[t, sl["quat"][0]:sl["quat"][1]]
                tip, R = eef_to_tip_pose(xyz, quat, cfg.tip_len)
                half = finger_half_open(g) # 计算手指横向偏移

                if closed[arm][t]: # 如果当前夹爪处于闭合状态，就认为可能发生抓取。
                    w = float(np.clip((cfg.close_thresh - g) / cfg.close_thresh,
                                      cfg.min_weight, 1.0)) # 夹爪闭合程度对应的接触置信度。不是力传感器测得的真实接触力。
                    for sgn in (+1.0, -1.0): # 生成左右两个指垫接触点
                        """其中：tip_len - pad_back：沿工具前方，但向后退一点, sgn * half：分别得到左右两根手指, 0：不在第三个局部方向偏移
                        然后：R @ local_point 把工具坐标系中的位置转换到世界坐标系。
                        最后加上：xyz 得到世界坐标下的指垫位置。"""
                        pad = np.asarray(xyz, dtype=np.float64) + R @ np.array(
                            [cfg.tip_len - cfg.pad_back, sgn * half, 0.0])
                        """接触位置：pad
                        接触权重：w
                        接触双方：当前夹爪和被抓物体
                        接触类型：grasp"""
                        pts.append(ContactPoint(position=pad.tolist(), weight=w,
                                                pair=[f"{arm}_gripper", "held_object"],
                                                kind="grasp"))
                # placement：释放后若干帧
                for t0 in list(release_at[arm]):
                    if 0 <= t - t0 < cfg.release_window: # 判断是否处于释放窗口
                        """如果释放发生在第 80 帧，那么第 80 到 84 帧都会被认为可能发生放置：
                        80, 81, 82, 83, 84
                        为什么不只标记释放那一帧？
                        因为真实放置可能存在：
                        - 物体落下延迟
                        - 夹爪打开延迟
                        - 接触稳定延迟
                        - 动作数据和物理事件的时间偏差"""
                        decay = 1.0 - (t - t0) / cfg.release_window # 放置权重衰减
                        pts.append(ContactPoint(
                            position=tip.tolist(),
                            weight=float(cfg.place_weight * decay),
                            pair=["held_object", "support"], kind="place"))
            frames.append(ContactFrame(frame=t, contacts=pts))
        return frames
