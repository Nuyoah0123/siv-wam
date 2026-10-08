#!/usr/bin/env python3
"""RoboTwin 仿真器适配器：导出真实物理接触点（Contact -> SIV 的正规来源）。

需要在**能渲染的 GPU 节点**上运行（本机容器 SAPIEN 起不了 Vulkan，见 README）。
用 LeRobot parquet 里的 16D 末端绝对位姿 replay 轨迹，逐帧取 scene.get_contacts()
得到接触点与接触对，落盘为与离线路径完全同构的 JSONL。

  cd <RoboTwin 仓库根目录>
  python <Contact2SIV>/scripts/build_contacts_sim.py \
      --robotwin_root . \
      --task_dir <.../lerobot_robotwin_eef_aug_500/lift_pot-aloha-agilex_randomized_500-1000> \
      --task_name lift_pot --task_config demo_clean \
      --episodes 0 1 --out_dir ./_c2s_contacts --save_depth

输出：
  <out_dir>/cameras.json                      # 各相机 OpenCV K / world_to_camera
  <out_dir>/ep<id>_contacts.jsonl             # 每帧接触点(world frame)
  <out_dir>/depth/<cam>/frame_%06d.npy        # 可选，米
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2s.contacts import ContactFrame, RobotwinSimContactSource, write_jsonl
from c2s.geometry import Camera
from c2s.lerobot_io import read_episode
from c2s.seeds import load_seed_map, seed_for_episode


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
def build_env(robotwin_root: str, task_name: str, task_config: str,
              seed: int = 0, ep_num: int = 0):
    """复刻 script/collect_data.py 的参数组装顺序（缺 embodiment 参数会直接 KeyError）。"""
    sys.path.insert(0, robotwin_root)
    os.chdir(robotwin_root)
    import importlib

    import yaml
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    def get_embodiment_config(robot_file):
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r",
              encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    mod = importlib.import_module(f"envs.{task_name}")
    task = getattr(mod, task_name)()
    cfg_path = os.path.join(robotwin_root, "task_config", f"{task_config}.yml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    emb = args["embodiment"]
    if len(emb) == 1:
        args["left_robot_file"] = embodiment_types[emb[0]]["file_path"]
        args["right_robot_file"] = embodiment_types[emb[0]]["file_path"]
        args["dual_arm_embodied"] = True
    elif len(emb) == 3:
        args["left_robot_file"] = embodiment_types[emb[0]]["file_path"]
        args["right_robot_file"] = embodiment_types[emb[1]]["file_path"]
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment 应为长度 1 或 3 的列表")
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = str(emb[0]) if len(emb) == 1 else f"{emb[0]}+{emb[1]}"

    task.setup_demo(now_ep_num=ep_num, seed=seed, is_test=True, **args)
    return task


def _camera_dicts(task) -> dict:
    """直接用 SAPIEN 的 OpenCV 内参/外参，避免手工推导坐标系写反。

    calib_size=None：由主点推断真实渲染分辨率（D435=320x240，不是 640x480）。
    """
    out = {}
    for name, item in task.cameras.get_config().items():
        cam = Camera.from_opencv(name, item["intrinsic_cv"], item["extrinsic_cv"],
                                 calib_size=None)
        out[name] = dict(
            K=cam.K.tolist(),
            world_to_camera=np.hstack([cam.R_cw, cam.t_cw[:, None]]).tolist(),
            image_size=[int(cam.calib_size[0]), int(cam.calib_size[1])],
        )
    return out


def dump_cameras(task, out_dir: str) -> dict:
    """首帧静态标定（head/front 相机用这个就够）。"""
    os.makedirs(out_dir, exist_ok=True)
    out = _camera_dicts(task)
    with open(os.path.join(out_dir, "cameras.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def dump_cameras_frame(task, frame: int) -> dict:
    """逐帧标定：wrist 相机装在臂上，外参每帧都变，必须逐帧存。"""
    return {"frame": frame, "cameras": _camera_dicts(task)}


def dump_depth(task, out_dir: str, frame: int, episode: int = 0) -> dict:
    """RoboTwin 的 get_depth 返回毫米，这里转成米后存 npy。

    必须先 update_picture() 再取图，否则 get_picture("Color") 报
    IndexError: _Map_base::at（渲染缓冲还没生成）。
    """
    task.cameras.update_picture()
    res = {}
    for name, item in task.cameras.get_depth().items():
        d = np.asarray(item["depth"], dtype=np.float32) / 1000.0
        # 带 episode 前缀，否则多 episode 会互相覆盖
        p = os.path.join(out_dir, "depth", f"ep{episode:06d}", name)
        os.makedirs(p, exist_ok=True)
        fn = os.path.join(p, f"frame_{frame:06d}.npy")
        np.save(fn, d)
        res[name] = fn
    return res


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_episode(task, eef: np.ndarray, source: RobotwinSimContactSource,
                out_dir: str, save_depth: bool = False,
                max_steps: int = 0, dump_cameras_every: int = 0,
                episode: int = 0) -> list:
    frames = []
    cam_fh = None
    if dump_cameras_every > 0:
        os.makedirs(out_dir, exist_ok=True)
        cam_fh = open(os.path.join(out_dir, f"cameras_ep{episode:06d}.jsonl"),
                      "w", encoding="utf-8")
    n = len(eef) if not max_steps else min(len(eef), max_steps)
    for t in range(n):
        task.take_action(eef[t], "ee")     # 16D: [L_xyz, L_quat, L_grip, R..., R_grip]
        depths = dump_depth(task, out_dir, t, episode) if save_depth else None
        if cam_fh is not None and t % dump_cameras_every == 0:
            cam_fh.write(json.dumps(dump_cameras_frame(task, t), ensure_ascii=False) + "\n")
        frames.append(ContactFrame(frame=t, contacts=source.extract(task.scene),
                                   depth_path=depths))
    if cam_fh is not None:
        cam_fh.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robotwin_root", required=True, help="RoboTwin 仓库根目录")
    ap.add_argument("--task_dir", required=True, help="LeRobot task 目录（提供 replay 动作）")
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--episodes", type=int, nargs="*", default=[0])
    ap.add_argument("--out_dir", default="./_c2s_contacts")
    ap.add_argument("--gripper_names", nargs="*",
                    default=["fl_link7", "fl_link8", "fr_link7", "fr_link8"])
    ap.add_argument("--exclude_names", nargs="*", default=None,
                    help="机器人本体非夹爪连杆，默认见 c2s.contacts.DEFAULT_ROBOT_EXCLUDE")
    ap.add_argument("--include_object_pairs", action="store_true",
                    help="额外保留不涉及夹爪的物体间接触(place 可行性)，可能含自接触噪声")
    ap.add_argument("--max_points", type=int, default=32)
    ap.add_argument("--save_depth", action="store_true")
    ap.add_argument("--dump_cameras_every", type=int, default=1,
                    help="每 N 帧写一条 cameras.jsonl（wrist 相机外参逐帧变，默认每帧）")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed_file", default=None,
                    help="seed.txt or JSON episode->seed mapping; required for faithful replay")
    ap.add_argument("--allow_episode_seed", action="store_true",
                    help="Explicitly allow the unsafe seed=episode fallback")
    ap.add_argument("--denoiser", choices=("none", "oidn", "optix"), default="none")
    args = ap.parse_args()

    seed_map = load_seed_map(args.seed_file) if args.seed_file else None
    if seed_map is None and not args.allow_episode_seed:
        ap.error("Formal replay requires --seed_file; use --allow_episode_seed only for debugging")
    import sapien
    sapien.render.set_ray_tracing_denoiser(args.denoiser)

    source = RobotwinSimContactSource(args.gripper_names, args.exclude_names,
                                      args.max_points,
                                      include_object_pairs=args.include_object_pairs)
    os.makedirs(args.out_dir, exist_ok=True)

    for ep in args.episodes:
        eef = read_episode(args.task_dir, ep)
        seed = seed_for_episode(seed_map, ep) if seed_map is not None else ep
        task = build_env(args.robotwin_root, args.task_name, args.task_config,
                         seed=seed, ep_num=ep)
        import sapien
        sapien.render.set_ray_tracing_denoiser(args.denoiser)
        cams = dump_cameras(task, args.out_dir)
        print(f"[env] {args.task_name} ep{ep} seed={seed}: cameras={list(cams)} steps={len(eef)}")
        frames = run_episode(task, eef, source, args.out_dir, args.save_depth,
                             args.max_steps, args.dump_cameras_every, ep)
        out = os.path.join(args.out_dir, f"ep{ep:06d}_contacts.jsonl")
        write_jsonl(frames, out)
        n_pts = sum(len(f.contacts) for f in frames)
        n_nonempty = sum(1 for f in frames if f.contacts)
        print(f"[contacts] {out} frames={len(frames)} points={n_pts} "
              f"nonempty_frames={n_nonempty}")


if __name__ == "__main__":
    main()
