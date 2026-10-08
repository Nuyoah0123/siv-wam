"""批量生产 Contact→SIV 数据集（单趟：仿真 replay + SIV + VAE，一气呵成）。

和 export_siv_dataset.py 的区别：不再把接触点/深度写盘再读回来，而是
**逐帧在内存里直接算 SIV map**，避免每 episode 139MB 深度 npy（8500 ep ≈ 1.2TB）。

分片并行（每进程占若干 GPU，靠 --shard_index/--num_shards 静态分片，天然可续跑）：
  python scripts/batch_export.py \
    --robotwin_root <RoboTwin 仓库> \
    --lerobot_root  <lerobot_robotwin_eef_aug_500> \
    --export_root   <.../SIV_WAM/contact_siv_dataset> \
    --shard_index 0 --num_shards 24 --gpus 0,1 \
    --episodes_per_task 500 --log_dir ./_c2s_out/batch_logs

已完成（3 个 mask latent 都在）的 episode 会自动跳过，中断后直接重跑同一条命令即可。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

from c2s.contacts import ContactFrame, RobotwinSimContactSource, write_jsonl
from c2s.geometry import Camera
from c2s.lerobot_io import read_episode, read_video_latent
from c2s.siv import SIVBuilder, SIVConfig
from c2s.spec import CAM_KEYS, MASK_KEYS, WAN_VAE_PATH
from c2s.seeds import load_seed_map, seed_for_episode

CAM_TO_SIM = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
# Formal SIV supervision uses the fixed head camera; all three image latents remain inputs.
DEFAULT_CAMS = ["cam_high"]


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
def build_env(robotwin_root, task_name, task_config, seed, ep_num):
    os.chdir(robotwin_root)
    sys.path.insert(0, robotwin_root)
    import yaml
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    def get_embodiment_config(robot_file):
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    with open(
        os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8"
    ) as f:
        types = yaml.load(f.read(), Loader=yaml.FullLoader)

    mod = importlib.import_module(f"envs.{task_name}")
    task = getattr(mod, task_name)()
    with open(
        os.path.join(robotwin_root, "task_config", f"{task_config}.yml"),
        "r",
        encoding="utf-8",
    ) as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    emb = args["embodiment"]
    if len(emb) == 1:
        args["left_robot_file"] = args["right_robot_file"] = types[emb[0]]["file_path"]
        args["dual_arm_embodied"] = True
    else:
        args["left_robot_file"] = types[emb[0]]["file_path"]
        args["right_robot_file"] = types[emb[1]]["file_path"]
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = str(emb[0]) if len(emb) == 1 else f"{emb[0]}+{emb[1]}"
    task.setup_demo(now_ep_num=ep_num, seed=seed, is_test=True, **args)
    return task


def camera_dicts(task, need):
    """只取需要的相机（跳过 front_camera，省一次渲染）。"""
    out = {}
    cfg = task.cameras.get_config()
    for sim_name in need:
        if sim_name not in cfg:
            continue
        item = cfg[sim_name]
        cam = Camera.from_opencv(
            sim_name, item["intrinsic_cv"], item["extrinsic_cv"], calib_size=None
        )
        out[sim_name] = cam
    return out


def depth_maps(task, need):
    """返回 {sim_name: (h,w) float32 米}。必须先 update_picture()。"""
    task.cameras.update_picture()
    got = task.cameras.get_depth()
    return {
        n: np.asarray(got[n]["depth"], dtype=np.float32) / 1000.0
        for n in need
        if n in got
    }


# --------------------------------------------------------------------------- #
# 单 episode 单趟处理
# --------------------------------------------------------------------------- #
def process_episode(
    task_dir,
    export_dir,
    episode,
    task,
    source,
    cams,
    cfg,
    encoder,
    device,
    siv_size,
    save_contacts_dir=None,
):
    eef = read_episode(task_dir, episode)
    sim_names = [CAM_TO_SIM[c] for c in cams]

    vds, fids, sizes = {}, {}, {}
    for cam in cams:
        vd = read_video_latent(task_dir, episode, CAM_KEYS[cam])
        if vd is None:
            return f"skip: {cam} 无 video latent"
        vds[cam] = vd
        fids[cam] = np.clip(np.asarray(list(vd["frame_ids"])), 0, len(eef) - 1)
        sizes[cam] = (int(vd["video_width"]), int(vd["video_height"]))

    builders, maps, metas, stats = {}, {c: [] for c in cams}, {c: [] for c in cams}, {}
    contact_frames = [] if save_contacts_dir else None

    for t in range(len(eef)):
        task.take_action(eef[t], "ee")
        contacts = source.extract(task.scene)
        cams_now = camera_dicts(task, sim_names)
        dms = depth_maps(task, sim_names) if cfg.depth_tolerance else {}
        if contact_frames is not None:
            contact_frames.append(ContactFrame(frame=t, contacts=contacts))
        for cam in cams:
            sim = CAM_TO_SIM[cam]
            camera = cams_now.get(sim)
            if camera is None:
                maps[cam].append(np.zeros((sizes[cam][1], sizes[cam][0]), np.float32))
                metas[cam].append({"frame": t, "accepted": 0, "depth_rejected": 0})
                continue
            camera.image_size = sizes[cam]
            b = builders.setdefault(cam, SIVBuilder(camera, cfg))
            b.camera = camera
            value, _, _, m = b.build_frame(ContactFrame(t, contacts), dms.get(sim))
            maps[cam].append(value)
            metas[cam].append(m)

    for cam in cams:
        st = {
            "frames": len(eef),
            "projected": sum(m["accepted"] for m in metas[cam]),
            "depth_rejected": sum(m["depth_rejected"] for m in metas[cam]),
            "empty_frames": sum(1 for m in metas[cam] if m["accepted"] == 0),
        }
        stats[cam] = st
        arr = np.asarray(maps[cam], dtype=np.float32)
        assert arr.shape[1:] == (sizes[cam][1], sizes[cam][0])
        if encoder is not None:
            from c2s.latents import save_mask_latent, siv_to_rgb

            lat = encoder.encode(siv_to_rgb(arr[fids[cam]]), device=device)
            vd = vds[cam]
            assert (lat.shape[0], lat.shape[1], lat.shape[2]) == (
                vd["latent_num_frames"],
                vd["latent_height"],
                vd["latent_width"],
            )
            base = os.path.basename(
                min(
                    glob.glob(
                        os.path.join(
                            task_dir,
                            "latents",
                            "chunk-000",
                            CAM_KEYS[cam],
                            f"episode_{episode:06d}_*.pth",
                        )
                    )
                )
            )
            save_mask_latent(
                os.path.join(export_dir, "latents", "chunk-000", MASK_KEYS[cam], base),
                lat,
                vd,
            )
    if contact_frames is not None:
        write_jsonl(
            contact_frames,
            os.path.join(save_contacts_dir, f"ep{episode:06d}_contacts.jsonl"),
        )
    return stats


# --------------------------------------------------------------------------- #
# 离线（无仿真）：只用录制的末端位姿 + 已验证的 head_camera 标定
# --------------------------------------------------------------------------- #
def process_episode_offline(
    task_dir, export_dir, episode, cams, cfg, encoder, device, save_contacts_dir=None
):
    """接触代理路径：只用 16D 末端位姿，天然与视频对齐（不需要仿真器/seed）。

    注意：这是接触**代理**（夹爪闭合 + 指垫几何），不是物理接触点；
    仿真真实接触需要能复现原场景的 seed，见 README「为什么批量走离线路径」。
    """
    from c2s.contacts import EEFContactEstimator, write_jsonl
    from c2s.geometry import Camera
    from c2s.latents import save_mask_latent, siv_to_rgb
    from c2s.lerobot_io import build_wrist_camera
    from c2s.siv import build_sequence

    eef = read_episode(task_dir, episode)
    frames = EEFContactEstimator()(eef)

    vds, fids, sizes = {}, {}, {}
    for cam in cams:
        vd = read_video_latent(task_dir, episode, CAM_KEYS[cam])
        if vd is None:
            return f"skip: {cam} 无 video latent"
        vds[cam] = vd
        fids[cam] = np.clip(np.asarray(list(vd["frame_ids"])), 0, len(eef) - 1)
        sizes[cam] = (int(vd["video_width"]), int(vd["video_height"]))

    stats, results = {}, {}
    for cam in cams:
        if cam == "cam_high":
            camera = Camera.head_camera(image_size=sizes[cam])
            res = SIVBuilder(camera, cfg).build(frames)
        else:
            # wrist 相机装在臂上：逐帧由末端位姿 + URDF 安装偏移构造（未渲染验证）
            res = build_sequence(
                frames,
                lambda i, c=cam: build_wrist_camera(
                    c, eef[i], (sizes[c][1], sizes[c][0])
                ),
                cfg,
            )
        results[cam] = res
        stats[cam] = res.stats

    if encoder is not None:
        for cam in cams:
            res = results[cam]
            arr = res.maps
            assert arr.shape[1:] == (sizes[cam][1], sizes[cam][0])
            lat = encoder.encode(siv_to_rgb(arr[fids[cam]]), device=device)
            vd = vds[cam]
            assert (lat.shape[0], lat.shape[1], lat.shape[2]) == (
                vd["latent_num_frames"],
                vd["latent_height"],
                vd["latent_width"],
            )
            base = os.path.basename(
                min(
                    glob.glob(
                        os.path.join(
                            task_dir,
                            "latents",
                            "chunk-000",
                            CAM_KEYS[cam],
                            f"episode_{episode:06d}_*.pth",
                        )
                    )
                )
            )
            save_mask_latent(
                os.path.join(export_dir, "latents", "chunk-000", MASK_KEYS[cam], base),
                lat,
                vd,
            )
    if save_contacts_dir:
        write_jsonl(
            frames, os.path.join(save_contacts_dir, f"ep{episode:06d}_contacts.jsonl")
        )
    return stats


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def task_name_of(dir_name: str) -> str:
    """任务目录名 -> envs 模块名。

    "lift_pot-aloha-agilex_randomized_500-1000" -> "lift_pot"
    "blocks_ranking_size"（无后缀）                -> "blocks_ranking_size"
    """
    import re

    return re.sub(
        r"-(aloha-agilex|aloha|piper|franka-panda|ur5-wsg|ARX-X5).*$", "", dir_name
    )


def task_dirs(lerobot_root, tasks):
    all_dirs = sorted(glob.glob(os.path.join(lerobot_root, "*")))
    all_dirs = [
        d for d in all_dirs if os.path.isfile(os.path.join(d, "meta", "info.json"))
    ]
    if tasks and "all" not in tasks:
        all_dirs = [
            d
            for d in all_dirs
            if os.path.basename(d).split("-")[0] in tasks
            or os.path.basename(d) in tasks
        ]
    return all_dirs


def done(export_dir, episode, cams) -> bool:
    for cam in cams:
        if not glob.glob(
            os.path.join(
                export_dir,
                "latents",
                "chunk-000",
                MASK_KEYS[cam],
                f"episode_{episode:06d}_*.pth",
            )
        ):
            return False
    return True


def prepare_export(task_dir, export_root, cams):
    name = os.path.basename(task_dir.rstrip("/"))
    dst = os.path.join(export_root, name)
    for sub in ("data", "meta", "videos"):
        src = os.path.join(task_dir, sub)
        link = os.path.join(dst, sub)
        if os.path.isdir(src) and not os.path.exists(link):
            os.makedirs(dst, exist_ok=True)
            try:
                os.symlink(os.path.abspath(src), link)
            except OSError:
                pass
    lat = os.path.join(dst, "latents", "chunk-000")
    os.makedirs(lat, exist_ok=True)
    for cam in CAM_KEYS:
        src = os.path.join(task_dir, "latents", "chunk-000", CAM_KEYS[cam])
        link = os.path.join(lat, CAM_KEYS[cam])
        if os.path.isdir(src) and not os.path.exists(link):
            try:
                os.symlink(os.path.abspath(src), link)
            except OSError:
                pass
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robotwin_root", required=True)
    ap.add_argument("--lerobot_root", required=True)
    ap.add_argument("--export_root", required=True)
    ap.add_argument("--tasks", nargs="*", default=["all"])
    ap.add_argument("--episodes_per_task", type=int, default=500)
    ap.add_argument("--cams", nargs="*", default=DEFAULT_CAMS)
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--contact_radius", type=float, default=0.02)
    ap.add_argument("--depth_tolerance", type=float, default=0.05)
    ap.add_argument("--max_points", type=int, default=16)
    ap.add_argument("--save_contacts", action="store_true")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="不用仿真器，走 EEF 接触代理路径：与视频天然对齐，快 ~20 倍",
    )
    ap.add_argument("--no_encode", action="store_true")
    ap.add_argument("--vae_path", default=WAN_VAE_PATH)
    ap.add_argument("--log_dir", default="./_c2s_out/batch_logs")
    ap.add_argument("--seed_file", default=None,
                    help="seed.txt or JSON episode->seed mapping for formal replay")
    ap.add_argument("--allow_episode_seed", action="store_true",
                    help="Explicitly allow the unsafe seed=episode fallback")
    ap.add_argument("--denoiser", choices=("none", "oidn", "optix"), default="none")
    args = ap.parse_args()

    if not args.offline and not args.seed_file and not args.allow_episode_seed:
        ap.error("Formal batch replay requires --seed_file")
    seed_map = load_seed_map(args.seed_file) if args.seed_file else None
    if not args.offline:
        import sapien
        sapien.render.set_ray_tracing_denoiser(args.denoiser)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"shard{args.shard_index:02d}.log")
    log = open(log_path, "a", encoding="utf-8", buffering=1)

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")

    cfg = SIVConfig(
        contact_radius=args.contact_radius,
        depth_tolerance=None if args.offline else args.depth_tolerance,
    )
    source = RobotwinSimContactSource(max_points=args.max_points)

    encoder = None
    device = "cpu"
    if not args.no_encode:
        import torch
        from c2s.latents import WanVAEEncoder, load_vae

        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = WanVAEEncoder(load_vae(args.vae_path, device=device))
    say(f"shard {args.shard_index}/{args.num_shards} gpus={args.gpus} device={device}")

    dirs = task_dirs(args.lerobot_root, args.tasks)
    jobs = []
    for d in dirs:
        name = os.path.basename(d)
        episodes = [
            int(os.path.basename(p).split("_")[1].split(".")[0])
            for p in sorted(
                glob.glob(os.path.join(d, "data", "chunk-*", "episode_*.parquet"))
            )
        ]
        for ep in episodes[: args.episodes_per_task]:
            jobs.append((d, name, ep))
    jobs = jobs[args.shard_index :: args.num_shards]
    say(f"jobs={len(jobs)} (of {len(dirs)} tasks)")

    cur_task = None
    task_obj = None
    ok = fail = skipped = 0
    t_start = time.time()
    for i, (task_dir, name, ep) in enumerate(jobs):
        export_dir = prepare_export(task_dir, args.export_root, args.cams)
        if done(export_dir, ep, args.cams):
            skipped += 1
            continue
        try:
            contacts_dir = (
                os.path.join(export_dir, "contacts") if args.save_contacts else None
            )
            if contacts_dir:
                os.makedirs(contacts_dir, exist_ok=True)
            if args.offline:
                st = process_episode_offline(
                    task_dir,
                    export_dir,
                    ep,
                    args.cams,
                    cfg,
                    encoder,
                    device,
                    contacts_dir,
                )
            else:
                if cur_task != name:
                    seed = seed_for_episode(seed_map, ep) if seed_map is not None else ep
                    task_obj = build_env(
                        args.robotwin_root,
                        task_name_of(name),
                        args.task_config,
                        seed=seed,
                        ep_num=ep,
                    )
                    import sapien
                    sapien.render.set_ray_tracing_denoiser(args.denoiser)
                    cur_task = name
                    say(f"env built: {name} seed={seed}")
                st = process_episode(
                    task_dir,
                    export_dir,
                    ep,
                    task_obj,
                    source,
                    args.cams,
                    cfg,
                    encoder,
                    device,
                    None,
                    contacts_dir,
                )
            if isinstance(st, str):
                skipped += 1
                say(f"[{i + 1}/{len(jobs)}] {name} ep{ep} {st}")
            else:
                ok += 1
                say(
                    f"[{i + 1}/{len(jobs)}] {name} ep{ep} ok "
                    + " ".join(f"{c}:{st[c]['projected']}" for c in args.cams)
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 长任务不能中断
            fail += 1
            say(f"[{i + 1}/{len(jobs)}] {name} ep{ep} FAIL {type(e).__name__}: {e}")
            log.write(traceback.format_exc() + "\n")
            cur_task, task_obj = None, None  # 环境可能已损坏，下一个 episode 重建
    dt = time.time() - t_start
    say(
        f"DONE ok={ok} fail={fail} skipped={skipped} in {dt / 3600:.2f}h "
        f"({dt / max(ok, 1):.1f}s/episode)"
    )


if __name__ == "__main__":
    main()
