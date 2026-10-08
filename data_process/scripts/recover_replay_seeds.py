#!/usr/bin/env python3
"""Recover episode -> RoboTwin seed mapping by first-frame similarity.

Run this only on a node where RoboTwin/SAPIEN rendering works. The result is a
JSON mapping that build_contacts_sim.py and batch_export.py can consume.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2s.lerobot_io import read_frames  # noqa: E402
from scripts.batch_export import build_env  # noqa: E402
from scripts.check_replay_alignment import compare  # noqa: E402


def task_name_from_dir(task_dir: str) -> str:
    return re.sub(r"-(aloha-agilex|aloha|piper|franka-panda|ur5-wsg|ARX-X5).*$", "",
                  Path(task_dir).name)


def configure_renderer(denoiser: str) -> None:
    import sapien
    sapien.render.set_ray_tracing_denoiser(denoiser)


def render_head(task, denoiser):
    # get_obs performs update_wrist_camera, scene.update_render, and
    # cameras.update_picture in RoboTwin's canonical order.
    configure_renderer(denoiser)
    obs = task.get_obs()
    return np.asarray(obs["observation"]["head_camera"]["rgb"])[:, :, :3]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robotwin_root", required=True)
    ap.add_argument("--task_dir", required=True)
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--seed_start", type=int, required=True)
    ap.add_argument("--seed_end", type=int, required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="Progress checkpoint; defaults to <out>.progress.json")
    ap.add_argument("--resume", action="store_true",
                    help="Resume candidates already stored in the checkpoint")
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--denoiser", choices=("none", "oidn", "optix"), default="none")
    args = ap.parse_args(argv)
    original_cwd = Path.cwd()
    if args.seed_end <= args.seed_start:
        ap.error("seed_end must be greater than seed_start")
    configure_renderer(args.denoiser)

    task_name = task_name_from_dir(args.task_dir)
    out_path = Path(args.out).resolve()
    checkpoint_path = Path(args.checkpoint or (str(out_path) + ".progress.json")).resolve()
    checkpoint = {}
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    results = dict(checkpoint.get("episodes", {}))
    for episode in args.episodes:
        real = read_frames(args.task_dir, episode, "observation.images.cam_high")[0]
        previous = checkpoint.get("running", {}).get(str(episode), {})
        candidates = list(previous.get("candidates", []))
        start_seed = int(previous.get("next_seed", args.seed_start))
        print(f"[episode {episode}] seeds {start_seed}..{args.seed_end - 1} "
              f"({args.seed_end - start_seed} candidates)", flush=True)
        for index, seed in enumerate(range(start_seed, args.seed_end), start=1):
            if index == 1 or index % max(args.progress_every, 1) == 0:
                print(f"[episode {episode}] trying seed={seed} "
                      f"({index}/{args.seed_end - start_seed})", flush=True)
            task = None
            try:
                task = build_env(args.robotwin_root, task_name, args.task_config,
                                 seed=seed, ep_num=episode)
                sim = render_head(task, args.denoiser)
                metrics = compare(sim, real)
                candidates.append({"seed": seed, **metrics})
            except Exception as exc:  # noqa: BLE001
                candidates.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                if task is not None:
                    try:
                        task.close_env()
                    except Exception:
                        pass
                os.chdir(original_cwd)
            ranked_live = sorted(
                (c for c in candidates if "gray_corr" in c),
                key=lambda c: (-c["gray_corr"], c["mse"]),
            )
            checkpoint = {
                "task_dir": args.task_dir,
                "task_name": task_name,
                "task_config": args.task_config,
                "seed_start": args.seed_start,
                "seed_end": args.seed_end,
                "running": {
                    str(episode): {
                        "next_seed": seed + 1,
                        "candidates": candidates,
                        "best": ranked_live[0] if ranked_live else None,
                    }
                },
                "episodes": results,
                "updated_at": time.time(),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=checkpoint_path.parent,
                                             delete=False) as fh:
                json.dump(checkpoint, fh, indent=2)
                temporary = Path(fh.name)
            temporary.replace(checkpoint_path)
        ranked = sorted(
            (c for c in candidates if "gray_corr" in c),
            key=lambda c: (-c["gray_corr"], c["mse"]),
        )
        if not ranked:
            raise RuntimeError(f"No renderable seed candidate for episode {episode}")
        results[str(episode)] = {
            "seed": ranked[0]["seed"],
            "mse": ranked[0]["mse"],
            "gray_corr": ranked[0]["gray_corr"],
            "top_candidates": ranked[:args.top_k],
        }
        checkpoint["episodes"] = results
        checkpoint.setdefault("running", {}).pop(str(episode), None)
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        print(f"episode {episode}: best seed={ranked[0]['seed']} "
              f"mse={ranked[0]['mse']} corr={ranked[0]['gray_corr']}", flush=True)

    report = {
        "task_dir": args.task_dir,
        "task_name": task_name,
        "task_config": args.task_config,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "episodes": results,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
