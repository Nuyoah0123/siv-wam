#!/usr/bin/env python3
"""Prepare a batch_export output for siv_wam.train_siv.

The data generator intentionally links large source directories. The current
dataset discovery code uses os.walk without followlinks, so this staging step
materializes each task's meta directory while keeping data/videos/latents as
links. It also copies empty_emb.pt and validates all required camera latents.
The source export is never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IMAGE_KEYS = tuple(f"observation.images.{cam}" for cam in CAMERAS)


def link_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(src.resolve()), str(dst))


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def validate_task(task_dir: Path, require_cameras: tuple[str, ...], allow_missing: bool) -> dict:
    info = json.loads((task_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    episodes_path = task_dir / "meta" / "episodes.jsonl"
    required_keys = IMAGE_KEYS + tuple(f"observation.masks.{c}" for c in require_cameras)
    valid_windows = 0
    missing = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        episode_index = int(episode["episode_index"])
        for action_cfg in episode.get("action_config", []):
            start = int(action_cfg["start_frame"])
            end = int(action_cfg["end_frame"])
            chunk = episode_chunk(episode_index, chunks_size)
            filename = f"episode_{episode_index:06d}_{start}_{end}.pth"
            paths = [task_dir / "latents" / f"chunk-{chunk:03d}" / key / filename
                     for key in required_keys]
            missing_paths = [str(path) for path in paths if not path.exists()]
            if missing_paths:
                missing.extend(missing_paths)
            else:
                valid_windows += 1
    return {
        "task": task_dir.name,
        "valid_windows": valid_windows,
        "missing_count": len(missing),
        "missing_examples": missing[:10],
    }


def prepare(args) -> int:
    source_root = Path(args.export_root).resolve()
    train_root = Path(args.train_root).resolve()
    if train_root.exists():
        raise FileExistsError(f"Output already exists: {train_root}")
    train_root.mkdir(parents=True)

    shutil.copy2(Path(args.empty_emb).resolve(), train_root / "empty_emb.pt")
    summaries = []
    for source_task in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if not (source_task / "meta" / "info.json").exists():
            continue
        target_task = train_root / source_task.name
        target_task.mkdir()
        for name in ("data", "videos", "latents"):
            src = source_task / name
            if src.exists():
                link_dir(src, target_task / name)
        # Materialize meta because the training loader discovers info.json via os.walk.
        shutil.copytree(source_task / "meta", target_task / "meta", symlinks=False)
        if (source_task / "contacts").exists():
            shutil.copytree(source_task / "contacts", target_task / "contacts", symlinks=False)
        summaries.append(validate_task(target_task, tuple(args.cams), args.allow_missing))

    report = {"source_root": str(source_root), "train_root": str(train_root), "tasks": summaries}
    (train_root / "siv_train_export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not summaries:
        raise RuntimeError("No task directory with meta/info.json found")
    if any(item["missing_count"] for item in summaries) and not args.allow_missing:
        raise RuntimeError("Export has incomplete video/mask latent windows; see report")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage a batch_export output for train_siv")
    parser.add_argument("--export_root", required=True, help="batch_export output root")
    parser.add_argument("--train_root", required=True, help="new training dataset root")
    parser.add_argument("--empty_emb", required=True)
    # All three observation.images.* latents remain required inputs.  Only the
    # fixed head-camera mask is required for SIV supervision.
    parser.add_argument("--cams", nargs="*", default=["cam_high"])
    parser.add_argument("--allow-missing", action="store_true",
                        help="Allow partial episode exports for overfit/debug datasets")
    return prepare(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
