#!/usr/bin/env python3
"""Run the data-generation pipeline one inspectable stage at a time.

The JSON manifest is intentionally a debug manifest rather than a training
configuration. Each stage writes artifacts that can be inspected before the
next stage is started:

    projection -> raw -> encode
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path: str) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"task_dir", "episode", "cams", "outputs", "runtime"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Missing manifest keys: {', '.join(missing)}")
    if not manifest["cams"]:
        raise ValueError("cams must contain at least one camera")
    if manifest.get("mode") != "offline_proxy":
        raise ValueError("This debug runner currently supports mode=offline_proxy only")
    return manifest


def python_bin(manifest: dict) -> str:
    return manifest["runtime"].get("python") or sys.executable


def common(manifest: dict) -> list[str]:
    return ["--task_dir", manifest["task_dir"], "--episode", str(manifest["episode"])]


def command_for(stage: str, manifest: dict) -> list[str]:
    py = python_bin(manifest)
    cams = manifest["cams"]
    outputs = manifest["outputs"]
    if stage == "projection":
        return [
            py,
            str(ROOT / "data_process/scripts/verify_projection.py"),
            *common(manifest),
            "--cams",
            *cams,
            "--num",
            str(manifest.get("projection", {}).get("num_frames", 8)),
            "--state_key",
            manifest.get("state_key", "action"),
            "--out_dir",
            outputs["projection"],
        ]
    if stage == "raw":
        siv = manifest.get("siv", {})
        return [
            py,
            str(ROOT / "data_process/scripts/export_siv_dataset.py"),
            "--task_dir",
            manifest["task_dir"],
            "--offline",
            "--export_root",
            outputs["raw_export"],
            "--episodes",
            str(manifest["episode"]),
            "--cams",
            *cams,
            "--contact_radius",
            str(siv.get("contact_radius", 0.02)),
        ]
    if stage == "encode":
        runtime = manifest["runtime"]
        task_name = Path(manifest["task_dir"]).name
        return [
            py,
            str(ROOT / "data_process/scripts/batch_export.py"),
            "--robotwin_root",
            runtime["robotwin_root"],
            "--lerobot_root",
            runtime["lerobot_root"],
            "--export_root",
            outputs["encoded_export"],
            "--tasks",
            task_name,
            "--episodes_per_task",
            str(manifest["episode"] + 1),
            "--cams",
            *cams,
            "--offline",
            "--vae_path",
            runtime["vae_path"],
            "--gpus",
            runtime.get("gpus", "0"),
            "--log_dir",
            outputs["logs"],
        ]
    raise ValueError(f"Unknown stage: {stage}")


def run(stage: str, manifest: dict, dry_run: bool) -> None:
    cmd = command_for(stage, manifest)
    print("[debug stage]", stage)
    print("[command]", subprocess.list2cmdline(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run inspectable SIV data-generation stages")
    parser.add_argument("--config", required=True, help="debug_data_generation.json")
    parser.add_argument("--stage", choices=("projection", "raw", "encode", "all"), default="projection")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.config)
    stages = ("projection", "raw", "encode") if args.stage == "all" else (args.stage,)
    for stage in stages:
        run(stage, manifest, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
