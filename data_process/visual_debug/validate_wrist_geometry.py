#!/usr/bin/env python3
"""Compare simulator wrist-camera geometry with the offline URDF estimate.

The simulator camera JSONL is treated as the reference geometry. The script
does not modify any original data-processing module; it only reads EEF poses,
RGB frames, optional contacts, and per-frame camera metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_process"))

from c2s.contacts import read_jsonl  # noqa: E402
from c2s.geometry import Camera  # noqa: E402
from c2s.lerobot_io import build_wrist_camera, map_size_for, read_episode, read_frames  # noqa: E402
from c2s.spec import ARM_SLICES, CAM_KEYS, FINGER_PAD_BACK, eef_to_tip_pose, finger_half_open  # noqa: E402


def load_per_frame_cameras(path: str) -> dict[int, dict]:
    out = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                item = json.loads(line)
                out[int(item["frame"])] = item["cameras"]
    return out


def frame_ids(text: str | None, total: int, count: int) -> list[int]:
    if text:
        values = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
        values = [x for x in values if 0 <= x < total]
        if values:
            return values
    return np.linspace(0, total - 1, min(count, total)).astype(int).tolist()


def landmarks(eef_row: np.ndarray, cam: str) -> tuple[np.ndarray, list[str]]:
    arm = "left" if "left" in cam else "right"
    sl = ARM_SLICES[arm]
    xyz = eef_row[sl["xyz"][0]:sl["xyz"][1]]
    quat = eef_row[sl["quat"][0]:sl["quat"][1]]
    grip = float(np.clip(eef_row[sl["gripper"]], 0.0, 1.0))
    tip, rotation = eef_to_tip_pose(xyz, quat)
    half = finger_half_open(grip)
    pads = [
        np.asarray(xyz) + rotation @ np.array([0.12 - FINGER_PAD_BACK, half, 0.0]),
        np.asarray(xyz) + rotation @ np.array([0.12 - FINGER_PAD_BACK, -half, 0.0]),
    ]
    return np.asarray([tip, *pads], dtype=np.float64), ["tip", "pad_plus", "pad_minus"]


def camera_center(camera: Camera) -> np.ndarray:
    return -camera.R_cw.T @ camera.t_cw


def rotation_error_deg(reference: Camera, estimate: Camera) -> float:
    relative = reference.R_cw.T @ estimate.R_cw
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def put_text(image, text, y, color=(240, 240, 240)):
    cv2.putText(image, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 1, cv2.LINE_AA)


def draw_projection(rgb, sim_pixels, sim_valid, off_pixels, off_valid, names, map_size):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    image_h, image_w = image.shape[:2]
    sx, sy = image_w / map_size[0], image_h / map_size[1]
    for i, name in enumerate(names):
        if sim_pixels is not None and sim_valid[i]:
            p = (int(round(sim_pixels[i, 0] * sx)), int(round(sim_pixels[i, 1] * sy)))
            cv2.circle(image, p, 7, (0, 255, 0), -1)
            cv2.putText(image, f"S:{name}", (p[0] + 8, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        if off_valid[i]:
            p = (int(round(off_pixels[i, 0] * sx)), int(round(off_pixels[i, 1] * sy)))
            cv2.circle(image, p, 6, (0, 0, 255), -1)
            cv2.putText(image, f"O:{name}", (p[0] + 8, p[1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        if sim_pixels is not None and sim_valid[i] and off_valid[i]:
            a = (int(round(sim_pixels[i, 0] * sx)), int(round(sim_pixels[i, 1] * sy)))
            b = (int(round(off_pixels[i, 0] * sx)), int(round(off_pixels[i, 1] * sy)))
            cv2.line(image, a, b, (255, 255, 0), 2)
    put_text(image, "green=simulator  red=offline  yellow=error", 28)
    return image


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate wrist-camera extrinsics")
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--cam", choices=("cam_left_wrist", "cam_right_wrist"), required=True)
    parser.add_argument("--cameras_jsonl", default=None,
                        help="build_contacts_sim.py cameras_epXXXX.jsonl")
    parser.add_argument("--offline_only", action="store_true",
                        help="Validate only offline projection against RGB; no simulator reference")
    parser.add_argument("--contacts", default=None, help="Optional contacts JSONL")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--frame_ids", default=None)
    parser.add_argument("--num", type=int, default=8)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eef = read_episode(args.task_dir, args.episode)
    rgb = read_frames(args.task_dir, args.episode, CAM_KEYS[args.cam])
    if rgb is None:
        raise FileNotFoundError(f"No video for {args.cam}")
    map_h, map_w = map_size_for(args.task_dir, args.episode, args.cam, (128, 160))
    map_size = (map_w, map_h)
    if not args.cameras_jsonl and not args.offline_only:
        parser.error("provide --cameras_jsonl or use --offline_only")
    sim_cameras = load_per_frame_cameras(args.cameras_jsonl) if args.cameras_jsonl else None
    contacts = read_jsonl(args.contacts) if args.contacts else None
    selected = frame_ids(args.frame_ids, min(len(eef), len(rgb)), args.num)

    rows = []
    for frame in selected:
        sim = None
        if sim_cameras is not None:
            item = sim_cameras.get(frame, {}).get("left_camera" if "left" in args.cam else "right_camera")
            if item is None:
                raise KeyError(f"No simulator camera metadata for frame {frame}")
            sim = Camera.from_opencv(args.cam, item["K"], item["world_to_camera"],
                                     tuple(item.get("image_size", (320, 240))), map_size)
        offline = build_wrist_camera(args.cam, eef[frame], (map_h, map_w))
        points, names = landmarks(eef[frame], args.cam)
        sim_px = sim_depth = sim_valid = None
        if sim is not None:
            sim_px, sim_depth, sim_valid = sim.project(points)
        off_px, off_depth, off_valid = offline.project(points)
        both = sim_valid & off_valid if sim is not None else np.zeros(len(points), dtype=bool)
        pixel_errors = np.linalg.norm(sim_px[both] - off_px[both], axis=1) if both.any() else np.empty(0)
        row = {
            "frame": frame,
            "offline_camera_center_world": camera_center(offline).tolist(),
            "landmarks": names,
            "offline_valid": off_valid.tolist(),
            "offline_depth_m": off_depth.tolist(),
        }
        if sim is not None:
            row.update({
                "sim_camera_center_world": camera_center(sim).tolist(),
                "translation_error_m": float(np.linalg.norm(camera_center(sim) - camera_center(offline))),
                "rotation_error_deg": rotation_error_deg(sim, offline),
                "sim_valid": sim_valid.tolist(),
                "mean_pixel_error_map_px": None if not len(pixel_errors) else float(pixel_errors.mean()),
                "max_pixel_error_map_px": None if not len(pixel_errors) else float(pixel_errors.max()),
                "sim_depth_m": sim_depth.tolist(),
            })
        if contacts is not None:
            points_c = np.asarray([c.position for c in contacts[frame].contacts], dtype=np.float64)
            if len(points_c):
                cp_off, _, cv_off = offline.project(points_c)
                row["contact_offline_valid"] = int(cv_off.sum())
                if sim is not None:
                    _, _, cv_sim = sim.project(points_c)
                    row["contact_sim_valid"] = int(cv_sim.sum())
        rows.append(row)
        image = draw_projection(rgb[frame], sim_px, sim_valid, off_px, off_valid, names, map_size)
        if sim is None:
            put_text(image, f"frame={frame} offline-only", 52)
        else:
            put_text(image, f"frame={frame} trans={row['translation_error_m']:.4f}m rot={row['rotation_error_deg']:.2f}deg", 52)
        cv2.imwrite(str(out_dir / f"frame_{frame:04d}_{args.cam}.png"), image)

    errors = [r["mean_pixel_error_map_px"] for r in rows if r.get("mean_pixel_error_map_px") is not None]
    offline_rates = [sum(r["offline_valid"]) / len(r["offline_valid"]) for r in rows]
    contact_rates = [r["contact_offline_valid"] / max(len(contacts[r["frame"]].contacts), 1)
                     for r in rows if "contact_offline_valid" in r]
    report = {
        "task_dir": args.task_dir,
        "episode": args.episode,
        "camera": args.cam,
        "map_size_wh": list(map_size),
        "frames": rows,
        "aggregate": {
            "offline_landmark_valid_rate": float(np.mean(offline_rates)),
            "offline_contact_valid_rate": None if not contact_rates else float(np.mean(contact_rates)),
            "mean_pixel_error_map_px": None if not errors else float(np.mean(errors)),
            "valid_frames": len(rows),
        },
    }
    if sim_cameras is not None:
        report["aggregate"].update({
            "mean_translation_error_m": float(np.mean([r["translation_error_m"] for r in rows])),
            "max_translation_error_m": float(np.max([r["translation_error_m"] for r in rows])),
            "mean_rotation_error_deg": float(np.mean([r["rotation_error_deg"] for r in rows])),
            "max_rotation_error_deg": float(np.max([r["rotation_error_deg"] for r in rows])),
        })
    (out_dir / "geometry_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
