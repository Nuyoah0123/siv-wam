"""Standalone visualization copy for the projection -> SIV pipeline.

This file imports the original data-processing modules without editing them.
It produces inspectable PNGs, an MP4 summary, and JSON traces.
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

from c2s.contacts import EEFContactEstimator, read_jsonl
from c2s.geometry import Camera
from c2s.lerobot_io import (
    build_wrist_camera,
    map_size_for,
    read_episode,
    read_frames,
)
from c2s.siv import SIVBuilder, SIVConfig
from c2s.spec import CAM_KEYS


def parse_frame_ids(text: str | None, total: int, num: int) -> list[int]:
    if text:
        ids = [int(x.strip()) for x in text.split(",") if x.strip()]
        ids = sorted({i for i in ids if 0 <= i < total})
        if ids:
            return ids
    return np.linspace(0, total - 1, min(num, total)).astype(int).tolist()


def color_for_kind(kind: str) -> tuple[int, int, int]:
    return {"grasp": (0, 255, 255), "place": (0, 255, 0),
            "support": (255, 255, 0)}.get(kind, (255, 128, 0))


def put_text(img: np.ndarray, text: str, x: int, y: int,
             color=(235, 235, 235), scale=0.58, thickness=1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def map_panel(siv: np.ndarray, width: int, height: int) -> np.ndarray:
    heat = (np.clip(siv, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_NEAREST)
    put_text(heat, "SIV map", 16, 28)
    return heat


def draw_points(rgb, pixels, valid, contacts, map_size):
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    image_h, image_w = out.shape[:2]
    map_w, map_h = map_size
    sx, sy = image_w / map_w, image_h / map_h
    for index, (pixel, is_valid, contact) in enumerate(zip(pixels, valid, contacts)):
        if not is_valid:
            continue
        point = (int(round(float(pixel[0]) * sx)), int(round(float(pixel[1]) * sy)))  # noqa: RUF046
        color = color_for_kind(contact.kind)
        cv2.circle(out, point, 7, color, -1)
        cv2.circle(out, point, 10, (20, 20, 20), 1)
        put_text(out, str(index), point[0] + 8, point[1] - 8, color, scale=0.48)
    put_text(out, "RGB + projected contacts", 16, 28)
    return out


def overlay_panel(rgb, siv):
    image_h, image_w = rgb.shape[:2]
    heat = cv2.resize(siv, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8),
                                   cv2.COLORMAP_JET)
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    strength = np.clip(heat, 0, 1)[..., None] * 0.58
    out = (base.astype(np.float32) * (1 - strength) +
           heat_color.astype(np.float32) * strength).clip(0, 255).astype(np.uint8)
    put_text(out, "RGB + SIV overlay", 16, 28)
    return out


def diagnostics_panel(frame_index, frame, pixels, depths, valid, siv, camera, contact_radius):
    out = np.full((480, 640, 3), 28, dtype=np.uint8)
    peak = np.unravel_index(int(np.argmax(siv)), siv.shape) if siv.max() > 0 else None
    valid_depth = depths[valid] if len(depths) else np.empty(0)
    sigmas = camera.sigma_px(valid_depth, contact_radius) if len(valid_depth) else np.empty(0)
    lines = [
        f"frame: {frame_index}",
        f"contacts: {len(frame.contacts)}",
        f"projected valid: {int(valid.sum())}",
        f"map shape: {siv.shape[1]}x{siv.shape[0]}",
        f"map min/max: {siv.min():.3f} / {siv.max():.3f}",
        f"peak (u,v): {None if peak is None else (int(peak[1]), int(peak[0]))}",
        f"depth min/max: {None if not len(valid_depth) else f'{valid_depth.min():.3f} / {valid_depth.max():.3f} m'}",
        f"sigma min/max: {None if not len(sigmas) else f'{sigmas.min():.2f} / {sigmas.max():.2f} px'}",
        "",
        "Contact rows:",
    ]
    for i, (contact, pixel, is_valid) in enumerate(zip(frame.contacts, pixels, valid)):
        if i >= 8:
            lines.append("... more contacts omitted")
            break
        xyz = ",".join(f"{v:.3f}" for v in contact.position)
        uv = "invalid" if not is_valid else f"({pixel[0]:.1f},{pixel[1]:.1f})"
        lines.append(f"{i}: {contact.kind:7s} xyz=({xyz}) uv={uv}")
    for row, line in enumerate(lines):
        put_text(out, line, 16, 28 + row * 24, scale=0.52)
    return out


def load_contacts(args, eef):
    if args.contacts:
        return read_jsonl(args.contacts)
    return EEFContactEstimator()(eef)


def camera_for_frame(args, eef_row, map_size):
    if args.cam == "cam_high":
        return Camera.head_camera(image_size=map_size)
    return build_wrist_camera(args.cam, eef_row, (map_size[1], map_size[0]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Visualize projection and SIV generation")
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--cam", choices=("cam_high", "cam_left_wrist", "cam_right_wrist"), default="cam_high")
    parser.add_argument("--contacts", help="Optional contacts.jsonl; omit to use EEF proxy")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--frame_ids", help="Comma-separated frame ids, e.g. 40,64,81,101")
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--contact_radius", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--live", action="store_true",
                        help="Write intermediate images after every processed frame")
    parser.add_argument("--pause_each", action="store_true",
                        help="After each frame, wait for Enter before continuing")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eef = read_episode(args.task_dir, args.episode)
    frames_rgb = read_frames(args.task_dir, args.episode, CAM_KEYS[args.cam])
    if frames_rgb is None:
        raise FileNotFoundError(f"No video for {args.cam}")
    contact_frames = load_contacts(args, eef)
    if len(contact_frames) != len(eef):
        raise ValueError(f"Contact frames {len(contact_frames)} != EEF frames {len(eef)}")

    map_h, map_w = map_size_for(args.task_dir, args.episode, args.cam, (256, 320))
    map_size = (map_w, map_h)
    frame_ids = parse_frame_ids(args.frame_ids, len(frames_rgb), args.num)
    cfg = SIVConfig(contact_radius=args.contact_radius)
    panels = []
    live_dir = out_dir / "live"
    if args.live:
        live_dir.mkdir(parents=True, exist_ok=True)
    summary = {"task_dir": args.task_dir, "episode": args.episode,
               "cam": args.cam,
               "source": "contacts_jsonl" if args.contacts else "eef_proxy",
               "map_size_wh": list(map_size), "frames": []}

    for frame_index in frame_ids:
        contact_frame = contact_frames[frame_index]
        camera = camera_for_frame(args, eef[frame_index], map_size)
        points = np.asarray([c.position for c in contact_frame.contacts], dtype=np.float64)
        pixels, depths, valid = camera.project(points)
        rgb = frames_rgb[frame_index]
        p1 = draw_points(rgb, pixels, valid, contact_frame.contacts, map_size)
        if args.live:
            projection_path = live_dir / f"frame_{frame_index:04d}_01_projection.png"
            cv2.imwrite(str(projection_path), p1)
            print(f"[live] projection saved: {projection_path}", flush=True)

        siv, _, _, meta = SIVBuilder(camera, cfg).build_frame(contact_frame)
        p2 = map_panel(siv, rgb.shape[1], rgb.shape[0])
        p3 = overlay_panel(rgb, siv)
        p4 = diagnostics_panel(frame_index, contact_frame, pixels, depths, valid,
                                siv, camera, args.contact_radius)
        panel = np.concatenate([
            np.concatenate([p1, p2], axis=1),
            np.concatenate([p3, p4], axis=1),
        ], axis=0)
        cv2.imwrite(str(out_dir / f"frame_{frame_index:04d}_pipeline.png"), panel)
        if args.live:
            live_outputs = {
                "02_siv_map": p2,
                "03_overlay": p3,
                "04_diagnostics": p4,
                "05_pipeline": panel,
            }
            for suffix, image in live_outputs.items():
                path = live_dir / f"frame_{frame_index:04d}_{suffix}.png"
                cv2.imwrite(str(path), image)
            print(f"[live] SIV/overlay/diagnostics saved for frame {frame_index}", flush=True)
            if args.pause_each:
                input("[live] Open the images in VS Code, inspect variables, then press Enter to continue... ")
        panels.append(panel)
        peak_index = int(np.argmax(siv)) if siv.max() > 0 else None
        summary["frames"].append({
            "frame": frame_index,
            "contacts": len(contact_frame.contacts),
            "projected": int(valid.sum()),
            "map_min": float(siv.min()),
            "map_max": float(siv.max()),
            "peak_uv_map": None if peak_index is None else [peak_index % siv.shape[1], peak_index // siv.shape[1]],
            "meta": meta,
        })

    video_path = out_dir / "pipeline_debug.mp4"
    if panels:
        h, w = panels[0].shape[:2]
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for panel in panels:
            writer.write(panel)
        writer.release()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"frames": frame_ids, "out_dir": str(out_dir), "video": str(video_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
