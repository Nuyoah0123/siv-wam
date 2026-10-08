"""Seed mapping utilities for faithful RoboTwin replay."""
from __future__ import annotations

import json
from pathlib import Path


def load_seed_map(path: str | None) -> dict[int, int]:
    if not path:
        raise ValueError("A seed mapping is required for simulator replay")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Seed mapping not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Seed mapping is empty: {p}")
    if p.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict) and "episodes" in obj:
            obj = obj["episodes"]
        if isinstance(obj, list):
            return {i: int(seed) for i, seed in enumerate(obj)}
        if isinstance(obj, dict):
            return {int(ep): int(seed) for ep, seed in obj.items()}
        raise ValueError("JSON seed mapping must be a list or object")
    values = [int(x) for x in text.split()]
    return {i: seed for i, seed in enumerate(values)}


def seed_for_episode(seed_map: dict[int, int], episode: int) -> int:
    if episode not in seed_map:
        raise KeyError(f"No seed mapping for episode {episode}")
    return int(seed_map[episode])
