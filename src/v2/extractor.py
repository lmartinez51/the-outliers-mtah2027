# -*- coding: utf-8 -*-
"""Self-contained primitive extractor for v2 deterministic pipeline (Stage 2B).

Extracts raw kinematic primitives directly from official optical tracking files
without importing any legacy v1 modules or loading v1 caches.
"""

import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional
import numpy as np

from src.v2.constants import (
    N_VIDEO_FRAMES,
    PITCH_LENGTH,
    PITCH_WIDTH,
)


def extract_v2_primitives(
    raw_txt_path: str = os.path.join("data", "tactical", "BAR-ATH_tactical_atd.txt"),
    raw_xml_path: str = os.path.join("data", "tactical", "BAR-ATH_tactical_atd.xml"),
    cache_path: Optional[str] = os.path.join("data", "cache_v2_primitives.npz"),
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
    n_frames: int = N_VIDEO_FRAMES,
    force_rebuild: bool = False,
) -> Dict[str, np.ndarray]:
    """Extract approved kinematic primitives directly from raw tracking files.

    Args:
        raw_txt_path: Path to raw tracking text file.
        raw_xml_path: Path to raw tracking XML metadata.
        cache_path: Optional path to v2-specific npz cache file.
        pitch_length: Field length in meters.
        pitch_width: Field width in meters.
        n_frames: Total video frames (default 199,319).
        force_rebuild: If True, ignores existing cache and re-extracts from raw files.

    Returns:
        Dict mapping primitive names -> np.ndarray.
    """
    if cache_path and os.path.isfile(cache_path) and not force_rebuild:
        data = np.load(cache_path)
        return {key: data[key] for key in data.files}

    if not os.path.isfile(raw_txt_path):
        raise FileNotFoundError(f"Raw tracking text file not found: {raw_txt_path}")
    if not os.path.isfile(raw_xml_path):
        raise FileNotFoundError(f"Raw tracking metadata file not found: {raw_xml_path}")

    # 1. Parse player team assignments (0 for Home, 1 for Away)
    root = ET.parse(raw_xml_path).getroot()
    team_elements = list(root.iter("Team"))
    if len(team_elements) != 2:
        raise ValueError(f"Expected 2 teams in {raw_xml_path}, found {len(team_elements)}")

    home_id = team_elements[0].get("id")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"No Player entries found in {raw_xml_path}")
    team_of_track = [0 if tid == home_id else 1 for tid in slots]

    # 2. Pre-allocate primitive arrays
    ball_valid = np.zeros(n_frames, dtype=np.float32)
    ball_x_m = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y_m = np.full(n_frames, np.nan, dtype=np.float32)
    d_h1 = np.full(n_frames, np.nan, dtype=np.float32)
    d_a1 = np.full(n_frames, np.nan, dtype=np.float32)
    p75_player_speed = np.zeros(n_frames, dtype=np.float32)

    # 3. Stream parse raw tracking file line-by-line
    with open(raw_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            c1 = line.find(":")
            if c1 == -1:
                continue
            c2 = line.find(":", c1 + 1)
            if c2 == -1:
                continue

            fid = int(line[:c1])
            idx = fid - 1
            if idx < 0 or idx >= n_frames:
                continue

            ball_str = line[c2 + 1 :].strip()
            players_str = line[c1 + 1 : c2]

            has_ball = bool(ball_str and ball_str[0] != "N")
            if has_ball:
                b_c1 = ball_str.find(",")
                b_c2 = ball_str.find(",", b_c1 + 1)
                bx_norm = float(ball_str[:b_c1])
                by_norm = float(ball_str[b_c1 + 1 : b_c2])
                bx_m = bx_norm * pitch_length
                by_m = by_norm * pitch_width
                ball_valid[idx] = 1.0
                ball_x_m[idx] = bx_m
                ball_y_m[idx] = by_m

            p_slots = players_str.split(";")
            h_dists = []
            a_dists = []
            valid_speeds = []

            for slot_idx, p in enumerate(p_slots):
                if not p or p.startswith("NaN"):
                    continue
                parts = p.split(",")
                px = float(parts[0]) * pitch_length
                py = float(parts[1]) * pitch_width
                ps_str = parts[2]
                if ps_str and ps_str[0] != "N":
                    valid_speeds.append(float(ps_str))

                if has_ball:
                    dist = math.hypot(px - bx_m, py - by_m)
                    if team_of_track[slot_idx] == 0:
                        h_dists.append(dist)
                    else:
                        a_dists.append(dist)

            if has_ball:
                if h_dists:
                    d_h1[idx] = min(h_dists)
                if a_dists:
                    d_a1[idx] = min(a_dists)

            if valid_speeds:
                if len(valid_speeds) >= 2:
                    valid_speeds.sort()
                    p75_player_speed[idx] = valid_speeds[int(0.75 * (len(valid_speeds) - 1))]
                else:
                    p75_player_speed[idx] = valid_speeds[0]

    primitives = {
        "ball_valid": ball_valid,
        "ball_x_m": ball_x_m,
        "ball_y_m": ball_y_m,
        "d_h1": d_h1,
        "d_a1": d_a1,
        "p75_player_speed": p75_player_speed,
    }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **primitives)

    return primitives
