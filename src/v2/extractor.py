# -*- coding: utf-8 -*-
"""Self-contained primitive extractor for v2 deterministic pipeline (Stage 2B).

Extracts raw kinematic primitives directly from official optical tracking files
without importing any legacy v1 modules or loading v1 caches.
"""

import hashlib
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


CACHE_SCHEMA_VERSION = 3
PRIMITIVE_DTYPES = {
    "ball_valid": np.dtype(np.float32),
    "ball_x_m": np.dtype(np.float32),
    "ball_y_m": np.dtype(np.float32),
    "d_h1": np.dtype(np.float32),
    "d_a1": np.dtype(np.float32),
    "nearest_home_track_id": np.dtype(np.int16),
    "nearest_away_track_id": np.dtype(np.int16),
    "p25_player_speed": np.dtype(np.float32),
    "median_player_speed": np.dtype(np.float32),
    "p75_player_speed": np.dtype(np.float32),
    "p90_player_speed": np.dtype(np.float32),
    "fraction_players_below_1ms": np.dtype(np.float32),
    "valid_player_speed_count": np.dtype(np.float32),
}


def _sha256_file(path: str) -> str:
    """Return a streaming SHA-256 digest for an input file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_metadata(
    raw_txt_path: str,
    raw_xml_path: str,
    pitch_length: float,
    pitch_width: float,
    n_frames: int,
) -> Dict[str, np.ndarray]:
    """Build provenance metadata stored alongside cached primitives."""
    return {
        "__cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
        "__raw_txt_sha256": np.asarray(_sha256_file(raw_txt_path)),
        "__raw_xml_sha256": np.asarray(_sha256_file(raw_xml_path)),
        "__pitch_length": np.asarray(pitch_length, dtype=np.float64),
        "__pitch_width": np.asarray(pitch_width, dtype=np.float64),
        "__n_frames": np.asarray(n_frames, dtype=np.int64),
    }


def _load_validated_cache(
    cache_path: str,
    raw_txt_path: str,
    raw_xml_path: str,
    pitch_length: float,
    pitch_width: float,
    n_frames: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Load a cache only when its schema, inputs, and array contract match."""
    if not os.path.isfile(cache_path):
        return None
    if not os.path.isfile(raw_txt_path) or not os.path.isfile(raw_xml_path):
        return None

    expected_metadata = _cache_metadata(
        raw_txt_path, raw_xml_path, pitch_length, pitch_width, n_frames
    )
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            required_keys = set(PRIMITIVE_DTYPES) | set(expected_metadata)
            if not required_keys.issubset(data.files):
                return None

            if int(data["__cache_schema_version"].item()) != CACHE_SCHEMA_VERSION:
                return None
            if str(data["__raw_txt_sha256"].item()) != str(expected_metadata["__raw_txt_sha256"].item()):
                return None
            if str(data["__raw_xml_sha256"].item()) != str(expected_metadata["__raw_xml_sha256"].item()):
                return None
            if int(data["__n_frames"].item()) != n_frames:
                return None
            if float(data["__pitch_length"].item()) != float(pitch_length):
                return None
            if float(data["__pitch_width"].item()) != float(pitch_width):
                return None

            primitives: Dict[str, np.ndarray] = {}
            for key, expected_dtype in PRIMITIVE_DTYPES.items():
                arr = data[key]
                if arr.shape != (n_frames,) or arr.dtype != expected_dtype:
                    return None
                primitives[key] = arr.copy()
            return primitives
    except (OSError, ValueError, TypeError, KeyError):
        return None


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
    if cache_path and not force_rebuild:
        cached = _load_validated_cache(
            cache_path=cache_path,
            raw_txt_path=raw_txt_path,
            raw_xml_path=raw_xml_path,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            n_frames=n_frames,
        )
        if cached is not None:
            return cached

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
    nearest_home_track_id = np.full(n_frames, -1, dtype=np.int16)
    nearest_away_track_id = np.full(n_frames, -1, dtype=np.int16)
    p25_player_speed = np.full(n_frames, np.nan, dtype=np.float32)
    median_player_speed = np.full(n_frames, np.nan, dtype=np.float32)
    p75_player_speed = np.full(n_frames, np.nan, dtype=np.float32)
    p90_player_speed = np.full(n_frames, np.nan, dtype=np.float32)
    fraction_players_below_1ms = np.full(n_frames, np.nan, dtype=np.float32)
    valid_player_speed_count = np.zeros(n_frames, dtype=np.float32)

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
                        h_dists.append((dist, slot_idx))
                    else:
                        a_dists.append((dist, slot_idx))

            if has_ball:
                if h_dists:
                    home_dist, home_track = min(h_dists)
                    d_h1[idx] = home_dist
                    nearest_home_track_id[idx] = home_track
                if a_dists:
                    away_dist, away_track = min(a_dists)
                    d_a1[idx] = away_dist
                    nearest_away_track_id[idx] = away_track

            if valid_speeds:
                speeds = np.asarray(valid_speeds, dtype=np.float32)
                p25_player_speed[idx] = float(np.percentile(speeds, 25))
                median_player_speed[idx] = float(np.percentile(speeds, 50))
                p75_player_speed[idx] = float(np.percentile(speeds, 75))
                p90_player_speed[idx] = float(np.percentile(speeds, 90))
                fraction_players_below_1ms[idx] = float(np.mean(speeds < 1.0))
                valid_player_speed_count[idx] = float(len(speeds))

    primitives = {
        "ball_valid": ball_valid,
        "ball_x_m": ball_x_m,
        "ball_y_m": ball_y_m,
        "d_h1": d_h1,
        "d_a1": d_a1,
        "nearest_home_track_id": nearest_home_track_id,
        "nearest_away_track_id": nearest_away_track_id,
        "p25_player_speed": p25_player_speed,
        "median_player_speed": median_player_speed,
        "p75_player_speed": p75_player_speed,
        "p90_player_speed": p90_player_speed,
        "fraction_players_below_1ms": fraction_players_below_1ms,
        "valid_player_speed_count": valid_player_speed_count,
    }

    if cache_path:
        cache_parent = os.path.dirname(os.path.abspath(cache_path))
        os.makedirs(cache_parent, exist_ok=True)
        metadata = _cache_metadata(
            raw_txt_path, raw_xml_path, pitch_length, pitch_width, n_frames
        )
        np.savez_compressed(cache_path, **primitives, **metadata)

    return primitives
