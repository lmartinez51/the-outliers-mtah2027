# -*- coding: utf-8 -*-
"""Deterministic control candidate detection (v2, Section 3.2)."""

import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.v2.constants import (
    LABEL_A,
    LABEL_H,
    MARGIN,
    PITCH_LENGTH,
    PITCH_WIDTH,
    R_TOUCH,
)


def load_player_teams_from_xml(xml_path: str) -> List[int]:
    """Parse player team assignments from tactical metadata XML.

    Returns:
        List where 0 indicates Home and 1 indicates Away for each player slot.
    """
    root = ET.parse(xml_path).getroot()
    team_elements = list(root.iter("Team"))
    if len(team_elements) != 2:
        raise ValueError(f"Expected 2 teams in {xml_path}, found {len(team_elements)}")

    home_id = team_elements[0].get("id")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"No Player entries found in {xml_path}")

    return [0 if tid == home_id else 1 for tid in slots]


def compute_player_distances_for_frames(
    raw_txt_path: str,
    raw_xml_path: str,
    target_frames: Dict[int, Tuple[float, float]],
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> Dict[int, Tuple[float, float, int, int]]:
    """Scan raw tactical tracking file to compute d_H and d_A for specific frames.

    Args:
        raw_txt_path: Path to BAR-ATH_tactical_atd.txt.
        raw_xml_path: Path to BAR-ATH_tactical_atd.xml.
        target_frames: Dict mapping 0-based frame index -> (ball_x_m, ball_y_m).
        pitch_length: Length of pitch in meters.
        pitch_width: Width of pitch in meters.

    Returns:
        Dict mapping frame index -> (d_H, d_A, home_track_id, away_track_id).
    """
    if not target_frames:
        return {}

    team_of_track = load_player_teams_from_xml(raw_xml_path)
    needed_fids = {idx + 1: idx for idx in target_frames.keys()}
    results: Dict[int, Tuple[float, float, int, int]] = {}

    with open(raw_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            c1 = line.find(":")
            if c1 == -1:
                continue
            fid = int(line[:c1])
            if fid not in needed_fids:
                continue

            frame_idx = needed_fids[fid]
            bx, by = target_frames[frame_idx]

            c2 = line.find(":", c1 + 1)
            players_str = line[c1 + 1 : c2]
            p_slots = players_str.split(";")

            h_dists: List[Tuple[float, int]] = []
            a_dists: List[Tuple[float, int]] = []

            for slot_idx, p in enumerate(p_slots):
                if not p or p.startswith("NaN"):
                    continue
                parts = p.split(",")
                px = float(parts[0]) * pitch_length
                py = float(parts[1]) * pitch_width
                dist = math.hypot(px - bx, py - by)

                if team_of_track[slot_idx] == 0:
                    h_dists.append((dist, slot_idx))
                else:
                    a_dists.append((dist, slot_idx))

            min_h, home_track = min(h_dists) if h_dists else (float("inf"), -1)
            min_a, away_track = min(a_dists) if a_dists else (float("inf"), -1)
            results[frame_idx] = (min_h, min_a, home_track, away_track)

            if len(results) == len(target_frames):
                break

    return results


def detect_control_candidates(
    ball_valid: np.ndarray,
    ball_known: np.ndarray,
    ball_x_m: np.ndarray,
    ball_y_m: np.ndarray,
    d_h1_cache: np.ndarray,
    d_a1_cache: np.ndarray,
    start_frame: int,
    end_frame: int,
    raw_txt_path: Optional[str] = None,
    raw_xml_path: Optional[str] = None,
    r_touch: float = R_TOUCH,
    margin: float = MARGIN,
    nearest_home_track_id_cache: Optional[np.ndarray] = None,
    nearest_away_track_id_cache: Optional[np.ndarray] = None,
    return_track_ids: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Determine control candidates on ball_known frames.

    Candidate team = the nearer team if its distance <= r_touch and the other team's
    nearest distance exceeds it by at least margin. Otherwise no candidate.

    Returns:
        candidates: Object array of shape (N,) with 'H', 'A', or None.
        d_H_all: Float array of shape (N,) with nearest distance to Home player.
        d_A_all: Float array of shape (N,) with nearest distance to Away player.
        candidate_track_ids: When return_track_ids is true, int16 array containing
            the candidate's frame-local tracking slot, or -1 when unavailable.
    """
    n_frames = len(ball_valid)
    candidates = np.full(n_frames, None, dtype=object)
    d_H_all = np.full(n_frames, np.nan, dtype=np.float32)
    d_A_all = np.full(n_frames, np.nan, dtype=np.float32)
    candidate_track_ids = np.full(n_frames, -1, dtype=np.int16)

    # 1. Identify interpolated frames that need distance computation
    interpolated_targets: Dict[int, Tuple[float, float]] = {}
    for f in range(start_frame, end_frame + 1):
        if ball_known[f] and ball_valid[f] == 0:
            interpolated_targets[f] = (float(ball_x_m[f]), float(ball_y_m[f]))

    # Compute distances for interpolated frames if any exist
    interp_dists: Dict[int, Tuple[float, float, int, int]] = {}
    if interpolated_targets:
        if raw_txt_path is None:
            raw_txt_path = os.path.join("data", "tactical", "BAR-ATH_tactical_atd.txt")
        if raw_xml_path is None:
            raw_xml_path = os.path.join("data", "tactical", "BAR-ATH_tactical_atd.xml")
        interp_dists = compute_player_distances_for_frames(
            raw_txt_path, raw_xml_path, interpolated_targets
        )

    # 2. Evaluate candidate condition for each frame in the half
    for f in range(start_frame, end_frame + 1):
        if not ball_known[f]:
            continue

        if ball_valid[f] > 0:
            dH = float(d_h1_cache[f]) if not np.isnan(d_h1_cache[f]) else float("inf")
            dA = float(d_a1_cache[f]) if not np.isnan(d_a1_cache[f]) else float("inf")
            home_track = (
                int(nearest_home_track_id_cache[f])
                if nearest_home_track_id_cache is not None else -1
            )
            away_track = (
                int(nearest_away_track_id_cache[f])
                if nearest_away_track_id_cache is not None else -1
            )
        else:
            dH, dA, home_track, away_track = interp_dists.get(
                f, (float("inf"), float("inf"), -1, -1)
            )

        d_H_all[f] = dH if not math.isinf(dH) else np.nan
        d_A_all[f] = dA if not math.isinf(dA) else np.nan

        # Candidate team condition
        min_d = min(dH, dA)
        if min_d <= r_touch:
            if dH + margin <= dA:
                candidates[f] = LABEL_H
                candidate_track_ids[f] = home_track
            elif dA + margin <= dH:
                candidates[f] = LABEL_A
                candidate_track_ids[f] = away_track

    if return_track_ids:
        return candidates, d_H_all, d_A_all, candidate_track_ids
    return candidates, d_H_all, d_A_all
