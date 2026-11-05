"""Fast Streaming Feature Extractor & Binary Cache for More than a Hack 2027.

Team: The Outliers
Authors: Lorenzo Martínez, Ana Karen Arellano, and Steven Ibarra
Repository: the-outliers-mtah2027
Target Competition: More than a Hack 2027

Extracts compact frame-level kinematic, proximity, temporal, and tactical features
from raw tactical tracking (ATD) and Smart Tagging XML without loading all data via kloppy.
Caches arrays in `data/cache_bar_ath_features.npz` for instant (<1s) subsequent loads.

Usage:
    python src/mtah_features.py            # builds cache or loads existing in <1s
    python src/mtah_features.py --rebuild  # forces re-extraction from raw files
"""

import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import (  # noqa: E402
    GAME,
    GAMES,
    DATA_DIR,
    half_video_blocks,
    require,
    source_frames,
)
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

# Constants
HOME, AWAY = 0, 1
DEFAULT_CACHE_NAME = "cache_bar_ath_features.npz"

# Tactical phases to encode for possession bias
ATTACKING_PHASES = {
    "BUILD UP",
    "PROGRESSION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "CREATING CHANCES",
    "LONG BALL",
    "ATTACKING TRANSITION",
}


def team_of_each_track(meta_path: str) -> np.ndarray:
    """Array over 46 track slots: 0 if Home (Barcelona), 1 if Away (Athletic Club)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"{meta_path}: no Player entries")
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def detect_half_kickoff_frame(
    half_start: int,
    half_end: int,
    events: Optional[List[Dict[str, Any]]] = None,
    ball_x_m: Optional[np.ndarray] = None,
    ball_y_m: Optional[np.ndarray] = None,
    ball_valid: Optional[np.ndarray] = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    game: str = GAME,
) -> int:
    """
    Algorithmic, generalizable detection of the actual kickoff frame for a match half.
    Eliminates hardcoded kickoff frame constants:
      1. Inspects Smart Tagging events for the kickoff event ('SET PIECES' with
         Type 'Kick Off (Start)') or first live play phase in [half_start, half_start + 7500].
         If the half start is already inside the kickoff event window (like Half 1 at 11185),
         kickoff is immediate at half_start.
         If kickoff occurs later (like Half 2), checks starter.load_smart_tagging.KICKOFF_VIDEO_FRAMES
         or detects when the ball is kicked from the center spot.
      2. Fallback if Smart Tagging is unavailable: detects when the ball is stationary within
         2.5m of the center spot (pitch_length/2, pitch_width/2) followed by sustained
         movement > 4.0m.
    """
    if events is None:
        try:
            import starter.load_smart_tagging as lst
            events = lst.load(game)
        except Exception:
            events = None

    if events is not None:
        kickoff_event = None
        for ev in events:
            code = (ev.get("code") or "").strip().upper()
            etype = (ev.get("tags", {}).get("Type") or "").strip().lower()
            if code == "SET PIECES" and "kick off" in etype:
                if half_start - 1000 <= ev["start_frame"] <= half_start + 7500:
                    kickoff_event = ev
                    break

        if kickoff_event is not None:
            if kickoff_event["start_frame"] <= half_start <= kickoff_event["end_frame"]:
                return half_start

            try:
                import starter.load_smart_tagging as lst
                ko_dict = getattr(lst, "KICKOFF_VIDEO_FRAMES", {})
                for _, ko_f in ko_dict.items():
                    if kickoff_event["start_frame"] <= ko_f <= kickoff_event["end_frame"]:
                        return ko_f
            except Exception:
                pass

            if ball_x_m is not None and ball_y_m is not None:
                cx, cy = pitch_length / 2.0, pitch_width / 2.0
                s_fr, e_fr = max(half_start, kickoff_event["start_frame"]), min(half_end, kickoff_event["end_frame"])
                for f in range(s_fr, e_fr - 10):
                    if not np.isnan(ball_x_m[f]) and not np.isnan(ball_y_m[f]):
                        dist_c = np.hypot(ball_x_m[f] - cx, ball_y_m[f] - cy)
                        if dist_c <= 3.5:
                            if not np.isnan(ball_x_m[f+10]) and not np.isnan(ball_y_m[f+10]):
                                dm = np.hypot(ball_x_m[f+10] - ball_x_m[f], ball_y_m[f+10] - ball_y_m[f])
                                if dm > 2.0:
                                    return f

            return kickoff_event["start_frame"]

    if ball_x_m is not None and ball_y_m is not None:
        cx, cy = pitch_length / 2.0, pitch_width / 2.0
        scan_end = min(half_end - 50, half_start + 7500)
        for f in range(half_start + 25, scan_end):
            if ball_valid is not None and ball_valid[f] == 0:
                continue
            if not np.isnan(ball_x_m[f]) and not np.isnan(ball_y_m[f]):
                dist_c = np.hypot(ball_x_m[f] - cx, ball_y_m[f] - cy)
                if dist_c < 2.5:
                    if not np.isnan(ball_x_m[f+50]) and not np.isnan(ball_y_m[f+50]):
                        disp = np.hypot(ball_x_m[f+50] - ball_x_m[f], ball_y_m[f+50] - ball_y_m[f])
                        if disp > 4.0:
                            return f

    return half_start


def build_masks(game: str = GAME) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build graded_mask, play_live_window_mask, and is_pre_2nd_half_kickoff algorithmically.

    Returns:
        graded_mask: bool array (199319,) True inside halves (11185..90651 and 110808..190197)
        play_live_window_mask: bool array (199319,) True in live play
        is_pre_2nd_half_kickoff: float32 array (199319,) 1.0 in pre-kickoff dead-ball interval, 0.0 elsewhere
    """
    n_frames = source_frames("tactical", game)
    graded_mask = np.zeros(n_frames, dtype=bool)
    play_live_window_mask = np.zeros(n_frames, dtype=bool)
    is_pre_2nd_half_kickoff = np.zeros(n_frames, dtype=np.float32)

    blocks = half_video_blocks(game)
    pitch_len, pitch_wid = GAMES[game]["pitch"]

    for half, first, last in blocks:
        graded_mask[first : last + 1] = True
        actual_kickoff = detect_half_kickoff_frame(
            half_start=first,
            half_end=last,
            pitch_length=pitch_len,
            pitch_width=pitch_wid,
            game=game,
        )
        if actual_kickoff > first:
            # Pre-kickoff dead-ball period (players walking into place before whistle)
            is_pre_2nd_half_kickoff[first:actual_kickoff] = 1.0
            play_live_window_mask[actual_kickoff : last + 1] = True
        else:
            play_live_window_mask[first : last + 1] = True

    return graded_mask, play_live_window_mask, is_pre_2nd_half_kickoff


def extract_streaming_atd(
    raw_path: str,
    team_of_track: np.ndarray,
    n_frames: int,
    pitch: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    """Single-pass streaming parse of BAR-ATH_tactical_atd.txt."""
    pitch_len, pitch_wid = pitch

    # Pre-allocate arrays
    ball_valid = np.zeros(n_frames, dtype=np.float32)
    ball_x_m = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y_m = np.full(n_frames, np.nan, dtype=np.float32)
    ball_speed_reported = np.full(n_frames, np.nan, dtype=np.float32)
    ball_out_of_bounds = np.zeros(n_frames, dtype=np.float32)

    n_home_players = np.zeros(n_frames, dtype=np.float32)
    n_away_players = np.zeros(n_frames, dtype=np.float32)
    mean_player_speed = np.zeros(n_frames, dtype=np.float32)
    max_player_speed = np.zeros(n_frames, dtype=np.float32)
    p75_player_speed = np.zeros(n_frames, dtype=np.float32)

    home_centroid_x = np.full(n_frames, np.nan, dtype=np.float32)
    home_centroid_y = np.full(n_frames, np.nan, dtype=np.float32)
    away_centroid_x = np.full(n_frames, np.nan, dtype=np.float32)
    away_centroid_y = np.full(n_frames, np.nan, dtype=np.float32)

    home_spread_x = np.full(n_frames, np.nan, dtype=np.float32)
    home_spread_y = np.full(n_frames, np.nan, dtype=np.float32)
    away_spread_x = np.full(n_frames, np.nan, dtype=np.float32)
    away_spread_y = np.full(n_frames, np.nan, dtype=np.float32)

    d_h1 = np.full(n_frames, np.nan, dtype=np.float32)
    d_h2 = np.full(n_frames, np.nan, dtype=np.float32)
    d_h3 = np.full(n_frames, np.nan, dtype=np.float32)
    d_a1 = np.full(n_frames, np.nan, dtype=np.float32)
    d_a2 = np.full(n_frames, np.nan, dtype=np.float32)
    d_a3 = np.full(n_frames, np.nan, dtype=np.float32)

    s_h1 = np.full(n_frames, np.nan, dtype=np.float32)
    s_a1 = np.full(n_frames, np.nan, dtype=np.float32)
    h1_x = np.full(n_frames, np.nan, dtype=np.float32)
    h1_y = np.full(n_frames, np.nan, dtype=np.float32)
    a1_x = np.full(n_frames, np.nan, dtype=np.float32)
    a1_y = np.full(n_frames, np.nan, dtype=np.float32)

    cnt_h_15 = np.zeros(n_frames, dtype=np.float32)
    cnt_h_30 = np.zeros(n_frames, dtype=np.float32)
    cnt_h_50 = np.zeros(n_frames, dtype=np.float32)
    cnt_a_15 = np.zeros(n_frames, dtype=np.float32)
    cnt_a_30 = np.zeros(n_frames, dtype=np.float32)
    cnt_a_50 = np.zeros(n_frames, dtype=np.float32)

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            c1 = line.find(":")
            if c1 == -1:
                continue
            c2 = line.find(":", c1 + 1)
            if c2 == -1:
                continue

            frame_id = int(line[:c1])
            idx = frame_id - 1
            if idx < 0 or idx >= n_frames:
                continue

            ball_str = line[c2 + 1 :].strip()
            players_str = line[c1 + 1 : c2]

            # 1. Parse ball observation
            has_ball = bool(ball_str and ball_str[0] != "N")
            bx_m, by_m = 0.0, 0.0
            if has_ball:
                b_c1 = ball_str.find(",")
                b_c2 = ball_str.find(",", b_c1 + 1)
                bx_norm = float(ball_str[:b_c1])
                by_norm = float(ball_str[b_c1 + 1 : b_c2])
                b_speed_str = ball_str[b_c2 + 1 :]
                b_speed = float(b_speed_str) if b_speed_str[0] != "N" else np.nan

                bx_m = bx_norm * pitch_len
                by_m = by_norm * pitch_wid

                ball_valid[idx] = 1.0
                ball_x_m[idx] = bx_m
                ball_y_m[idx] = by_m
                ball_speed_reported[idx] = b_speed

                if bx_norm < 0.0 or bx_norm > 1.0 or by_norm < 0.0 or by_norm > 1.0:
                    ball_out_of_bounds[idx] = 1.0

            # 2. Parse player slots
            home_x, home_y, home_s = [], [], []
            away_x, away_y, away_s = [], [], []
            valid_speeds = []

            for slot, p in enumerate(players_str.split(";")):
                if not p or p[0] == "N":
                    continue
                p_c1 = p.find(",")
                p_c2 = p.find(",", p_c1 + 1)
                px = float(p[:p_c1]) * pitch_len
                py = float(p[p_c1 + 1 : p_c2]) * pitch_wid
                ps_str = p[p_c2 + 1 :]
                ps = float(ps_str) if ps_str[0] != "N" else np.nan

                if not math.isnan(ps):
                    valid_speeds.append(ps)

                if team_of_track[slot] == HOME:
                    home_x.append(px)
                    home_y.append(py)
                    home_s.append(ps)
                else:
                    away_x.append(px)
                    away_y.append(py)
                    away_s.append(ps)

            nh = len(home_x)
            na = len(away_x)
            n_home_players[idx] = float(nh)
            n_away_players[idx] = float(na)

            # Player collective speed metrics
            if valid_speeds:
                mean_player_speed[idx] = sum(valid_speeds) / len(valid_speeds)
                max_player_speed[idx] = max(valid_speeds)
                if len(valid_speeds) >= 2:
                    valid_speeds.sort()
                    p75_player_speed[idx] = valid_speeds[int(0.75 * (len(valid_speeds) - 1))]
                else:
                    p75_player_speed[idx] = valid_speeds[0]

            # Team centroids & spread
            if nh > 0:
                hcx = sum(home_x) / nh
                hcy = sum(home_y) / nh
                home_centroid_x[idx] = hcx
                home_centroid_y[idx] = hcy
                home_spread_x[idx] = math.sqrt(sum((x - hcx) ** 2 for x in home_x) / nh)
                home_spread_y[idx] = math.sqrt(sum((y - hcy) ** 2 for y in home_y) / nh)

            if na > 0:
                acx = sum(away_x) / na
                acy = sum(away_y) / na
                away_centroid_x[idx] = acx
                away_centroid_y[idx] = acy
                away_spread_x[idx] = math.sqrt(sum((x - acx) ** 2 for x in away_x) / na)
                away_spread_y[idx] = math.sqrt(sum((y - acy) ** 2 for y in away_y) / na)

            # 3. Ball-relative player proximity features
            if has_ball:
                if nh > 0:
                    h_dists = []
                    for i in range(nh):
                        dx = home_x[i] - bx_m
                        dy = home_y[i] - by_m
                        d = math.sqrt(dx * dx + dy * dy)
                        h_dists.append((d, home_s[i], home_x[i], home_y[i]))
                        if d <= 1.5:
                            cnt_h_15[idx] += 1.0
                        if d <= 3.0:
                            cnt_h_30[idx] += 1.0
                        if d <= 5.0:
                            cnt_h_50[idx] += 1.0

                    h_dists.sort(key=lambda item: item[0])
                    d_h1[idx] = h_dists[0][0]
                    s_h1[idx] = h_dists[0][1]
                    h1_x[idx] = h_dists[0][2]
                    h1_y[idx] = h_dists[0][3]
                    if nh > 1:
                        d_h2[idx] = h_dists[1][0]
                    if nh > 2:
                        d_h3[idx] = h_dists[2][0]

                if na > 0:
                    a_dists = []
                    for i in range(na):
                        dx = away_x[i] - bx_m
                        dy = away_y[i] - by_m
                        d = math.sqrt(dx * dx + dy * dy)
                        a_dists.append((d, away_s[i], away_x[i], away_y[i]))
                        if d <= 1.5:
                            cnt_a_15[idx] += 1.0
                        if d <= 3.0:
                            cnt_a_30[idx] += 1.0
                        if d <= 5.0:
                            cnt_a_50[idx] += 1.0

                    a_dists.sort(key=lambda item: item[0])
                    d_a1[idx] = a_dists[0][0]
                    s_a1[idx] = a_dists[0][1]
                    a1_x[idx] = a_dists[0][2]
                    a1_y[idx] = a_dists[0][3]
                    if na > 1:
                        d_a2[idx] = a_dists[1][0]
                    if na > 2:
                        d_a3[idx] = a_dists[2][0]

    # Uniform kinematic ball speed (25 Hz finite difference) across all 199,319 frames,
    # solving the 100% NaN reported ball speed in the first 54,835 frames (including public window).
    vx_ball = np.diff(ball_x_m, prepend=ball_x_m[0]) * 25.0
    vy_ball = np.diff(ball_y_m, prepend=ball_y_m[0]) * 25.0
    v_ball_kin = np.hypot(vx_ball, vy_ball).astype(np.float32)
    invalid_v = (ball_valid == 0.0) | (np.roll(ball_valid, 1) == 0.0) | (v_ball_kin > 38.0)
    v_ball_kin[invalid_v] = np.nan
    ball_speed_reported = v_ball_kin

    # Proximity-attenuated nearest team advantage:
    # Attenuates raw distance difference when the ball is in open space (min(d_h1, d_a1) > 1.8m)
    # so dead-ball rolling or long clearances in empty space do not create artificial control bias.
    d_min = np.minimum(d_h1, d_a1)
    atten = np.exp(-np.maximum(0.0, d_min - 1.8) / 1.8)
    raw_diff = np.clip(d_a1 - d_h1, -4.5, 4.5)
    diff2 = np.clip(d_a2 - d_h2, -3.5, 3.5) * np.exp(
        -np.maximum(0.0, np.minimum(d_h2, d_a2) - 2.5) / 2.5
    )
    nearest_team_adv = np.where(
        ball_valid == 1.0, raw_diff * atten + 0.25 * diff2, np.nan
    ).astype(np.float32)

    return {
        "ball_valid": ball_valid,
        "ball_x_m": ball_x_m,
        "ball_y_m": ball_y_m,
        "ball_speed_reported": ball_speed_reported,
        "ball_out_of_bounds": ball_out_of_bounds,
        "n_home_players": n_home_players,
        "n_away_players": n_away_players,
        "mean_player_speed": mean_player_speed,
        "max_player_speed": max_player_speed,
        "p75_player_speed": p75_player_speed,
        "home_centroid_x": home_centroid_x,
        "home_centroid_y": home_centroid_y,
        "away_centroid_x": away_centroid_x,
        "away_centroid_y": away_centroid_y,
        "home_spread_x": home_spread_x,
        "home_spread_y": home_spread_y,
        "away_spread_x": away_spread_x,
        "away_spread_y": away_spread_y,
        "d_h1": d_h1,
        "d_h2": d_h2,
        "d_h3": d_h3,
        "d_a1": d_a1,
        "d_a2": d_a2,
        "d_a3": d_a3,
        "s_h1": s_h1,
        "s_a1": s_a1,
        "h1_x": h1_x,
        "h1_y": h1_y,
        "a1_x": a1_x,
        "a1_y": a1_y,
        "cnt_h_15": cnt_h_15,
        "cnt_h_30": cnt_h_30,
        "cnt_h_50": cnt_h_50,
        "cnt_a_15": cnt_a_15,
        "cnt_a_30": cnt_a_30,
        "cnt_a_50": cnt_a_50,
        "nearest_team_adv": nearest_team_adv,
    }


def compute_temporal_and_bridge_features(
    raw: Dict[str, np.ndarray],
    half_blocks: List[Tuple[int, int, int]],
    n_frames: int,
) -> Dict[str, np.ndarray]:
    """Computes ball interpolation, pass anchors, flight bridges, and rolling windows."""
    ball_valid = raw["ball_valid"]
    ball_x_m = raw["ball_x_m"]
    ball_y_m = raw["ball_y_m"]
    d_h1 = raw["d_h1"]
    d_a1 = raw["d_a1"]
    home_centroid_x = raw["home_centroid_x"]
    home_centroid_y = raw["home_centroid_y"]
    away_centroid_x = raw["away_centroid_x"]
    away_centroid_y = raw["away_centroid_y"]
    nearest_team_adv = raw["nearest_team_adv"]
    ball_speed = raw["ball_speed_reported"]
    player_speed = raw["mean_player_speed"]

    # 1. Temporal Ball Interpolation (<= 75 frames / 3.0s per half)
    ball_x_interp = ball_x_m.copy()
    ball_y_interp = ball_y_m.copy()
    ball_interpolated = np.zeros(n_frames, dtype=np.float32)
    ball_gap_len = np.full(n_frames, np.nan, dtype=np.float32)
    frames_since_ball_seen = np.full(n_frames, 9999.0, dtype=np.float32)
    frames_until_ball_seen = np.full(n_frames, 9999.0, dtype=np.float32)

    for half, start_f, end_f in half_blocks:
        sub_valid = np.where(ball_valid[start_f : end_f + 1] == 1.0)[0] + start_f
        for i in range(len(sub_valid) - 1):
            a = sub_valid[i]
            b = sub_valid[i + 1]
            gap = b - a - 1
            if 1 <= gap <= 75:
                alpha = np.linspace(0, 1, gap + 2, dtype=np.float32)[1:-1]
                ball_x_interp[a + 1 : b] = ball_x_m[a] + alpha * (ball_x_m[b] - ball_x_m[a])
                ball_y_interp[a + 1 : b] = ball_y_m[a] + alpha * (ball_y_m[b] - ball_y_m[a])
                ball_interpolated[a + 1 : b] = 1.0
                ball_gap_len[a + 1 : b] = float(gap)
            elif gap > 75:
                ball_gap_len[a + 1 : b] = float(gap)

        # Set gap_len to 0 where ball is directly seen
        ball_gap_len[sub_valid] = 0.0

        # frames_since_ball_seen (forward scan)
        last_seen = -999999
        for t in range(start_f, end_f + 1):
            if ball_valid[t] == 1.0:
                last_seen = t
                frames_since_ball_seen[t] = 0.0
            else:
                if last_seen >= start_f:
                    frames_since_ball_seen[t] = float(t - last_seen)

        # frames_until_ball_seen (backward scan)
        next_seen = 999999
        for t in range(end_f, start_f - 1, -1):
            if ball_valid[t] == 1.0:
                next_seen = t
                frames_until_ball_seen[t] = 0.0
            else:
                if next_seen <= end_f:
                    frames_until_ball_seen[t] = float(next_seen - t)

    # 2. Proximity to centroids using interpolated ball
    dist_ball_interp_home_centroid = np.sqrt(
        (ball_x_interp - home_centroid_x) ** 2 + (ball_y_interp - home_centroid_y) ** 2
    ).astype(np.float32)
    dist_ball_interp_away_centroid = np.sqrt(
        (ball_x_interp - away_centroid_x) ** 2 + (ball_y_interp - away_centroid_y) ** 2
    ).astype(np.float32)
    centroid_advantage = (
        dist_ball_interp_away_centroid - dist_ball_interp_home_centroid
    ).astype(np.float32)

    # 3. Pass / Touch Anchors & Bridges (1.45m touch radius with clear separation margin)
    anchor_team = np.zeros(n_frames, dtype=np.float32)
    h_touch = (
        (ball_valid == 1.0)
        & (d_h1 <= 1.45)
        & (np.isnan(d_a1) | ((d_a1 - d_h1) >= 0.25) | (d_h1 <= 1.15))
        & (np.isnan(d_a1) | (d_h1 < d_a1))
    )
    a_touch = (
        (ball_valid == 1.0)
        & (d_a1 <= 1.45)
        & (np.isnan(d_h1) | ((d_h1 - d_a1) >= 0.25) | (d_a1 <= 1.15))
        & (np.isnan(d_h1) | (d_a1 < d_h1))
    )
    anchor_team[h_touch] = 1.0
    anchor_team[a_touch] = -1.0

    last_anchor_team = np.zeros(n_frames, dtype=np.float32)
    frames_since_anchor = np.full(n_frames, 9999.0, dtype=np.float32)
    next_anchor_team = np.zeros(n_frames, dtype=np.float32)
    frames_until_anchor = np.full(n_frames, 9999.0, dtype=np.float32)
    same_team_bridge = np.zeros(n_frames, dtype=np.float32)
    pass_bridge_len = np.full(n_frames, 9999.0, dtype=np.float32)

    for half, start_f, end_f in half_blocks:
        # Forward pass for anchor
        curr_anc = 0.0
        last_anc_f = -999999
        for t in range(start_f, end_f + 1):
            if anchor_team[t] != 0.0:
                curr_anc = anchor_team[t]
                last_anc_f = t
            last_anchor_team[t] = curr_anc
            if last_anc_f >= start_f:
                frames_since_anchor[t] = float(t - last_anc_f)

        # Backward pass for anchor
        curr_next_anc = 0.0
        next_anc_f = 999999
        for t in range(end_f, start_f - 1, -1):
            if anchor_team[t] != 0.0:
                curr_next_anc = anchor_team[t]
                next_anc_f = t
            next_anchor_team[t] = curr_next_anc
            if next_anc_f <= end_f:
                frames_until_anchor[t] = float(next_anc_f - t)

        # Bridge features
        for t in range(start_f, end_f + 1):
            b_len = frames_since_anchor[t] + frames_until_anchor[t]
            pass_bridge_len[t] = b_len
            if (
                (last_anchor_team[t] == next_anchor_team[t])
                and (last_anchor_team[t] != 0.0)
                and (b_len <= 150)
            ):
                same_team_bridge[t] = last_anchor_team[t]

    # 4. Rolling Windows (w = 15, 35, 75 frames centered radius)
    rolling_feats = {}
    s_adv = pd.Series(nearest_team_adv)
    s_ball_spd = pd.Series(ball_speed)
    s_play_spd = pd.Series(player_speed)

    for w in (15, 35, 75):
        win_size = 2 * w + 1
        rolling_feats[f"adv_rolling_w{w}"] = (
            s_adv.rolling(window=win_size, center=True, min_periods=1)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        rolling_feats[f"ball_speed_rolling_w{w}"] = (
            s_ball_spd.rolling(window=win_size, center=True, min_periods=1)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        rolling_feats[f"player_speed_rolling_w{w}"] = (
            s_play_spd.rolling(window=win_size, center=True, min_periods=1)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

    out = {
        "ball_x_interp": ball_x_interp,
        "ball_y_interp": ball_y_interp,
        "ball_interpolated": ball_interpolated,
        "ball_gap_len": ball_gap_len,
        "frames_since_ball_seen": frames_since_ball_seen,
        "frames_until_ball_seen": frames_until_ball_seen,
        "dist_ball_interp_home_centroid": dist_ball_interp_home_centroid,
        "dist_ball_interp_away_centroid": dist_ball_interp_away_centroid,
        "centroid_advantage": centroid_advantage,
        "anchor_team": anchor_team,
        "last_anchor_team": last_anchor_team,
        "frames_since_anchor": frames_since_anchor,
        "next_anchor_team": next_anchor_team,
        "frames_until_anchor": frames_until_anchor,
        "same_team_bridge": same_team_bridge,
        "pass_bridge_len": pass_bridge_len,
    }
    out.update(rolling_feats)
    return out


def encode_smart_tagging(game: str = GAME, n_frames: int = 199319) -> Dict[str, np.ndarray]:
    """Maps Smart Tagging XML intervals into frame-level tactical signals."""
    events = sorted(load_smart_tagging(game=game), key=lambda x: x["start_frame"])

    bias_sum = np.zeros(n_frames, dtype=np.float32)
    bias_count = np.zeros(n_frames, dtype=np.int32)
    st_set_piece_active = np.zeros(n_frames, dtype=np.float32)
    st_set_piece_phase = np.zeros(n_frames, dtype=np.float32)

    for ev in events:
        code = (ev["code"] or "").strip().upper()
        sf = max(0, ev["start_frame"])
        ef = min(n_frames, ev["end_frame"])
        if sf >= ef:
            continue

        if code == "SET PIECES":
            st_set_piece_active[sf:ef] = 1.0
            dur = max(1, ef - sf)
            st_set_piece_phase[sf:ef] = np.linspace(
                0.0, 1.0, dur, endpoint=False, dtype=np.float32
            )
        elif (
            code in ATTACKING_PHASES
            or code.startswith("DEFENDING IN")
            or code == "DEFENSIVE TRANSITION"
        ):
            team = ev["tags"].get("Team")
            bias = 0.0
            if team == "Barcelona":
                bias = 1.0
            elif team in ("Atletic Club", "Athletic Club"):
                bias = -1.0

            # Invert sign for defending phases because the tagged team is defending
            if code.startswith("DEFENDING IN") or code == "DEFENSIVE TRANSITION":
                bias = -bias

            if bias != 0.0:
                bias_sum[sf:ef] += bias
                bias_count[sf:ef] += 1

    st_team_bias = np.where(
        bias_count > 0, bias_sum / np.maximum(1, bias_count), 0.0
    ).astype(np.float32)

    return {
        "st_team_bias": st_team_bias,
        "st_set_piece_active": st_set_piece_active,
        "st_set_piece_phase": st_set_piece_phase,
    }


class FeatureDataset:
    """Wrapper providing dict-like access to the 2D feature matrix and individual columns."""

    def __init__(
        self,
        features: np.ndarray,
        feature_names: List[str],
        graded_mask: np.ndarray,
        play_live_window_mask: np.ndarray,
    ):
        self.features = features  # (199319, N_features)
        self.feature_names = list(feature_names)
        self.name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        self.graded_mask = graded_mask
        self.play_live_window_mask = play_live_window_mask

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self.name_to_idx:
            return self.features[:, self.name_to_idx[key]]
        if key == "features":
            return self.features
        if key == "feature_names":
            return np.array(self.feature_names)
        if key == "graded_mask":
            return self.graded_mask
        if key == "play_live_window_mask":
            return self.play_live_window_mask
        raise KeyError(f"Key '{key}' not found in FeatureDataset. Features: {self.feature_names}")

    def __contains__(self, key: str) -> bool:
        return key in self.name_to_idx or key in (
            "features",
            "feature_names",
            "graded_mask",
            "play_live_window_mask",
        )

    def get_feature(self, name: str) -> np.ndarray:
        return self[name]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.features.shape


def load_cached_features(cache_path: str) -> FeatureDataset:
    """Fast (<1s) binary loader from npz cache."""
    with np.load(cache_path) as data:
        features = data["features"]
        feature_names = list(data["feature_names"])
        graded_mask = data["graded_mask"]
        play_live_window_mask = data["play_live_window_mask"]
    return FeatureDataset(features, feature_names, graded_mask, play_live_window_mask)


def extract_all_features(
    game: str = GAME,
    data_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
    rebuild: bool = False,
) -> FeatureDataset:
    """Extracts or loads cached features for the match."""
    effective_data_dir = data_dir or DATA_DIR
    effective_cache_path = cache_path or os.path.join(
        effective_data_dir, DEFAULT_CACHE_NAME
    )

    if not rebuild and os.path.isfile(effective_cache_path):
        t0 = time.time()
        dataset = load_cached_features(effective_cache_path)
        load_time = time.time() - t0
        print(f"Loaded cached features from {effective_cache_path} in {load_time:.3f}s")
        return dataset

    print(f"Building features from raw match data ({game}) ...")
    t0 = time.time()

    n_frames = source_frames("tactical", game)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, effective_data_dir)
    pitch = GAMES[game]["pitch"]
    half_blocks = half_video_blocks(game)

    # 1. Build masks
    graded_mask, play_live_window_mask, is_pre_2nd_half_kickoff = build_masks(game)

    # 2. Extract streaming ATD
    print(f"  Streaming {os.path.basename(raw_path)} ({n_frames} frames) ...")
    team_of_track = team_of_each_track(meta_path)
    raw_feats = extract_streaming_atd(raw_path, team_of_track, n_frames, pitch)

    # 3. Temporal, bridge & rolling features
    print("  Computing temporal interpolation, pass anchors, and rolling windows ...")
    temp_feats = compute_temporal_and_bridge_features(raw_feats, half_blocks, n_frames)

    # 4. Smart Tagging tactical encoding
    print("  Encoding Smart Tagging tactical events ...")
    st_feats = encode_smart_tagging(game=game, n_frames=n_frames)

    # 5. Assemble feature dictionary
    all_dict = {
        "is_pre_2nd_half_kickoff": is_pre_2nd_half_kickoff,
    }
    all_dict.update(raw_feats)
    all_dict.update(temp_feats)
    all_dict.update(st_feats)

    feature_names = list(all_dict.keys())
    matrix_list = [all_dict[k] for k in feature_names]
    features = np.column_stack(matrix_list).astype(np.float32)

    elapsed = time.time() - t0
    print(f"  Feature extraction complete in {elapsed:.2f}s. Matrix shape: {features.shape}")

    # 6. Save cache
    os.makedirs(os.path.dirname(effective_cache_path), exist_ok=True)
    print(f"  Saving cache to {effective_cache_path} ...")
    t_save = time.time()
    np.savez_compressed(
        effective_cache_path,
        features=features,
        feature_names=np.array(feature_names),
        graded_mask=graded_mask,
        play_live_window_mask=play_live_window_mask,
    )
    print(f"  Saved in {time.time() - t_save:.2f}s ({os.path.getsize(effective_cache_path) / (1024*1024):.1f} MB)")

    return FeatureDataset(features, feature_names, graded_mask, play_live_window_mask)


def print_diagnostic_summary(dataset: FeatureDataset, gt_path: Optional[str] = None):
    """Prints verification and diagnostic stats on public ground truth."""
    features = dataset.features
    names = dataset.feature_names
    graded = dataset.graded_mask
    n_frames, n_features = features.shape

    ball_valid = dataset["ball_valid"]
    st_bias = dataset["st_team_bias"]
    st_sp = dataset["st_set_piece_active"]

    total_ball_vis = 100.0 * np.sum(ball_valid == 1.0) / n_frames
    graded_ball_vis = 100.0 * np.sum((ball_valid == 1.0) & graded) / np.sum(graded)

    st_bias_cov = 100.0 * np.sum(st_bias != 0.0) / n_frames
    st_sp_cov = 100.0 * np.sum(st_sp == 1.0) / n_frames

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY: BAR-ATH FAST STREAMING FEATURE CACHE")
    print("=" * 80)
    print(f"Feature matrix shape:      {n_frames:,} frames x {n_features} features (dtype: {features.dtype})")
    print(f"Graded frames:             {np.sum(graded):,} ({100.0 * np.sum(graded) / n_frames:.1f}%)")
    print(f"Ball visibility (total):   {total_ball_vis:.2f}%")
    print(f"Ball visibility (graded):  {graded_ball_vis:.2f}% (unseen on {100.0 - graded_ball_vis:.2f}%)")
    print(f"Smart Tagging phase bias:  {st_bias_cov:.2f}% coverage")
    print(f"Smart Tagging set pieces:  {st_sp_cov:.2f}% coverage")

    # Public ground truth verification (frames 11185..18684, 7500 frames)
    if gt_path is None:
        gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

    if os.path.isfile(gt_path):
        gt_df = pd.read_csv(gt_path)
        public_slice = (gt_df["frame"] >= 11185) & (gt_df["frame"] <= 18684)
        gt_public = gt_df[public_slice]
        labels = gt_public["label"].value_counts().to_dict()

        pub_start, pub_end = 11185, 18684 + 1
        pub_ball_vis = 100.0 * np.mean(ball_valid[pub_start:pub_end] == 1.0)

        print("\n" + "-" * 80)
        print("PUBLIC GROUND TRUTH SLICE (frames 11185..18684, 7,500 frames = 5 minutes):")
        print(f"Ground Truth labels:       {labels}")
        print(f"Public ball visibility:    {pub_ball_vis:.2f}%")

        key_cols = [
            "ball_valid",
            "d_h1",
            "d_a1",
            "nearest_team_adv",
            "mean_player_speed",
            "max_player_speed",
            "st_team_bias",
            "st_set_piece_active",
            "adv_rolling_w35",
            "same_team_bridge",
        ]

        print("\nFeature statistics on public 5-minute slice:")
        print(f"{'Feature':<24} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'NaNs':>8}")
        print("-" * 75)
        for col in key_cols:
            if col in dataset:
                vals = dataset[col][pub_start:pub_end]
                nan_cnt = int(np.sum(np.isnan(vals)))
                clean_vals = vals[~np.isnan(vals)]
                if len(clean_vals) > 0:
                    v_mean = float(np.mean(clean_vals))
                    v_std = float(np.std(clean_vals))
                    v_min = float(np.min(clean_vals))
                    v_max = float(np.max(clean_vals))
                    print(f"{col:<24} {v_mean:>10.3f} {v_std:>10.3f} {v_min:>10.3f} {v_max:>10.3f} {nan_cnt:>8}")
                else:
                    print(f"{col:<24} {'NaN':>10} {'NaN':>10} {'NaN':>10} {'NaN':>10} {nan_cnt:>8}")
    print("=" * 80 + "\n")


def main(
    rebuild: bool = False,
    data_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> FeatureDataset:
    """Master entrypoint for streaming feature extraction and diagnostic reporting."""
    t_start = time.time()
    ds = extract_all_features(
        data_dir=data_dir,
        cache_path=cache_path,
        rebuild=rebuild,
    )
    print_diagnostic_summary(ds)
    print(f"Total script elapsed time: {time.time() - t_start:.2f}s")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="force rebuild cache")
    parser.add_argument("--data-dir", default=None, help="override MTAH_DATA_DIR")
    parser.add_argument("--cache-path", default=None, help="override cache npz path")
    args = parser.parse_args()

    main(
        rebuild=args.rebuild,
        data_dir=args.data_dir,
        cache_path=args.cache_path,
    )
