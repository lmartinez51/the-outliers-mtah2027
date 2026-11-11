# -*- coding: utf-8 -*-
"""Deterministic tuning harness for external SkillCorner open data.

Governed by Stage 3 rules (S1-S5) and PLAN.md Amendments 1, 2, and 3:
- External data resides outside the repository.
- Baseline theta0 assertions enforce exact alignment with src/v2/constants.py.
- Grid points use multiples of 0.2s for integer frame alignment at 10 and 25 fps.
- Cleanup rules (i) and (ii) are disabled during external evaluation (Amendment 1).
- Adoption uses paired difference bootstrap (B=10000, seed 42) on 8 hold-out matches.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from evaluation.possession_metrics import SPELL_IOUS, label_codes, spell_scores, valid_runs
from src.v2.ball_tracker import build_ball_track
from src.v2.constants import (
    APRIORI,
    MIN_LIVE_FLICKER,
)
from src.v2.state_machine import compose_half_possession

# -----------------------------------------------------------------------------
# 1. Constants, Match Partitions & Assertions
# -----------------------------------------------------------------------------

# Frozen a priori baseline configuration theta0
THETA_0 = {
    "R_TOUCH": 2.0,
    "MARGIN": 0.5,
    "T_CONFIRM": 2.0,
    "G_INTERP": 2.0,
    "MIN_LIVE_FLICKER": 1.0,
}

# Pre-registered tuning grids (Amendment 1: MIN_LIVE_FLICKER fixed a priori at 1.0s)
R_TOUCH_GRID = [1.5, 2.0, 2.5]
MARGIN_GRID = [0.25, 0.50, 0.75]
T_CONFIRM_GRID = [1.2, 2.0, 2.8]
G_INTERP_GRID = [1.2, 2.0, 2.8]

# Match partition (PLAN.md Section 3)
TUNING_MATCH_IDS = [
    1874553,
    1886347,
    1899585,
    1925299,
    1927964,
    1959846,
    1986691,
    1996435,
    1996436,
    2006229,
    2006363,
]

HOLDOUT_MATCH_IDS = [
    2007448,
    2007721,
    2010085,
    2011166,
    2013725,
    2015213,
    2016236,
    2017461,
]

EXCLUDED_MATCH_IDS = [1953632]

ALL_20_MATCH_IDS = sorted(TUNING_MATCH_IDS + HOLDOUT_MATCH_IDS + EXCLUDED_MATCH_IDS)


def verify_harness_assertions() -> None:
    """Verify mandatory theta0 and grid integrity assertions."""
    # 4.3 Mandatory Harness Assertion against frozen a priori baseline
    assert THETA_0["R_TOUCH"] == APRIORI["R_TOUCH"], f"R_TOUCH mismatch: {THETA_0['R_TOUCH']} != {APRIORI['R_TOUCH']}"
    assert THETA_0["MARGIN"] == APRIORI["MARGIN"], f"MARGIN mismatch: {THETA_0['MARGIN']} != {APRIORI['MARGIN']}"
    assert round(THETA_0["T_CONFIRM"] * 25) == APRIORI["T_CONFIRM"], (
        f"T_CONFIRM mismatch: {round(THETA_0['T_CONFIRM'] * 25)} != {APRIORI['T_CONFIRM']}"
    )
    assert round(THETA_0["G_INTERP"] * 25) == APRIORI["G_INTERP"], (
        f"G_INTERP mismatch: {round(THETA_0['G_INTERP'] * 25)} != {APRIORI['G_INTERP']}"
    )
    assert round(THETA_0["MIN_LIVE_FLICKER"] * 25) == MIN_LIVE_FLICKER, (
        f"MIN_LIVE_FLICKER mismatch: {round(THETA_0['MIN_LIVE_FLICKER'] * 25)} != {MIN_LIVE_FLICKER}"
    )

    # Frame count integrity check at 10 fps (SkillCorner) and 25 fps
    for val in T_CONFIRM_GRID + G_INTERP_GRID + [1.0]:
        for fps in (10, 25):
            exact_frames = val * fps
            assert abs(exact_frames - round(exact_frames)) < 1e-9, (
                f"Grid value {val}s does not yield integer frames at {fps} fps"
            )


# -----------------------------------------------------------------------------
# 2. External Data Loading & Ground-Truth Derivation
# -----------------------------------------------------------------------------


def load_raw_match(match_id: int, ext_data_dir: str) -> Dict[str, Any]:
    """Load match metadata, phases of play, and tracking JSONL."""
    match_dir = os.path.join(ext_data_dir, "data", "matches", str(match_id))
    json_path = os.path.join(match_dir, f"{match_id}_match.json")
    phases_path = os.path.join(match_dir, f"{match_id}_phases_of_play.csv")
    jsonl_path = os.path.join(match_dir, f"{match_id}_tracking_extrapolated.jsonl")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing match json: {json_path}")
    if not os.path.exists(phases_path):
        raise FileNotFoundError(f"Missing phases CSV: {phases_path}")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Missing tracking JSONL: {jsonl_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    home_id = meta["home_team"]["id"]
    away_id = meta["away_team"]["id"]
    pitch_len = float(meta.get("pitch_length", 105.0))
    pitch_wid = float(meta.get("pitch_width", 68.0))

    player_team: Dict[int, str] = {}
    for p in meta.get("players", []):
        tid = p.get("team_id")
        if tid == home_id:
            player_team[p["id"]] = "H"
        elif tid == away_id:
            player_team[p["id"]] = "A"

    df_phases = pd.read_csv(phases_path)
    if "period" in df_phases.columns and "frame_start" in df_phases.columns:
        df_phases = df_phases.sort_values(["period", "frame_start", "frame_end"])

    periods_meta = meta.get("match_periods", [])
    if not periods_meta:
        periods_meta = []
        for p in sorted(df_phases["period"].unique()):
            sub = df_phases[df_phases["period"] == p]
            periods_meta.append({
                "period": int(p),
                "start_frame": int(sub["frame_start"].min()),
                "end_frame": int(sub["frame_end"].max()),
            })

    tracking_frames: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            tracking_frames.append(json.loads(line))

    n_frames = len(tracking_frames)

    # 1. Ground truth derivation (PLAN.md Section 7)
    gt_labels = np.full(n_frames, "X", dtype=object)

    for p_info in periods_meta:
        p_num = p_info["period"]
        p_start = p_info["start_frame"]
        p_end = min(p_info["end_frame"], n_frames - 1)

        gt_labels[p_start : p_end + 1] = "D"

        p_df = df_phases[df_phases["period"] == p_num]
        if len(p_df) == 0:
            continue

        # Merge contiguous / overlapping phases into contiguous live stretches (s <= cur_end)
        stretches: List[Tuple[int, int, List[Any]]] = []
        cur_start: Optional[int] = None
        cur_end: Optional[int] = None
        cur_phases: List[Any] = []

        for _, row in p_df.iterrows():
            s = int(row["frame_start"])
            e = min(int(row["frame_end"]), n_frames - 1)
            if cur_start is None:
                cur_start, cur_end = s, e
                cur_phases = [row]
            elif s <= cur_end:
                cur_end = max(cur_end, e)
                cur_phases.append(row)
            else:
                stretches.append((cur_start, cur_end, cur_phases))
                cur_start, cur_end = s, e
                cur_phases = [row]
        if cur_start is not None and cur_end is not None:
            stretches.append((cur_start, cur_end, cur_phases))

        # Assign team possession in each live stretch
        for s, e, phases in stretches:
            stretch_touches: List[Tuple[int, str]] = []
            for f in range(s, e + 1):
                poss = tracking_frames[f].get("possession", {})
                grp = poss.get("group") if poss else None
                if grp == "home team":
                    stretch_touches.append((f, "H"))
                elif grp == "away team":
                    stretch_touches.append((f, "A"))

            if stretch_touches:
                first_f, first_team = stretch_touches[0]
                gt_labels[s : first_f + 1] = first_team
                cur_team = first_team
                for f in range(first_f + 1, e + 1):
                    poss = tracking_frames[f].get("possession", {})
                    grp = poss.get("group") if poss else None
                    if grp == "home team":
                        cur_team = "H"
                    elif grp == "away team":
                        cur_team = "A"
                    gt_labels[f] = cur_team
            else:
                first_tip = phases[0].get("team_in_possession_id")
                fb_team = "H" if first_tip == home_id else "A"
                gt_labels[s : e + 1] = fb_team

    # 2. Extract ball and player tracks
    ball_valid = np.zeros(n_frames, dtype=int)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)

    # Per frame: list of (pid, is_detected, x, y, team)
    player_data_per_frame: List[List[Tuple[int, bool, float, float, str]]] = []
    # Map pid -> dict of frame -> (x, y) for detected players
    detected_trajectories: Dict[int, Dict[int, Tuple[float, float]]] = {}

    for f, d in enumerate(tracking_frames):
        bd = d.get("ball_data")
        if bd and bd.get("is_detected") and bd.get("x") is not None and bd.get("y") is not None:
            ball_valid[f] = 1
            ball_x[f] = float(bd["x"])
            ball_y[f] = float(bd["y"])

        p_list: List[Tuple[int, bool, float, float, str]] = []
        for pl in d.get("player_data", []):
            px, py = pl.get("x"), pl.get("y")
            if px is not None and py is not None:
                pid = pl.get("player_id")
                tm = player_team.get(pid)
                is_det = bool(pl.get("is_detected", False))
                if tm in ("H", "A"):
                    p_list.append((pid, is_det, float(px), float(py), tm))
                if is_det and pid is not None:
                    if pid not in detected_trajectories:
                        detected_trajectories[pid] = {}
                    detected_trajectories[pid][f] = (float(px), float(py))
        player_data_per_frame.append(p_list)

    return {
        "match_id": match_id,
        "n_frames": n_frames,
        "periods_meta": periods_meta,
        "gt_labels": gt_labels,
        "ball_valid": ball_valid,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "player_data": player_data_per_frame,
        "detected_trajectories": detected_trajectories,
        "pitch_length": pitch_len,
        "pitch_width": pitch_wid,
        "df_phases": df_phases,
    }


# -----------------------------------------------------------------------------
# 3. Ball Tracking, Candidate Extraction & Pipeline Execution
# -----------------------------------------------------------------------------


def compute_match_ball_tracks(
    match_data: Dict[str, Any], g_interp_grid: List[float]
) -> Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute interpolated ball track for each G_INTERP value (in seconds)."""
    n_frames = match_data["n_frames"]
    ball_valid = match_data["ball_valid"]
    ball_x = match_data["ball_x"]
    ball_y = match_data["ball_y"]
    periods_meta = match_data["periods_meta"]

    ball_tracks: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for g_sec in g_interp_grid:
        g_frames = round(g_sec * 10)  # 10 fps
        full_known = np.zeros(n_frames, dtype=bool)
        full_bx = np.full(n_frames, np.nan, dtype=np.float32)
        full_by = np.full(n_frames, np.nan, dtype=np.float32)

        for p_info in periods_meta:
            p_start = p_info["start_frame"]
            p_end = min(p_info["end_frame"], n_frames - 1)
            b_known, bx, by = build_ball_track(
                ball_valid, ball_x, ball_y, p_start, p_end, g_interp=g_frames
            )
            full_known[p_start : p_end + 1] = b_known[p_start : p_end + 1]
            full_bx[p_start : p_end + 1] = bx[p_start : p_end + 1]
            full_by[p_start : p_end + 1] = by[p_start : p_end + 1]

        ball_tracks[g_sec] = (full_known, full_bx, full_by)

    return ball_tracks


def compute_match_distances(
    match_data: Dict[str, Any],
    ball_tracks: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Compute d_H and d_A for each G_INTERP value."""
    n_frames = match_data["n_frames"]
    ball_valid = match_data["ball_valid"]
    ball_x = match_data["ball_x"]
    ball_y = match_data["ball_y"]
    player_data = match_data["player_data"]

    # 1. Base distances on valid ball frames
    base_dH = np.full(n_frames, np.nan, dtype=np.float32)
    base_dA = np.full(n_frames, np.nan, dtype=np.float32)

    for f in range(n_frames):
        if ball_valid[f] == 1:
            bx, by = ball_x[f], ball_y[f]
            min_h = 1e9
            min_a = 1e9
            for _, _, px, py, tm in player_data[f]:
                d = math.hypot(px - bx, py - by)
                if tm == "H":
                    if d < min_h:
                        min_h = d
                elif tm == "A":
                    if d < min_a:
                        min_a = d
            if min_h < 1e8:
                base_dH[f] = min_h
            if min_a < 1e8:
                base_dA[f] = min_a

    # 2. Add interpolated frames for each g_interp
    distances_by_g: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    for g_sec, (b_known, bx_arr, by_arr) in ball_tracks.items():
        dH = base_dH.copy()
        dA = base_dA.copy()

        interp_mask = b_known & (ball_valid == 0)
        for f in np.flatnonzero(interp_mask):
            bx, by = bx_arr[f], by_arr[f]
            min_h = 1e9
            min_a = 1e9
            for _, _, px, py, tm in player_data[f]:
                d = math.hypot(px - bx, py - by)
                if tm == "H":
                    if d < min_h:
                        min_h = d
                elif tm == "A":
                    if d < min_a:
                        min_a = d
            if min_h < 1e8:
                dH[f] = min_h
            if min_a < 1e8:
                dA[f] = min_a

        distances_by_g[g_sec] = (dH, dA)

    return distances_by_g


def evaluate_single_match(
    match_data: Dict[str, Any],
    distances_by_g: Dict[float, Tuple[np.ndarray, np.ndarray]],
    r_touch: float,
    margin: float,
    t_confirm_sec: float,
    g_interp_sec: float,
) -> float:
    """Evaluate pipeline on one match for given parameters and return spell_f1_mean."""
    n_frames = match_data["n_frames"]
    gt_labels = match_data["gt_labels"]
    periods_meta = match_data["periods_meta"]

    dH, dA = distances_by_g[g_interp_sec]
    t_confirm_frames = round(t_confirm_sec * 10)  # 10 fps

    candidates = np.full(n_frames, None, dtype=object)
    valid_d = ~np.isnan(dH) & ~np.isnan(dA)
    min_d = np.minimum(dH, dA)

    h_mask = valid_d & (min_d <= r_touch) & (dH + margin <= dA)
    a_mask = valid_d & (min_d <= r_touch) & (dA + margin <= dH)
    candidates[h_mask] = "H"
    candidates[a_mask] = "A"

    pred_labels = np.full(n_frames, "X", dtype=object)

    for p_info in periods_meta:
        p_start = p_info["start_frame"]
        p_end = min(p_info["end_frame"], n_frames - 1)

        period_dead_mask = np.zeros(n_frames, dtype=bool)
        period_dead_mask[p_start : p_end + 1] = gt_labels[p_start : p_end + 1] == "D"

        # Core state machine (Amendment 1: cleanup rules disabled)
        p_pred, _ = compose_half_possession(
            dead_mask=period_dead_mask,
            candidates=candidates,
            start_frame=p_start,
            end_frame=p_end,
            initial_possessor=None,
            t_confirm=t_confirm_frames,
        )
        pred_labels[p_start : p_end + 1] = p_pred[p_start : p_end + 1]

    gt_codes = label_codes(gt_labels)
    pred_codes = label_codes(pred_labels)
    runs = valid_runs(gt_codes)

    scores = spell_scores(gt_codes, pred_codes, runs, ious=SPELL_IOUS)
    return float(scores["mean_f1"])


# -----------------------------------------------------------------------------
# 4. Bootstrap Inference (Hold-out Paired Differences)
# -----------------------------------------------------------------------------


def compute_paired_bootstrap_ci(
    deltas: np.ndarray,
    n_draws: int = 10000,
    seed: int = 42,
    percentiles: Tuple[float, float] = (2.5, 97.5),
) -> Tuple[float, float]:
    """Compute empirical bootstrap confidence interval of the mean paired difference."""
    rng = np.random.RandomState(seed)
    n = len(deltas)
    picks = rng.randint(0, n, size=(n_draws, n))
    sampled_means = np.mean(deltas[picks], axis=1)
    low, high = np.percentile(sampled_means, percentiles)
    return float(low), float(high)


# -----------------------------------------------------------------------------
# 5. Amendment 2 Diagnostics
# -----------------------------------------------------------------------------


def compute_diagnostics_2a(
    match_data: Dict[str, Any], distances_by_g: Dict[float, Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, float]:
    """Compute report-only diagnostic (a) for one match under baseline theta0."""
    n_frames = match_data["n_frames"]
    gt_labels = match_data["gt_labels"]
    ball_valid = match_data["ball_valid"]
    periods_meta = match_data["periods_meta"]

    period_mask = np.zeros(n_frames, dtype=bool)
    for p_info in periods_meta:
        p_start = p_info["start_frame"]
        p_end = min(p_info["end_frame"], n_frames - 1)
        period_mask[p_start : p_end + 1] = True

    total_period_frames = int(period_mask.sum())
    live_mask = period_mask & (gt_labels != "D") & (gt_labels != "X")
    total_live_frames = int(live_mask.sum())

    dead_period_frames = int((period_mask & (gt_labels == "D")).sum())
    out_of_play_share = (
        float(dead_period_frames) / total_period_frames if total_period_frames > 0 else 0.0
    )

    ball_det_in_phase = int((live_mask & (ball_valid == 1)).sum())
    ball_detected_share = (
        float(ball_det_in_phase) / total_live_frames if total_live_frames > 0 else 0.0
    )

    dH, dA = distances_by_g[2.0]
    valid_d = ~np.isnan(dH) & ~np.isnan(dA)
    min_d = np.minimum(dH, dA)
    cand_mask = valid_d & (min_d <= 2.0) & ((dH + 0.5 <= dA) | (dA + 0.5 <= dH))

    cand_in_phase = int((live_mask & cand_mask).sum())
    candidate_share = float(cand_in_phase) / total_live_frames if total_live_frames > 0 else 0.0

    return {
        "total_period_frames": total_period_frames,
        "total_live_frames": total_live_frames,
        "out_of_play_share": out_of_play_share,
        "ball_detected_share": ball_detected_share,
        "candidate_share": candidate_share,
    }


def _mark_runs_at_least(condition_array: np.ndarray, min_length: int) -> np.ndarray:
    """Find contiguous runs of True with length >= min_length and return mask."""
    n = len(condition_array)
    out_mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if condition_array[i]:
            start = i
            while i < n and condition_array[i]:
                i += 1
            if (i - start) >= min_length:
                out_mask[start:i] = True
        else:
            i += 1
    return out_mask


def compute_diagnostics_2b(
    match_data: Dict[str, Any],
    ball_tracks: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """Compute report-only diagnostic (b) for one match using v2's a priori dead-ball rules.

    Applies out-of-bounds (1.0m margin, T_OOB=5 frames @ 10fps) and collective stoppage
    (T_STOP=15 frames @ 10fps, V_STOP=1.35 m/s, player speeds as 1s centered displacement
    of detected players only).
    """
    n_frames = match_data["n_frames"]
    gt_labels = match_data["gt_labels"]
    detected_trajectories = match_data["detected_trajectories"]
    periods_meta = match_data["periods_meta"]
    pitch_len = match_data["pitch_length"]
    pitch_wid = match_data["pitch_width"]

    # Ball track under baseline G_INTERP = 2.0s
    b_known, bx_arr, by_arr = ball_tracks[2.0]

    # Pre-compute p75 speeds using 1s centered displacement (delta = 5 frames at 10 fps)
    delta = 5
    p75_speeds = np.full(n_frames, np.nan, dtype=np.float32)
    for f in range(delta, n_frames - delta):
        speeds_f: List[float] = []
        for _, pos_dict in detected_trajectories.items():
            if f in pos_dict and (f - delta) in pos_dict and (f + delta) in pos_dict:
                x0, y0 = pos_dict[f - delta]
                x1, y1 = pos_dict[f + delta]
                spd = math.hypot(x1 - x0, y1 - y0)  # / 1.0s
                speeds_f.append(spd)
        if speeds_f:
            p75_speeds[f] = float(np.percentile(speeds_f, 75))

    pred_dead = np.zeros(n_frames, dtype=bool)
    ref_dead = np.zeros(n_frames, dtype=bool)
    period_mask = np.zeros(n_frames, dtype=bool)

    half_l = pitch_len / 2.0
    half_w = pitch_wid / 2.0
    half_win = 5  # 1.0s window for speed smoothing

    for p_info in periods_meta:
        s_fr = p_info["start_frame"]
        e_fr = min(p_info["end_frame"], n_frames - 1)
        period_mask[s_fr : e_fr + 1] = True
        ref_dead[s_fr : e_fr + 1] = gt_labels[s_fr : e_fr + 1] == "D"

        # 1. Out of bounds (1.0m margin, T_OOB = 5 frames)
        oob_inst = np.zeros(n_frames, dtype=bool)
        for f in range(s_fr, e_fr + 1):
            if b_known[f]:
                if abs(bx_arr[f]) > (half_l + 1.0) or abs(by_arr[f]) > (half_w + 1.0):
                    oob_inst[f] = True
        mask_oob = _mark_runs_at_least(oob_inst[s_fr : e_fr + 1], 5)

        # 2. Collective stoppage (V_STOP = 1.35 m/s, T_STOP = 15 frames)
        p75_smooth = np.zeros(n_frames, dtype=np.float32)
        for f in range(s_fr, e_fr + 1):
            w_start = max(s_fr, f - half_win)
            w_end = min(e_fr + 1, f + half_win + 1)
            slice_s = p75_speeds[w_start:w_end]
            valid_s = slice_s[~np.isnan(slice_s)]
            p75_smooth[f] = float(np.mean(valid_s)) if len(valid_s) > 0 else 0.0

        stop_inst = np.zeros(n_frames, dtype=bool)
        for f in range(s_fr, e_fr + 1):
            p_stop = p75_smooth[f] < 1.35
            t1 = f - half_win
            t2 = f + half_win
            if t1 >= s_fr and t2 <= e_fr and b_known[t1] and b_known[t2]:
                b_spd = math.hypot(bx_arr[t2] - bx_arr[t1], by_arr[t2] - by_arr[t1])
                b_stop = b_spd < 1.5
            else:
                b_stop = True
            if p_stop and b_stop:
                stop_inst[f] = True
        mask_stop = _mark_runs_at_least(stop_inst[s_fr : e_fr + 1], 15)

        pred_dead[s_fr : e_fr + 1] = mask_oob | mask_stop

    # Compute agreement over all period frames
    p_pred = pred_dead[period_mask]
    p_ref = ref_dead[period_mask]

    tp = int((p_pred & p_ref).sum())
    fp = int((p_pred & ~p_ref).sum())
    fn = int((~p_pred & p_ref).sum())

    prec = float(tp) / (tp + fp) if (tp + fp) else 0.0
    rec = float(tp) / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tot = len(p_ref)

    return {
        "ref_dead_share": float(p_ref.sum()) / tot if tot else 0.0,
        "pred_dead_share": float(p_pred.sum()) / tot if tot else 0.0,
        "precision_d": prec,
        "recall_d": rec,
        "f1_d": f1,
        "tp_frames": tp,
        "fp_frames": fp,
        "fn_frames": fn,
    }


# -----------------------------------------------------------------------------
# 6. Full Experiment Protocol
# -----------------------------------------------------------------------------


def run_tuning_protocol(
    ext_data_dir: Optional[str] = None,
    results_dir: str = os.path.join("tools", "tuning", "results"),
) -> Dict[str, Any]:
    """Execute complete Stage 3 tuning protocol."""
    if ext_data_dir is None:
        ext_data_dir = os.environ.get(
            "SKILLCORNER_DATA_DIR",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "ext_data", "skillcorner")
        )
    start_time = time.time()
    os.makedirs(results_dir, exist_ok=True)

    print("===================================================================")
    print("STAGE 3: EXTERNAL TUNING PROTOCOL (SkillCorner Open Data)")
    print("===================================================================")

    verify_harness_assertions()
    print("[PASS] Harness theta0 and grid integrity assertions verified.")

    print("\n--- Phase 1: Preloading Match Data ---")
    loaded_matches: Dict[int, Dict[str, Any]] = {}
    match_distances: Dict[int, Dict[float, Tuple[np.ndarray, np.ndarray]]] = {}
    match_ball_tracks: Dict[
        int, Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = {}

    for mid in ALL_20_MATCH_IDS:
        t_m0 = time.time()
        m_data = load_raw_match(mid, ext_data_dir)
        b_tracks = compute_match_ball_tracks(m_data, G_INTERP_GRID)
        dists = compute_match_distances(m_data, b_tracks)
        loaded_matches[mid] = m_data
        match_ball_tracks[mid] = b_tracks
        match_distances[mid] = dists
        print(
            f"  Match {mid} loaded: {m_data['n_frames']} frames in {time.time() - t_m0:.2f}s"
        )

    print("\n--- Phase 2: Baseline theta0 Evaluation ---")
    tuning_theta0_scores: Dict[int, float] = {}
    for mid in TUNING_MATCH_IDS:
        sc = evaluate_single_match(
            loaded_matches[mid],
            match_distances[mid],
            r_touch=2.0,
            margin=0.5,
            t_confirm_sec=2.0,
            g_interp_sec=2.0,
        )
        tuning_theta0_scores[mid] = sc

    holdout_theta0_scores: Dict[int, float] = {}
    for mid in HOLDOUT_MATCH_IDS:
        sc = evaluate_single_match(
            loaded_matches[mid],
            match_distances[mid],
            r_touch=2.0,
            margin=0.5,
            t_confirm_sec=2.0,
            g_interp_sec=2.0,
        )
        holdout_theta0_scores[mid] = sc

    mean_tuning_theta0 = float(np.mean(list(tuning_theta0_scores.values())))
    mean_holdout_theta0 = float(np.mean(list(holdout_theta0_scores.values())))
    print(f"  Tuning Set baseline theta0 mean F1:  {mean_tuning_theta0:.5f}")
    print(f"  Hold-out Set baseline theta0 mean F1: {mean_holdout_theta0:.5f}")

    print(
        f"\n--- Phase 3: Evaluating 81-Configuration Grid on {len(TUNING_MATCH_IDS)} Tuning Matches ---"
    )
    grid_configs: List[Dict[str, float]] = []
    for r in R_TOUCH_GRID:
        for m in MARGIN_GRID:
            for t in T_CONFIRM_GRID:
                for g in G_INTERP_GRID:
                    grid_configs.append({
                        "R_TOUCH": r,
                        "MARGIN": m,
                        "T_CONFIRM": t,
                        "G_INTERP": g,
                    })

    assert len(grid_configs) == 81, f"Expected 81 configs, got {len(grid_configs)}"

    grid_results: List[Dict[str, Any]] = []
    grid_start_time = time.time()

    for idx, cfg in enumerate(grid_configs, start=1):
        r = cfg["R_TOUCH"]
        m = cfg["MARGIN"]
        t = cfg["T_CONFIRM"]
        g = cfg["G_INTERP"]

        scores = [
            evaluate_single_match(
                loaded_matches[mid],
                match_distances[mid],
                r_touch=r,
                margin=m,
                t_confirm_sec=t,
                g_interp_sec=g,
            )
            for mid in TUNING_MATCH_IDS
        ]
        mean_score = float(np.mean(scores))

        dist_to_theta0 = math.sqrt(
            ((r - 2.0) / 0.5) ** 2
            + ((m - 0.5) / 0.25) ** 2
            + ((t - 2.0) / 0.8) ** 2
            + ((g - 2.0) / 0.8) ** 2
        )

        record = {
            "config_id": idx,
            "R_TOUCH": r,
            "MARGIN": m,
            "T_CONFIRM": t,
            "G_INTERP": g,
            "mean_spell_f1": mean_score,
            "dist_to_theta0": dist_to_theta0,
            "scores_by_match": {mid: sc for mid, sc in zip(TUNING_MATCH_IDS, scores)},
        }
        grid_results.append(record)

    print(
        f"  81-grid completed ({81 * len(TUNING_MATCH_IDS)} match evaluations) in {time.time() - grid_start_time:.2f}s"
    )

    grid_results.sort(key=lambda item: (-item["mean_spell_f1"], item["dist_to_theta0"]))
    best_config_record = grid_results[0]
    best_theta = {
        "R_TOUCH": best_config_record["R_TOUCH"],
        "MARGIN": best_config_record["MARGIN"],
        "T_CONFIRM": best_config_record["T_CONFIRM"],
        "G_INTERP": best_config_record["G_INTERP"],
    }
    print(
        f"  Top candidate theta*: {best_theta} with tuning mean F1 = {best_config_record['mean_spell_f1']:.5f}"
    )

    print(
        f"\n--- Phase 4: Hold-Out Evaluation ({len(HOLDOUT_MATCH_IDS)} Matches) ---"
    )
    holdout_star_scores: Dict[int, float] = {}
    paired_deltas: List[float] = []

    for mid in HOLDOUT_MATCH_IDS:
        sc_star = evaluate_single_match(
            loaded_matches[mid],
            match_distances[mid],
            r_touch=best_theta["R_TOUCH"],
            margin=best_theta["MARGIN"],
            t_confirm_sec=best_theta["T_CONFIRM"],
            g_interp_sec=best_theta["G_INTERP"],
        )
        sc_0 = holdout_theta0_scores[mid]
        delta = sc_star - sc_0
        holdout_star_scores[mid] = sc_star
        paired_deltas.append(delta)

    deltas_arr = np.array(paired_deltas, dtype=np.float64)
    mean_delta = float(np.mean(deltas_arr))

    ci42_low, ci42_high = compute_paired_bootstrap_ci(
        deltas_arr, n_draws=10000, seed=42
    )
    ci0_low, ci0_high = compute_paired_bootstrap_ci(
        deltas_arr, n_draws=10000, seed=0
    )
    ci1_low, ci1_high = compute_paired_bootstrap_ci(
        deltas_arr, n_draws=10000, seed=1
    )

    print(f"  Hold-out mean paired difference delta_bar: {mean_delta:+.6f}")
    print(f"  Bootstrap 95% CI (seed 42 - DECISION):     [{ci42_low:+.6f}, {ci42_high:+.6f}]")
    print(f"  Bootstrap 95% CI (seed 0 - robustness):    [{ci0_low:+.6f}, {ci0_high:+.6f}]")
    print(f"  Bootstrap 95% CI (seed 1 - robustness):    [{ci1_low:+.6f}, {ci1_high:+.6f}]")

    adopted = bool(mean_delta > 0 and ci42_low > 0)
    print(f"\n--- Phase 5: Mechanized Adoption Decision ---")
    print(
        f"  Criteria: mean_delta > 0 ({mean_delta > 0}) AND ci42_low > 0 ({ci42_low > 0})"
    )
    print(f"  ADOPTION VERDICT: {'ADOPTED' if adopted else 'NOT ADOPTED (REJECTED)'}")

    print("\n--- Phase 6: Amendment 2 Diagnostics across All 20 Matches ---")
    # 6.1 Diagnostic 2(a)
    diag_2a_records: List[Dict[str, Any]] = []
    tot_period_f = 0
    tot_live_f = 0
    tot_dead_period_f = 0
    tot_ball_det_f = 0
    tot_cand_f = 0

    for mid in ALL_20_MATCH_IDS:
        d2a = compute_diagnostics_2a(loaded_matches[mid], match_distances[mid])
        tot_period_f += d2a["total_period_frames"]
        tot_live_f += d2a["total_live_frames"]
        tot_dead_period_f += round(
            d2a["out_of_play_share"] * d2a["total_period_frames"]
        )
        tot_ball_det_f += round(
            d2a["ball_detected_share"] * d2a["total_live_frames"]
        )
        tot_cand_f += round(d2a["candidate_share"] * d2a["total_live_frames"])

        rec = {
            "match_id": mid,
            "partition": (
                "tuning"
                if mid in TUNING_MATCH_IDS
                else ("holdout" if mid in HOLDOUT_MATCH_IDS else "excluded")
            ),
            "period_frames": d2a["total_period_frames"],
            "live_frames": d2a["total_live_frames"],
            "out_of_play_share": d2a["out_of_play_share"],
            "ball_detected_share": d2a["ball_detected_share"],
            "candidate_share": d2a["candidate_share"],
        }
        diag_2a_records.append(rec)

    aggregate_2a = {
        "aggregate_out_of_play_share": (
            float(tot_dead_period_f) / tot_period_f if tot_period_f > 0 else 0.0
        ),
        "aggregate_ball_detected_share": (
            float(tot_ball_det_f) / tot_live_f if tot_live_f > 0 else 0.0
        ),
        "aggregate_candidate_share": (
            float(tot_cand_f) / tot_live_f if tot_live_f > 0 else 0.0
        ),
    }

    # 6.2 Diagnostic 2(b)
    diag_2b_records: List[Dict[str, Any]] = []
    tot_ref_dead_f = 0
    tot_pred_dead_f = 0
    tot_tp_f = 0
    tot_fp_f = 0
    tot_fn_f = 0

    for mid in ALL_20_MATCH_IDS:
        d2b = compute_diagnostics_2b(
            loaded_matches[mid], match_ball_tracks[mid]
        )
        tot_tp_f += d2b["tp_frames"]
        tot_fp_f += d2b["fp_frames"]
        tot_fn_f += d2b["fn_frames"]
        rec_b = {
            "match_id": mid,
            "partition": (
                "tuning"
                if mid in TUNING_MATCH_IDS
                else ("holdout" if mid in HOLDOUT_MATCH_IDS else "excluded")
            ),
            "ref_dead_share": d2b["ref_dead_share"],
            "pred_dead_share": d2b["pred_dead_share"],
            "precision_d": d2b["precision_d"],
            "recall_d": d2b["recall_d"],
            "f1_d": d2b["f1_d"],
        }
        diag_2b_records.append(rec_b)

    overall_prec = (
        float(tot_tp_f) / (tot_tp_f + tot_fp_f) if (tot_tp_f + tot_fp_f) else 0.0
    )
    overall_rec = (
        float(tot_tp_f) / (tot_tp_f + tot_fn_f) if (tot_tp_f + tot_fn_f) else 0.0
    )
    overall_f1 = (
        2.0 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec)
        else 0.0
    )
    aggregate_2b = {
        "aggregate_precision_d": overall_prec,
        "aggregate_recall_d": overall_rec,
        "aggregate_f1_d": overall_f1,
    }

    print(
        f"  Aggregate Out-of-play share:    {aggregate_2a['aggregate_out_of_play_share'] * 100:.2f}%"
    )
    print(
        f"  Aggregate Ball-detected share:  {aggregate_2a['aggregate_ball_detected_share'] * 100:.2f}%"
    )
    print(
        f"  Aggregate Candidate share (t0): {aggregate_2a['aggregate_candidate_share'] * 100:.2f}%"
    )
    print(
        f"  Aggregate Dead-ball Precision:  {aggregate_2b['aggregate_precision_d'] * 100:.2f}%"
    )
    print(
        f"  Aggregate Dead-ball Recall:     {aggregate_2b['aggregate_recall_d'] * 100:.2f}%"
    )
    print(
        f"  Aggregate Dead-ball F1:         {aggregate_2b['aggregate_f1_d'] * 100:.2f}%"
    )

    # 7. Save outputs
    df_grid = pd.DataFrame([
        {
            "config_id": r["config_id"],
            "r_touch_m": r["R_TOUCH"],
            "margin_m": r["MARGIN"],
            "t_confirm_s": r["T_CONFIRM"],
            "g_interp_s": r["G_INTERP"],
            "mean_spell_f1": r["mean_spell_f1"],
            "dist_to_theta0": r["dist_to_theta0"],
        }
        for r in grid_results
    ])
    grid_csv_path = os.path.join(results_dir, "grid_tuning_results.csv")
    df_grid.to_csv(grid_csv_path, index=False)

    df_holdout = pd.DataFrame([
        {
            "match_id": mid,
            "theta0_f1": holdout_theta0_scores[mid],
            "thetastar_f1": holdout_star_scores[mid],
            "delta": holdout_star_scores[mid] - holdout_theta0_scores[mid],
        }
        for mid in HOLDOUT_MATCH_IDS
    ])
    holdout_csv_path = os.path.join(results_dir, "holdout_comparison.csv")
    df_holdout.to_csv(holdout_csv_path, index=False)

    df_diag2a = pd.DataFrame(diag_2a_records)
    diag2a_csv_path = os.path.join(results_dir, "diagnostics_amendment_2a.csv")
    df_diag2a.to_csv(diag2a_csv_path, index=False)

    df_diag2b = pd.DataFrame(diag_2b_records)
    diag2b_csv_path = os.path.join(results_dir, "diagnostics_amendment_2b.csv")
    df_diag2b.to_csv(diag2b_csv_path, index=False)

    total_runtime = time.time() - start_time
    summary_data = {
        "theta0": THETA_0,
        "best_theta": best_theta,
        "mean_tuning_theta0": mean_tuning_theta0,
        "best_tuning_mean_f1": best_config_record["mean_spell_f1"],
        "mean_holdout_theta0": mean_holdout_theta0,
        "mean_holdout_thetastar": float(
            np.mean(list(holdout_star_scores.values()))
        ),
        "mean_paired_difference": mean_delta,
        "bootstrap_ci_seed42": [ci42_low, ci42_high],
        "bootstrap_ci_seed0": [ci0_low, ci0_high],
        "bootstrap_ci_seed1": [ci1_low, ci1_high],
        "adopted": adopted,
        "aggregate_diagnostics_2a": aggregate_2a,
        "aggregate_diagnostics_2b": aggregate_2b,
        "runtime_seconds": total_runtime,
    }
    summary_json_path = os.path.join(results_dir, "tuning_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nAll results saved to: {results_dir}")
    print(f"Total protocol execution time: {total_runtime:.2f}s")
    print("===================================================================")

    return summary_data


if __name__ == "__main__":
    run_tuning_protocol()
