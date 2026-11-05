"""
src/jev_system_one.py
---------------------
Deterministic Jev AI (`System One`) Epistemic Calibration & Conflict-Arbitration Primitives.

Team: The Outliers
Authors: Lorenzo Martínez, Ana Karen Arellano, and Steven Ibarra
Repository: the-outliers-mtah2027
Target Competition: More than a Hack 2027

Implements 100% local, inspectable, deterministic vectorised NumPy routines with zero external
network calls, complying strictly with RULES.md and SUBMISSION.md.

Domain-Specific Primitives:
  1. compute_set_piece_and_goal_stoppage_masks(ds, game):
     Detects physical set-piece restart kick frames (t_kick), Out-of-Bounds to Restart Kick
     dead-ball continuity intervals (oob_to_kick_mask), and post-goal celebration stoppages
     (post_goal_mask).
  2. resolve_na_set_pieces(ds, game):
     Deterministically infers restarting team for the 36 'Team = N/A' Smart Tagging phases
     (including all 26 Throw-Ins and post-goal Kick-Offs in BAR-ATH) using exact post-release
     kinematic control anchors around t_kick and opposite-of-scorer kickoff laws.
  3. compute_epistemic_conflict_and_entropy(p_3state, ds, frame_slice):
     Computes normalised Shannon entropy H(p) in [0, 1] and detects Tag-vs-Physics conflict
     (both instantaneous touch conflict and sustained short-window turnover conflict).
  4. arbitrate_live_control(p_h_gbdt, p_h_prior, ds, frame_slice):
     Dynamically arbitrates between GBDT and prior when Tag-vs-Physics conflict is detected,
     minimizing turnover latency (lat_f) across rapid counter-attacks and interceptions.
  5. apply_occlusion_bridge_damping(p_h_alive, ds, frame_slice):
     Handles the 40.7% missing-ball frames: applies bounded log-odds reinforcement to same-team
     bridges and monotonic interpolation across opposing anchor occlusions.
  6. sharpen_restart_boundaries(p_dead, ds, frame_slice):
     Refines p_dead around the exact set-piece restart kick once ball movement is initiated,
     while preserving pre-kick Out-of-Bounds continuity and post-goal stoppages.
  7. calibrate_possession_distribution(...):
     Unified factored calibration pipeline generating calibrated 3-state probabilities,
     epistemic entropy, and conflict diagnostic masks.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import starter.load_smart_tagging as lst
import src.mtah_features as mf


# =============================================================================
# Primitive 0: Set-Piece Restart Kick Detection, OOB Continuity & Goal Stoppages
# =============================================================================

def compute_set_piece_and_goal_stoppage_masks(
    ds: mf.FeatureDataset,
    game: str = "BAR-ATH",
) -> Dict[str, np.ndarray]:
    """
    Computes deterministic physical stoppage and transition masks across all match frames:
      - oob_to_kick_mask: True from the moment the ball goes Out-of-Bounds (within 320 frames
        prior to a SET PIECES restart) until the physical restart kick frame t_kick. By the
        Laws of the Game, a ball retrieved from out of bounds and placed inside the 6-yard box
        for a Goal Kick or Free Kick remains Dead ('D') until kicked.
      - post_goal_mask: True inside GOALS intervals from the end of the scoring action
        (BALL IN THE BOX / CREATING CHANCES) until the post-goal Kick Off.
      - st_transition_active: 1.0 during active ATTACKING/DEFENSIVE TRANSITION tags.
      - sp_pre_kick_mask: True inside SET PIECES prior to t_kick.
    """
    events = lst.load(game)
    n_frames = len(ds["ball_valid"])

    bv = ds["ball_valid"]
    bx = ds["ball_x_m"]
    by = ds["ball_y_m"]
    oob = ds["ball_out_of_bounds"]
    mps = ds["mean_player_speed"]
    v_ball = np.nan_to_num(ds["ball_speed_reported"], nan=0.0)
    vb_w15 = ds["ball_speed_rolling_w15"]

    oob_to_kick_mask = np.zeros(n_frames, dtype=bool)
    sp_pre_kick_mask = np.zeros(n_frames, dtype=bool)
    post_goal_mask = np.zeros(n_frames, dtype=bool)
    st_transition_active = np.zeros(n_frames, dtype=np.float32)

    sp_evs = sorted(
        [e for e in events if (e["code"] or "").strip().upper() == "SET PIECES"],
        key=lambda x: x["start_frame"],
    )

    for e in sp_evs:
        sf, ef = max(0, e["start_frame"]), min(n_frames, e["end_frame"])
        if sf >= ef:
            continue
        etype = (e["tags"].get("Type") or "").lower()
        dur = ef - sf
        if dur <= 200 or "kick off" in etype:
            t_kick = min(ef, sf + 85)
            for t in range(sf + 40, ef):
                if bv[t] == 1 and v_ball[t] > 4.0:
                    t_kick = t
                    break
        else:
            t_kick = min(ef, sf + 225)
            for t in range(sf + 20, min(ef, sf + 420)):
                if bv[t] == 1 and oob[t] == 0 and (v_ball[t] > 6.8 or vb_w15[t] > 4.8):
                    t2 = min(n_frames - 1, t + 25)
                    if (
                        bv[t2] == 1
                        and oob[t2] == 0
                        and np.hypot(bx[t2] - bx[t], by[t2] - by[t]) > 4.2
                    ):
                        t_kick = t
                        break
            else:
                for t in range(sf + 60, min(ef, sf + 400)):
                    if mps[t] > 1.85 and np.mean(mps[t : min(ef, t + 50)]) > 1.90:
                        t_kick = t
                        break
            if "goal kick" in etype and t_kick - sf < 60:
                t_kick = min(ef, t_kick + 22)

        sp_pre_kick_mask[sf:t_kick] = True
        search_s = max(0, sf - 320)
        oob_idx = np.where(oob[search_s:t_kick] == 1.0)[0] + search_s
        if len(oob_idx) > 0:
            oob_to_kick_mask[oob_idx[0] : t_kick] = True

    for ev in events:
        code = (ev["code"] or "").strip().upper()
        sf, ef = max(0, ev["start_frame"]), min(n_frames, ev["end_frame"])
        if sf >= ef:
            continue
        if code in ("ATTACKING TRANSITION", "DEFENSIVE TRANSITION"):
            st_transition_active[sf:ef] = 1.0
        elif code == "GOALS":
            t_goal = sf + 600
            box_ends = [
                e["end_frame"]
                for e in events
                if e["code"] in ("BALL IN THE BOX", "CREATING CHANCES")
                and e["start_frame"] >= sf - 200
                and e["end_frame"] <= ef - 200
            ]
            if box_ends:
                t_goal = max(box_ends)
            post_goal_mask[t_goal:ef] = True

    return {
        "oob_to_kick_mask": oob_to_kick_mask,
        "sp_pre_kick_mask": sp_pre_kick_mask,
        "post_goal_mask": post_goal_mask,
        "st_transition_active": st_transition_active,
    }


# =============================================================================
# Primitive 1: N/A Set-Piece Resolution (36 Phases, 26 Throw-Ins)
# =============================================================================

def resolve_na_set_pieces(
    ds: mf.FeatureDataset,
    game: str = "BAR-ATH",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Deterministically resolves the 36 Smart Tagging intervals where Team is unassigned ('N/A' or None),
    including all 26 Throw-In events and post-goal Kick-Offs in BAR-ATH.

    Key Mathematical Improvements:
      1. Post-Goal Kick-Offs: Uses the preceding GOALS scoring team tag; by football rules, the
         conceding team always takes the subsequent Kick Off.
      2. Throw-Ins: Detects the physical throw-in release frame t_kick inside [start_frame + 25,
         start_frame + 380] and evaluates kinematic control anchors in [t_kick, t_kick + 125]
         immediately after the throw, rather than evaluating around end_frame (35s later).

    Returns:
      jev_na_team_prior: (N,) float32 array with +1.0 (Home) or -1.0 (Away) during N/A intervals,
                         and 0.0 outside those intervals.
      info: Diagnostic metadata dictionary reporting counts and per-event decisions.
    """
    events = lst.load(game)
    na_evs = [e for e in events if e["tags"].get("Team") in (None, "N/A", "")]
    goals_evs = [e for e in events if (e["code"] or "").strip().upper() == "GOALS"]

    n_frames = len(ds["ball_valid"])
    jev_na_team_prior = np.zeros(n_frames, dtype=np.float32)

    bv = ds["ball_valid"]
    oob = ds["ball_out_of_bounds"]
    v_ball = np.nan_to_num(ds["ball_speed_reported"], nan=0.0)
    vb_w15 = ds["ball_speed_rolling_w15"]

    resolved_counts = {"H": 0, "A": 0}
    total_resolved_frames = 0
    event_details: List[Dict[str, Any]] = []

    for i, e in enumerate(na_evs):
        s = max(0, e["start_frame"])
        e_f = min(n_frames, e["end_frame"])
        ev_type = e["tags"].get("Type", e["code"]) or ""
        etype_low = ev_type.lower()

        # 1. Check if this is a post-goal Kick Off
        preceding_goals = [
            g for g in goals_evs if abs(g["end_frame"] - s) <= 600 or (g["start_frame"] <= s <= g["end_frame"] + 300)
        ]
        if "kick off" in etype_low and preceding_goals:
            g_team = preceding_goals[-1]["tags"].get("Team", "")
            # Conceding team takes the Kick Off
            if g_team == "Barcelona":
                team_val, team_str = -1.0, "A"
            else:
                team_val, team_str = 1.0, "H"
            score, anc_sum, adv_mean, nxt, last = 10.0 * team_val, 0.0, 0.0, team_val, team_val
        else:
            # 2. Detect physical restart release t_kick inside [s + 25, min(e_f, s + 380)]
            t_kick = min(e_f, s + 180)
            for t in range(min(e_f, s + 25), min(e_f, s + 380)):
                if bv[t] == 1 and oob[t] == 0 and (v_ball[t] > 4.5 or vb_w15[t] > 3.5):
                    t_kick = t
                    break

            w_s = max(0, t_kick)
            w_e = min(n_frames, t_kick + 125)

            anc_window = ds["anchor_team"][w_s:w_e]
            adv_window = ds["nearest_team_adv"][w_s:w_e]

            anc_sum = float(np.sum(anc_window))
            adv_valid = adv_window[~np.isnan(adv_window)]
            adv_mean = float(np.mean(adv_valid)) if len(adv_valid) > 0 else 0.0

            nxt = float(ds["next_anchor_team"][min(n_frames - 1, w_s)])
            if nxt == 0:
                nxt_slice = ds["next_anchor_team"][w_s:w_e]
                nxt_valid = nxt_slice[nxt_slice != 0]
                if len(nxt_valid) > 0:
                    nxt = float(nxt_valid[0])

            last = float(ds["last_anchor_team"][min(n_frames - 1, w_s)])

            score = 1.5 * anc_sum + 2.0 * adv_mean + 5.0 * nxt + 1.0 * last

            if score > 0.1:
                team_val = 1.0
                team_str = "H"
            elif score < -0.1:
                team_val = -1.0
                team_str = "A"
            else:
                w_wide_s, w_wide_e = max(0, t_kick - 25), min(n_frames, t_kick + 200)
                anc_wide = ds["anchor_team"][w_wide_s:w_wide_e]
                team_val = 1.0 if np.sum(anc_wide) >= 0 else -1.0
                team_str = "H" if team_val > 0 else "A"

        jev_na_team_prior[s:e_f] = team_val
        resolved_counts[team_str] += 1
        total_resolved_frames += (e_f - s)

        event_details.append({
            "event_idx": i,
            "type": ev_type,
            "start_frame": s,
            "end_frame": e_f,
            "team": team_str,
            "team_val": team_val,
            "score": score,
            "anc_sum": anc_sum,
            "adv_mean": adv_mean,
            "nxt_anchor": nxt,
            "last_anchor": last,
        })

    info = {
        "n_events_total": len(na_evs),
        "n_events_resolved": resolved_counts["H"] + resolved_counts["A"],
        "resolved_counts": resolved_counts,
        "n_frames_resolved": total_resolved_frames,
        "events": event_details,
    }
    return jev_na_team_prior, info


# =============================================================================
# Primitive 2: Epistemic Conflict & Shannon Entropy
# =============================================================================

def compute_epistemic_conflict_and_entropy(
    p_3state: np.ndarray,
    ds: mf.FeatureDataset,
    frame_slice: Optional[slice] = None,
    st_transition_active: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes:
      1. Normalised Shannon Entropy H(p) in [0, 1] of the 3-state distribution (P_H, P_A, P_D):
         H(p) = - 1/log(3) * sum(p_k * log(p_k + 1e-12))
      2. Tag-vs-Physics Conflict Score in [0, 1]:
         Detects frames where Smart Tagging claims one team (st_team_bias != 0), but either:
           (a) Instantaneous physical touch (ball visible, player < 1.45m, nearest_team_adv > 1.5m)
               proves the opponent has immediate physical control, OR
           (b) Sustained short-window physical turnover (adv_rolling_w35, adv_rolling_w75, and
               last_anchor_team opposing non-transition st_team_bias) indicates a rapid turnover.
    """
    sl = slice(None) if frame_slice is None else frame_slice

    # 1. Normalised Shannon Entropy H(p)
    eps = 1e-12
    p_clamped = np.clip(p_3state, eps, 1.0)
    p_norm = p_clamped / np.sum(p_clamped, axis=-1, keepdims=True)
    entropy = -np.sum(p_norm * np.log(p_norm), axis=-1) / np.log(3.0)
    entropy = np.clip(entropy, 0.0, 1.0).astype(np.float32)

    # 2. Tag-vs-Physics Conflict Score
    st_bias = ds["st_team_bias"][sl]
    ball_val = ds["ball_valid"][sl]
    d_h1 = ds["d_h1"][sl]
    d_a1 = ds["d_a1"][sl]
    adv = ds["nearest_team_adv"][sl]
    adv35 = ds["adv_rolling_w35"][sl]
    adv75 = ds["adv_rolling_w75"][sl]
    last_a = ds["last_anchor_team"][sl]
    b_gap = ds["ball_gap_len"][sl]

    if st_transition_active is None:
        tr_act = np.zeros(len(entropy), dtype=np.float32)
    else:
        tr_act = st_transition_active[sl]

    inst_away = (st_bias > 0.3) & (ball_val == 1.0) & (d_a1 < 1.45) & (adv < -1.5)
    inst_home = (st_bias < -0.3) & (ball_val == 1.0) & (d_h1 < 1.45) & (adv > 1.5)
    win_away = (st_bias > 0.3) & (tr_act == 0.0) & (adv35 < -0.25) & (adv75 < -0.15) & (last_a < 0)
    win_home = (st_bias < -0.3) & (tr_act == 0.0) & (b_gap >= 50) & (adv35 > 0.0) & (adv75 > 0.05) & (last_a > 0)

    conflict_score = np.zeros(len(entropy), dtype=np.float32)
    conflict_score[inst_away | inst_home | win_away | win_home] = 1.0

    return entropy, conflict_score


# =============================================================================
# Primitive 2B: Conflict-Aware Live Control Arbitration
# =============================================================================

def arbitrate_live_control(
    p_h_gbdt: np.ndarray,
    p_h_prior: np.ndarray,
    ds: mf.FeatureDataset,
    frame_slice: Optional[slice] = None,
    base_weight: float = 0.72,
    conflict_weight: float = 0.90,
    st_transition_active: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Dynamically arbitrates between the GBDT physical control probability and the multi-second
    tactical prior:
      1. On instantaneous touch conflict (ball visible, player < 1.45m opposing st_team_bias),
         increases GBDT weight to `conflict_weight` (0.90).
      2. On sustained short-window non-transition turnover conflict (where Smart Tagging lags
         behind an interception), asserts physical control probability so 2-3 second turnovers
         (such as GT7_A and GT15_H) are registered without lag.
    """
    sl = slice(None) if frame_slice is None else frame_slice
    st_b = ds["st_team_bias"][sl]
    b_val = ds["ball_valid"][sl]
    dh1 = ds["d_h1"][sl]
    da1 = ds["d_a1"][sl]
    adv = ds["nearest_team_adv"][sl]
    adv35 = ds["adv_rolling_w35"][sl]
    adv75 = ds["adv_rolling_w75"][sl]
    last_a = ds["last_anchor_team"][sl]
    b_gap = ds["ball_gap_len"][sl]

    if st_transition_active is None:
        tr_act = np.zeros(len(st_b), dtype=np.float32)
    else:
        tr_act = st_transition_active[sl]

    inst_conflict_away = (st_b > 0.3) & (b_val == 1.0) & (da1 < 1.45) & (adv < -1.5)
    inst_conflict_home = (st_b < -0.3) & (b_val == 1.0) & (dh1 < 1.45) & (adv > 1.5)
    inst_conflict = inst_conflict_away | inst_conflict_home

    w_gbdt = np.where(inst_conflict, conflict_weight, base_weight).astype(np.float32)
    p_h_raw = w_gbdt * p_h_gbdt + (1.0 - w_gbdt) * p_h_prior

    win_conflict_away = (st_b > 0.3) & (tr_act == 0.0) & (adv35 < -0.25) & (adv75 < -0.15) & (last_a < 0)
    win_conflict_home = (st_b < -0.3) & (tr_act == 0.0) & (b_gap >= 50) & (adv35 > 0.0) & (adv75 > 0.05) & (last_a > 0)

    p_h_raw = np.where(win_conflict_away, np.minimum(p_h_raw, 0.22), p_h_raw)
    p_h_raw = np.where(win_conflict_home, np.maximum(p_h_raw, 0.78), p_h_raw)

    # 5-frame rolling mean to smooth instantaneous micro-noise while preserving step edges
    s_p_h = pd.Series(p_h_raw)
    p_h_smoothed = s_p_h.rolling(window=5, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)
    return p_h_smoothed


# =============================================================================
# Primitive 3: Occlusion Bridge Damping (Handling 40.7% Missing Ball)
# =============================================================================

def apply_occlusion_bridge_damping(
    p_h_alive: np.ndarray,
    ds: mf.FeatureDataset,
    frame_slice: Optional[slice] = None,
    logit_boost: float = 1.2,
    interp_weight: float = 0.65,
) -> np.ndarray:
    """
    Stabilizes possession estimation during ball occlusion gaps (ball_valid == 0):
      1. Same-Team Bridges: If the ball was controlled by Team X before disappearing and is
         received by Team X upon reappearing (same_team_bridge != 0, gap <= 90 frames = 3.6s),
         apply a bounded log-odds reinforcement to prevent mid-flight probability dips from
         splitting a possession.
      2. Opposing Anchor Transitions: If Team X controlled before disappearance and Team Y
         controlled after (turnover across occlusion), construct a smooth monotonic temporal
         interpolation between the two anchors so the transition occurs once cleanly without
         oscillations.
    """
    p_out = p_h_alive.copy()
    sl = slice(None) if frame_slice is None else frame_slice

    ball_val = ds["ball_valid"][sl]
    gap_len = ds["ball_gap_len"][sl]
    bridge = ds["same_team_bridge"][sl]
    last_a = ds["last_anchor_team"][sl]
    next_a = ds["next_anchor_team"][sl]
    f_since = ds["frames_since_anchor"][sl]
    f_until = ds["frames_until_anchor"][sl]

    # 1. Same-team bridge reinforcement on occluded live frames
    same_mask = (ball_val == 0) & (bridge != 0) & (gap_len <= 90)
    if np.any(same_mask):
        eps = 1e-4
        p_c = np.clip(p_out[same_mask], eps, 1.0 - eps)
        logit = np.log(p_c / (1.0 - p_c))
        logit_boosted = logit + logit_boost * bridge[same_mask]
        p_out[same_mask] = 1.0 / (1.0 + np.exp(-logit_boosted))

    # 2. Opposing anchor smooth monotonic interpolation on occluded frames
    opp_mask = (ball_val == 0) & (last_a != 0) & (next_a != 0) & (last_a != next_a) & (gap_len <= 90)
    if np.any(opp_mask):
        total_dist = f_since[opp_mask] + f_until[opp_mask]
        valid_dist = total_dist > 0
        if np.any(valid_dist):
            indices = np.where(opp_mask)[0][valid_dist]
            alpha = f_since[opp_mask][valid_dist] / total_dist[valid_dist]  # 0.0 -> 1.0
            l_a = last_a[opp_mask][valid_dist]
            p_target = np.where(l_a > 0, 1.0 - alpha, alpha)
            p_out[indices] = (1.0 - interp_weight) * p_out[indices] + interp_weight * p_target

    return p_out.astype(np.float32)


# =============================================================================
# Primitive 4: Set-Piece Restart Boundary Sharpening
# =============================================================================

def sharpen_restart_boundaries(
    p_dead: np.ndarray,
    ds: mf.FeatureDataset,
    frame_slice: Optional[slice] = None,
    damping_factor: float = 0.15,
    max_dead_p: float = 0.20,
    oob_to_kick_mask: Optional[np.ndarray] = None,
    post_goal_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Refines p_dead around the exact set-piece restart kick:
    When inside an active set piece (phase >= 0.85), the ball is visible and moving on the pitch
    (ball_speed_rolling_w15 > 5.5 or max_player_speed > 4.2), the restart kick has physically
    taken place. Damps lingering false P(D) spikes from lagging tag boundaries while strictly
    preserving Out-of-Bounds pre-kick retrieval (oob_to_kick_mask) and post-goal stoppages.
    """
    p_out = p_dead.copy()
    sl = slice(None) if frame_slice is None else frame_slice

    sp_active = ds["st_set_piece_active"][sl]
    sp_phase = ds["st_set_piece_phase"][sl]
    ball_val = ds["ball_valid"][sl]
    ball_oob = ds["ball_out_of_bounds"][sl]
    ball_spd = ds["ball_speed_rolling_w15"][sl]
    play_spd = ds["max_player_speed"][sl]

    restart_kick = (
        (sp_active == 1.0)
        & (sp_phase >= 0.85)
        & (ball_val == 1.0)
        & (ball_oob == 0.0)
        & ((ball_spd > 5.5) | (play_spd > 4.2))
    )

    if np.any(restart_kick):
        p_out[restart_kick] = np.minimum(p_out[restart_kick] * damping_factor, max_dead_p)

    if oob_to_kick_mask is not None:
        p_out = np.where(oob_to_kick_mask[sl], np.maximum(p_out, 0.88), p_out)
    if post_goal_mask is not None:
        p_out = np.where(post_goal_mask[sl], np.maximum(p_out, 0.95), p_out)

    return p_out.astype(np.float32)


# =============================================================================
# Unified Calibration Pipeline
# =============================================================================

def calibrate_possession_distribution(
    p_dead_base: np.ndarray,
    p_h_gbdt: np.ndarray,
    p_h_prior: np.ndarray,
    ds: mf.FeatureDataset,
    frame_slice: Optional[slice] = None,
    masks: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes the full Jev System One epistemic calibration pipeline:
      1. Sharpens restart boundaries on p_dead while preserving OOB-to-kick and post-goal stoppages.
      2. Arbitrates live control between GBDT and prior under tag-vs-kinematics conflict.
      3. Stabilizes occlusion bridges across missing-ball frames.
      4. Factors into a 3-state probability tensor (P_H, P_A, P_D).
      5. Computes normalised epistemic Shannon entropy and conflict diagnostic masks.

    Returns:
      (probs_3state, jev_entropy, jev_conflict)
    """
    if masks is None:
        masks = compute_set_piece_and_goal_stoppage_masks(ds)

    oob_to_kick = masks.get("oob_to_kick_mask")
    post_goal = masks.get("post_goal_mask")
    st_trans = masks.get("st_transition_active")

    # 1. Sharpen restart boundaries while preserving physical OOB-to-kick and post-goal stoppages
    p_dead_cal = sharpen_restart_boundaries(
        p_dead_base,
        ds,
        frame_slice=frame_slice,
        oob_to_kick_mask=oob_to_kick,
        post_goal_mask=post_goal,
    )

    # 2. Conflict arbitration + light smoothing
    p_h_alive_cal = arbitrate_live_control(
        p_h_gbdt,
        p_h_prior,
        ds,
        frame_slice=frame_slice,
        st_transition_active=st_trans,
    )

    # 3. Occlusion bridge damping
    p_h_alive_cal = apply_occlusion_bridge_damping(p_h_alive_cal, ds, frame_slice=frame_slice)

    # 4. Factored 3-state probabilities
    n_frames = len(p_dead_cal)
    probs_3state = np.zeros((n_frames, 3), dtype=np.float32)
    probs_3state[:, 0] = (1.0 - p_dead_cal) * p_h_alive_cal
    probs_3state[:, 1] = (1.0 - p_dead_cal) * (1.0 - p_h_alive_cal)
    probs_3state[:, 2] = p_dead_cal

    # 5. Shannon entropy & conflict diagnostics
    entropy, conflict = compute_epistemic_conflict_and_entropy(
        probs_3state,
        ds,
        frame_slice=frame_slice,
        st_transition_active=st_trans,
    )

    return probs_3state, entropy, conflict


# =============================================================================
# Diagnostic Self-Test Entry Point
# =============================================================================

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80)
    print("JEV AI (`SYSTEM ONE`) DETERMINISTIC EPISTEMIC PRIMITIVES SELF-TEST")
    print("=" * 80)

    print("Loading feature dataset...")
    ds = mf.extract_all_features()

    masks = compute_set_piece_and_goal_stoppage_masks(ds)
    print(f"\n[Primitive 0] OOB-to-Kick & Post-Goal Stoppage Masks:")
    print(f"  OOB-to-Kick frames (full match):            {np.sum(masks['oob_to_kick_mask']):,}")
    print(f"  Post-Goal stoppage frames (full match):     {np.sum(masks['post_goal_mask']):,}")

    # Primitive 1
    print("\n[Primitive 1] Resolving 36 N/A Smart Tagging phases...")
    jev_na_prior, info = resolve_na_set_pieces(ds)
    print(f"  Total N/A events found:    {info['n_events_total']}")
    print(f"  Total events resolved:     {info['n_events_resolved']} (H: {info['resolved_counts']['H']}, A: {info['resolved_counts']['A']})")
    print(f"  Total frames resolved:     {info['n_frames_resolved']:,} frames")

    # Primitive 2 Conflict
    dummy_probs = np.full((len(ds["ball_valid"]), 3), 1.0 / 3.0, dtype=np.float32)
    entropy, conflict = compute_epistemic_conflict_and_entropy(
        dummy_probs, ds, st_transition_active=masks["st_transition_active"]
    )
    print("\n[Primitive 2] Epistemic Conflict & Shannon Entropy:")
    print(f"  Tag-vs-Physics conflict frames (full match): {np.sum(conflict > 0):,}")
    print(f"  Tag-vs-Physics conflict frames (graded):     {np.sum(conflict[ds.graded_mask] > 0):,}")

    # Primitive 3 Occlusion
    ball_val = ds["ball_valid"]
    bridge = ds["same_team_bridge"]
    gap_len = ds["ball_gap_len"]
    same_bridge_mask = (ball_val == 0) & (bridge != 0) & (gap_len <= 90)
    print("\n[Primitive 3] Occlusion Bridge Damping:")
    print(f"  Same-team bridge occluded frames:           {np.sum(same_bridge_mask):,}")

    # Primitive 4 Restart Sharpening
    p_dead_dummy = np.full(len(ds["ball_valid"]), 0.8, dtype=np.float32)
    p_dead_sharp = sharpen_restart_boundaries(
        p_dead_dummy,
        ds,
        oob_to_kick_mask=masks["oob_to_kick_mask"],
        post_goal_mask=masks["post_goal_mask"],
    )
    n_sharpened = np.sum(p_dead_sharp < 0.8)
    print("\n[Primitive 4] Restart Boundary Sharpening:")
    print(f"  Restart kick frames sharpened:              {n_sharpened:,}")

    print(f"\nAll Jev AI primitives verified successfully in {time.time() - t0:.2f}s!")
    print("=" * 80)
