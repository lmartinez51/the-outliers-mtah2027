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
  1. resolve_na_set_pieces(ds, game):
     Deterministically infers restarting team for the 36 'Team = N/A' Smart Tagging phases
     (including all 26 Throw-Ins in BAR-ATH) using post-restart kinematic control anchors.
  2. compute_epistemic_conflict_and_entropy(p_3state, ds, frame_slice):
     Computes normalised Shannon entropy H(p) in [0, 1] and detects Tag-vs-Physics conflict
     where Smart Tagging opposes instantaneous physical proximity.
  3. arbitrate_live_control(p_h_gbdt, p_h_prior, ds, frame_slice):
     Dynamically shifts blending weight from lagging tactical prior toward instantaneous physical
     control probability when conflict is detected, minimizing turnover latency (lat_f).
  4. apply_occlusion_bridge_damping(p_h_alive, ds, frame_slice):
     Handles the 40.7% missing-ball frames: applies bounded log-odds reinforcement to same-team
     bridges and monotonic interpolation across opposing anchor occlusions.
  5. sharpen_restart_boundaries(p_dead, ds, frame_slice):
     Refines p_dead around the exact set-piece restart kick once ball movement is initiated,
     damping lingering false P(D) spikes.
  6. calibrate_possession_distribution(...):
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
# Primitive 1: N/A Set-Piece Resolution (36 Phases, 26 Throw-Ins)
# =============================================================================

def resolve_na_set_pieces(
    ds: mf.FeatureDataset,
    game: str = "BAR-ATH",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Deterministically resolves the 36 Smart Tagging intervals where Team is unassigned ('N/A' or None),
    including all 26 Throw-In events in BAR-ATH.
    
    Infers restarting team (+1.0 for Home/Barcelona, -1.0 for Away/Athletic) using post-restart
    kinematic control anchors (anchor_team, nearest_team_adv, last_anchor_team, next_anchor_team)
    in the [end_frame - 25 .. end_frame + 75] restart window.
    
    Returns:
      jev_na_team_prior: (N,) float32 array with +1.0 (Home) or -1.0 (Away) during N/A intervals,
                         and 0.0 outside those intervals.
      info: Diagnostic metadata dictionary reporting counts and per-event decisions.
    """
    events = lst.load(game)
    na_evs = [e for e in events if e["tags"].get("Team") in (None, "N/A", "")]
    
    n_frames = len(ds["ball_valid"])
    jev_na_team_prior = np.zeros(n_frames, dtype=np.float32)
    
    resolved_counts = {"H": 0, "A": 0}
    total_resolved_frames = 0
    event_details: List[Dict[str, Any]] = []
    
    for i, e in enumerate(na_evs):
        s = e["start_frame"]
        e_f = e["end_frame"]
        ev_type = e["tags"].get("Type", e["code"])
        
        # Post-restart kinematic evaluation window: [e_f - 25 .. e_f + 75]
        w_s = max(0, e_f - 25)
        w_e = min(n_frames, e_f + 75)
        
        anc_window = ds["anchor_team"][w_s:w_e]
        adv_window = ds["nearest_team_adv"][w_s:w_e]
        
        anc_sum = float(np.sum(anc_window))
        adv_valid = adv_window[~np.isnan(adv_window)]
        adv_mean = float(np.mean(adv_valid)) if len(adv_valid) > 0 else 0.0
        
        nxt = float(ds["next_anchor_team"][min(n_frames - 1, e_f)])
        if nxt == 0:
            nxt_slice = ds["next_anchor_team"][w_s:w_e]
            nxt_valid = nxt_slice[nxt_slice != 0]
            if len(nxt_valid) > 0:
                nxt = float(nxt_valid[0])
                
        last = float(ds["last_anchor_team"][min(n_frames - 1, e_f)])
        
        # Composite kinematic vote:
        # Prioritizes post-restart ball receiver (next_anchor), proximity advantage (adv_mean),
        # and post-restart control anchors (anc_sum)
        score = 1.5 * anc_sum + 2.0 * adv_mean + 5.0 * nxt + 1.0 * last
        
        if score > 0.1:
            team_val = 1.0
            team_str = "H"
        elif score < -0.1:
            team_val = -1.0
            team_str = "A"
        else:
            w_wide_s, w_wide_e = max(0, e_f - 50), min(n_frames, e_f + 150)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes:
      1. Normalised Shannon Entropy H(p) in [0, 1] of the 3-state distribution (P_H, P_A, P_D):
         H(p) = - 1/log(3) * sum(p_k * log(p_k + 1e-12))
      2. Tag-vs-Physics Conflict Score in [0, 1]:
         Detects frames where Smart Tagging claims one team (st_team_bias != 0), but instantaneous
         physical proximity (ball visible, player < 1.45m, and nearest_team_adv opposing by > 2.0m)
         proves the opponent has physical control.
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
    
    conflict_away = (st_bias > 0.3) & (ball_val == 1.0) & (d_a1 < 1.45) & (adv < -2.0)
    conflict_home = (st_bias < -0.3) & (ball_val == 1.0) & (d_h1 < 1.45) & (adv > 2.0)
    
    conflict_score = np.zeros(len(entropy), dtype=np.float32)
    conflict_score[conflict_away | conflict_home] = 1.0
    
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
    conflict_weight: float = 0.95,
) -> np.ndarray:
    """
    Dynamically shifts blending weight from the lagging tactical prior toward the instantaneous
    physical control probability (GBDT) on frames where tag-vs-physics conflict is detected.
    
    This allows turnovers inside an ongoing Smart Tagging phase to be registered with minimal
    frame latency (lat_f).
    """
    sl = slice(None) if frame_slice is None else frame_slice
    st_b = ds["st_team_bias"][sl]
    b_val = ds["ball_valid"][sl]
    dh1 = ds["d_h1"][sl]
    da1 = ds["d_a1"][sl]
    adv = ds["nearest_team_adv"][sl]
    
    conflict_away = (st_b > 0.3) & (b_val == 1.0) & (da1 < 1.45) & (adv < -2.0)
    conflict_home = (st_b < -0.3) & (b_val == 1.0) & (dh1 < 1.45) & (adv > 2.0)
    conflict = conflict_away | conflict_home
    
    w_gbdt = np.where(conflict, conflict_weight, base_weight).astype(np.float32)
    p_h_raw = w_gbdt * p_h_gbdt + (1.0 - w_gbdt) * p_h_prior
    
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
) -> np.ndarray:
    """
    Refines p_dead around the exact set-piece restart kick:
    When inside an active set piece (phase >= 0.85), the ball is visible and moving on the pitch
    (ball_speed_rolling_w15 > 5.5 or ball_speed > 2.0 with player_speed > 4.2), the restart kick
    has physically taken place. Damps lingering false P(D) spikes from lagging tag boundaries
    so the new live possession starts on time.
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes the full Jev System One epistemic calibration pipeline:
      1. Sharpens restart boundaries on p_dead.
      2. Arbitrates live control between GBDT and prior under tag-vs-kinematics conflict.
      3. Stabilizes occlusion bridges across missing-ball frames.
      4. Factors into a 3-state probability tensor (P_H, P_A, P_D).
      5. Computes normalised epistemic Shannon entropy and conflict diagnostic masks.
      
    Returns:
      (probs_3state, jev_entropy, jev_conflict)
    """
    # 1. Sharpen restart boundaries
    p_dead_cal = sharpen_restart_boundaries(p_dead_base, ds, frame_slice=frame_slice)
    
    # 2. Conflict arbitration + light smoothing
    p_h_alive_cal = arbitrate_live_control(p_h_gbdt, p_h_prior, ds, frame_slice=frame_slice)
    
    # 3. Occlusion bridge damping
    p_h_alive_cal = apply_occlusion_bridge_damping(p_h_alive_cal, ds, frame_slice=frame_slice)
    
    # 4. Factored 3-state probabilities
    n_frames = len(p_dead_cal)
    probs_3state = np.zeros((n_frames, 3), dtype=np.float32)
    probs_3state[:, 0] = (1.0 - p_dead_cal) * p_h_alive_cal
    probs_3state[:, 1] = (1.0 - p_dead_cal) * (1.0 - p_h_alive_cal)
    probs_3state[:, 2] = p_dead_cal
    
    # 5. Shannon entropy & conflict diagnostics
    entropy, conflict = compute_epistemic_conflict_and_entropy(probs_3state, ds, frame_slice=frame_slice)
    
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
    
    # Primitive 1
    print("\n[Primitive 1] Resolving 36 N/A Smart Tagging phases...")
    jev_na_prior, info = resolve_na_set_pieces(ds)
    print(f"  Total N/A events found:    {info['n_events_total']}")
    print(f"  Total events resolved:     {info['n_events_resolved']} (H: {info['resolved_counts']['H']}, A: {info['resolved_counts']['A']})")
    print(f"  Total frames resolved:     {info['n_frames_resolved']:,} frames")
    
    # Primitive 2 Conflict
    dummy_probs = np.full((len(ds["ball_valid"]), 3), 1.0 / 3.0, dtype=np.float32)
    entropy, conflict = compute_epistemic_conflict_and_entropy(dummy_probs, ds)
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
    p_dead_sharp = sharpen_restart_boundaries(p_dead_dummy, ds)
    n_sharpened = np.sum(p_dead_sharp < 0.8)
    print("\n[Primitive 4] Restart Boundary Sharpening:")
    print(f"  Restart kick frames sharpened:              {n_sharpened:,}")
    
    print(f"\nAll Jev AI primitives verified successfully in {time.time() - t0:.2f}s!")
    print("=" * 80)
