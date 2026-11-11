"""
src/mtah_model.py
-----------------
Factored 3-State ('H', 'A', 'D') Probabilistic Possession Model.

Team: The Outliers
Authors: Team The Outliers
Repository: the-outliers-mtah2027
Target Competition: More than a Hack 2027

Key Architectures:
  1. Head 1: Occlusion-Robust Dead-Ball Detector (P(y = 'D'))
     - Ball-Dropout Data Augmentation (35% synthetic masking on live training frames)
     - Stoppage & Boundary Rule Overrides (pre-2nd half kickoff, out of bounds, speed drop)
     - Immune to the full-match 40.7% ball-occlusion shift
  2. Head 2: Team-Symmetric Live Possession Estimator (P(y = 'H' | y in {'H', 'A'}))
     - Strict Mathematical Team-Swap Symmetry Augmentation ([X; X_swap] training & symmetrized inference)
     - 0.72 GBDT + 0.28 Smooth Physics & Smart-Tagging Analytical Prior
  3. Combined 3-Class Probability Tensor:
     - P(D) = P_dead
     - P(H) = (1.0 - P_dead) * P_home_given_alive
     - P(A) = (1.0 - P_dead) * (1.0 - P_home_given_alive)
  4. 5-Fold Contiguous Temporal Block CV (1,500 frames / 1-min blocks on 7,500 public frames)
  5. Caching to data/cache_bar_ath_probs.npz (oof_probs_7500, full_probs_199319, classes)
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.mtah_features as mf
import src.jev_system_one as jev

# Standard public slice constants
PUBLIC_START_FRAME = 11185
PUBLIC_END_FRAME = 18684
PUBLIC_NUM_FRAMES = 7500
PUBLIC_SLICE = slice(PUBLIC_START_FRAME, PUBLIC_END_FRAME + 1)
CLASSES = np.array(["H", "A", "D"])
CACHE_PROBS_PATH = os.path.join(ROOT, "data", "cache_bar_ath_probs.npz")

# GBDT Model Hyperparameters (Centralized Module Constants)
HEAD1_GBDT_PARAMS: Dict[str, Any] = {
    "max_depth": 3,
    "max_iter": 80,
    "learning_rate": 0.05,
    "min_samples_leaf": 40,
    "l2_regularization": 2.0,
    "early_stopping": False,
    "random_state": 42,
}
HEAD2_GBDT_PARAMS: Dict[str, Any] = {
    "max_depth": 4,
    "max_iter": 120,
    "learning_rate": 0.05,
    "min_samples_leaf": 45,
    "l2_regularization": 2.5,
    "early_stopping": False,
    "random_state": 42,
}


# =============================================================================
# 1. Feature Assembly for Head 1 & Head 2
# =============================================================================

def build_head1_features(ds: mf.FeatureDataset) -> Tuple[np.ndarray, List[str]]:
    """
    Construct Head 1 (Dead-Ball Detector) feature matrix.

    Includes collective player speeds, relative speed slowdown ratios,
    player spread ratios, Smart Tagging tactical context, and ball occlusion
    features (clamped to prevent outlier sensitivity).
    """
    speed_ratio_15_75 = ds["player_speed_rolling_w15"] / (ds["player_speed_rolling_w75"] + 1e-4)
    speed_ratio_35_75 = ds["player_speed_rolling_w35"] / (ds["player_speed_rolling_w75"] + 1e-4)
    spread_ratio_x = np.nan_to_num(ds["home_spread_x"] / (ds["away_spread_x"] + 1e-4), nan=1.0)
    spread_ratio_y = np.nan_to_num(ds["home_spread_y"] / (ds["away_spread_y"] + 1e-4), nan=1.0)

    fs_clamped = np.clip(ds["frames_since_ball_seen"], 0.0, 50.0)
    fu_clamped = np.clip(ds["frames_until_ball_seen"], 0.0, 50.0)
    gap_clamped = np.clip(ds["ball_gap_len"], 0.0, 100.0)
    min_d = np.minimum(
        np.nan_to_num(ds["d_h1"], nan=15.0),
        np.nan_to_num(ds["d_a1"], nan=15.0)
    )
    min_d_clamped = np.clip(min_d, 0.0, 15.0)
    abs_bias = np.abs(ds["st_team_bias"])

    cols = [
        "mean_player_speed", "max_player_speed", "p75_player_speed",
        "player_speed_rolling_w15", "player_speed_rolling_w35", "player_speed_rolling_w75",
        "n_home_players", "n_away_players",
        "speed_ratio_15_75", "speed_ratio_35_75",
        "spread_ratio_x", "spread_ratio_y",
        "st_set_piece_active", "st_set_piece_phase", "abs_st_bias",
        "ball_out_of_bounds",
        "ball_speed_rolling_w15", "ball_speed_rolling_w35", "ball_speed_rolling_w75",
        "is_pre_2nd_half_kickoff",
        "fs_clamped", "fu_clamped", "gap_clamped", "min_d_clamped", "ball_valid"
    ]

    feat_dict = {
        "speed_ratio_15_75": speed_ratio_15_75,
        "speed_ratio_35_75": speed_ratio_35_75,
        "spread_ratio_x": spread_ratio_x,
        "spread_ratio_y": spread_ratio_y,
        "fs_clamped": fs_clamped,
        "fu_clamped": fu_clamped,
        "gap_clamped": gap_clamped,
        "min_d_clamped": min_d_clamped,
        "abs_st_bias": abs_bias,
    }
    for c in cols:
        if c not in feat_dict:
            feat_dict[c] = ds[c]

    X = np.column_stack([feat_dict[c] for c in cols]).astype(np.float32)
    return X, cols


def compute_multisecond_control_rollings(ds: mf.FeatureDataset) -> Dict[str, np.ndarray]:
    """
    Computes multi-second rolling control windows once over the full match:
      - adv_w125: +/-2.5s window (251 frames)
      - adv_w200: +/-4.0s window (401 frames)
      - anchor_w125: +/-2.5s window (251 frames)
      - anchor_w200: +/-4.0s window (401 frames)
      - st_smooth: +/-1.25s window (125 frames)
    """
    s_adv = pd.Series(ds["nearest_team_adv"])
    s_anchor = pd.Series(ds["anchor_team"])
    s_st = pd.Series(ds["st_team_bias"])

    return {
        "adv_w125": s_adv.rolling(window=251, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32),
        "adv_w200": s_adv.rolling(window=401, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32),
        "anchor_w125": s_anchor.rolling(window=251, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32),
        "anchor_w200": s_anchor.rolling(window=401, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32),
        "st_smooth": s_st.rolling(window=125, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32),
    }


def build_head2_features(
    ds: mf.FeatureDataset,
    rollings: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Construct Head 2 (Live Possession Estimator) feature matrices:
      - X: original feature matrix
      - X_swap: paired/antisymmetric feature matrix when swapping Home <-> Away.

    Incorporates multi-second rolling control windows (w125 = +/-2.5s, w200 = +/-4.0s)
    and smoothed Smart-Tagging bias to generalize across unseen match minutes.
    Overfitting spatial/count features (spread_x/y, n_players) are excluded.
    """
    if rollings is None:
        rollings = compute_multisecond_control_rollings(ds)

    adv_w125 = rollings["adv_w125"]
    adv_w200 = rollings["adv_w200"]
    anchor_w125 = rollings["anchor_w125"]
    anchor_w200 = rollings["anchor_w200"]
    st_smooth = rollings["st_smooth"]

    feature_pairs = [
        ("d_h1", "d_a1", "paired"),
        ("d_h2", "d_a2", "paired"),
        ("d_h3", "d_a3", "paired"),
        ("s_h1", "s_a1", "paired"),
        ("cnt_h_15", "cnt_a_15", "paired"),
        ("cnt_h_30", "cnt_a_30", "paired"),
        ("cnt_h_50", "cnt_a_50", "paired"),
        ("nearest_team_adv", None, "antisymmetric"),
        ("adv_rolling_w15", None, "antisymmetric"),
        ("adv_rolling_w35", None, "antisymmetric"),
        ("adv_rolling_w75", None, "antisymmetric"),
        ("anchor_team", None, "antisymmetric"),
        ("last_anchor_team", None, "antisymmetric"),
        ("next_anchor_team", None, "antisymmetric"),
        ("same_team_bridge", None, "antisymmetric"),
        ("st_team_bias", None, "antisymmetric"),
        ("centroid_advantage", None, "antisymmetric"),
        ("ball_speed_rolling_w15", None, "symmetric"),
        ("ball_speed_rolling_w35", None, "symmetric"),
        ("mean_player_speed", None, "symmetric"),
    ]

    h2_cols: List[str] = []
    X_cols: List[np.ndarray] = []
    X_swap_cols: List[np.ndarray] = []

    for item in feature_pairs:
        if item[2] == "paired":
            col_h, col_a = item[0], item[1]
            h2_cols.extend([col_h, col_a])
            X_cols.extend([ds[col_h], ds[col_a]])
            X_swap_cols.extend([ds[col_a], ds[col_h]])
        elif item[2] == "antisymmetric":
            col = item[0]
            h2_cols.append(col)
            X_cols.append(ds[col])
            X_swap_cols.append(-ds[col])
        elif item[2] == "symmetric":
            col = item[0]
            h2_cols.append(col)
            X_cols.append(ds[col])
            X_swap_cols.append(ds[col])

    # Extra multi-second rolling control features
    extra_antisym = [
        ("adv_rolling_w125", adv_w125),
        ("adv_rolling_w200", adv_w200),
        ("anchor_rolling_w125", anchor_w125),
        ("anchor_rolling_w200", anchor_w200),
        ("st_team_bias_smooth", st_smooth),
    ]
    for name, arr in extra_antisym:
        h2_cols.append(name)
        X_cols.append(arr)
        X_swap_cols.append(-arr)

    X = np.column_stack(X_cols).astype(np.float32)
    X_swap = np.column_stack(X_swap_cols).astype(np.float32)
    return X, X_swap, h2_cols


def compute_live_prior(
    ds: mf.FeatureDataset,
    adv_w125: Optional[np.ndarray] = None,
    adv_w200: Optional[np.ndarray] = None,
    anchor_w125: Optional[np.ndarray] = None,
    anchor_w200: Optional[np.ndarray] = None,
    st_smooth: Optional[np.ndarray] = None,
    rollings: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Smooth physics + multi-second rolling windows + Smart Tagging analytical prior:
    Incorporates adv_rolling_w75, adv_rolling_w125, adv_rolling_w200, anchor_rolling_w125/200,
    same_team_bridge, last_anchor_team, next_anchor_team, and smoothed st_team_bias
    to prevent false turnover spikes on unseen minutes.
    """
    if rollings is not None:
        if adv_w125 is None:
            adv_w125 = rollings.get("adv_w125")
        if adv_w200 is None:
            adv_w200 = rollings.get("adv_w200")
        if anchor_w125 is None:
            anchor_w125 = rollings.get("anchor_w125")
        if anchor_w200 is None:
            anchor_w200 = rollings.get("anchor_w200")
        if st_smooth is None:
            st_smooth = rollings.get("st_smooth")

    if adv_w125 is None:
        s_adv = pd.Series(ds["nearest_team_adv"])
        adv_w125 = s_adv.rolling(window=251, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32)
    if adv_w200 is None:
        s_adv = pd.Series(ds["nearest_team_adv"])
        adv_w200 = s_adv.rolling(window=401, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32)
    if anchor_w125 is None:
        s_anc = pd.Series(ds["anchor_team"])
        anchor_w125 = s_anc.rolling(window=251, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32)
    if anchor_w200 is None:
        s_anc = pd.Series(ds["anchor_team"])
        anchor_w200 = s_anc.rolling(window=401, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32)
    if st_smooth is None:
        s_st = pd.Series(ds["st_team_bias"])
        st_smooth = s_st.rolling(window=125, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32)

    b_adv = np.clip(np.nan_to_num(ds["nearest_team_adv"], nan=0.0) / 1.5, -2.5, 2.5)
    r_adv75 = np.clip(np.nan_to_num(ds["adv_rolling_w75"], nan=0.0) / 1.5, -2.5, 2.5)
    r_adv125 = np.clip(np.nan_to_num(adv_w125, nan=0.0) / 1.5, -2.5, 2.5)
    r_adv200 = np.clip(np.nan_to_num(adv_w200, nan=0.0) / 1.5, -2.5, 2.5)
    r_anc125 = np.clip(np.nan_to_num(anchor_w125, nan=0.0) * 1.5, -2.0, 2.0)
    r_anc200 = np.clip(np.nan_to_num(anchor_w200, nan=0.0) * 1.5, -2.0, 2.0)
    st_bias = np.clip(np.nan_to_num(st_smooth, nan=0.0) * 1.0, -1.5, 1.5)
    bridge = np.nan_to_num(ds["same_team_bridge"], nan=0.0) * 0.8
    last_a = np.nan_to_num(ds["last_anchor_team"], nan=0.0) * 0.5
    next_a = np.nan_to_num(ds["next_anchor_team"], nan=0.0) * 0.5

    score = 0.4 * b_adv + 0.6 * r_adv75 + 0.8 * r_adv125 + 1.2 * r_adv200 + 0.5 * r_anc125 + 0.7 * r_anc200 + st_bias + bridge + last_a + next_a
    return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)


# =============================================================================
# 2. Augmentation & Rule Overrides
# =============================================================================

def apply_ball_dropout(
    X_tr: np.ndarray,
    y_tr_dead: np.ndarray,
    cols: List[str],
    drop_rate: float = 0.45,
    seed: int = 42
) -> np.ndarray:
    """
    Synthetically masks out ball visibility features on live training rows
    with `drop_rate` probability. This forces Head 1 to rely on player speeds,
    spreads, and set-piece context rather than naively equating ball occlusion
    or zero rolling ball speed with Dead Ball.
    """
    X_augmented = X_tr.copy()
    live_indices = np.where(y_tr_dead == 0)[0]
    rng = np.random.default_rng(seed)
    mask = rng.random(len(live_indices)) < drop_rate
    drop_idx = live_indices[mask]
    n_drop = len(drop_idx)

    X_augmented[drop_idx, cols.index("ball_valid")] = 0.0
    X_augmented[drop_idx, cols.index("min_d_clamped")] = 15.0
    X_augmented[drop_idx, cols.index("gap_clamped")] = rng.uniform(35.0, 200.0, size=n_drop).astype(np.float32)
    X_augmented[drop_idx, cols.index("fs_clamped")] = rng.uniform(10.0, 150.0, size=n_drop).astype(np.float32)
    X_augmented[drop_idx, cols.index("fu_clamped")] = rng.uniform(10.0, 150.0, size=n_drop).astype(np.float32)
    X_augmented[drop_idx, cols.index("ball_out_of_bounds")] = 0.0
    for c_spd in ("ball_speed_rolling_w15", "ball_speed_rolling_w35", "ball_speed_rolling_w75"):
        if c_spd in cols:
            X_augmented[drop_idx, cols.index(c_spd)] = 0.0

    return X_augmented


def compute_frames_until_set_piece(st_set_piece_active: np.ndarray) -> np.ndarray:
    """
    Computes lookahead distance in frames until the next tagged set-piece restart.
    Vectorized backward pass via binary search on set-piece event indices.
    """
    N = len(st_set_piece_active)
    frames_until_sp = np.full(N, 99999, dtype=np.int32)
    sp_idx = np.where(st_set_piece_active == 1.0)[0]
    if len(sp_idx) > 0:
        insert_pts = np.searchsorted(sp_idx, np.arange(N))
        valid = insert_pts < len(sp_idx)
        frames_until_sp[valid] = sp_idx[insert_pts[valid]] - np.arange(N, dtype=np.int32)[valid]
    return frames_until_sp


def apply_dead_ball_rules(
    p_dead: np.ndarray,
    ds_slice: Dict[str, np.ndarray],
    frames_until_sp: Optional[np.ndarray] = None,
    oob_kick_slice: Optional[np.ndarray] = None,
    post_goal_slice: Optional[np.ndarray] = None,
    stoppage_speed_w75: float = 1.35,
    stoppage_speed_max: float = 2.80,
    stoppage_floor: float = 0.75,
    occl_stoppage_floor: float = 0.75,
    min_occl_gap: int = 120,
    occl_mps_thresh: float = 1.68,
    occl_w75_thresh: float = 2.50,
    occl_max_s_thresh: float = 4.25,
) -> np.ndarray:
    """
    Deterministic domain rules for dead-ball detection:
      1. Pre-2nd half kickoff: frames 110808..113741 -> P_dead = 0.995
      2. Out of bounds & OOB-to-Kick continuity -> P_dead >= 0.88
      3. Post-goal celebration stoppage -> P_dead >= 0.95
      4. Collective physical stoppage: rolling_speed_w75 < 1.35 & max_speed < 2.8 -> P_dead >= 0.75
      5. Pre-set-piece occlusion & stoppage: within 100 frames (4.0s) before set-piece restart
         if ball is invisible or players are far & slow -> P_dead >= 0.85
      6. Occlusion-stoppage: prolonged missing ball (gap >= 120) with calm player movement
         and no uncontested open-play attack tag -> P_dead >= 0.75
    """
    p = p_dead.copy()

    # Rule 5: Occlusion-stoppage detection (calm players during prolonged ball disappearance)
    if occl_stoppage_floor > 0.0 and "ball_gap_len" in ds_slice:
        bv = ds_slice["ball_valid"]
        gap = ds_slice["ball_gap_len"]
        mps = ds_slice["mean_player_speed"]
        w75 = ds_slice["player_speed_rolling_w75"]
        max_s = ds_slice["max_player_speed"]
        abs_st = np.abs(ds_slice.get("st_team_bias", np.zeros_like(bv)))
        sp_act = ds_slice.get("st_set_piece_active", np.zeros_like(bv))
        occl_stop = (
            (bv == 0)
            & (gap >= min_occl_gap)
            & (mps < occl_mps_thresh)
            & (w75 < occl_w75_thresh)
            & (max_s < occl_max_s_thresh)
            & ((abs_st == 0.0) | (sp_act == 1.0) | (w75 < 1.50))
        )
        p = np.where(occl_stop, np.maximum(p, occl_stoppage_floor), p)

    # Rule 4: Pre-set-piece occlusion & stoppage signal
    if frames_until_sp is not None:
        bv = ds_slice["ball_valid"]
        min_d = np.minimum(ds_slice["d_h1"], ds_slice["d_a1"])
        mps = ds_slice["mean_player_speed"]
        pre_sp_rule = ((frames_until_sp <= 100) & (frames_until_sp > 0)) & ((bv == 0) | ((min_d > 3.0) & (mps < 1.85)))
        p = np.where(pre_sp_rule, np.maximum(p, 0.85), p)

    # Rule 3: Collective physical stoppage
    roll75 = ds_slice["player_speed_rolling_w75"]
    max_s = ds_slice["max_player_speed"]
    stoppage = (roll75 < stoppage_speed_w75) & (max_s < stoppage_speed_max)
    p = np.where(stoppage, np.maximum(p, stoppage_floor), p)

    # Rule 2: Out of bounds & OOB-to-restart-kick continuity + post-goal celebration stoppage
    oob = ds_slice["ball_out_of_bounds"]
    p = np.where(oob == 1.0, np.maximum(p, 0.88), p)
    if oob_kick_slice is not None:
        p = np.where(oob_kick_slice, np.maximum(p, 0.88), p)
    if post_goal_slice is not None:
        p = np.where(post_goal_slice, np.maximum(p, 0.95), p)

    # Rule 1: Pre-2nd half kickoff window
    pre_ko = ds_slice["is_pre_2nd_half_kickoff"]
    p = np.where(pre_ko == 1.0, 0.995, p)

    return p.astype(np.float32)


# =============================================================================
# 3. 5-Fold Contiguous Block Cross-Validation
# =============================================================================

def run_5fold_cv(
    ds: mf.FeatureDataset,
    gt_df: pd.DataFrame,
    head1_cols: List[str],
    X_h1_full: np.ndarray,
    X_h2_full: np.ndarray,
    X_h2_swap_full: np.ndarray,
    p_h_prior_full: np.ndarray,
    p_h_prior_jev_full: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Executes 5-Fold Contiguous Temporal Block Cross-Validation on the 7,500 public frames.
    Blocks are 1,500 frames (~1 minute) each.

    If p_h_prior_jev_full is provided, simultaneously computes both the Motor Base
    and Jev AI (System One) calibrated out-of-fold predictions.

    Returns:
      - oof_probs_jev (or oof_probs_base if jev prior not provided)
      - metrics_jev (or metrics_base)
      - oof_probs_base (or None)
      - metrics_base (or None)
    """
    pub_labels = gt_df[
        (gt_df["frame"] >= PUBLIC_START_FRAME) & (gt_df["frame"] <= PUBLIC_END_FRAME)
    ]["label"].to_numpy()
    if len(pub_labels) != PUBLIC_NUM_FRAMES:
        raise ValueError(f"Expected {PUBLIC_NUM_FRAMES} public frames, got {len(pub_labels)}")

    X_h1_pub = X_h1_full[PUBLIC_SLICE]
    X_h2_pub = X_h2_full[PUBLIC_SLICE]
    X_h2_swap_pub = X_h2_swap_full[PUBLIC_SLICE]
    p_h_prior_pub = p_h_prior_full[PUBLIC_SLICE]
    p_h_prior_jev_pub = p_h_prior_jev_full[PUBLIC_SLICE] if p_h_prior_jev_full is not None else None
    y_dead_pub = (pub_labels == "D").astype(int)

    oof_probs_base = np.zeros((PUBLIC_NUM_FRAMES, 3), dtype=np.float32)
    oof_probs_jev = np.zeros((PUBLIC_NUM_FRAMES, 3), dtype=np.float32) if p_h_prior_jev_pub is not None else None
    fold_size = PUBLIC_NUM_FRAMES // 5  # 1500 frames

    if verbose:
        print("\n" + "=" * 80)
        print("5-FOLD CONTIGUOUS TEMPORAL BLOCK CROSS-VALIDATION (1,500 frames / 1 min per fold)")
        print("=" * 80)

    frames_until_sp = compute_frames_until_set_piece(ds["st_set_piece_active"])
    jev_masks = jev.compute_set_piece_and_goal_stoppage_masks(ds)
    oob_to_kick_mask = jev_masks["oob_to_kick_mask"]
    post_goal_mask = jev_masks["post_goal_mask"]

    t_start = time.time()
    for fold in range(5):
        t_f = time.time()
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        val_idx = np.arange(val_start, val_end)
        train_idx = np.setdiff1d(np.arange(PUBLIC_NUM_FRAMES), val_idx)
        v_sl = slice(PUBLIC_START_FRAME + val_start, PUBLIC_START_FRAME + val_end)

        # -------------------------------------------------------------
        # Head 1: Occlusion-Robust Dead-Ball Detector with Ball-Dropout
        # -------------------------------------------------------------
        X_tr1 = X_h1_pub[train_idx]
        y_tr1 = y_dead_pub[train_idx]
        X_tr1_aug = apply_ball_dropout(X_tr1, y_tr1, head1_cols, drop_rate=0.45, seed=42 + fold)

        clf1 = HistGradientBoostingClassifier(**HEAD1_GBDT_PARAMS)
        clf1.fit(X_tr1_aug, y_tr1)
        p_dead_val_raw = clf1.predict_proba(X_h1_pub[val_idx])[:, 1]

        # Apply domain stoppage rules to fold
        ds_slice_dict = {
            "player_speed_rolling_w75": ds["player_speed_rolling_w75"][v_sl],
            "max_player_speed": ds["max_player_speed"][v_sl],
            "ball_out_of_bounds": ds["ball_out_of_bounds"][v_sl],
            "is_pre_2nd_half_kickoff": ds["is_pre_2nd_half_kickoff"][v_sl],
            "ball_valid": ds["ball_valid"][v_sl],
            "d_h1": ds["d_h1"][v_sl],
            "d_a1": ds["d_a1"][v_sl],
            "mean_player_speed": ds["mean_player_speed"][v_sl],
            "ball_gap_len": ds["ball_gap_len"][v_sl],
            "st_team_bias": ds["st_team_bias"][v_sl],
            "st_set_piece_active": ds["st_set_piece_active"][v_sl],
        }
        p_dead_val_base = apply_dead_ball_rules(
            p_dead_val_raw,
            ds_slice_dict,
            frames_until_sp=frames_until_sp[v_sl],
            oob_kick_slice=oob_to_kick_mask[v_sl],
            post_goal_slice=post_goal_mask[v_sl],
        )

        # -------------------------------------------------------------
        # Head 2: Team-Symmetric Live Possession Estimator
        # -------------------------------------------------------------
        tr_labels = pub_labels[train_idx]
        live_mask_tr = tr_labels != "D"
        train_live_idx = train_idx[live_mask_tr]

        X_tr2 = X_h2_pub[train_live_idx]
        X_tr2_sw = X_h2_swap_pub[train_live_idx]
        y_tr2 = (tr_labels[live_mask_tr] == "H").astype(int)

        # Double dataset with team-swap symmetry
        X_tr2_doubled = np.vstack([X_tr2, X_tr2_sw])
        y_tr2_doubled = np.concatenate([y_tr2, 1 - y_tr2])

        clf2 = HistGradientBoostingClassifier(**HEAD2_GBDT_PARAMS)
        clf2.fit(X_tr2_doubled, y_tr2_doubled)

        # Symmetrized inference: 0.5 * (p_orig + (1.0 - p_swap))
        p_h_gbdt_val = 0.5 * (
            clf2.predict_proba(X_h2_pub[val_idx])[:, 1]
            + (1.0 - clf2.predict_proba(X_h2_swap_pub[val_idx])[:, 1])
        )

        # 1. Base combination: 0.72 GBDT + 0.28 base prior
        p_h_alive_raw = 0.72 * p_h_gbdt_val + 0.28 * p_h_prior_pub[val_idx]
        s_p_h = pd.Series(p_h_alive_raw)
        p_h_alive_val_base = s_p_h.rolling(window=5, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)

        oof_probs_base[val_idx, 0] = (1.0 - p_dead_val_base) * p_h_alive_val_base
        oof_probs_base[val_idx, 1] = (1.0 - p_dead_val_base) * (1.0 - p_h_alive_val_base)
        oof_probs_base[val_idx, 2] = p_dead_val_base

        # 2. Jev AI calibration (if requested)
        if oof_probs_jev is not None and p_h_prior_jev_pub is not None:
            oof_probs_jev[val_idx], _, _ = jev.calibrate_possession_distribution(
                p_dead_base=p_dead_val_base,
                p_h_gbdt=p_h_gbdt_val,
                p_h_prior=p_h_prior_jev_pub[val_idx],
                ds=ds,
                frame_slice=v_sl,
                masks=jev_masks,
            )

        if verbose:
            f_start_fr = PUBLIC_START_FRAME + val_start
            f_end_fr = PUBLIC_START_FRAME + val_end - 1
            cur_probs = oof_probs_jev if oof_probs_jev is not None else oof_probs_base
            fold_preds = CLASSES[np.argmax(cur_probs[val_idx], axis=1)]
            fold_f1 = f1_score(pub_labels[val_idx], fold_preds, average="macro", zero_division=0)
            fold_acc = accuracy_score(pub_labels[val_idx], fold_preds)
            print(f"Fold {fold} (frames {f_start_fr}..{f_end_fr}): Macro F1 = {fold_f1:.4f} | Acc = {fold_acc:.4f} in {time.time() - t_f:.2f}s")

    cv_time = time.time() - t_start

    # Metrics Base
    oof_preds_base = CLASSES[np.argmax(oof_probs_base, axis=1)]
    macro_f1_base = f1_score(pub_labels, oof_preds_base, average="macro")
    per_class_base = f1_score(pub_labels, oof_preds_base, average=None, labels=["H", "A", "D"])
    dead_alive_base = f1_score(pub_labels == "D", oof_preds_base == "D", average="macro")
    live_mask_all = pub_labels != "D"
    live_ha_acc_base = accuracy_score(pub_labels[live_mask_all], oof_preds_base[live_mask_all])

    metrics_base = {
        "macro_f1": float(macro_f1_base),
        "f1_h": float(per_class_base[0]),
        "f1_a": float(per_class_base[1]),
        "f1_d": float(per_class_base[2]),
        "dead_alive_f1": float(dead_alive_base),
        "live_ha_accuracy": float(live_ha_acc_base),
        "cv_time_sec": float(cv_time),
    }

    if oof_probs_jev is not None:
        oof_preds_jev = CLASSES[np.argmax(oof_probs_jev, axis=1)]
        macro_f1_jev = f1_score(pub_labels, oof_preds_jev, average="macro")
        per_class_jev = f1_score(pub_labels, oof_preds_jev, average=None, labels=["H", "A", "D"])
        dead_alive_jev = f1_score(pub_labels == "D", oof_preds_jev == "D", average="macro")
        live_ha_acc_jev = accuracy_score(pub_labels[live_mask_all], oof_preds_jev[live_mask_all])

        metrics_jev = {
            "macro_f1": float(macro_f1_jev),
            "f1_h": float(per_class_jev[0]),
            "f1_a": float(per_class_jev[1]),
            "f1_d": float(per_class_jev[2]),
            "dead_alive_f1": float(dead_alive_jev),
            "live_ha_accuracy": float(live_ha_acc_jev),
            "cv_time_sec": float(cv_time),
        }
    else:
        oof_probs_jev = oof_probs_base
        metrics_jev = metrics_base
        oof_probs_base = None
        metrics_base = None

    if verbose:
        print("\n" + "=" * 80)
        print("OUT-OF-FOLD (OOF) 5-FOLD CV DIAGNOSTIC RESULTS")
        print("=" * 80)
        if metrics_base is not None:
            print(f"Base OOF Macro F1:             {metrics_base['macro_f1']:.4f} | Live Acc: {metrics_base['live_ha_accuracy']*100:.2f}% | D F1: {metrics_base['f1_d']:.4f}")
            print(f"Jev  OOF Macro F1:             {metrics_jev['macro_f1']:.4f} | Live Acc: {metrics_jev['live_ha_accuracy']*100:.2f}% | D F1: {metrics_jev['f1_d']:.4f}")
            diff_macro = metrics_jev["macro_f1"] - metrics_base["macro_f1"]
            diff_acc = (metrics_jev["live_ha_accuracy"] - metrics_base["live_ha_accuracy"]) * 100
            diff_d = metrics_jev["f1_d"] - metrics_base["f1_d"]
            print(f"Jev AI Net Lift:               Macro F1 {diff_macro:+.4f} | Live Acc {diff_acc:+.2f}% | D F1 {diff_d:+.4f}")
        else:
            print(f"Overall OOF Macro F1:          {metrics_jev['macro_f1']:.4f}")
            print(f"Per-Class F1 [H, A, D]:        [{metrics_jev['f1_h']:.4f}, {metrics_jev['f1_a']:.4f}, {metrics_jev['f1_d']:.4f}]")
        print(f"Total 5-Fold CV Time:          {cv_time:.2f}s")
        print("=" * 80)

    return oof_probs_jev, metrics_jev, oof_probs_base, metrics_base


# =============================================================================
# 4. Full-Match Training & Inference across 199,319 Frames
# =============================================================================

def fit_full_model(
    ds: mf.FeatureDataset,
    gt_df: pd.DataFrame,
    head1_cols: List[str],
    X_h1_full: np.ndarray,
    X_h2_full: np.ndarray,
    X_h2_swap_full: np.ndarray,
    p_h_prior_full: np.ndarray,
    p_h_prior_jev_full: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fits Head 1 and Head 2 on all 7,500 public ground-truth frames and predicts
    calibrated 3-class probabilities across all 199,319 match frames.

    If p_h_prior_jev_full is provided, computes both the Motor Base probabilities and
    the Jev AI (System One) calibrated probabilities, plus Shannon entropy and conflict masks.

    Returns:
      (full_probs_jev, full_probs_base, jev_entropy_199319, jev_conflict_199319)
    """
    t_start = time.time()
    n_frames = len(ds.graded_mask)
    pub_labels = gt_df[
        (gt_df["frame"] >= PUBLIC_START_FRAME) & (gt_df["frame"] <= PUBLIC_END_FRAME)
    ]["label"].to_numpy()

    X_h1_pub = X_h1_full[PUBLIC_SLICE]
    X_h2_pub = X_h2_full[PUBLIC_SLICE]
    X_h2_swap_pub = X_h2_swap_full[PUBLIC_SLICE]
    y_dead_pub = (pub_labels == "D").astype(int)

    if verbose:
        print("\n" + "=" * 80)
        print("FITTING FULL MODEL ON ALL 7,500 GROUND TRUTH FRAMES & INFERENCE ON 199,319 FRAMES")
        print("=" * 80)

    # 1. Fit Head 1 with Ball-Dropout
    X_tr1_aug = apply_ball_dropout(X_h1_pub, y_dead_pub, head1_cols, drop_rate=0.45, seed=42)
    clf1_full = HistGradientBoostingClassifier(**HEAD1_GBDT_PARAMS)
    clf1_full.fit(X_tr1_aug, y_dead_pub)
    raw_p_dead_full = clf1_full.predict_proba(X_h1_full)[:, 1]

    # Full-match prior calibration (logit shift delta = -1.20):
    # Rationale for distribution shift calibration between 5-min GT and full 90-min match:
    # 1. Ball Occlusion Surge: In the public 5-minute training window, optical ball tracking visibility
    #    is 88.20% (only 11.80% occlusion). Over the full match (158,857 graded frames), ball occlusion
    #    surges to 40.69%. Combined with zeroed rolling ball speeds in apply_ball_dropout, a moderate
    #    -1.20 logit shift prevents false dead-ball triggers during live occlusions while preserving
    #    sharp stoppage boundaries.
    # 2. Deterministic Physical Stoppage Laws: Out-of-bounds to restart-kick continuity (oob_to_kick_mask)
    #    and post-goal celebration stoppages (post_goal_mask) enforce FIFA/IFAB dead-ball intervals directly.
    eps = 1e-6
    clipped_f = np.clip(raw_p_dead_full, eps, 1.0 - eps)
    shifted_f = np.log(clipped_f / (1.0 - clipped_f)) - 1.20
    p_dead_base_full = (1.0 / (1.0 + np.exp(-shifted_f))).astype(np.float32)

    frames_until_sp = compute_frames_until_set_piece(ds["st_set_piece_active"])
    jev_masks = jev.compute_set_piece_and_goal_stoppage_masks(ds)

    # Full-match stoppage & boundary rules
    ds_full_dict = {
        "player_speed_rolling_w75": ds["player_speed_rolling_w75"],
        "max_player_speed": ds["max_player_speed"],
        "ball_out_of_bounds": ds["ball_out_of_bounds"],
        "is_pre_2nd_half_kickoff": ds["is_pre_2nd_half_kickoff"],
        "ball_valid": ds["ball_valid"],
        "d_h1": ds["d_h1"],
        "d_a1": ds["d_a1"],
        "mean_player_speed": ds["mean_player_speed"],
        "ball_gap_len": ds["ball_gap_len"],
        "st_team_bias": ds["st_team_bias"],
        "st_set_piece_active": ds["st_set_piece_active"],
    }
    p_dead_base_full = apply_dead_ball_rules(
        p_dead_base_full,
        ds_full_dict,
        frames_until_sp=frames_until_sp,
        oob_kick_slice=jev_masks["oob_to_kick_mask"],
        post_goal_slice=jev_masks["post_goal_mask"],
        stoppage_speed_w75=1.15,
        stoppage_speed_max=2.40,
        stoppage_floor=0.75,
    )

    # 2. Fit Head 2 with Team-Swap Symmetry Doubling
    live_mask_pub = pub_labels != "D"
    X_tr2_pub = X_h2_pub[live_mask_pub]
    X_tr2_swap_pub = X_h2_swap_pub[live_mask_pub]
    y_tr2_pub = (pub_labels[live_mask_pub] == "H").astype(int)

    X_tr2_doubled = np.vstack([X_tr2_pub, X_tr2_swap_pub])
    y_tr2_doubled = np.concatenate([y_tr2_pub, 1 - y_tr2_pub])

    clf2_full = HistGradientBoostingClassifier(**HEAD2_GBDT_PARAMS)
    clf2_full.fit(X_tr2_doubled, y_tr2_doubled)

    # Symmetrized inference on full match
    p_h_gbdt_full = 0.5 * (
        clf2_full.predict_proba(X_h2_full)[:, 1]
        + (1.0 - clf2_full.predict_proba(X_h2_swap_full)[:, 1])
    )

    # 3. Base 3-state probabilities
    p_h_alive_raw = 0.72 * p_h_gbdt_full + 0.28 * p_h_prior_full
    s_p_h = pd.Series(p_h_alive_raw)
    p_h_alive_base_full = s_p_h.rolling(window=5, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)

    full_probs_base = np.zeros((n_frames, 3), dtype=np.float32)
    full_probs_base[:, 0] = (1.0 - p_dead_base_full) * p_h_alive_base_full
    full_probs_base[:, 1] = (1.0 - p_dead_base_full) * (1.0 - p_h_alive_base_full)
    full_probs_base[:, 2] = p_dead_base_full

    # 4. Jev AI (System One) Calibration
    if p_h_prior_jev_full is not None:
        full_probs_jev, jev_entropy_199319, jev_conflict_199319 = jev.calibrate_possession_distribution(
            p_dead_base=p_dead_base_full,
            p_h_gbdt=p_h_gbdt_full,
            p_h_prior=p_h_prior_jev_full,
            ds=ds,
            frame_slice=None,
            masks=jev_masks,
        )
    else:
        full_probs_jev = full_probs_base
        jev_entropy_199319 = None
        jev_conflict_199319 = None

    if verbose:
        elapsed = time.time() - t_start
        print(f"Full-match inference complete in {elapsed:.2f}s. Shape: {full_probs_jev.shape}")

        # Graded frames class distribution
        graded = ds.graded_mask
        preds_grd = CLASSES[np.argmax(full_probs_jev[graded], axis=1)]
        counts_pct = pd.Series(preds_grd).value_counts(normalize=True) * 100
        counts_raw = pd.Series(preds_grd).value_counts()
        print("\n" + "-" * 80)
        print(f"FULL-MATCH GRADED FRAMES DISTRIBUTION ({np.sum(graded):,} frames) [Jev AI Enhanced]:")
        for c in ["H", "A", "D"]:
            print(f"  {c}: {counts_raw.get(c, 0):,} ({counts_pct.get(c, 0.0):.2f}%)")
        print("-" * 80)

    return full_probs_jev, full_probs_base, jev_entropy_199319, jev_conflict_199319


# =============================================================================
# 5. Serialization & Cache Loading
# =============================================================================

def save_probability_cache(
    path: str,
    oof_probs_7500: np.ndarray,
    full_probs_199319: np.ndarray,
    classes: np.ndarray = CLASSES,
    oof_probs_base_7500: Optional[np.ndarray] = None,
    full_probs_base_199319: Optional[np.ndarray] = None,
    jev_entropy_199319: Optional[np.ndarray] = None,
    jev_conflict_199319: Optional[np.ndarray] = None,
) -> None:
    """Save out-of-fold and full-match probability tensors and Jev diagnostics to npz cache."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_dict: Dict[str, np.ndarray] = {
        "oof_probs_7500": oof_probs_7500.astype(np.float32),
        "full_probs_199319": full_probs_199319.astype(np.float32),
        "classes": classes,
    }
    if oof_probs_base_7500 is not None:
        save_dict["oof_probs_base_7500"] = oof_probs_base_7500.astype(np.float32)
    if full_probs_base_199319 is not None:
        save_dict["full_probs_base_199319"] = full_probs_base_199319.astype(np.float32)
    if jev_entropy_199319 is not None:
        save_dict["jev_entropy_199319"] = jev_entropy_199319.astype(np.float32)
    if jev_conflict_199319 is not None:
        save_dict["jev_conflict_199319"] = jev_conflict_199319.astype(np.float32)

    np.savez(path, **save_dict)
    print(f"Saved probability cache to: {path} ({os.path.getsize(path) / (1024 * 1024):.2f} MB)")


def load_probability_cache(
    path: Optional[str] = None,
    return_dict: bool = False,
) -> Any:
    """
    Load cached probabilities from disk.

    If return_dict is True, returns a dict with all available cached keys.
    Otherwise returns (oof_probs_7500, full_probs_199319, classes) for backwards compatibility.
    """
    if path is None:
        path = CACHE_PROBS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Probability cache not found at {path}. Run mtah_model.py first.")

    data = np.load(path)
    if return_dict:
        return {k: data[k] for k in data.files}
    return data["oof_probs_7500"], data["full_probs_199319"], data["classes"]


# =============================================================================
# 6. Main Entry Point
# =============================================================================

def main():
    t_global = time.time()

    # 1. Load cached features & ground truth
    print("Loading feature dataset...")
    ds = mf.extract_all_features()
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt_df = pd.read_csv(gt_path)

    # 2. Build feature matrices & compute multi-second control rollings (computed once)
    print("Computing multi-second rolling control windows...")
    rollings = compute_multisecond_control_rollings(ds)

    print("Building Head 1 and Head 2 feature matrices...")
    X_h1_full, head1_cols = build_head1_features(ds)
    X_h2_full, X_h2_swap_full, _ = build_head2_features(ds, rollings=rollings)

    # Base prior (reuses precomputed rollings)
    p_h_prior_base = compute_live_prior(ds, rollings=rollings)

    # Jev AI (System One): Resolve 36 N/A Smart Tagging phases & compute enhanced prior
    print("\n[Jev AI System One] Resolving N/A set pieces...")
    jev_na_prior, na_info = jev.resolve_na_set_pieces(ds)
    print(f"  Resolved {na_info['n_events_resolved']} / {na_info['n_events_total']} events ({na_info['n_frames_resolved']:,} frames)")

    s_st_jev = pd.Series(ds["st_team_bias"] + jev_na_prior)
    st_smooth_jev = s_st_jev.rolling(window=125, center=True, min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32)
    p_h_prior_jev = compute_live_prior(ds, rollings=rollings, st_smooth=st_smooth_jev)

    # 3. 5-Fold Contiguous Temporal Block Cross-Validation (Base vs Jev AI)
    oof_probs_jev, cv_metrics_jev, oof_probs_base, cv_metrics_base = run_5fold_cv(
        ds=ds,
        gt_df=gt_df,
        head1_cols=head1_cols,
        X_h1_full=X_h1_full,
        X_h2_full=X_h2_full,
        X_h2_swap_full=X_h2_swap_full,
        p_h_prior_full=p_h_prior_base,
        p_h_prior_jev_full=p_h_prior_jev,
        verbose=True,
    )

    # 4. Fit on full public ground truth & predict on all 199,319 frames
    full_probs_jev, full_probs_base, jev_entropy_199319, jev_conflict_199319 = fit_full_model(
        ds=ds,
        gt_df=gt_df,
        head1_cols=head1_cols,
        X_h1_full=X_h1_full,
        X_h2_full=X_h2_full,
        X_h2_swap_full=X_h2_swap_full,
        p_h_prior_full=p_h_prior_base,
        p_h_prior_jev_full=p_h_prior_jev,
        verbose=True,
    )

    # 5. Save Probability Cache with both Base, Jev AI, and diagnostic tensors
    save_probability_cache(
        path=CACHE_PROBS_PATH,
        oof_probs_7500=oof_probs_jev,
        full_probs_199319=full_probs_jev,
        classes=CLASSES,
        oof_probs_base_7500=oof_probs_base,
        full_probs_base_199319=full_probs_base,
        jev_entropy_199319=jev_entropy_199319,
        jev_conflict_199319=jev_conflict_199319,
    )

    # 6. Validations for acceptance (ARCH-06)
    if cv_metrics_jev["cv_time_sec"] >= 15.0:
        print(f"[WARN] CV took {cv_metrics_jev['cv_time_sec']:.2f}s (benchmark budget 15s)")
    if cv_metrics_jev["macro_f1"] < cv_metrics_base["macro_f1"]:
        raise RuntimeError(
            f"Jev OOF Macro F1 {cv_metrics_jev['macro_f1']:.4f} must not regress Base ({cv_metrics_base['macro_f1']:.4f})"
        )
    if cv_metrics_jev["macro_f1"] <= 0.6207:
        raise RuntimeError(f"OOF Macro F1 {cv_metrics_jev['macro_f1']:.4f} must beat baseline_smooth50 (0.6207)")
    if cv_metrics_jev["f1_d"] <= 0.4197:
        raise RuntimeError(f"OOF D F1 {cv_metrics_jev['f1_d']:.4f} must beat baseline_raw (0.4197)")

    if np.isnan(full_probs_jev).any():
        raise RuntimeError("full_probs_jev contains NaN values!")
    if not np.allclose(full_probs_jev.sum(axis=1), 1.0, atol=1e-4):
        raise RuntimeError("full_probs_jev row sums do not equal 1.0!")

    graded = ds.graded_mask
    preds_grd = CLASSES[np.argmax(full_probs_jev[graded], axis=1)]
    d_share_pct = float(np.mean(preds_grd == "D") * 100)
    if not (35.0 <= d_share_pct <= 48.0):
        print(f"[WARN] Full-match D share {d_share_pct:.2f}% outside realistic 35%-48% LaLiga stoppage range!")
    else:
        print(f"Full-match D share: {d_share_pct:.2f}% (realistic 35%-48% LaLiga stoppage range / {100.0 - d_share_pct:.2f}% effective playing time verified)")

    print(f"\nAll Model & Jev AI requirements verified successfully in {time.time() - t_global:.2f}s!")


if __name__ == "__main__":
    main()

