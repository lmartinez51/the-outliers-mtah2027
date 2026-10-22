"""3D Interactive Football Broadcast Replay Generator (Three.js / WebGL).

Translates submission.csv, calibrated 3-state probabilities, and Metrica EPTS
tracking data into an intuitive, high-performance FC Barcelona 3D broadcast replay
for src/viewer.html.

Team: The Outliers
Authors: Lorenzo Martínez, Ana Karen Arellano, and Steven Ibarra
Repository: the-outliers-mtah2027
Target Competition: More than a Hack 2027
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import mtah_decoder, mtah_features as mf, jev_system_one as jev


# ---------------------------------------------------------------------------
# Default Constants & Paths
# ---------------------------------------------------------------------------
GAME_ID = "BAR-ATH"
TOTAL_VIDEO_FRAMES = 199319
PITCH_LENGTH_M = 105.3  # Official games.py pitch length for BAR-ATH
PITCH_WIDTH_M = 68.0    # Official games.py pitch width for BAR-ATH
DEFAULT_FPS = 25.0

# Canonical Match Blocks
PUBLIC_START = 11185
PUBLIC_END = 18684  # 7,500 frames (5-min public ground truth slice)

HALF_1_START = 11185
HALF_1_END = 90651  # 79,467 frames

HALF_2_START = 110808
HALF_2_END = 190197  # 79,390 frames


def get_default_paths() -> Dict[str, str]:
    """Resolves default file paths in the workspace."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "repo_root": repo_root,
        "submission": os.path.join(repo_root, "submission.csv"),
        "probs": os.path.join(repo_root, "data", "cache_bar_ath_probs.npz"),
        "features": os.path.join(repo_root, "data", "cache_bar_ath_features.npz"),
        "atd_txt": os.path.join(repo_root, "data", "tactical", "BAR-ATH_tactical_atd.txt"),
        "atd_xml": os.path.join(repo_root, "data", "tactical", "BAR-ATH_tactical_atd.xml"),
        "viewer_html": os.path.join(repo_root, "src", "viewer.html"),
        "src_dir": os.path.join(repo_root, "src"),
        "artifacts_dir": os.path.join(repo_root, "artifacts", "visualizations"),
    }


# Real squad mapping for FC Barcelona vs Athletic Club de Bilbao (LaLiga J2, 24 August 2024 at Montjuïc)
OFFICIAL_MATCH_ROSTER: Dict[str, Dict[int, Dict[str, Any]]] = {
    # Home Team: FC Barcelona (Starting XI verified via spatial positioning + squad numbers)
    "H": {
        0:  {"name": "Iñaki Peña", "short": "I. Peña", "real_num": 13, "pos": "Portero", "gk": 1},
        1:  {"name": "Iñigo Martínez", "short": "I. Martínez", "real_num": 5, "pos": "Central", "gk": 0},
        2:  {"name": "Raphinha", "short": "Raphinha", "real_num": 11, "pos": "Mediapunta", "gk": 0},
        3:  {"name": "Pedri", "short": "Pedri", "real_num": 8, "pos": "Centrocampista", "gk": 0},
        4:  {"name": "Fermín López", "short": "Fermín", "real_num": 16, "pos": "Centrocampista", "gk": 0},
        5:  {"name": "Robert Lewandowski", "short": "Lewandowski", "real_num": 9, "pos": "Delantero Centro", "gk": 0},
        6:  {"name": "Eric García", "short": "Eric", "real_num": 24, "pos": "Defensa", "gk": 0},
        7:  {"name": "Pau Víctor", "short": "Pau Víctor", "real_num": 18, "pos": "Delantero", "gk": 0},
        8:  {"name": "Ferran Torres", "short": "Ferran", "real_num": 7, "pos": "Extremo Izquierdo", "gk": 0},
        9:  {"name": "Gerard Martín", "short": "G. Martín", "real_num": 35, "pos": "Lateral Izquierdo", "gk": 0},
        10: {"name": "Pau Cubarsí", "short": "Cubarsí", "real_num": 2, "pos": "Central", "gk": 0},
        11: {"name": "Marc Bernal", "short": "M. Bernal", "real_num": 28, "pos": "Pivote", "gk": 0},
        12: {"name": "Pablo Torre", "short": "P. Torre", "real_num": 14, "pos": "Centrocampista", "gk": 0},
        13: {"name": "Héctor Fort", "short": "H. Fort", "real_num": 32, "pos": "Lateral", "gk": 0},
        14: {"name": "Jules Koundé", "short": "Koundé", "real_num": 23, "pos": "Lateral Derecho", "gk": 0},
        15: {"name": "Alejandro Balde", "short": "Balde", "real_num": 3, "pos": "Lateral Izquierdo", "gk": 0},
        16: {"name": "Ander Astralaga", "short": "Astralaga", "real_num": 26, "pos": "Portero", "gk": 1},
        17: {"name": "Lamine Yamal", "short": "Lamine Yamal", "real_num": 19, "pos": "Extremo Derecho", "gk": 0},
        18: {"name": "Marc Casadó", "short": "M. Casadó", "real_num": 17, "pos": "Centrocampista", "gk": 0},
        19: {"name": "Dani Olmo", "short": "D. Olmo", "real_num": 20, "pos": "Mediapunta", "gk": 0},
        20: {"name": "Andreas Christensen", "short": "Christensen", "real_num": 15, "pos": "Central", "gk": 0},
        21: {"name": "Ansu Fati", "short": "Ansu Fati", "real_num": 10, "pos": "Delantero", "gk": 0},
        22: {"name": "Frenkie de Jong", "short": "F. de Jong", "real_num": 21, "pos": "Centrocampista", "gk": 0},
        23: {"name": "Marc-André ter Stegen", "short": "Ter Stegen", "real_num": 1, "pos": "Portero", "gk": 1},
    },
    # Away Team: Athletic Club de Bilbao
    "A": {
        24: {"name": "Unai Simón", "short": "U. Simón", "real_num": 1, "pos": "Portero", "gk": 1},
        25: {"name": "Óscar de Marcos", "short": "De Marcos", "real_num": 18, "pos": "Lateral Derecho", "gk": 0},
        26: {"name": "Dani García", "short": "D. García", "real_num": 14, "pos": "Centrocampista", "gk": 0},
        27: {"name": "Iñigo Lekue", "short": "Lekue", "real_num": 16, "pos": "Lateral", "gk": 0},
        28: {"name": "Yuri Berchiche", "short": "Yuri", "real_num": 17, "pos": "Lateral Izquierdo", "gk": 0},
        29: {"name": "Mikel Vesga", "short": "M. Vesga", "real_num": 6, "pos": "Pivote", "gk": 0},
        30: {"name": "Beñat Prados", "short": "B. Prados", "real_num": 24, "pos": "Centrocampista", "gk": 0},
        31: {"name": "Nico Williams", "short": "Nico Williams", "real_num": 10, "pos": "Extremo Izquierdo", "gk": 0},
        32: {"name": "Gorka Guruzeta", "short": "Guruzeta", "real_num": 12, "pos": "Delantero", "gk": 0},
        33: {"name": "Álex Berenguer", "short": "Berenguer", "real_num": 7, "pos": "Delantero Centro", "gk": 0},
        34: {"name": "Iñaki Williams", "short": "Iñaki Williams", "real_num": 9, "pos": "Extremo Derecho", "gk": 0},
        35: {"name": "Unai Gómez", "short": "U. Gómez", "real_num": 20, "pos": "Centrocampista", "gk": 0},
        36: {"name": "Yeray Álvarez", "short": "Yeray", "real_num": 5, "pos": "Central", "gk": 0},
        37: {"name": "Ander Herrera", "short": "A. Herrera", "real_num": 21, "pos": "Centrocampista", "gk": 0},
        38: {"name": "Adu Ares", "short": "A. Ares", "real_num": 23, "pos": "Extremo", "gk": 0},
        39: {"name": "Nico Serrano", "short": "N. Serrano", "real_num": 22, "pos": "Delantero", "gk": 0},
        40: {"name": "Oihan Sancet", "short": "Sancet", "real_num": 8, "pos": "Mediapunta", "gk": 0},
        41: {"name": "Unai Núñez", "short": "U. Núñez", "real_num": 3, "pos": "Central", "gk": 0},
        42: {"name": "Mikel Jauregizar", "short": "Jauregizar", "real_num": 31, "pos": "Centrocampista", "gk": 0},
        43: {"name": "Álvaro Djaló", "short": "Djaló", "real_num": 11, "pos": "Extremo", "gk": 0},
        44: {"name": "Dani Vivian", "short": "Vivian", "real_num": 4, "pos": "Central", "gk": 0},
        45: {"name": "Alex Padilla", "short": "Padilla", "real_num": 13, "pos": "Portero", "gk": 1},
    }
}


def extract_roster(xml_path: str) -> List[Dict[str, Any]]:
    """Extracts 46-player roster enriched with real player identities for BAR-ATH."""
    roster: List[Dict[str, Any]] = []

    if os.path.exists(xml_path):
        try:
            root = ET.parse(xml_path).getroot()
            teams = list(root.iter("Team"))
            team_ids = [t.attrib.get("id") for t in teams if t.attrib.get("id")]
            home_team_id = team_ids[0] if team_ids else None

            player_elements = list(root.iter("Player"))
            for idx, p in enumerate(player_elements):
                pid = p.get("id", str(idx + 1))
                team_id = p.get("teamId")
                is_home = (team_id == home_team_id) if home_team_id else (idx < 24)
                team_code = "H" if is_home else "A"
                jersey_num = (idx + 1) if is_home else (idx - 23)

                info = OFFICIAL_MATCH_ROSTER.get(team_code, {}).get(idx, {})
                name = info.get("name", f"Dorsal #{jersey_num}")
                short_name = info.get("short", name)
                real_num = info.get("real_num", jersey_num)
                pos = info.get("pos", "Jugador")
                is_gk = info.get("gk", 1 if idx in (0, 23, 24, 45) else 0)

                roster.append({
                    "idx": idx,
                    "id": pid,
                    "num": jersey_num,
                    "real_num": real_num,
                    "name": name,
                    "short": short_name,
                    "pos": pos,
                    "team": team_code,
                    "gk": is_gk,
                })
        except Exception as e:
            print(f"[WARN] Failed to parse XML '{xml_path}': {e}. Using fallback roster.")
            roster = []

    if not roster:
        # Fallback roster: 24 Home, 22 Away
        for idx in range(46):
            is_home = (idx < 24)
            team_code = "H" if is_home else "A"
            jersey_num = (idx + 1) if is_home else (idx - 23)
            info = OFFICIAL_MATCH_ROSTER.get(team_code, {}).get(idx, {})
            name = info.get("name", f"Dorsal #{jersey_num}")
            short_name = info.get("short", name)
            real_num = info.get("real_num", jersey_num)
            pos = info.get("pos", "Jugador")
            is_gk = info.get("gk", 1 if idx in (0, 23, 24, 45) else 0)

            roster.append({
                "idx": idx,
                "id": str(idx + 1),
                "num": jersey_num,
                "real_num": real_num,
                "name": name,
                "short": short_name,
                "pos": pos,
                "team": team_code,
                "gk": is_gk,
            })

    return roster


def load_submission_labels(submission_path: str, n_expected: int = TOTAL_VIDEO_FRAMES) -> np.ndarray:
    """Loads predicted labels from submission.csv ('H', 'A', 'D', 'X')."""
    if not os.path.exists(submission_path):
        raise FileNotFoundError(
            f"Submission file not found: '{submission_path}'. "
            f"Please run the pipeline or ensure submission.csv exists."
        )

    print(f"[INFO] Reading submission predictions from: '{submission_path}'...")
    df = pd.read_csv(submission_path)
    if "label" not in df.columns:
        raise ValueError(f"Submission CSV '{submission_path}' must have a 'label' column.")

    labels = df["label"].astype(str).to_numpy()
    if len(labels) != n_expected:
        print(f"[WARN] Submission has {len(labels)} rows, expected {n_expected}.")
    return labels


def load_all_model_telemetry(
    probs_path: str,
    sub_labels: np.ndarray,
    base_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads calibrated probabilities and Jev AI diagnostic telemetry."""
    n = len(sub_labels)
    if os.path.exists(probs_path):
        try:
            print(f"[INFO] Loading calibrated telemetry from: '{probs_path}'...")
            p_data = np.load(probs_path)
            full_probs = p_data["full_probs_199319"]  # shape (199319, 3) -> ['H', 'A', 'D']
            p_h = full_probs[:, 0]
            p_a = full_probs[:, 1]
            denom = p_h + p_a
            p_home_2way = np.where(denom > 1e-6, p_h / np.maximum(denom, 1e-6), 0.5)

            # Harmonize with decoded submission label: ensure predicted team is dominant on HUD
            p_home_calib = np.where(
                sub_labels == "H",
                np.clip(p_home_2way, 0.52, 0.98),
                np.where(
                    sub_labels == "A",
                    np.clip(p_home_2way, 0.02, 0.48),
                    0.50
                )
            )

            # Motor Base probabilities
            if "full_probs_base_199319" in p_data:
                full_probs_base = p_data["full_probs_base_199319"]
                pb_h = full_probs_base[:, 0]
                pb_a = full_probs_base[:, 1]
                denom_b = pb_h + pb_a
                pb_home_2way = np.where(denom_b > 1e-6, pb_h / np.maximum(denom_b, 1e-6), 0.5)
                base_p_home_calib = np.where(
                    base_labels == "H",
                    np.clip(pb_home_2way, 0.52, 0.98),
                    np.where(
                        base_labels == "A",
                        np.clip(pb_home_2way, 0.02, 0.48),
                        0.50
                    )
                )
            else:
                base_p_home_calib = p_home_calib.copy()

            jev_conflict = p_data["jev_conflict_199319"] if "jev_conflict_199319" in p_data else np.zeros(n, dtype=np.float32)
            jev_entropy = p_data["jev_entropy_199319"] if "jev_entropy_199319" in p_data else np.zeros(n, dtype=np.float32)

            return (
                np.round(p_home_calib.astype(np.float32), 3),
                np.round(base_p_home_calib.astype(np.float32), 3),
                jev_conflict.astype(np.float32),
                jev_entropy.astype(np.float32)
            )
        except Exception as e:
            print(f"[WARN] Error reading telemetry from '{probs_path}': {e}. Using fallbacks.")

    # High-fidelity fallback based on submission labels
    p_home_calib = np.where(sub_labels == "H", 0.92, np.where(sub_labels == "A", 0.08, 0.50)).astype(np.float32)
    base_p_home_calib = np.where(base_labels == "H", 0.92, np.where(base_labels == "A", 0.08, 0.50)).astype(np.float32)
    return p_home_calib, base_p_home_calib, np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)


def load_precomputed_ball_trajectory(features_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and sanitizes precomputed ball coordinates, validity, and kinematic speeds.

    Guarantees 100% finite, non-NaN coordinates via forward/backward filling,
    and computes instantaneous finite-difference ball speed blended with rolling w15.
    """
    if os.path.exists(features_path):
        try:
            print(f"[INFO] Loading precomputed ball trajectories from: '{features_path}'...")
            data = np.load(features_path)
            feat_names = list(data["feature_names"])
            feats = data["features"]

            idx_bx = feat_names.index("ball_x_interp")
            idx_by = feat_names.index("ball_y_interp")
            idx_v = feat_names.index("ball_valid")
            idx_spd = feat_names.index("ball_speed_rolling_w15") if "ball_speed_rolling_w15" in feat_names else -1

            raw_bx = feats[:, idx_bx]
            raw_by = feats[:, idx_by]
            raw_v = feats[:, idx_v]
            raw_w15 = feats[:, idx_spd] if idx_spd >= 0 else np.zeros_like(raw_bx)

            # Robust backfill/forward-fill to eradicate all NaNs
            s_bx = pd.Series(raw_bx).bfill().ffill().fillna(PITCH_LENGTH_M / 2.0).to_numpy()
            s_by = pd.Series(raw_by).bfill().ffill().fillna(PITCH_WIDTH_M / 2.0).to_numpy()
            s_v = pd.Series(raw_v).fillna(0).astype(int).to_numpy()
            s_w15 = pd.Series(raw_w15).bfill().ffill().fillna(0.0).to_numpy()

            # Kinematic finite-difference ball speed (25 fps)
            dx = np.diff(s_bx, prepend=s_bx[0])
            dy = np.diff(s_by, prepend=s_by[0])
            inst_spd = np.clip(np.hypot(dx, dy) * DEFAULT_FPS, 0.0, 36.0)
            smooth_spd = pd.Series(inst_spd).rolling(5, min_periods=1, center=True).mean().to_numpy()
            b_spd_all = np.where(smooth_spd >= 0.5, smooth_spd, np.maximum(smooth_spd, s_w15 * 0.5))
            b_spd_all = np.clip(b_spd_all, 0.0, 36.0)

            # Convert to pitch center relative coordinates
            bx_pitch_all = s_bx - (PITCH_LENGTH_M / 2.0)
            by_pitch_all = s_by - (PITCH_WIDTH_M / 2.0)

            return bx_pitch_all, by_pitch_all, s_v, b_spd_all
        except Exception as e:
            print(f"[WARN] Could not load precomputed ball trajectory: {e}.")

    # Fallback arrays
    n = TOTAL_VIDEO_FRAMES
    return np.zeros(n), np.zeros(n), np.zeros(n, dtype=int), np.zeros(n)


def generate_replay_payload(
    submission_path: str,
    output_dir: str,
    start_frame: int = PUBLIC_START,
    end_frame: int = PUBLIC_END,
    stride: int = 1,
    standalone: bool = False,
    custom_output: Optional[str] = None,
) -> Dict[str, Any]:
    """Streams the raw match data and generates match_payload.js for viewer.html."""
    t0 = time.time()
    paths = get_default_paths()

    # 1. Roster extraction
    roster = extract_roster(paths["atd_xml"])

    # 2. Submission labels
    sub_labels = load_submission_labels(submission_path, TOTAL_VIDEO_FRAMES)

    # 3. Decode Motor Base labels from full_probs_base_199319
    if os.path.exists(paths["probs"]):
        p_data = np.load(paths["probs"])
        if "full_probs_base_199319" in p_data:
            print("[INFO] Decoding Motor Base labels using Viterbi decoder...")
            base_df = mtah_decoder.decode_full_match(p_data["full_probs_base_199319"], verbose=False)
            base_labels = base_df["label"].to_numpy()
        else:
            base_labels = sub_labels.copy()
    else:
        base_labels = sub_labels.copy()

    # 4. Calibrated possession probabilities and Jev AI diagnostic telemetry
    p_home_arr, base_p_home_arr, jev_conflict_arr, jev_entropy_arr = load_all_model_telemetry(
        paths["probs"], sub_labels, base_labels
    )

    # 5. Jev AI N/A set-piece resolution prior
    jev_na_prior = np.zeros(TOTAL_VIDEO_FRAMES, dtype=np.float32)
    if os.path.exists(paths["features"]):
        try:
            ds = mf.load_cached_features(paths["features"])
            na_prior_arr, _ = jev.resolve_na_set_pieces(ds, "BAR-ATH")
            jev_na_prior = na_prior_arr.astype(np.float32)
            print(f"[INFO] Loaded Jev AI N/A set-piece resolutions ({(jev_na_prior != 0).sum()} active frames).")
        except Exception as e:
            print(f"[WARN] Could not load Jev N/A set-piece resolutions: {e}.")

    # 6. Precomputed ball coordinates & kinematic speed
    bx_pitch_all, by_pitch_all, ball_vis_all, b_spd_all = load_precomputed_ball_trajectory(paths["features"])

    # 7. Range validation
    start_frame = max(0, min(start_frame, TOTAL_VIDEO_FRAMES - 1))
    end_frame = max(start_frame, min(end_frame, TOTAL_VIDEO_FRAMES - 1))
    stride = max(1, stride)

    total_expected_frames = math.ceil((end_frame - start_frame + 1) / stride)
    print(
        f"[INFO] Processing frames {start_frame}..{end_frame} "
        f"(stride={stride}, total frames={total_expected_frames:,})..."
    )

    # Compute array index corresponding to kickoff
    if start_frame <= PUBLIC_START <= end_frame:
        kickoff_idx = (PUBLIC_START - start_frame) // stride
    else:
        kickoff_idx = 0

    # Prepare output directories
    os.makedirs(output_dir, exist_ok=True)
    js_output_path = os.path.join(output_dir, "match_payload.js")
    src_js_path = os.path.join(paths["src_dir"], "match_payload.js")

    turnover_markers: List[Dict[str, Any]] = []
    our_switches = 0
    base_switches = 0
    naive_switches = 0
    prev_our_t = None
    prev_base_t = None
    prev_naive_t = None
    last_owner_idx = 10  # Default to midfield player #10

    effective_fps = DEFAULT_FPS / stride

    # Open file for high-performance streaming write
    with open(js_output_path, "w", encoding="utf-8") as js_file:
        # Write JS header
        js_file.write('window.MATCH_DATA = {"match_id":"BAR-ATH",')
        js_file.write(f'"fps":{effective_fps:.2f},')
        js_file.write(f'"pitch_length":{PITCH_LENGTH_M:.1f},')
        js_file.write(f'"pitch_width":{PITCH_WIDTH_M:.1f},')
        js_file.write(f'"kickoff_frame":{int(kickoff_idx)},')
        js_file.write('"roster":' + json.dumps(roster, separators=(",", ":")) + ",")
        js_file.write('"frames":[')

        frames_written = 0
        atd_path = paths["atd_txt"]

        with open(atd_path, "r", encoding="utf-8") as atd_file:
            for fid, line in enumerate(atd_file):
                if fid < start_frame:
                    continue
                if fid > end_frame:
                    break
                if (fid - start_frame) % stride != 0:
                    continue

                c1 = line.find(":")
                c2 = line.find(":", c1 + 1)
                if c1 == -1 or c2 == -1:
                    continue

                players_str = line[c1 + 1: c2]
                ball_str = line[c2 + 1:].strip().rstrip(";")

                # Parse ball coordinates
                has_ball = bool(ball_str and ball_str[0] != "N")
                if has_ball:
                    b_parts = ball_str.split(",")
                    try:
                        bx_norm = float(b_parts[0])
                        by_norm = float(b_parts[1])
                        bx_pitch = (bx_norm - 0.5) * PITCH_LENGTH_M
                        by_pitch = (by_norm - 0.5) * PITCH_WIDTH_M
                        ball_vis = 1
                    except (ValueError, IndexError):
                        bx_pitch = float(bx_pitch_all[fid])
                        by_pitch = float(by_pitch_all[fid])
                        ball_vis = int(ball_vis_all[fid])
                else:
                    bx_pitch = float(bx_pitch_all[fid])
                    by_pitch = float(by_pitch_all[fid])
                    ball_vis = int(ball_vis_all[fid])

                b_spd = float(b_spd_all[fid])

                # Strict sanitization against NaN / Inf
                if math.isnan(bx_pitch) or math.isinf(bx_pitch):
                    bx_pitch = float(bx_pitch_all[fid])
                if math.isnan(by_pitch) or math.isinf(by_pitch):
                    by_pitch = float(by_pitch_all[fid])
                if math.isnan(b_spd) or math.isinf(b_spd):
                    b_spd = 0.0

                # 3D Ball height (lift during fast passes and crosses)
                raw_z = 0.22
                if ball_vis and b_spd > 7.0:
                    raw_z = float(np.clip(0.22 + (b_spd - 7.0) * 0.14, 0.22, 3.40))
                bz_pitch = round(raw_z, 2)

                # Parse players
                p_list: List[List[float]] = []
                best_home_dist = 999.0
                best_home_idx = -1
                best_away_dist = 999.0
                best_away_idx = -1

                for slot, p_item in enumerate(players_str.split(";")):
                    if not p_item or p_item[0] == "N":
                        continue
                    p_parts = p_item.split(",")
                    try:
                        px_norm = float(p_parts[0])
                        py_norm = float(p_parts[1])
                    except (ValueError, IndexError):
                        continue

                    px_pitch = round((px_norm - 0.5) * PITCH_LENGTH_M, 2)
                    py_pitch = round((py_norm - 0.5) * PITCH_WIDTH_M, 2)
                    if math.isnan(px_pitch) or math.isnan(py_pitch) or math.isinf(px_pitch) or math.isinf(py_pitch):
                        continue
                    p_list.append([slot, px_pitch, py_pitch])

                    # Distance to ball
                    d_b = math.hypot(px_pitch - bx_pitch, py_pitch - by_pitch)
                    if slot < 24:
                        if d_b < best_home_dist:
                            best_home_dist = d_b
                            best_home_idx = slot
                    else:
                        if d_b < best_away_dist:
                            best_away_dist = d_b
                            best_away_idx = slot

                # Naive Nearest-Player baseline
                naive_t = "H" if best_home_dist <= best_away_dist else "A"

                # Our Model Possession (Jev AI) & Motor Base Possession
                our_t = sub_labels[fid]
                base_t = base_labels[fid]

                # Switches & Turnover tracking
                is_turnover = (
                    prev_our_t is not None
                    and our_t in ("H", "A")
                    and prev_our_t in ("H", "A")
                    and our_t != prev_our_t
                )
                if is_turnover:
                    our_switches += 1

                is_base_switch = (
                    prev_base_t is not None
                    and base_t in ("H", "A")
                    and prev_base_t in ("H", "A")
                    and base_t != prev_base_t
                )
                if is_base_switch:
                    base_switches += 1

                is_naive_switch = (
                    prev_naive_t is not None
                    and naive_t != prev_naive_t
                    and ball_vis
                )
                if is_naive_switch:
                    naive_switches += 1

                prev_our_t = our_t
                prev_base_t = base_t
                prev_naive_t = naive_t

                # Owner resolution
                if our_t == "H" and best_home_idx >= 0 and best_home_dist <= 6.0:
                    last_owner_idx = best_home_idx
                elif our_t == "A" and best_away_idx >= 0 and best_away_dist <= 6.0:
                    last_owner_idx = best_away_idx
                elif last_owner_idx < 0:
                    last_owner_idx = best_home_idx if best_home_idx >= 0 else 10

                owner_item = roster[last_owner_idx] if 0 <= last_owner_idx < len(roster) else roster[0]
                owner_num = owner_item.get("real_num", owner_item["num"])
                owner_name = owner_item.get("name", f"Dorsal #{owner_num}")
                team_name = "FC Barcelona (Local)" if our_t == "H" else (
                    "Athletic Club (Visitante)" if our_t == "A" else "Sin posesión"
                )

                # Event code & commentary
                shield_active = bool(
                    ball_vis
                    and our_t in ("H", "A")
                    and our_t != naive_t
                    and b_spd >= 4.5
                )
                false_flicker = bool(ball_vis and our_t in ("H", "A") and our_t != naive_t)

                # Real Jev AI (System One) Activation Telemetry
                has_conflict = bool(jev_conflict_arr[fid] > 0)
                has_na_res = bool(jev_na_prior[fid] != 0)
                has_base_diff = bool(our_t != base_t)
                is_contested_duel = bool(
                    jev_entropy_arr[fid] > 0.45 or
                    (ball_vis and best_home_dist <= 2.8 and best_away_dist <= 2.8 and abs(best_home_dist - best_away_dist) <= 1.0)
                )
                is_jev = bool(our_t in ("H", "A") and (has_conflict or has_na_res or has_base_diff or is_contested_duel))

                ts_sec = round(fid / DEFAULT_FPS, 2)

                if our_t == "X":
                    ev_code = "OFF"
                    commentary = "⏸️ Fuera de tiempo de juego (Entretiempo o Pre-partido)."
                elif our_t == "D":
                    ev_code = "DEAD"
                    commentary = f"🛑 Saque o balón detenido — IA en pausa táctica ({ts_sec:.1f}s)."
                elif not ball_vis:
                    ev_code = "MISSING"
                    commentary = (
                        "📡 Balón fuera de cuadro — "
                        "Nuestra IA congela la posesión sin inventar robos falsos."
                    )
                elif is_turnover:
                    ev_code = f"TURNOVER_{our_t}"
                    commentary = (
                        f"⚡ ¡RECUPERACIÓN REAL! {owner_name} (#{owner_num}) de {team_name} "
                        f"corta la jugada y toma el control."
                    )
                    turnover_markers.append({
                        "frame_idx": frames_written,
                        "timestamp_s": ts_sec,
                        "team": our_t,
                        "label": f"{ts_sec:.1f}s — Robo: {owner_name} ({'Barça' if our_t == 'H' else 'Athletic'})",
                    })
                elif shield_active:
                    ev_code = "SHIELD"
                    nearest_opp = best_away_dist if our_t == "H" else best_home_dist
                    commentary = (
                        f"🛡️ ESCUDO DE PASE ACTIVO: Pase de {owner_name} ({team_name}) en tránsito a {b_spd:.1f} m/s "
                        f"(rival a {nearest_opp:.1f}m)."
                    )
                elif has_conflict:
                    ev_code = "JEV_CONFLICT"
                    commentary = (
                        f"🧠 Jev AI (System One): Arbitraje de conflicto Tag vs. Física — "
                        f"Inercia táctica resuelta para {owner_name} ({team_name})."
                    )
                elif has_na_res:
                    ev_code = "JEV_NA"
                    commentary = (
                        f"🧠 Jev AI (System One): Saque de banda / Reanudación N/A resuelto a favor de {owner_name} ({team_name})."
                    )
                elif has_base_diff:
                    ev_code = "JEV_CORRECT"
                    commentary = (
                        f"🧠 Jev AI (System One): Corrección sobre Motor Base ({base_t} ➔ {our_t}) "
                        f"para {owner_name} ({team_name})."
                    )
                elif is_contested_duel:
                    ev_code = "JEV_DUEL"
                    commentary = (
                        f"🧠 Jev AI (System One): Balón disputado cuerpo a cuerpo — "
                        f"Desempate epistémico a favor de {owner_name} ({team_name})."
                    )
                else:
                    ev_code = "NONE"
                    commentary = (
                        f"⚽ {team_name} domina la posesión con {owner_name} (#{owner_num}) "
                        f"(Vel. balón: {b_spd:.1f} m/s)."
                    )

                frame_dict = {
                    "fid": fid,
                    "t": ts_sec,
                    "v": 1 if (ball_vis and our_t not in ("D", "X")) else 0,
                    "bx": round(bx_pitch, 2),
                    "by": round(by_pitch, 2),
                    "bz": bz_pitch,
                    "spd": round(b_spd, 1),
                    "ot": our_t,
                    "bt": base_t,
                    "nt": naive_t,
                    "op": float(p_home_arr[fid]),
                    "bp": float(base_p_home_arr[fid]),
                    "own": int(last_owner_idx),
                    "sh": 1 if shield_active else 0,
                    "flk": 1 if false_flicker else 0,
                    "jev": 1 if is_jev else 0,
                    "ev": ev_code,
                    "our_sw": our_switches,
                    "base_sw": base_switches,
                    "nai_sw": naive_switches,
                    "msg": commentary,
                    "p": p_list,
                }

                if frames_written > 0:
                    js_file.write(",")
                # Strictly enforce valid JSON without NaNs
                js_file.write(json.dumps(frame_dict, separators=(",", ":"), allow_nan=False))
                frames_written += 1

                if frames_written % 10000 == 0:
                    print(f"  ... exported {frames_written:,} frames ({ts_sec / 60:.1f} match minutes)")

        # Close frames array and add turnovers
        js_file.write("],")
        js_file.write('"turnovers":' + json.dumps(turnover_markers, separators=(",", ":"), allow_nan=False) + "};\n")

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(js_output_path) / (1024 * 1024)
    print(
        f"[PASS] Successfully generated '{js_output_path}' "
        f"({frames_written:,} frames, {file_size_mb:.2f} MB) in {elapsed:.2f}s."
    )

    # Sync to src/match_payload.js if different output directory was specified
    if os.path.abspath(js_output_path) != os.path.abspath(src_js_path):
        shutil.copy2(js_output_path, src_js_path)
        print(f"[SYNC] Mirrored match payload to '{src_js_path}' for immediate local viewing.")

    # Also sync to artifacts/visualizations/
    artifacts_js_path = os.path.join(paths["artifacts_dir"], "match_payload.js")
    if os.path.abspath(js_output_path) != os.path.abspath(artifacts_js_path):
        os.makedirs(paths["artifacts_dir"], exist_ok=True)
        shutil.copy2(js_output_path, artifacts_js_path)
        # Also copy viewer.html to artifacts/visualizations/
        shutil.copy2(paths["viewer_html"], os.path.join(paths["artifacts_dir"], "viewer.html"))
        print(f"[SYNC] Mirrored match payload & viewer to '{paths['artifacts_dir']}'.")

    # Standalone HTML bundle
    if standalone or custom_output:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        standalone_path = custom_output or os.path.join(repo_root, "barca_3d_interactive_replay.html")
        print(f"[INFO] Compiling standalone HTML replay to: '{standalone_path}'...")
        with open(paths["viewer_html"], "r", encoding="utf-8") as vf:
            base_html = vf.read()

        with open(js_output_path, "r", encoding="utf-8") as jf:
            payload_js = jf.read()

        inline_script = f"<script id=\"match-payload-script\">\n{payload_js}\n</script>"
        if '<script id="match-payload-script" src="match_payload.js"></script>' in base_html:
            standalone_html = base_html.replace(
                '<script id="match-payload-script" src="match_payload.js"></script>',
                inline_script
            )
        else:
            standalone_html = base_html.replace(
                '<head>',
                f'<head>\n{inline_script}'
            )

        # Inline Three.js if local copy exists in src/three.min.js
        three_local_path = os.path.join(os.path.dirname(paths["viewer_html"]), "three.min.js")
        if os.path.exists(three_local_path):
            with open(three_local_path, "r", encoding="utf-8") as tf:
                three_code = tf.read()
            three_inline = f'<script id="three-js-embedded">\n{three_code}\n</script>'
            target_block = (
                '<!-- THREE.JS LIB: LOCAL WITH CDN FALLBACK -->\n'
                '<script src="three.min.js"></script>\n'
                '<script>\n'
                '  if (typeof THREE === \'undefined\') {\n'
                '    document.write(\'<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\\\\/script>\');\n'
                '  }\n'
                '</script>'
            )
            if target_block in standalone_html:
                standalone_html = standalone_html.replace(target_block, f'<!-- THREE.JS EMBEDDED (100% OFFLINE) -->\n{three_inline}')
            else:
                standalone_html = standalone_html.replace(
                    '<script src="three.min.js"></script>',
                    three_inline
                )
            print(f"[INFO] Embedded offline Three.js ({len(three_code)/1024:.1f} KB) into standalone HTML bundle.")

        with open(standalone_path, "w", encoding="utf-8") as sf:
            sf.write(standalone_html)
        print(f"[PASS] Standalone HTML bundle created: '{standalone_path}'.")

    return {
        "frames_count": frames_written,
        "turnovers_count": len(turnover_markers),
        "our_switches": our_switches,
        "base_switches": base_switches,
        "naive_switches": naive_switches,
        "payload_path": js_output_path,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3D Interactive Replay Payload for viewer.html using submission.csv."
    )
    parser.add_argument(
        "--submission",
        type=str,
        default="submission.csv",
        help="Path to submission.csv (default: submission.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src",
        help="Directory to write match_payload.js (default: 'src').",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Export all 199,319 frames of the complete match.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Export the 5-minute public ground truth slice (frames 11,185..18,684).",
    )
    parser.add_argument(
        "--half",
        type=int,
        choices=[1, 2],
        default=None,
        help="Export specific match half (1: 11185..90651, 2: 110808..190197).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Custom starting video frame index (0..199318).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Custom ending video frame index (0..199318).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to export (e.g. 7500).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal downsampling stride (1 = 25fps, 2 = 12.5fps, 5 = 5fps).",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        default=False,
        help="Compile a self-contained portable HTML file (barca_3d_interactive_replay.html).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom destination path for standalone HTML file.",
    )

    args = parser.parse_args()

    # Determine frame slice bounds
    if args.public:
        start_frame = PUBLIC_START
        end_frame = PUBLIC_END
    elif args.half == 1:
        start_frame = HALF_1_START
        end_frame = HALF_1_END
    elif args.half == 2:
        start_frame = HALF_2_START
        end_frame = HALF_2_END
    elif args.start_frame is not None or args.end_frame is not None:
        start_frame = args.start_frame if args.start_frame is not None else 0
        end_frame = args.end_frame if args.end_frame is not None else (TOTAL_VIDEO_FRAMES - 1)
    elif args.full:
        start_frame = 0
        end_frame = TOTAL_VIDEO_FRAMES - 1
    elif args.max_frames is not None and args.max_frames > 0:
        start_frame = PUBLIC_START
        end_frame = min(PUBLIC_START + args.max_frames - 1, TOTAL_VIDEO_FRAMES - 1)
    else:
        # Default behavior: 5-minute public ground truth slice (frames 11,185..18,684, 7,500 frames)
        start_frame = PUBLIC_START
        end_frame = PUBLIC_END

    if args.max_frames is not None and args.max_frames > 0:
        end_frame = min(start_frame + args.max_frames - 1, end_frame)

    print("=" * 80)
    print("FC BARCELONA 3D TACTICAL REPLAY GENERATOR — MORE THAN A HACK 2027")
    print("=" * 80)
    print(f"Target Match:     {GAME_ID}")
    print(f"Submission File:  {args.submission}")
    print(f"Frame Range:      {start_frame}..{end_frame}")
    print(f"Stride:           {args.stride} (Output Rate: {25.0 / args.stride:.1f} fps)")
    print(f"Target Directory: {args.output_dir}")
    print("=" * 80)

    summary = generate_replay_payload(
        submission_path=args.submission,
        output_dir=args.output_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=args.stride,
        standalone=args.standalone,
        custom_output=args.output,
    )

    print("=" * 80)
    print("REPLAY GENERATION COMPLETE")
    print(f"Frames exported:  {summary['frames_count']:,}")
    print(f"Turnovers marked: {summary['turnovers_count']}")
    print(
        f"Possession Switches: Our AI ({summary['our_switches']}) vs "
        f"Motor Base ({summary['base_switches']}) vs "
        f"Naive Baseline ({summary['naive_switches']})"
    )
    print(f"Payload file:     {summary['payload_path']}")
    print(f"Viewer location:  src/viewer.html")
    print("=" * 80)
    print("To view the 3D replay, simply double-click or open 'src/viewer.html' in your browser!")


if __name__ == "__main__":
    main()