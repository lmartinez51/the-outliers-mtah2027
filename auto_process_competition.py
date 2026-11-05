"""Unified Master Orchestrator for More than a Hack 2027 (FC Barcelona & Metrica Sports).

Executes the complete end-to-end competition pipeline in a single command:
  1. High-Performance Feature Extraction & Cache Management (src/mtah_features.py)
  2. Factored Probabilistic Modeling & Jev AI (System One) Epistemic Calibration (src/mtah_model.py)
  3. 3-State Soft Viterbi Sequence Decoding & Side-by-Side Ablation (src/mtah_decoder.py)
  4. Official Submission Verification & Leaderboard Ranking (evaluation/)
  5. Interactive 3D Tactical Broadcast Replay Generation (src/generate_3d_replay_html.py)

Team: The Outliers
Authors: Lorenzo Martínez, Ana Karen Arellano, and Steven Ibarra
Repository: the-outliers-mtah2027
Target Competition: More than a Hack 2027
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src import generate_3d_replay_html as gen3d
from src import mtah_decoder
from src import mtah_features as mf
from src import mtah_model as mm

# Canonical Path Constants
PATH_FEATURES_CACHE = os.path.join(ROOT, "data", "cache_bar_ath_features.npz")
PATH_PROBS_CACHE = os.path.join(ROOT, "data", "cache_bar_ath_probs.npz")
PATH_SUBMISSION_CSV = os.path.join(ROOT, "submission.csv")
PATH_RESULTS_CSV = os.path.join(ROOT, "evaluation", "results", "The_Outliers.csv")
PATH_STANDALONE_HTML = os.path.join(ROOT, "barca_3d_interactive_replay.html")


def compute_file_md5(file_path: str) -> str:
    """Computes hexadecimal MD5 checksum of a file."""
    if not os.path.exists(file_path):
        return "N/A"
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def format_duration(seconds: float) -> str:
    """Formats duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"


# =============================================================================
# Step 1: Feature Extraction
# =============================================================================

def step_features(rebuild: bool = False) -> float:
    """Executes or verifies Step 1: Feature Extraction."""
    t0 = time.time()
    print("\n" + "=" * 80)
    print("STEP 1: FEATURE EXTRACTION & CACHING (src/mtah_features.py)")
    print("=" * 80)

    cache_exists = os.path.exists(PATH_FEATURES_CACHE)
    if not cache_exists or rebuild:
        reason = "User requested --rebuild-cache" if rebuild else "Cache not found"
        print(f"[ACTION] Running full streaming feature extraction ({reason})...")
        mf.main(rebuild=True)
    else:
        cache_size_mb = os.path.getsize(PATH_FEATURES_CACHE) / (1024 * 1024)
        print(f"[CACHE HIT] Features cache detected: '{PATH_FEATURES_CACHE}' ({cache_size_mb:.2f} MB)")
        ds = mf.load_cached_features(PATH_FEATURES_CACHE)
        print(f"  Loaded dataset: {len(ds['ball_valid']):,} frames, {len(ds.feature_names)} features in {time.time() - t0:.2f}s.")

    elapsed = time.time() - t0
    print(f"[STEP 1 COMPLETED] in {format_duration(elapsed)}")
    return elapsed


# =============================================================================
# Step 2: Probabilistic Model & Jev AI Epistemic Primitives
# =============================================================================

def step_model(rebuild: bool = False) -> float:
    """Executes or verifies Step 2: Probabilistic Model + Jev AI System One."""
    t0 = time.time()
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING & JEV AI EPISTEMIC CALIBRATION (src/mtah_model.py)")
    print("=" * 80)

    cache_exists = os.path.exists(PATH_PROBS_CACHE)
    if not cache_exists or rebuild:
        reason = "User requested --rebuild-cache" if rebuild else "Probabilities cache not found"
        print(f"[ACTION] Training factored possession model ({reason})...")
        mm.main()
    else:
        cache_size_mb = os.path.getsize(PATH_PROBS_CACHE) / (1024 * 1024)
        print(f"[CACHE HIT] Probabilities cache detected: '{PATH_PROBS_CACHE}' ({cache_size_mb:.2f} MB)")
        p_data = np.load(PATH_PROBS_CACHE)
        keys = list(p_data.keys())
        print(f"  Validated probability tensors: {keys} in {time.time() - t0:.2f}s.")

    elapsed = time.time() - t0
    print(f"[STEP 2 COMPLETED] in {format_duration(elapsed)}")
    return elapsed


# =============================================================================
# Step 3: Viterbi Sequence Decoding & Side-by-Side Ablation
# =============================================================================

def step_decoder(
    use_jev: bool = True,
    run_ablation: bool = True,
    team_name: str = "The_Outliers",
) -> Tuple[pd.DataFrame, float]:
    """Executes Step 3: 3-State Soft Viterbi Decoding and Ablation Study."""
    t0 = time.time()
    print("\n" + "=" * 80)
    print("STEP 3: 3-STATE SEQUENCE DECODER & ABLATION STUDY (src/mtah_decoder.py)")
    print("=" * 80)

    clean_team = (team_name or "The_Outliers").strip().replace(" ", "_")
    output_csvs = [
        PATH_SUBMISSION_CSV,
        os.path.join(ROOT, f"{clean_team}.csv"),
        os.path.join(ROOT, "evaluation", "results", f"{clean_team}.csv"),
        os.path.join(ROOT, "deliverables", f"{clean_team}.csv"),
        os.path.join(ROOT, "deliverables", "The Outliers.csv"),
    ]
    # Deduplicate while preserving order
    output_csvs = list(dict.fromkeys(output_csvs))
    print(f"[CONFIG] Team submission target: '{clean_team}.csv'")
    print(f"[CONFIG] Outputs to write: {[os.path.relpath(p, ROOT) for p in output_csvs]}")

    df_sub = mtah_decoder.generate_and_verify_submission(
        probs_path=PATH_PROBS_CACHE,
        output_csvs=output_csvs,
        use_jev=use_jev,
        run_ablation=run_ablation,
    )

    elapsed = time.time() - t0
    print(f"[STEP 3 COMPLETED] in {format_duration(elapsed)}")
    return df_sub, elapsed


# =============================================================================
# Step 4: Official Validation & Leaderboard Evaluation
# =============================================================================

def step_evaluation() -> float:
    """Executes Step 4: Official Validation & Leaderboard Scoreboard."""
    t0 = time.time()
    print("\n" + "=" * 80)
    print("STEP 4: OFFICIAL COMPETITION SCOREBOARD (evaluation/run_evaluation.py)")
    print("=" * 80)

    # 1. Official format validator
    val_script = os.path.join(ROOT, "evaluation", "validate_submission.py")
    val_cmd = [sys.executable, val_script, PATH_SUBMISSION_CSV]
    val_res = subprocess.run(val_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(val_res.stdout)
    if val_res.returncode != 0:
        print(val_res.stderr)
        raise RuntimeError(f"Submission validation failed with exit code {val_res.returncode}!")

    # 2. Official competition evaluator
    eval_script = os.path.join(ROOT, "evaluation", "run_evaluation.py")
    eval_cmd = [sys.executable, eval_script]
    eval_res = subprocess.run(eval_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(eval_res.stdout)
    if eval_res.returncode != 0:
        print(eval_res.stderr)
        raise RuntimeError(f"Scoreboard evaluation failed with exit code {eval_res.returncode}!")

    elapsed = time.time() - t0
    print(f"[STEP 4 COMPLETED] in {format_duration(elapsed)}")
    return elapsed


# =============================================================================
# Step 5: 3D Broadcast Replay Generation
# =============================================================================

def step_3d_replay(
    slice_name: str = "public",
    stride: int = 1,
    skip: bool = False,
) -> float:
    """Executes Step 5: 3D Tactical Interactive Replay Generation."""
    t0 = time.time()
    if skip:
        print("\n" + "=" * 80)
        print("STEP 5: 3D INTERACTIVE BROADCAST REPLAY GENERATOR (SKIPPED)")
        print("=" * 80)
        return 0.0

    print("\n" + "=" * 80)
    print("STEP 5: 3D INTERACTIVE BROADCAST REPLAY GENERATOR (src/generate_3d_replay_html.py)")
    print("=" * 80)

    slice_flags: Dict[str, Any] = {}
    if slice_name == "public":
        slice_flags["start_frame"] = gen3d.PUBLIC_START
        slice_flags["end_frame"] = gen3d.PUBLIC_END
    elif slice_name == "half1":
        slice_flags["start_frame"] = gen3d.HALF_1_START
        slice_flags["end_frame"] = gen3d.HALF_1_END
    elif slice_name == "half2":
        slice_flags["start_frame"] = gen3d.HALF_2_START
        slice_flags["end_frame"] = gen3d.HALF_2_END
    elif slice_name == "full":
        slice_flags["start_frame"] = 0
        slice_flags["end_frame"] = gen3d.TOTAL_VIDEO_FRAMES - 1
    else:
        slice_flags["start_frame"] = gen3d.PUBLIC_START
        slice_flags["end_frame"] = gen3d.PUBLIC_END

    print(
        f"[INFO] Exporting 3D replay (Slice: '{slice_name}', "
        f"Frames: {slice_flags['start_frame']}..{slice_flags['end_frame']}, Stride: {stride})..."
    )

    summary_3d = gen3d.generate_replay_payload(
        submission_path=PATH_SUBMISSION_CSV,
        output_dir=os.path.join(ROOT, "src"),
        start_frame=slice_flags["start_frame"],
        end_frame=slice_flags["end_frame"],
        stride=stride,
        standalone=True,
        custom_output=PATH_STANDALONE_HTML,
    )

    elapsed = time.time() - t0
    print(f"[STEP 5 COMPLETED] in {format_duration(elapsed)}")
    return elapsed


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Master Orchestrator for More than a Hack 2027 Pipeline."
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        default=False,
        help="Force full feature extraction and model training from raw tactical data.",
    )
    parser.add_argument(
        "--no-jev",
        action="store_true",
        default=False,
        help="Disable Jev AI (System One) primitives and use raw Motor Base predictions.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        default=False,
        help="Skip printing the side-by-side ablation comparison table.",
    )
    parser.add_argument(
        "--skip-3d",
        action="store_true",
        default=False,
        help="Skip regenerating the 3D replay payload and standalone HTML replay.",
    )
    parser.add_argument(
        "--3d-slice",
        dest="slice_3d",
        type=str,
        choices=["public", "half1", "half2", "full"],
        default="public",
        help="Match slice for 3D replay export (default: 'public' [7,500 frames]).",
    )
    parser.add_argument(
        "--3d-stride",
        dest="stride_3d",
        type=int,
        default=1,
        help="Temporal downsampling stride for 3D replay export (default: 1 [25 fps]).",
    )
    parser.add_argument(
        "--team-name",
        type=str,
        default="The_Outliers",
        help="Competition team name (default: 'The_Outliers'). Writes <team_name>.csv as required by SUBMISSION.md.",
    )

    args = parser.parse_args()

    t_start = time.time()

    print("=" * 80)
    print("MORE THAN A HACK 2027 — UNIFIED MASTER PIPELINE ORCHESTRATOR")
    print("=" * 80)
    print(f"Working Directory: {ROOT}")
    print(f"Jev AI Enabled:    {not args.no_jev}")
    print(f"Rebuild Cache:     {args.rebuild_cache}")
    print(f"Team Name:         {args.team_name}")
    print(f"3D Slice:          {args.slice_3d} (stride={args.stride_3d})")
    print("=" * 80)

    # Step 1: Features
    t_feat = step_features(rebuild=args.rebuild_cache)

    # Step 2: Model & Jev AI
    t_model = step_model(rebuild=args.rebuild_cache)

    # Step 3: Sequence Decoder & Ablation
    df_sub, t_dec = step_decoder(
        use_jev=not args.no_jev,
        run_ablation=not args.skip_ablation,
        team_name=args.team_name,
    )

    # Step 4: Official Validation & Leaderboard
    t_eval = step_evaluation()

    # Step 5: 3D Replay
    t_3d = step_3d_replay(
        slice_name=args.slice_3d,
        stride=args.stride_3d,
        skip=args.skip_3d,
    )

    total_time = time.time() - t_start
    sub_md5 = compute_file_md5(PATH_SUBMISSION_CSV)

    # Extract class distribution
    label_counts = df_sub["label"].value_counts().to_dict()
    h_cnt = label_counts.get("H", 0)
    a_cnt = label_counts.get("A", 0)
    d_cnt = label_counts.get("D", 0)
    x_cnt = label_counts.get("X", 0)

    print("\n" + "=" * 80)
    print("MASTER ORCHESTRATION PIPELINE COMPLETE — EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"Total Execution Time:    {format_duration(total_time)}")
    print(f"  - Step 1 (Features):   {format_duration(t_feat)}")
    print(f"  - Step 2 (Model & Jev):{format_duration(t_model)}")
    print(f"  - Step 3 (Decoding):   {format_duration(t_dec)}")
    print(f"  - Step 4 (Evaluation): {format_duration(t_eval)}")
    print(f"  - Step 5 (3D Replay):  {format_duration(t_3d)}")
    print("-" * 80)
    print(f"Primary Submission File: {PATH_SUBMISSION_CSV}")
    print(f"MD5 Checksum:            {sub_md5}")
    print(f"Total Rows:              {len(df_sub):,} (Frames 0..{len(df_sub) - 1})")
    print(f"Class Breakdown:         H={h_cnt:,} ({h_cnt/len(df_sub)*100:.1f}%), "
          f"A={a_cnt:,} ({a_cnt/len(df_sub)*100:.1f}%), "
          f"D={d_cnt:,} ({d_cnt/len(df_sub)*100:.1f}%), "
          f"X={x_cnt:,} ({x_cnt/len(df_sub)*100:.1f}%)")
    print("-" * 80)
    sb_path = os.path.join(ROOT, "evaluation", "scoreboard.csv")
    if os.path.exists(sb_path):
        sb_df = pd.read_csv(sb_path)
        clean_team = (args.team_name or "The_Outliers").strip().replace(" ", "_")
        row_df = sb_df[sb_df["submission"] == clean_team]
        if not row_df.empty:
            r0 = row_df.iloc[0]
            rank_val = int(r0["rank"])
            spell_f1_val = float(r0["spell_f1_mean"])
            macro_f1_val = float(r0["macro_f1"])
            f1_h_val = float(r0["f1_H"])
            f1_a_val = float(r0["f1_A"])
            f1_d_val = float(r0["f1_D"])
            lat_val = float(r0["mean_latency"])
            poss_val = int(r0["poss"])
            gt_poss_val = int(r0["n_gt_spells"])
            abstain_val = int(r0["n_abstain_on_graded"])
            print(f"  [RANK]                   #{rank_val} on Leaderboard")
            print(f"  [SPELL F1] (Rank Metric): {spell_f1_val:.4f} (vs. 0.4242 nearest baseline, +{(spell_f1_val/0.424242 - 1)*100:.1f}% relative)")
            print(f"  [MACRO F1]               {macro_f1_val:.4f} (H: {f1_h_val:.4f}, A: {f1_a_val:.4f}, D: {f1_d_val:.4f})")
            print(f"  [LATENCY]                {lat_val:+.1f} frames")
            print(f"  [POSSESSIONS]            {poss_val} (Ground Truth: {gt_poss_val})")
            print(f"  [ABSTENTIONS]            {abstain_val} frames on graded ground truth (0.00%)")
    print("-" * 80)
    print("3D TACTICAL INTERACTIVE REPLAY:")
    if not args.skip_3d and os.path.exists(PATH_STANDALONE_HTML):
        html_size_mb = os.path.getsize(PATH_STANDALONE_HTML) / (1024 * 1024)
        print(f"  Viewer Replay HTML:    {PATH_STANDALONE_HTML} ({html_size_mb:.2f} MB)")
        print("  To view: Double-click or open 'barca_3d_interactive_replay.html' in your browser!")
        print("  Controls: [Space] Play/Pause, [M] Cycle Brains, [H] Cinema Mode, [1-4] Cameras.")
    else:
        print("  (3D Replay generation skipped)")
    print("=" * 80)


if __name__ == "__main__":
    main()
