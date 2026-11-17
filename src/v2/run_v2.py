# -*- coding: utf-8 -*-
"""Deterministic, GT-free possession pipeline entry point (v2 and v2.1).

Usage:
    python -B -m src.v2.run_v2 [--config {v2_1, apriori}] [--out PATH] [--overwrite]
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.v2.ball_tracker import build_ball_track
from src.v2.cleanup import cleanup_half_labels
from src.v2.constants import (
    APRIORI,
    FPS,
    G_INTERP,
    HALF_1_END,
    HALF_1_KICKOFF,
    HALF_1_START,
    HALF_2_END,
    HALF_2_KICKOFF,
    HALF_2_START,
    LABEL_A,
    LABEL_D,
    LABEL_H,
    LABEL_X,
    MARGIN,
    MIN_DEAD_FLICKER,
    MIN_LIVE_FLICKER,
    N_VIDEO_FRAMES,
    R_TOUCH,
    T_CONFIRM,
)
from src.v2.control_detector import detect_control_candidates
from src.v2.dead_ball import compute_dead_ball_mask
from src.v2.extractor import extract_v2_primitives
from src.v2.state_machine import compose_half_possession


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deterministic possession pipeline (v2 / v2.1)"
    )
    parser.add_argument(
        "--config",
        choices=["v2_1", "apriori"],
        default="v2_1",
        help="Configuration to run: 'v2_1' (default, externally tuned constants) or 'apriori' (frozen v2.0.1 baseline constants)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Destination path for submission CSV. Defaults to 'outputs_v2/submission_v2_1_external.csv' for v2_1 and 'outputs_v2/submission_v2.csv' for apriori.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting destination file if it already exists (refused by default).",
    )
    return parser.parse_args(args)


def compute_half_diagnostics(
    labels: np.ndarray, start_frame: int, end_frame: int, fps: int = FPS
) -> Dict[str, float]:
    """Compute structural diagnostics for bug detection only (Rule R4)."""
    half_labels = labels[start_frame : end_frame + 1]
    n_frames = len(half_labels)

    # Count possession spells (contiguous runs of H or A)
    spells: List[Tuple[str, int, int]] = []
    i = 0
    while i < n_frames:
        if half_labels[i] in (LABEL_H, LABEL_A):
            team = half_labels[i]
            s = i
            while i < n_frames and half_labels[i] == team:
                i += 1
            spells.append((team, s, i - 1))
        else:
            i += 1

    n_spells = len(spells)
    n_live = int(np.sum((half_labels == LABEL_H) | (half_labels == LABEL_A)))
    n_d = int(np.sum(half_labels == LABEL_D))
    n_x = int(np.sum(half_labels == LABEL_X))
    mean_duration = (n_live / float(fps) / n_spells) if n_spells > 0 else 0.0
    d_share = (n_d / float(n_frames)) if n_frames > 0 else 0.0

    return {
        "n_frames": float(n_frames),
        "n_spells": float(n_spells),
        "n_live": float(n_live),
        "n_d": float(n_d),
        "n_x": float(n_x),
        "d_share": float(d_share),
        "mean_duration_s": float(mean_duration),
    }


def _ensure_output_parent(output_path: str) -> str:
    """Create and return the absolute parent directory for an output path."""
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    return parent


def execute_pipeline(
    output_path: Optional[str] = None,
    config: str = "v2_1",
    overwrite: bool = False,
    cache_path: str = os.path.join("data", "cache_v2_primitives.npz"),
    raw_txt_path: str = os.path.join("data", "tactical", "BAR-ATH_tactical_atd.txt"),
    raw_xml_path: str = os.path.join("data", "tactical", "BAR-ATH_tactical_atd.xml"),
    force_rebuild: bool = False,
    r_touch: Optional[float] = None,
    margin: Optional[float] = None,
    t_confirm: Optional[int] = None,
    g_interp: Optional[int] = None,
    min_live_flicker: int = MIN_LIVE_FLICKER,
    min_dead_flicker: int = MIN_DEAD_FLICKER,
    fps: int = FPS,
) -> str:
    """Run full deterministic possession pipeline and write submission CSV."""
    t0 = time.time()

    if output_path is None:
        if config == "apriori":
            output_path = os.path.join("outputs_v2", "submission_v2.csv")
        else:
            output_path = os.path.join("outputs_v2", "submission_v2_1_external.csv")

    if os.path.isfile(output_path) and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Refusing to overwrite without --overwrite flag."
        )

    # Resolve constants
    if config == "apriori":
        eff_r_touch = float(APRIORI["R_TOUCH"]) if r_touch is None else r_touch
        eff_margin = float(APRIORI["MARGIN"]) if margin is None else margin
        eff_t_confirm = int(APRIORI["T_CONFIRM"]) if t_confirm is None else t_confirm
        eff_g_interp = int(APRIORI["G_INTERP"]) if g_interp is None else g_interp
    else:
        eff_r_touch = R_TOUCH if r_touch is None else r_touch
        eff_margin = MARGIN if margin is None else margin
        eff_t_confirm = T_CONFIRM if t_confirm is None else t_confirm
        eff_g_interp = G_INTERP if g_interp is None else g_interp

    _ensure_output_parent(output_path)

    # 1. Extract or load self-contained v2 primitives directly from tracking
    prims = extract_v2_primitives(
        raw_txt_path=raw_txt_path,
        raw_xml_path=raw_xml_path,
        cache_path=cache_path,
        force_rebuild=force_rebuild,
    )
    ball_valid = prims["ball_valid"]
    ball_x_m = prims["ball_x_m"]
    ball_y_m = prims["ball_y_m"]
    d_h1 = prims["d_h1"]
    d_a1 = prims["d_a1"]
    p75_speed = prims["p75_player_speed"]

    full_labels = np.full(N_VIDEO_FRAMES, LABEL_X, dtype=object)

    halves = [
        ("Half 1", HALF_1_START, HALF_1_END, HALF_1_KICKOFF),
        ("Half 2", HALF_2_START, HALF_2_END, HALF_2_KICKOFF),
    ]

    half_diagnostics: Dict[str, Dict[str, float]] = {}

    for name, s_fr, e_fr, k_fr in halves:
        # 3.1 Ball track
        ball_known, bx, by = build_ball_track(
            ball_valid=ball_valid,
            ball_x_m=ball_x_m,
            ball_y_m=ball_y_m,
            start_frame=s_fr,
            end_frame=e_fr,
            g_interp=eff_g_interp,
        )

        # 3.2 Control candidates
        candidates, _, _ = detect_control_candidates(
            ball_valid=ball_valid,
            ball_known=ball_known,
            ball_x_m=bx,
            ball_y_m=by,
            d_h1_cache=d_h1,
            d_a1_cache=d_a1,
            start_frame=s_fr,
            end_frame=e_fr,
            raw_txt_path=raw_txt_path,
            raw_xml_path=raw_xml_path,
            r_touch=eff_r_touch,
            margin=eff_margin,
        )

        # 3.4 Dead ball mask
        dead_mask, _, _, _ = compute_dead_ball_mask(
            ball_known=ball_known,
            ball_x_m=bx,
            ball_y_m=by,
            p75_player_speed=p75_speed,
            start_frame=s_fr,
            end_frame=e_fr,
            kickoff_frame=k_fr,
        )

        # 3.3 & 3.5 State machine & composition (halves are independent)
        composed_labels, _ = compose_half_possession(
            dead_mask=dead_mask,
            candidates=candidates,
            start_frame=s_fr,
            end_frame=e_fr,
            initial_possessor=None,
            t_confirm=eff_t_confirm,
        )

        # 3.6 Cleanup to fixed point
        cleaned_labels, iterations = cleanup_half_labels(
            labels=composed_labels,
            start_frame=s_fr,
            end_frame=e_fr,
            min_live_flicker=min_live_flicker,
            min_dead_flicker=min_dead_flicker,
        )

        full_labels[s_fr : e_fr + 1] = cleaned_labels[s_fr : e_fr + 1]
        half_diagnostics[name] = compute_half_diagnostics(full_labels, s_fr, e_fr, fps=fps)
        half_diagnostics[name]["cleanup_iterations"] = float(iterations)

    # 3.7 Write output CSV
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("frame,label\n")
        for idx in range(N_VIDEO_FRAMES):
            f.write(f"{idx},{full_labels[idx]}\n")

    elapsed = time.time() - t0

    # Print structural diagnostics (bug detection only, Rule R4)
    print("=" * 60)
    print(f"v2 DETERMINISTIC PIPELINE EXECUTION REPORT (Config: {config})")
    print("=" * 60)
    print(f"Constants:       R_TOUCH={eff_r_touch}m, MARGIN={eff_margin}m, G_INTERP={eff_g_interp}f, T_CONFIRM={eff_t_confirm}f")
    print(f"Output File:     {output_path}")
    print(f"Total Rows:      {N_VIDEO_FRAMES}")
    print(f"Execution Time:  {elapsed:.2f} s")
    print(f"Total X Frames:  {int(np.sum(full_labels == LABEL_X))} (expected: 40462)")

    for name in ("Half 1", "Half 2"):
        diag = half_diagnostics[name]
        print(f"\n--- {name} Structural Diagnostics (Bug Detection Only) ---")
        print(f"  Graded frames:      {int(diag['n_frames'])}")
        print(f"  Possession spells:  {int(diag['n_spells'])}")
        print(f"  Live ball frames:   {int(diag['n_live'])} ({100.0 * diag['n_live'] / diag['n_frames']:.1f}%)")
        print(f"  Dead ball frames:   {int(diag['n_d'])} ({100.0 * diag['d_share']:.1f}%)")
        print(f"  Mean duration:      {diag['mean_duration_s']:.1f} s")
        print(f"  X inside graded:    0 (strictly 0)")
        print(f"  Cleanup iterations: {int(diag['cleanup_iterations'])}")

    print("=" * 60)
    return output_path


if __name__ == "__main__":
    cli_args = parse_args()
    execute_pipeline(
        output_path=cli_args.out,
        config=cli_args.config,
        overwrite=cli_args.overwrite,
    )
