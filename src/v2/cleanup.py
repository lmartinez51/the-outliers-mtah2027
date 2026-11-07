# -*- coding: utf-8 -*-
"""Deterministic cleanup and flicker suppression (v2, Section 3.6)."""

from typing import List, Tuple
import numpy as np

from src.v2.constants import (
    LABEL_A,
    LABEL_D,
    LABEL_H,
    MIN_DEAD_FLICKER,
    MIN_LIVE_FLICKER,
)


def cleanup_half_labels(
    labels: np.ndarray,
    start_frame: int,
    end_frame: int,
    min_live_flicker: int = MIN_LIVE_FLICKER,
    min_dead_flicker: int = MIN_DEAD_FLICKER,
) -> Tuple[np.ndarray, int]:
    """Iterate deterministic cleanup rules to a fixed point with no pass cap.

    Rules applied in fixed order:
        (i) A live segment shorter than min_live_flicker between two dead segments becomes D.
        (ii) A dead segment shorter than min_dead_flicker between two live segments with the
             same possessor becomes live with that team.

    Args:
        labels: String array of labels.
        start_frame: Start of half (inclusive).
        end_frame: End of half (inclusive).
        min_live_flicker: Minimum length of live segment between dead segments (default: 25).
        min_dead_flicker: Minimum length of dead segment between same possessor (default: 12).

    Returns:
        out_labels: Fixed-point cleaned label array.
        iterations: Number of passes until convergence.
    """
    out_labels = labels.copy()
    iterations = 0

    while True:
        iterations += 1
        changed = False

        # Pass (i): Live segments < min_live_flicker between two dead segments become D
        t = start_frame
        while t <= end_frame:
            if out_labels[t] != LABEL_D:
                seg_start = t
                while t <= end_frame and out_labels[t] != LABEL_D:
                    t += 1
                seg_end = t - 1
                length = seg_end - seg_start + 1

                has_prev_dead = (seg_start > start_frame and out_labels[seg_start - 1] == LABEL_D)
                has_next_dead = (seg_end < end_frame and out_labels[seg_end + 1] == LABEL_D)

                if has_prev_dead and has_next_dead and length < min_live_flicker:
                    for k in range(seg_start, seg_end + 1):
                        out_labels[k] = LABEL_D
                    changed = True
            else:
                t += 1

        # Pass (ii): Dead segments < min_dead_flicker between identical possessors become live
        t = start_frame
        while t <= end_frame:
            if out_labels[t] == LABEL_D:
                seg_start = t
                while t <= end_frame and out_labels[t] == LABEL_D:
                    t += 1
                seg_end = t - 1
                length = seg_end - seg_start + 1

                has_prev_live = (seg_start > start_frame and out_labels[seg_start - 1] in (LABEL_H, LABEL_A))
                has_next_live = (seg_end < end_frame and out_labels[seg_end + 1] in (LABEL_H, LABEL_A))

                if has_prev_live and has_next_live and length < min_dead_flicker:
                    prev_team = out_labels[seg_start - 1]
                    next_team = out_labels[seg_end + 1]
                    if prev_team == next_team:
                        for k in range(seg_start, seg_end + 1):
                            out_labels[k] = prev_team
                        changed = True
            else:
                t += 1

        if not changed:
            break

    return out_labels, iterations
