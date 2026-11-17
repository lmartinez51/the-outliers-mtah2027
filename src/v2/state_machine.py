# -*- coding: utf-8 -*-
"""Deterministic possession state machine and composition (v2, Sections 3.3 and 3.5)."""

from typing import List, Optional, Tuple
import numpy as np

from src.v2.constants import (
    LABEL_A,
    LABEL_D,
    LABEL_H,
    T_CONFIRM,
)


def compose_half_possession(
    dead_mask: np.ndarray,
    candidates: np.ndarray,
    start_frame: int,
    end_frame: int,
    initial_possessor: Optional[str] = None,
    t_confirm: int = T_CONFIRM,
    candidate_track_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str]:
    """Compose dead-ball and live possession labels for one half.

    Args:
        dead_mask: Bool array True on dead frames.
        candidates: Object array with 'H', 'A', or None for each frame.
        start_frame: First frame of the half.
        end_frame: Last frame of the half.
        initial_possessor: Possessor carried from previous half or None.
        t_confirm: Candidate threshold to confirm possession change.
        candidate_track_ids: Optional frame-local candidate slot IDs. When present,
            evidence from different opponent tracks cannot be combined into one
            turnover confirmation run.

    Returns:
        labels: String array of shape (N,) containing 'H', 'A', or 'D' in the half.
        final_possessor: Last active possessor at the end of the half.
    """
    n_frames = len(dead_mask)
    labels = np.full(n_frames, LABEL_D, dtype=object)

    pre_stoppage_possessor: Optional[str] = initial_possessor

    # Find live segments where dead_mask is False inside [start_frame, end_frame]
    is_live = ~dead_mask[start_frame : end_frame + 1]
    live_indices = np.arange(start_frame, end_frame + 1)[is_live]

    if len(live_indices) == 0:
        return labels, (pre_stoppage_possessor or LABEL_H)

    # Break live frames into contiguous segments
    segments: List[Tuple[int, int]] = []
    seg_start = live_indices[0]
    prev_f = live_indices[0]

    for f in live_indices[1:]:
        if f == prev_f + 1:
            prev_f = f
        else:
            segments.append((seg_start, prev_f))
            seg_start = f
            prev_f = f
    segments.append((seg_start, prev_f))

    # Process each live segment
    for t_start, t_end in segments:
        # Check for control candidates in this live segment
        first_cand_idx: Optional[int] = None
        for t in range(t_start, t_end + 1):
            if candidates[t] is not None:
                first_cand_idx = t
                break

        if first_cand_idx is None:
            # "If no candidate frame occurs before the next dead interval or the end of the half,
            # hold the pre-stoppage possessor."
            hold_team = pre_stoppage_possessor or LABEL_H
            for t in range(t_start, t_end + 1):
                labels[t] = hold_team
            pre_stoppage_possessor = hold_team
            continue

        # "After each restart the state is none; the first candidate frame's team becomes possessor
        # and labels are back-filled from the restart frame."
        current_possessor = candidates[first_cand_idx]
        for t in range(t_start, first_cand_idx + 1):
            labels[t] = current_possessor

        opp_counter = 0
        opp_start: Optional[int] = None
        opp_track_id = -1

        for t in range(first_cand_idx + 1, t_end + 1):
            c = candidates[t]
            if c == current_possessor:
                opp_counter = 0
                opp_start = None
                opp_track_id = -1
                labels[t] = current_possessor
            elif c is not None and c != current_possessor:
                candidate_track = (
                    int(candidate_track_ids[t]) if candidate_track_ids is not None else -1
                )
                track_changed = (
                    opp_counter > 0
                    and candidate_track >= 0
                    and opp_track_id >= 0
                    and candidate_track != opp_track_id
                )
                if opp_counter == 0 or track_changed:
                    opp_start = t
                    opp_counter = 0
                    opp_track_id = candidate_track
                opp_counter += 1
                if opp_counter >= t_confirm:
                    opp_team = c
                    assert opp_start is not None
                    for k in range(opp_start, t + 1):
                        labels[k] = opp_team
                    current_possessor = opp_team
                    opp_counter = 0
                    opp_start = None
                    opp_track_id = -1
                else:
                    labels[t] = current_possessor
            else:
                # Candidate is None: possessor holds
                labels[t] = current_possessor

        pre_stoppage_possessor = current_possessor

    return labels, (pre_stoppage_possessor or LABEL_H)
