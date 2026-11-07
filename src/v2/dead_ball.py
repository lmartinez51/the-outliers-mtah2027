# -*- coding: utf-8 -*-
"""Deterministic dead-ball mask generation (v2, Section 3.4)."""

import math
from typing import Tuple
import numpy as np

from src.v2.constants import (
    FPS,
    OOB_MARGIN,
    PITCH_LENGTH,
    PITCH_WIDTH,
    SPEED_SMOOTH,
    T_OOB,
    T_STOP,
    V_BALL_STOP,
    V_STOP,
)


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
            end = i
            if (end - start) >= min_length:
                out_mask[start:end] = True
        else:
            i += 1

    return out_mask


def compute_dead_ball_mask(
    ball_known: np.ndarray,
    ball_x_m: np.ndarray,
    ball_y_m: np.ndarray,
    p75_player_speed: np.ndarray,
    start_frame: int,
    end_frame: int,
    kickoff_frame: int,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
    oob_margin: float = OOB_MARGIN,
    t_oob: int = T_OOB,
    speed_smooth: int = SPEED_SMOOTH,
    v_stop: float = V_STOP,
    v_ball_stop: float = V_BALL_STOP,
    t_stop: int = T_STOP,
    fps: int = FPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the union dead-ball mask per half.

    Components:
        a) PRE_KICKOFF: from start_frame to kickoff_frame - 1.
        b) OUT_OF_BOUNDS: ball_known and outside pitch + oob_margin for >= t_oob frames.
        c) COLLECTIVE_STOP: rolling p75 player speed < v_stop and ball speed < v_ball_stop
           for >= t_stop frames.
        d) SMART_TAGGING set-piece windows: not used (unsupported by organizer docs).

    Returns:
        dead_mask: Combined boolean array True on dead-ball frames.
        mask_pre_kickoff: Pre-kickoff dead mask.
        mask_oob: Out-of-bounds dead mask.
        mask_stop: Collective stoppage dead mask.
    """
    n_frames = len(ball_known)
    mask_pre_kickoff = np.zeros(n_frames, dtype=bool)
    mask_oob = np.zeros(n_frames, dtype=bool)
    mask_stop = np.zeros(n_frames, dtype=bool)

    # 3.4(a) PRE_KICKOFF
    if kickoff_frame > start_frame:
        mask_pre_kickoff[start_frame:kickoff_frame] = True

    # 3.4(b) OUT_OF_BOUNDS
    oob_instant = np.zeros(n_frames, dtype=bool)
    for f in range(start_frame, end_frame + 1):
        if ball_known[f]:
            bx = ball_x_m[f]
            by = ball_y_m[f]
            if (
                bx < -oob_margin
                or bx > (pitch_length + oob_margin)
                or by < -oob_margin
                or by > (pitch_width + oob_margin)
            ):
                oob_instant[f] = True

    # Find runs >= t_oob inside [start_frame, end_frame]
    sub_oob = _mark_runs_at_least(oob_instant[start_frame : end_frame + 1], t_oob)
    mask_oob[start_frame : end_frame + 1] = sub_oob

    # 3.4(c) COLLECTIVE_STOP
    # Compute centered rolling mean of p75_player_speed
    half_w = speed_smooth // 2
    p75_smooth = np.zeros(n_frames, dtype=np.float32)

    for f in range(start_frame, end_frame + 1):
        w_start = max(start_frame, f - half_w)
        w_end = min(end_frame + 1, f + half_w + 1)
        slice_speeds = p75_player_speed[w_start:w_end]
        valid_speeds = slice_speeds[~np.isnan(slice_speeds)]
        p75_smooth[f] = float(np.mean(valid_speeds)) if len(valid_speeds) > 0 else 0.0

    stop_instant = np.zeros(n_frames, dtype=bool)
    ball_time_delta = float(speed_smooth) / float(fps)

    for f in range(start_frame, end_frame + 1):
        player_is_stopped = (p75_smooth[f] < v_stop)

        # Ball speed evaluation
        t1 = f - half_w
        t2 = f + half_w
        if t1 >= start_frame and t2 <= end_frame and ball_known[t1] and ball_known[t2]:
            dx = ball_x_m[t2] - ball_x_m[t1]
            dy = ball_y_m[t2] - ball_y_m[t1]
            dist = math.hypot(dx, dy)
            ball_speed = dist / ball_time_delta
            ball_is_stopped = (ball_speed < v_ball_stop)
        else:
            # "if either end is unknown the ball condition is not evaluated"
            ball_is_stopped = True

        if player_is_stopped and ball_is_stopped:
            stop_instant[f] = True

    # Find runs >= t_stop inside [start_frame, end_frame]
    sub_stop = _mark_runs_at_least(stop_instant[start_frame : end_frame + 1], t_stop)
    mask_stop[start_frame : end_frame + 1] = sub_stop

    # Combined dead-ball mask
    dead_mask = mask_pre_kickoff | mask_oob | mask_stop
    return dead_mask, mask_pre_kickoff, mask_oob, mask_stop
