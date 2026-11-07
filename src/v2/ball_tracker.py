# -*- coding: utf-8 -*-
"""Deterministic ball tracking and linear gap interpolation (v2, Section 3.1)."""

from typing import Tuple
import numpy as np

from src.v2.constants import G_INTERP


def build_ball_track(
    ball_valid: np.ndarray,
    ball_x_m: np.ndarray,
    ball_y_m: np.ndarray,
    start_frame: int,
    end_frame: int,
    g_interp: int = G_INTERP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate ball (x, y) linearly across gaps of at most g_interp frames.

    Args:
        ball_valid: Array of shape (N,) with >0 indicating valid observation.
        ball_x_m: Array of shape (N,) with x position in meters.
        ball_y_m: Array of shape (N,) with y position in meters.
        start_frame: Start of half video block (inclusive).
        end_frame: End of half video block (inclusive).
        g_interp: Maximum gap length in frames to interpolate (default: 50).

    Returns:
        ball_known: Bool array of shape (N,) True where ball is observed or interpolated.
        out_x: Float array of shape (N,) with known x coordinates.
        out_y: Float array of shape (N,) with known y coordinates.
    """
    n_frames = len(ball_valid)
    ball_known = np.zeros(n_frames, dtype=bool)
    out_x = np.full(n_frames, np.nan, dtype=np.float32)
    out_y = np.full(n_frames, np.nan, dtype=np.float32)

    # Copy raw valid frames inside the window
    valid_mask = (ball_valid[start_frame : end_frame + 1] > 0)
    window_indices = np.arange(start_frame, end_frame + 1)
    valid_frame_indices = window_indices[valid_mask]

    for f in valid_frame_indices:
        ball_known[f] = True
        out_x[f] = float(ball_x_m[f])
        out_y[f] = float(ball_y_m[f])

    if len(valid_frame_indices) < 2:
        return ball_known, out_x, out_y

    # Interpolate gaps of length <= g_interp
    for i in range(len(valid_frame_indices) - 1):
        f0 = valid_frame_indices[i]
        f1 = valid_frame_indices[i + 1]
        gap = f1 - f0 - 1

        if 0 < gap <= g_interp:
            x0, y0 = out_x[f0], out_y[f0]
            x1, y1 = out_x[f1], out_y[f1]
            for step in range(1, gap + 1):
                f_interp = f0 + step
                alpha = float(step) / float(gap + 1)
                ball_known[f_interp] = True
                out_x[f_interp] = (1.0 - alpha) * x0 + alpha * x1
                out_y[f_interp] = (1.0 - alpha) * y0 + alpha * y1

    return ball_known, out_x, out_y
