# -*- coding: utf-8 -*-
"""Validity-aware, multi-signal dead-ball mask generation."""

import math
from typing import Optional, Tuple

import numpy as np

from src.v2.constants import (
    DECEL_DELTA, FPS, FRACTION_LOW_STOP, MIN_VALID_PLAYERS, OOB_MARGIN,
    PITCH_LENGTH, PITCH_WIDTH, SPEED_SMOOTH, T_EVENT_CONTEXT, T_OOB,
    T_RESTART, T_STOP, V_BALL_RESTART, V_BALL_STOP, V_MEDIAN_STOP,
    V_P90_STOP, V_PLAYER_RESTART, V_STOP,
)

BALL_UNKNOWN = np.int8(0)
BALL_STOPPED = np.int8(1)
BALL_MOVING = np.int8(2)


def _mark_runs_at_least(condition_array: np.ndarray, min_length: int) -> np.ndarray:
    out = np.zeros(len(condition_array), dtype=bool)
    i = 0
    while i < len(condition_array):
        if not condition_array[i]:
            i += 1
            continue
        start = i
        while i < len(condition_array) and condition_array[i]:
            i += 1
        if i - start >= min_length:
            out[start:i] = True
    return out


def _rolling_nanmean(values: np.ndarray, start: int, end: int, half_w: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    for frame in range(start, end + 1):
        lo, hi = max(start, frame - half_w), min(end + 1, frame + half_w + 1)
        valid = values[lo:hi][np.isfinite(values[lo:hi])]
        if len(valid):
            out[frame] = float(np.mean(valid))
    return out


def _extend_events(events: np.ndarray, start: int, end: int, duration: int) -> np.ndarray:
    out = np.zeros(len(events), dtype=bool)
    last = start - duration - 1
    for frame in range(start, end + 1):
        if events[frame]:
            last = frame
        out[frame] = frame - last <= duration
    return out


def _hysteretic_stop_mask(entry: np.ndarray, live: np.ndarray, start: int, end: int,
                          t_entry: int, t_exit: int) -> np.ndarray:
    out = np.zeros(len(entry), dtype=bool)
    dead, entry_run, exit_run = False, 0, 0
    for frame in range(start, end + 1):
        if not dead:
            entry_run = entry_run + 1 if entry[frame] else 0
            if entry_run >= t_entry:
                out[frame - entry_run + 1:frame + 1] = True
                dead, exit_run = True, 0
        else:
            out[frame] = True
            exit_run = exit_run + 1 if live[frame] else 0
            if exit_run >= t_exit:
                out[frame - exit_run + 1:frame + 1] = False
                dead, entry_run, exit_run = False, 0, 0
    return out


def compute_dead_ball_mask(
    ball_known: np.ndarray, ball_x_m: np.ndarray, ball_y_m: np.ndarray,
    p75_player_speed: np.ndarray, start_frame: int, end_frame: int,
    kickoff_frame: int, pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH, oob_margin: float = OOB_MARGIN,
    t_oob: int = T_OOB, speed_smooth: int = SPEED_SMOOTH,
    v_stop: float = V_STOP, v_ball_stop: float = V_BALL_STOP,
    t_stop: int = T_STOP, fps: int = FPS,
    median_player_speed: Optional[np.ndarray] = None,
    p90_player_speed: Optional[np.ndarray] = None,
    fraction_players_below_1ms: Optional[np.ndarray] = None,
    valid_player_speed_count: Optional[np.ndarray] = None,
    smart_tag_support: Optional[np.ndarray] = None,
    v_median_stop: float = V_MEDIAN_STOP, v_p90_stop: float = V_P90_STOP,
    fraction_low_stop: float = FRACTION_LOW_STOP,
    min_valid_players: int = MIN_VALID_PLAYERS, decel_delta: float = DECEL_DELTA,
    event_context: int = T_EVENT_CONTEXT, v_player_restart: float = V_PLAYER_RESTART,
    v_ball_restart: float = V_BALL_RESTART, t_restart: int = T_RESTART,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine positive boundary, motion-transition, ball, and tag evidence.

    Unknown ball/player measurements are neutral. Smart Tagging only supports an
    independently observed low-motion candidate and never labels its broad window.
    """
    n = len(ball_known)
    mask_pre = np.zeros(n, dtype=bool)
    if kickoff_frame > start_frame:
        mask_pre[start_frame:kickoff_frame] = True

    oob_instant = np.zeros(n, dtype=bool)
    for frame in range(start_frame, end_frame + 1):
        if ball_known[frame]:
            x, y = ball_x_m[frame], ball_y_m[frame]
            oob_instant[frame] = (x < -oob_margin or x > pitch_length + oob_margin
                                  or y < -oob_margin or y > pitch_width + oob_margin)
    mask_oob = np.zeros(n, dtype=bool)
    mask_oob[start_frame:end_frame + 1] = _mark_runs_at_least(
        oob_instant[start_frame:end_frame + 1], t_oob)

    half_w = speed_smooth // 2
    p75 = _rolling_nanmean(p75_player_speed, start_frame, end_frame, half_w)
    median = _rolling_nanmean(p75_player_speed if median_player_speed is None else median_player_speed,
                              start_frame, end_frame, half_w)
    p90 = _rolling_nanmean(p75_player_speed if p90_player_speed is None else p90_player_speed,
                           start_frame, end_frame, half_w)
    if fraction_players_below_1ms is None:
        fraction_source = np.where(np.isfinite(p75_player_speed),
                                   (p75_player_speed < v_stop).astype(np.float32), np.nan)
    else:
        fraction_source = fraction_players_below_1ms
    low_fraction = _rolling_nanmean(fraction_source, start_frame, end_frame, half_w)
    valid_count = (np.where(np.isfinite(p75_player_speed), 22.0, 0.0).astype(np.float32)
                   if valid_player_speed_count is None
                   else _rolling_nanmean(valid_player_speed_count, start_frame, end_frame, half_w))
    player_low = ((valid_count >= min_valid_players) & np.isfinite(median)
                  & (median < v_median_stop) & np.isfinite(p75) & (p75 < v_stop)
                  & np.isfinite(p90) & (p90 < v_p90_stop) & np.isfinite(low_fraction)
                  & (low_fraction >= fraction_low_stop))

    ball_state = np.full(n, BALL_UNKNOWN, dtype=np.int8)
    ball_speed = np.full(n, np.nan, dtype=np.float32)
    for frame in range(start_frame, end_frame + 1):
        t1, t2 = frame - half_w, frame + half_w
        if t1 < start_frame or t2 > end_frame or not (ball_known[t1] and ball_known[t2]):
            continue
        dt = max(1, t2 - t1) / float(fps)
        speed = math.hypot(ball_x_m[t2] - ball_x_m[t1], ball_y_m[t2] - ball_y_m[t1]) / dt
        ball_speed[frame] = speed
        ball_state[frame] = BALL_STOPPED if speed < v_ball_stop else BALL_MOVING

    deceleration = np.zeros(n, dtype=bool)
    lag = max(1, fps)
    for frame in range(start_frame + lag, end_frame + 1):
        before, now = p75[frame - lag], p75[frame]
        deceleration[frame] = (np.isfinite(before) and np.isfinite(now)
                               and before >= v_player_restart and before - now >= decel_delta)
    initiating_event = deceleration | mask_oob
    if smart_tag_support is not None:
        initiating_event |= smart_tag_support.astype(bool)
    event_support = _extend_events(initiating_event, start_frame, end_frame, event_context)

    entry = player_low & ((ball_state == BALL_STOPPED) | event_support)
    live = ((np.isfinite(p75) & (p75 >= v_player_restart))
            | (np.isfinite(ball_speed) & (ball_speed >= v_ball_restart)))
    mask_stop = _hysteretic_stop_mask(entry, live, start_frame, end_frame, t_stop, t_restart)
    return mask_pre | mask_oob | mask_stop, mask_pre, mask_oob, mask_stop
