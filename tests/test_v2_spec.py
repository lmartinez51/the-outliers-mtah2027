# -*- coding: utf-8 -*-
"""Spec-conformance unit tests on synthetic data only (Stage 2B, Section 4).

All tests use hand-made synthetic arrays. Zero match data, zero ground-truth data,
and zero file labels are used.
"""

import unittest
import numpy as np

from src.v2.ball_tracker import build_ball_track
from src.v2.cleanup import cleanup_half_labels
from src.v2.constants import (
    FPS,
    G_INTERP,
    HALF_1_KICKOFF,
    HALF_1_START,
    HALF_2_KICKOFF,
    HALF_2_START,
    LABEL_A,
    LABEL_D,
    LABEL_H,
    MARGIN,
    MIN_DEAD_FLICKER,
    MIN_LIVE_FLICKER,
    OOB_MARGIN,
    PITCH_LENGTH,
    PITCH_WIDTH,
    R_TOUCH,
    SPEED_SMOOTH,
    T_CONFIRM,
    T_OOB,
    T_STOP,
    V_BALL_STOP,
    V_STOP,
)
from src.v2.control_detector import detect_control_candidates
from src.v2.dead_ball import compute_dead_ball_mask
from src.v2.state_machine import compose_half_possession


class TestV2SpecConformance(unittest.TestCase):
    """Synthetic unit tests enforcing exact specification compliance."""

    def test_a_flip_dated_at_opp_start(self):
        """(a) A flip is dated at opp_start (back-dated), not at T_CONFIRM."""
        n = 100
        dead = np.zeros(n, dtype=bool)
        candidates = np.full(n, None, dtype=object)

        # Frame 0..9: H is candidate and establishes possession
        for f in range(10):
            candidates[f] = LABEL_H

        # At frame 20, Opponent A candidate appears and continues for exactly T_CONFIRM (50) frames
        opp_start = 20
        for f in range(opp_start, opp_start + T_CONFIRM):
            candidates[f] = LABEL_A

        labels, _ = compose_half_possession(dead, candidates, 0, n - 1, initial_possessor=LABEL_H)

        # Before opp_start, possession is H
        for f in range(0, opp_start):
            self.assertEqual(labels[f], LABEL_H, f"Frame {f} should be H before opp_start")

        # From opp_start (frame 20) onwards, possession MUST be back-dated to A
        for f in range(opp_start, opp_start + T_CONFIRM):
            self.assertEqual(labels[f], LABEL_A, f"Frame {f} should be back-dated to A")

    def test_b_candidate_resets_opponent_counter(self):
        """(b) Current possessor resets opponent counter; no-candidate frames neither reset nor increment."""
        n = 150
        dead = np.zeros(n, dtype=bool)
        candidates = np.full(n, None, dtype=object)

        # Establish H
        candidates[0] = LABEL_H

        # A appears for 20 frames (counter = 20)
        for f in range(10, 30):
            candidates[f] = LABEL_A

        # 10 frames with no candidate (None): counter neither resets nor increments
        # A appears for another 20 frames (counter = 40)
        for f in range(40, 60):
            candidates[f] = LABEL_A

        # Now 1 candidate frame of H appears at frame 60: MUST reset counter to 0!
        candidates[60] = LABEL_H

        # Now A appears for 35 frames (frames 70..104). Counter reaches 35 < T_CONFIRM (50)
        for f in range(70, 105):
            candidates[f] = LABEL_A

        labels, _ = compose_half_possession(dead, candidates, 0, n - 1, initial_possessor=LABEL_H)

        # Turnover to A should NEVER occur because H reset the counter at frame 60
        for f in range(n):
            self.assertEqual(labels[f], LABEL_H, f"Frame {f} should have remained H")

    def test_c_unknown_ball_holds_possessor(self):
        """(c) Unknown ball or no candidate: possessor holds, even across gap > G_INTERP."""
        n = 300
        dead = np.zeros(n, dtype=bool)
        candidates = np.full(n, None, dtype=object)

        # Establish H
        candidates[0] = LABEL_H

        # No candidate frames from 1 to 299 (including gap of 200 > G_INTERP)
        labels, _ = compose_half_possession(dead, candidates, 0, n - 1, initial_possessor=LABEL_H)

        for f in range(n):
            self.assertEqual(labels[f], LABEL_H, f"Frame {f} should hold possessor H")

    def test_d_ball_gap_interpolation_boundary(self):
        """(d) Gaps of exactly G_INTERP (50) are interpolated; G_INTERP + 1 (51) stay unknown."""
        # Gap of exactly 50: frames 0 and 51 valid, frames 1..50 missing (len 50)
        n1 = 60
        bv1 = np.zeros(n1, dtype=np.float32)
        bx1 = np.full(n1, np.nan, dtype=np.float32)
        by1 = np.full(n1, np.nan, dtype=np.float32)
        bv1[0], bx1[0], by1[0] = 1.0, 10.0, 20.0
        bv1[51], bx1[51], by1[51] = 1.0, 61.0, 71.0

        known1, obx1, oby1 = build_ball_track(bv1, bx1, by1, 0, n1 - 1, g_interp=G_INTERP)
        self.assertTrue(np.all(known1[0:52]), "Gap of exactly G_INTERP must be interpolated")
        self.assertAlmostEqual(obx1[25], 10.0 + (61.0 - 10.0) * (25.0 / 51.0), places=4)

        # Gap of 51: frames 0 and 52 valid, frames 1..51 missing (len 51)
        n2 = 60
        bv2 = np.zeros(n2, dtype=np.float32)
        bx2 = np.full(n2, np.nan, dtype=np.float32)
        by2 = np.full(n2, np.nan, dtype=np.float32)
        bv2[0], bx2[0], by2[0] = 1.0, 10.0, 20.0
        bv2[52], bx2[52], by2[52] = 1.0, 62.0, 72.0

        known2, obx2, oby2 = build_ball_track(bv2, bx2, by2, 0, n2 - 1, g_interp=G_INTERP)
        self.assertTrue(known2[0])
        self.assertTrue(known2[52])
        self.assertFalse(np.any(known2[1:52]), "Gap of G_INTERP + 1 must stay unknown")

    def test_e_margin_and_rtouch_boundaries(self):
        """(e) MARGIN (0.5) and R_TOUCH (2.0) boundary conditions."""
        n = 5
        bv = np.ones(n, dtype=np.float32)
        known = np.ones(n, dtype=bool)
        bx = np.zeros(n, dtype=np.float32)
        by = np.zeros(n, dtype=np.float32)

        dh = np.zeros(n, dtype=np.float32)
        da = np.zeros(n, dtype=np.float32)

        # Frame 0: dH = 2.0 (== R_TOUCH), dA = 2.5 (dA - dH == 0.5 == MARGIN) -> H
        dh[0], da[0] = 2.0, 2.5
        # Frame 1: dH = 2.001 (> R_TOUCH), dA = 2.6 -> None
        dh[1], da[1] = 2.001, 2.6
        # Frame 2: dH = 1.9, dA = 2.399 (dA - dH = 0.499 < MARGIN) -> None
        dh[2], da[2] = 1.9, 2.399
        # Frame 3: dH = 1.9, dA = 2.4 (dA - dH = 0.5 == MARGIN) -> H
        dh[3], da[3] = 1.9, 2.4
        # Frame 4: dA = 1.0, dH = 1.5 (dH - dA = 0.5 == MARGIN) -> A
        dh[4], da[4] = 1.5, 1.0

        cands, _, _ = detect_control_candidates(bv, known, bx, by, dh, da, 0, n - 1)
        self.assertEqual(cands[0], LABEL_H, "dH == R_TOUCH and diff == MARGIN should be H")
        self.assertIsNone(cands[1], "dH > R_TOUCH should be None")
        self.assertIsNone(cands[2], "diff < MARGIN should be None")
        self.assertEqual(cands[3], LABEL_H, "diff == MARGIN should be H")
        self.assertEqual(cands[4], LABEL_A, "diff == MARGIN should be A")

    def test_f_restart_backfill_and_hold_rule(self):
        """(f) After dead-to-live transition, state resets; first candidate backfills; if none, hold."""
        n = 100
        dead = np.zeros(n, dtype=bool)
        candidates = np.full(n, None, dtype=object)

        # 0..19: Live 1 (H is possessor)
        candidates[0] = LABEL_H

        # 20..39: Dead interval
        dead[20:40] = True

        # 40..69: Live 2, first candidate appears at frame 50 and is A
        candidates[50] = LABEL_A

        # 70..79: Dead interval
        dead[70:80] = True

        # 80..99: Live 3, NO candidates appear throughout Live 3
        # (Should hold pre-stoppage possessor, which was A)

        labels, _ = compose_half_possession(dead, candidates, 0, n - 1, initial_possessor=LABEL_H)

        # In Live 2, frames 40..50 must be back-filled to A
        for f in range(40, 70):
            self.assertEqual(labels[f], LABEL_A, f"Live 2 frame {f} should be A (back-filled from restart)")

        # Dead intervals must be D
        for f in range(20, 40):
            self.assertEqual(labels[f], LABEL_D)
        for f in range(70, 80):
            self.assertEqual(labels[f], LABEL_D)

        # In Live 3, frames 80..99 must hold pre-stoppage possessor (A)
        for f in range(80, 100):
            self.assertEqual(labels[f], LABEL_A, f"Live 3 frame {f} should hold pre-stoppage possessor A")

    def test_g_oob_persistence_and_margin(self):
        """(g) OUT_OF_BOUNDS: T_OOB - 1 (11) vs T_OOB (12); OOB_MARGIN boundary."""
        n = 50
        known = np.ones(n, dtype=bool)
        bx = np.full(n, 50.0, dtype=np.float32)  # Inside pitch
        by = np.full(n, 30.0, dtype=np.float32)
        p75 = np.full(n, 3.0, dtype=np.float32)

        # OOB boundary: pitch_len = 105.3, oob_margin = 1.0 -> boundary is 106.3
        # Exactly on boundary 106.3 -> not strictly greater, so not OOB
        bx[0:15] = 106.3
        _, _, oob_boundary, _ = compute_dead_ball_mask(
            known, bx, by, p75, 0, n - 1, kickoff_frame=0
        )
        self.assertFalse(np.any(oob_boundary[0:15]), "Exactly on OOB_MARGIN boundary is not OOB")

        # 11 frames of 106.31 (T_OOB - 1) -> should NOT trigger
        bx[0:50] = 50.0
        bx[10:21] = 106.31
        _, _, oob_11, _ = compute_dead_ball_mask(
            known, bx, by, p75, 0, n - 1, kickoff_frame=0
        )
        self.assertFalse(np.any(oob_11[10:21]), "T_OOB - 1 frames must not trigger OOB")

        # 12 frames of 106.31 (T_OOB) -> MUST trigger
        bx[25:37] = 106.31
        _, _, oob_12, _ = compute_dead_ball_mask(
            known, bx, by, p75, 0, n - 1, kickoff_frame=0
        )
        self.assertTrue(np.all(oob_12[25:37]), "T_OOB frames must trigger OOB")

    def test_h_collective_stop_boundaries_and_ball_skip(self):
        """(h) COLLECTIVE_STOP: T_STOP - 1 vs T_STOP; V_STOP boundary; uncomputable ball skipped."""
        n = 200
        known = np.ones(n, dtype=bool)
        bx = np.zeros(n, dtype=np.float32)
        by = np.zeros(n, dtype=np.float32)

        # 1. Persistence T_STOP - 1 (74) vs T_STOP (75) with speed_smooth=1
        p75_74 = np.full(n, 3.0, dtype=np.float32)
        p75_74[20:94] = 0.5  # Exactly 74 frames
        _, _, _, stop_74 = compute_dead_ball_mask(
            known, bx, by, p75_74, 0, n - 1, 0, speed_smooth=1, t_stop=T_STOP
        )
        self.assertEqual(int(np.sum(stop_74)), 0, "T_STOP - 1 frames must not trigger stop")

        p75_75 = np.full(n, 3.0, dtype=np.float32)
        p75_75[20:95] = 0.5  # Exactly 75 frames
        _, _, _, stop_75 = compute_dead_ball_mask(
            known, bx, by, p75_75, 0, n - 1, 0, speed_smooth=1, t_stop=T_STOP
        )
        self.assertEqual(int(np.sum(stop_75)), 75, "T_STOP frames must trigger stop")
        self.assertTrue(np.all(stop_75[20:95]))

        # 2. V_STOP boundary (1.5 m/s)
        p75_boundary = np.full(n, V_STOP, dtype=np.float32)
        _, _, _, stop_at_b = compute_dead_ball_mask(known, bx, by, p75_boundary, 0, n - 1, 0)
        self.assertEqual(int(np.sum(stop_at_b)), 0, "p75 == V_STOP is not stopped")

        p75_below_b = np.full(n, V_STOP - 0.01, dtype=np.float32)
        _, _, _, stop_below_b = compute_dead_ball_mask(known, bx, by, p75_below_b, 0, n - 1, 0)
        self.assertEqual(int(np.sum(stop_below_b)), n, "p75 < V_STOP must trigger stop")

        # 3. V_BALL_STOP boundary (1.0 m/s)
        # Moving at 1.0 m/s: dx = 1.0 m across 24 frames
        bx_fast = np.array([f * 1.0 for f in range(n)], dtype=np.float32)
        _, _, _, stop_ball_fast = compute_dead_ball_mask(known, bx_fast, by, p75_below_b, 0, n - 1, 0)
        self.assertEqual(int(np.sum(stop_ball_fast)), 0, "Ball speed >= V_BALL_STOP must block stop")

        # 4. Unknown ball skip:
        known_empty = np.zeros(n, dtype=bool)
        _, _, _, stop_unknown = compute_dead_ball_mask(known_empty, bx_fast, by, p75_below_b, 0, n - 1, 0)
        self.assertEqual(int(np.sum(stop_unknown)), n, "Unknown ball must skip ball condition")

    def test_i_cleanup_rules_and_pass_cap(self):
        """(i) Cleanup rules (i), (ii), and 300-interleaved flicker fixed-point convergence."""
        # Rule (i): Live < MIN_LIVE_FLICKER (25) between dead becomes D; 25 stays live
        seq1 = [LABEL_D] * 30 + [LABEL_H] * 24 + [LABEL_D] * 30 + [LABEL_H] * 25 + [LABEL_D] * 30
        c1, _ = cleanup_half_labels(np.array(seq1, dtype=object), 0, len(seq1) - 1)
        self.assertTrue(np.all(c1[30:54] == LABEL_D), "Live segment of 24 between dead must become D")
        self.assertTrue(np.all(c1[84:109] == LABEL_H), "Live segment of 25 between dead must stay H")

        # Rule (ii): Dead < MIN_DEAD_FLICKER (12) between same possessor becomes live; between different stays D
        seq2 = [LABEL_H] * 30 + [LABEL_D] * 11 + [LABEL_H] * 30 + [LABEL_D] * 11 + [LABEL_A] * 30
        c2, _ = cleanup_half_labels(np.array(seq2, dtype=object), 0, len(seq2) - 1)
        self.assertTrue(np.all(c2[30:41] == LABEL_H), "Dead segment of 11 between H and H must become H")
        self.assertTrue(np.all(c2[71:82] == LABEL_D), "Dead segment of 11 between H and A must stay D")

        # (iii) 300 interleaved flickers: must clean completely with no pass cap
        interleaved = [LABEL_D] * 30
        for _ in range(300):
            interleaved.extend([LABEL_H] * 5)
            interleaved.extend([LABEL_D] * 30)
        c3, iters = cleanup_half_labels(np.array(interleaved, dtype=object), 0, len(interleaved) - 1)
        self.assertTrue(np.all(c3 == LABEL_D), "All 300 flickers must be cleaned to D")
        self.assertGreater(iters, 1, "Should take multiple passes to reach fixed point")

    def test_j_half_independence(self):
        """(j) Half independence: possessor state does not carry over from half 1 to half 2."""
        n = 100
        dead = np.zeros(n, dtype=bool)
        candidates = np.full(n, None, dtype=object)

        # Half 1 ends with A as possessor
        candidates[0] = LABEL_A
        labels1, final_poss1 = compose_half_possession(dead, candidates, 0, n - 1, initial_possessor=None)
        self.assertEqual(final_poss1, LABEL_A)

        # Half 2 starts with pre-kickoff dead block (0..20), then live restart with candidate H at frame 30
        dead2 = np.zeros(n, dtype=bool)
        dead2[0:21] = True
        candidates2 = np.full(n, None, dtype=object)
        candidates2[30] = LABEL_H

        # Half 2 MUST be executed with initial_possessor=None (independent)
        labels2, _ = compose_half_possession(dead2, candidates2, 0, n - 1, initial_possessor=None)

        # Pre-kickoff frames are D
        for f in range(21):
            self.assertEqual(labels2[f], LABEL_D)

        # Live frames 21..30 MUST be backfilled to H, NOT carrying over A from Half 1
        for f in range(21, 31):
            self.assertEqual(labels2[f], LABEL_H, f"Frame {f} in Half 2 should be H (independent of Half 1)")

    def test_k_pre_kickoff_boundaries(self):
        """(k) Pre-kickoff dead block spans half_start..kickoff-1 for half 2 and is empty for half 1."""
        known = np.ones(200000, dtype=bool)
        bx = np.full(200000, 50.0, dtype=np.float32)
        by = np.full(200000, 30.0, dtype=np.float32)
        p75 = np.full(200000, 3.0, dtype=np.float32)

        # Half 1: start=11185, kickoff=11185 -> empty
        _, pk1, _, _ = compute_dead_ball_mask(
            known, bx, by, p75, HALF_1_START, 20000, kickoff_frame=HALF_1_KICKOFF
        )
        self.assertEqual(int(np.sum(pk1)), 0, "Half 1 pre-kickoff mask must be empty")

        # Half 2: start=110808, kickoff=113742 -> exactly 2934 frames (110808..113741)
        _, pk2, _, _ = compute_dead_ball_mask(
            known, bx, by, p75, HALF_2_START, 120000, kickoff_frame=HALF_2_KICKOFF
        )
        pk2_indices = np.where(pk2)[0]
        self.assertEqual(len(pk2_indices), 2934, f"Expected 2934 frames, got {len(pk2_indices)}")
        self.assertEqual(pk2_indices[0], 110808)
        self.assertEqual(pk2_indices[-1], 113741)


if __name__ == "__main__":
    unittest.main()
