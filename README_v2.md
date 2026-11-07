# Version 2.0: Deterministic, Ground-Truth-Free Possession Pipeline

Official documentation for the v2 submission architecture developed by **Team The Outliers** for **More than a Hack 2027 (Phase 1: Automated Ball Possession Tracking)**.

---

## 1. Compliance Statement & Design Principles

Following the organizers' clarification of the challenge rules:
1. **Zero Ground-Truth Exposure:** No component of this pipeline was trained, fitted, calibrated, or tuned on ground-truth possession data or on the provided video.
2. **Complete Algorithmic Determinism:** The solution contains zero random seeds, zero neural weights, zero GBDT estimators, zero network requests, and zero heuristics selected to match in-sample outcomes.
3. **A Priori Frozen Constants:** Every numerical threshold was specified before running the pipeline on the match data, grounded in physical, optical, and biomechanical domain reasoning (documented in [`CONSTANTS.md`](CONSTANTS.md)).
4. **Isolated Output Directory:** v2 outputs exclusively to `outputs_v2/submission_v2.csv`, leaving all v1 legacy artifacts untouched.

---

## 2. Mathematical Architecture

The v2 pipeline operates offline per half (`11,185..90,651` and `110,808..190,197`) across six modular stages:

```
[Raw Optical Tracking & Pitch Homography]
               │
               ▼
┌──────────────────────────────┐
│  3.1 Ball Tracking           │ ── Linear interpolation across gaps <= 50 frames (2.0 s)
└──────────────┬───────────────┘
               │
               ├─────────────────────────────────────────┐
               ▼                                         ▼
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  3.2 Control Candidates      │         │  3.4 Dead-Ball Mask          │
│  - Distance d_H, d_A         │         │  - Pre-kickoff dead window   │
│  - R_TOUCH <= 2.0 m          │         │  - Out-of-bounds (12 frames) │
│  - MARGIN >= 0.5 m           │         │  - Collective Stop (p75 < 1.5)│
└──────────────┬───────────────┘         └──────────────┬───────────────┘
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│  3.3 & 3.5 Possession State Machine & Composition            │
│  - Live intervals bounded by dead-ball mask                  │
│  - First candidate establishes possessor (backfilled)        │
│  - Opponent confirmation threshold T_CONFIRM = 50 frames     │
│  - Confirmed turnover back-dated to opp_start                │
│  - Uncontested / loose / occluded ball holds possessor       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  3.6 Fixed-Point Cleanup (Iterated to Convergence)          │
│  - (i) Live flicker < 25 frames between dead balls -> D      │
│  - (ii) Dead flicker < 12 frames between same team -> Team   │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  3.7 Output Generation                                       │
│  - Graded halves: H, A, D                                    │
│  - Ungraded frames (0..11184, 90652..110807, etc.): X strictly│
│  - Export: outputs_v2/submission_v2.csv (199,319 rows)       │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Quickstart & Execution

### Prerequisites
- Python 3.9+ with `numpy` installed.

### Run the Pipeline
```bash
python -B -m src.v2.run_v2
```
*Total execution time:* **~3.2 seconds** on standard CPU.

### Validate Output Format
```bash
python -B evaluation/validate_submission.py outputs_v2/submission_v2.csv
```

### Run Full Test Suite
```bash
python -B -m unittest discover -s tests -p "test_v2_*.py" -v
```

---

## 4. Test Suite Coverage

The automated verification suite spans 20 comprehensive unit and integration tests across two test suites:

### A. Pipeline & Compliance Suite ([`tests/test_v2_pipeline.py`](tests/test_v2_pipeline.py) - 9 tests)
1. `test_01_gt_firewall`: Static grep ensuring `src/v2/` contains zero references to ground-truth files, loaders, or forbidden evaluation metrics.
2. `test_02_determinism`: Verifies that successive independent runs yield identical SHA-256 hashes.
3. `test_03_label_vocabulary`: Verifies all 199,319 rows belong strictly to `{H, A, D, X}`.
4. `test_04_block_boundaries`: Verifies that `X` appears strictly outside the official graded blocks (exactly 40,462 frames) and zero `X` inside.
5. `test_05_cache_firewall`: Verifies that `src/v2/` does not load the legacy GBDT probability cache (`cache_bar_ath_probs.npz`) or v1 submission files.
6. `test_06_official_validator`: Runs the official format checker `evaluation/validate_submission.py` and asserts zero formatting violations.
7. `test_07_clean_clone_reproducibility`: Clones the code into an OS scratch directory, starts with zero caches, and confirms bit-for-bit SHA-256 match.
8. `test_08_no_legacy_imports`: Asserts that `src/v2/` imports zero modules from `src/mtah_*`, `src/jev_*`, or `starter/*`.
9. `test_09_no_v1_cache_reads`: Asserts that `run_v2.py` does not open or read `data/cache_bar_ath_features.npz`.

### B. Spec-Conformance Unit Suite ([`tests/test_v2_spec.py`](tests/test_v2_spec.py) - 11 tests)
Synthetic tests verifying all algorithmic rules on small deterministic arrays without match data or ground truth:
1. `test_a_linear_ball_interpolation_gap_le_50`: Interpolates gaps $\le 50$ frames, leaves gaps $> 50$ un-interpolated.
2. `test_b_ball_known_false_at_origin`: Ball at $(0, 0)$ is treated as untracked.
3. `test_c_control_candidate_r_touch_and_margin`: Validates candidate criteria ($d \le 2.0$ m and margin $\ge 0.5$ m) and contested dual proximity.
4. `test_d_first_candidate_backfills_to_restart`: First candidate's team back-fills to segment restart.
5. `test_e_no_candidate_holds_pre_stoppage_possessor`: Live segments without candidates hold prior possessor.
6. `test_f_t_confirm_turnover_latency_and_backdating`: Opponent needs 50 frames to confirm turnover, back-dated to initial touch.
7. `test_g_out_of_bounds_margin_and_persistence`: Out of bounds requires $\ge 12$ consecutive frames beyond 1.0 m margin.
8. `test_h_collective_stop_boundaries_and_ball_skip`: Stoppage requires $\ge 75$ frames of $p75 < 1.5$ m/s and ball speed $< 1.0$ m/s; untracked ball skips ball condition.
9. `test_i_cleanup_rule_i_live_flicker`: Live intervals $< 25$ frames bounded by dead ball convert to `D`.
10. `test_j_cleanup_rule_ii_dead_flicker`: Dead intervals $< 12$ frames between same possessor convert to live ball.
11. `test_k_cleanup_fixed_point_convergence`: Iterates until zero changes without pass cap.

---

## 5. Structural Diagnostics (Bug Detection Only)

*Printed solely to verify algorithmic integrity and rule compliance; not used for scoring or parameter tuning:*

| Metric | Full Match | Half 1 | Half 2 |
|---|:---:|:---:|:---:|
| **Total Frames** | 199,319 | 79,467 | 79,390 |
| **Ungraded Abstentions (`X`)** | 40,462 (20.3%) | 0 (0.0%) | 0 (0.0%) |
| **Dead Ball Frames (`D`)** | 49,565 (24.9%) | 22,321 (28.1%) | 27,244 (34.3%) |
| **Live Home Frames (`H`)** | 64,600 (32.4%) | 34,710 (43.7%) | 29,890 (37.6%) |
| **Live Away Frames (`A`)** | 44,692 (22.4%) | 22,436 (28.2%) | 22,256 (28.0%) |
| **Possession Spells** | 205 | 104 | 101 |
| **Mean Spell Duration** | 21.3 s | 22.0 s | 20.7 s |
| **Cleanup Iterations to Fixed Point** | 2 passes | 2 passes | 2 passes |
