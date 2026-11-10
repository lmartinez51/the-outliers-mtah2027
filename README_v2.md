# Version 2.0: Deterministic, Ground-Truth-Free Possession Pipeline

Official documentation for the v2 submission architecture developed by **Team The Outliers** for **More than a Hack 2027 (Phase 1: Automated Ball Possession Tracking)**.

---

## 1. Compliance Statement & Design Principles

Following the organizers' clarification of the challenge rules:
1. **No ground-truth dependence:** No component of v2 reads, was trained on, or was selected using ground-truth possession data or the provided video. The team's earlier v1 work did use the public 5-minute ground-truth window (see the Status notice in README.md). The v2 constants were fixed before v2 was first run on the match and were not selected by optimizing any metric.
2. **Complete Algorithmic Determinism:** The solution contains zero random seeds, zero neural weights, zero GBDT estimators, zero network requests, and no parameter search or metric-based selection was performed for v2.
3. **A Priori Frozen Constants:** Every numerical threshold was specified before running the pipeline on the match data, grounded in physical, optical, and biomechanical domain reasoning (documented in [`CONSTANTS.md`](CONSTANTS.md)).
4. **Isolated Output Directory:** v2 outputs exclusively to `outputs_v2/submission_v2.csv` (and tuned candidate output to `outputs_v2/submission_v2_1_external.csv`), leaving all v1 legacy artifacts untouched.
5. **External Data Attribution:** External parameter tuning conducted in Stage 3 utilizes open tracking data provided by **SkillCorner** under the MIT License ([SkillCorner Open Data](https://github.com/SkillCorner/opendata.git)).

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
- Tested and executed with **Python 3.12.0** (`.venv\Scripts\python.exe`) and **NumPy 2.2.6**.

### Run the Pipeline

The entry point `src.v2.run_v2` executes the deterministic pipeline. By default, it runs the promoted primary configuration `v2_1` (`outputs_v2/submission_v2_1_external.csv`).

To protect generated files from accidental modification or deletion, **the runner refuses to overwrite existing destination files by default** (`FileExistsError`).

#### First Run (Fresh Output)
```bash
# Primary submission (v2.1-external, default):
python -B -m src.v2.run_v2

# Baseline fallback (v2.0.1-apriori):
python -B -m src.v2.run_v2 --config apriori
```

#### Re-Running with Overwrite
If destination files already exist in `outputs_v2/`, pass `--overwrite` to re-generate them:
```bash
# Re-run primary submission:
python -B -m src.v2.run_v2 --overwrite

# Re-run baseline fallback:
python -B -m src.v2.run_v2 --config apriori --overwrite
```

#### Custom Output Path
To write to an alternate path:
```bash
python -B -m src.v2.run_v2 --out path/to/custom_submission.csv
```

*Measured execution times on standard CPU:*
- **Cached execution (`data/cache_v2_primitives.npz` present):** **~2.75 seconds**. Loads pre-extracted kinematic primitives directly into memory.
- **Fresh extraction (cache absent):** **~8.64 seconds**. Parses raw ASCII tracking (`BAR-ATH_tactical_atd.txt`) and XML structure from scratch and rebuilds the local primitive cache.

### Validate Output Format
```bash
# Validate primary submission:
python -B evaluation/validate_submission.py outputs_v2/submission_v2_1_external.csv

# Validate baseline fallback:
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
7. `test_07_clean_clone_no_cache_reproducibility`: Clones the code into an OS scratch directory, starts with zero caches, and confirms bit-for-bit SHA-256 match.
8. `test_08_no_legacy_imports`: Asserts that `src/v2/` imports zero modules from `src/mtah_*`, `src/jev_*`, or `starter/*`.
9. `test_09_no_v1_cache_read`: Asserts that `run_v2.py` does not open or read `data/cache_bar_ath_features.npz`.

### B. Spec-Conformance Unit Suite ([`tests/test_v2_spec.py`](tests/test_v2_spec.py) - 11 tests)
Synthetic tests verifying all algorithmic rules on small deterministic arrays without match data or ground truth:
1. `test_a_flip_dated_at_opp_start`: Asserts turnover is back-dated to initial opponent touch (`opp_start`) once confirmed at `T_CONFIRM`.
2. `test_b_candidate_resets_opponent_counter`: Current possessor resets opponent candidate counter; no-candidate frames neither reset nor increment.
3. `test_c_unknown_ball_holds_possessor`: Asserts unknown ball or absence of candidate holds prior possessor across gaps $> G_{INTERP}$.
4. `test_d_ball_gap_interpolation_boundary`: Gaps of exactly $G_{INTERP} = 50$ frames are interpolated; $51$ remain un-interpolated.
5. `test_e_margin_and_rtouch_boundaries`: Validates candidate criteria ($d \le 2.0$ m and margin $\ge 0.5$ m) boundary conditions.
6. `test_f_restart_backfill_and_hold_rule`: First candidate's team back-fills to segment restart; if none, prior possessor holds.
7. `test_g_oob_persistence_and_margin`: Out-of-bounds requires $\ge 12$ consecutive frames beyond 1.0 m margin.
8. `test_h_collective_stop_boundaries_and_ball_skip`: Stoppage requires $\ge 75$ frames of $p75 < 1.5$ m/s and ball speed $< 1.0$ m/s; untracked ball skips ball condition.
9. `test_i_cleanup_rules_and_pass_cap`: Validates Rule (i), Rule (ii), and 300-interleaved flicker fixed-point convergence.
10. `test_j_half_independence`: Possessor state does not carry over from Half 1 to Half 2.
11. `test_k_pre_kickoff_boundaries`: Pre-kickoff dead block spans `HALF_2_START..HALF_2_KICKOFF - 1` in Half 2 and is empty in Half 1.

---

## 5. Structural Diagnostics (Bug Detection Only)

*Printed solely to verify algorithmic integrity and rule compliance; not used for scoring or parameter tuning:*

| Metric | Full Match | Half 1 | Half 2 |
|---|:---:|:---:|:---:|
| **Total Frames** | 199,319 | 79,467 | 79,390 |
| **Ungraded Abstentions (`X`)** | 40,462 (20.30% of total) · (0.00% of graded) | 0 (0.00% of 79,467) | 0 (0.00% of 79,390) |
| **Dead Ball Frames (`D`)** | 49,565 (24.87% of total) · (31.20% of 158,857 graded) | 22,321 (28.09% of 79,467) | 27,244 (34.32% of 79,390) |
| **Live Home Frames (`H`)** | 64,600 (32.41% of total) · (40.67% of 158,857 graded) | 34,710 (43.68% of 79,467) | 29,890 (37.65% of 79,390) |
| **Live Away Frames (`A`)** | 44,692 (22.42% of total) · (28.13% of 158,857 graded) | 22,436 (28.23% of 79,467) | 22,256 (28.03% of 79,390) |
| **Possession Spells** | 205 | 104 | 101 |
| **Mean Spell Duration** | 21.3 s | 22.0 s | 20.7 s |
| **Cleanup Iterations to Fixed Point** | 2 passes | 2 passes | 2 passes |

---

## 6. Known Limitations and Validation Status

1. **Validation Status:** v2 has not been scored against ground truth at this commit, so no accuracy, macroF1, or spellF1 metric exists for this version.
2. **Smart Tagging Excluded from Dead Ball:** Smart Tagging event intervals are not used for dead-ball detection because, as documented in [`starter/load_smart_tagging.py:38-39`](starter/load_smart_tagging.py):
   > *"The tagging marks a ~5 s window around the restart rather than the instant, so the check below is containment, not equality."*
   and lines 77-78:
   > *"the tagging marks a window around the restart, not the instant of it"*.
   Applying these windows directly would mislabel several seconds of live play following restarts.
3. **Ball Tracking Gaps & Occlusions:** The ball position is unknown on **40.65% of graded frames** (94,287 / 158,857 known; 64,570 unknown). Over these missing-ball intervals, the collective stoppage detector relies exclusively on the 75th percentile of player speeds.
4. **Dominance of Collective Stoppage:** The collective stop rule accounts for **44,637 of the 49,291 union dead-ball frames** (~90.6% of dead-ball frames; 28.1% of graded match time).
5. **External Data Attribution & Parameter Tuning (Stage 3):** Tier B constants were tuned on external broadcast tracking data open-sourced by **SkillCorner** under the MIT License ([SkillCorner Open Data](https://github.com/SkillCorner/opendata.git), commit `4340d274572876239c154c90bc507a9b3250a656`, covering 20 Australian A-League 2024/25 fixtures). In accordance with Stage 3 Amendment 2(a) diagnostics, the external out-of-play share was a pooled **46.52%** of period frames across all 20 SkillCorner matches (549,951 / 1,182,244 discrete frames; or 46.62%, 523,558 / 1,123,013 frames across the 19 included matches), compared to v2's physical dead-ball (`D`) share of **31.20%** of graded frames (49,565 / 158,857 frames) on BAR-ATH; competitions, tactical tempos, and stoppage definitions differ between datasets, and no performance claim is made regarding BAR-ATH ground truth.
6. **Player Speed Units:** Documented in [`DATA.md:84-85`](DATA.md) as position and speed in **m/s** (`"position, and speed in m/s"`), verified by empirical aggregates across graded frames (median 1.41 m/s, p90 3.85 m/s, max 9.43 m/s), despite vendor XML metadata template carrying `<Unit>kph</Unit>` ([`BAR-ATH_tactical_atd.xml:1136`](data/tactical/BAR-ATH_tactical_atd.xml)).
