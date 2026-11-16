# More than a Hack 2027 — Team The Outliers

Official competition repository for **More than a Hack 2027 (Phase 1: Automated Ball Possession Tracking)**, organized by **[FC Barcelona](https://www.fcbarcelona.com)**, **[Metrica Sports](https://metrica-sports.com)**, and **[Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**.

- **Team Name:** `The Outliers`
- **Competition Track:** Phase 1 (Automated Ball Possession Tracking)
- **Target Match:** `BAR-ATH` (FC Barcelona vs. Athletic Club, 199,319 video frames @ 25 fps)
- **Primary Submission Deliverable:** `outputs_v2/submission_v2_1_external.csv` (SHA-256 `5b4c3db7ede46f497822604eda83d837836ea336a4da5dad2312492439c98c5a`)
- **Baseline Fallback:** `outputs_v2/submission_v2.csv` (SHA-256 `89ff92a62a6e571430c0ff06716f6cf1136c8367cbf3eb1c3be4dfcfb04ef6f2`, accessible via `--config apriori`)

**Status notice (Nov 2026).** Earlier versions of this repository (`v1`, now archived in [`legacy_v1/`](legacy_v1/)) fitted models and parameters using the public 5-minute ground-truth window. Following the organizers' rules clarification, `v1` was retired and superseded. The active `src/v2/` pipeline (`v2.0.1-apriori` and `v2.1-external`) never loads ground-truth labels or match video; all engineering constants were fixed *a priori* or tuned exclusively on external open data ([SkillCorner Open Data](https://github.com/SkillCorner/opendata.git)).

---

## 1. Documentation Index

- [`README.md`](README.md) (this file): Unified repository guide — architecture, quickstart, reproducibility, test coverage, Stage 3 external tuning summary, structural diagnostics, limitations, and retired v1 archive inventory.
- [`TECHNICAL_DOCUMENT.md`](TECHNICAL_DOCUMENT.md): Concise official technical document prepared for submission.
- [`CONSTANTS.md`](CONSTANTS.md): Parameter tables, classification tiers (Tier F fixed vs. Tier B engineering), physical rationales, and the formal Decision Record.
- [`tools/tuning/PLAN.md`](tools/tuning/PLAN.md): Pre-registered external tuning protocol (Stage 3).

---

## 2. Compliance Principles & Mathematical Architecture (`src/v2/`)

### Design & Compliance Principles
1. **No Ground-Truth Dependence:** No component of `src/v2/` reads, trains on, or selects parameters using ground-truth possession labels or the provided match video.
2. **Complete Algorithmic Determinism:** Zero random seeds, zero neural weights, zero GBDT estimators, and zero network calls.
3. **A Priori & Externally Tuned Constants:** Tier F constants come from competition definitions and tracking geometry; Tier B thresholds were specified *a priori* (`v2.0.1-apriori`) and four Tier B possession thresholds (`R_TOUCH`, `MARGIN`, `T_CONFIRM`, `G_INTERP`) were tuned strictly on disjoint SkillCorner open broadcast data under [`tools/tuning/PLAN.md`](tools/tuning/PLAN.md) (`v2.1-external`). Full provenance is in [`CONSTANTS.md`](CONSTANTS.md).
4. **Isolated Output Directory:** Outputs write exclusively to `outputs_v2/` (`outputs_v2/submission_v2_1_external.csv` by default; `outputs_v2/submission_v2.csv` with `--config apriori`).

### Pipeline Stages

The `src/v2/` pipeline processes each graded half (`11,185..90,651` and `110,808..190,197`) independently across six deterministic stages (shown with active `v2.1-external` defaults; `--config apriori` uses `G_INTERP=50f / 2.0s`, `R_TOUCH<=2.0m`, `MARGIN>=0.5m`, `T_CONFIRM=50f / 2.0s`):

```
[Raw Optical Tracking & Pitch Homography (25 fps)]
               │
               ▼
┌──────────────────────────────┐
│  3.1 Ball Tracking           │ ── Linear interpolation across gaps <= 30 frames (1.2 s)
└──────────────┬───────────────┘
               │
               ├─────────────────────────────────────────┐
               ▼                                         ▼
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  3.2 Control Candidates      │         │  3.4 Dead-Ball Mask          │
│  - Distance d_H, d_A         │         │  - Pre-kickoff dead window   │
│  - R_TOUCH <= 2.5 m          │         │  - Out-of-bounds (12 frames) │
│  - MARGIN >= 0.75 m          │         │  - Collective Stop (p75<1.5) │
└──────────────┬───────────────┘         └──────────────┬───────────────┘
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│  3.3 & 3.5 Possession State Machine & Composition            │
│  - Live intervals bounded by dead-ball mask                  │
│  - First candidate establishes possessor (backfilled)        │
│  - Opponent confirmation threshold T_CONFIRM = 30 f (1.2 s)  │
│  - Confirmed turnover back-dated to opp_start                │
│  - Uncontested / loose / occluded ball holds possessor       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  3.6 Fixed-Point Cleanup (Iterated to Convergence)           │
│  - (i) Live flicker < 25 frames (1.0 s) between dead -> D    │
│  - (ii) Dead flicker < 12 frames (0.48 s) between same team  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  3.7 Output Generation                                       │
│  - Graded halves: H, A, D                                    │
│  - Ungraded frames (0..11184, 90652..110807, 190198..199318) │
│    assigned X exclusively (40,462 frames)                    │
│  - Export: 199,319 rows (frame,label)                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Quickstart & Execution

### Prerequisites
- Tested with **Python 3.12.0** (`.venv/Scripts/python.exe`) and **NumPy 2.2.6** (see [`requirements.txt`](requirements.txt)).

### Running the Pipeline
To protect generated files from accidental overwriting, the runner refuses to overwrite existing destination files by default (`FileExistsError`).

#### First Run (Fresh Outputs)
```bash
# Generate primary submission (v2.1-external):
python -B -m src.v2.run_v2

# Generate baseline fallback (v2.0.1-apriori):
python -B -m src.v2.run_v2 --config apriori
```

#### Re-Running with Overwrite
```bash
# Overwrite existing primary submission:
python -B -m src.v2.run_v2 --overwrite

# Overwrite existing baseline fallback:
python -B -m src.v2.run_v2 --config apriori --overwrite
```

#### Custom Destination Path
```bash
python -B -m src.v2.run_v2 --out path/to/output.csv
```

*Measured runtime on CPU:*
- **Cached execution (`data/cache_v2_primitives.npz` present):** ~2.75 s (loads pre-extracted kinematic primitives).
- **Fresh extraction (cache absent):** ~8.64 s (parses raw `BAR-ATH_tactical_atd.txt` and XML metadata from scratch).

### Validate Output Format
```bash
# Validate primary submission:
python -B evaluation/validate_submission.py outputs_v2/submission_v2_1_external.csv

# Validate baseline fallback:
python -B evaluation/validate_submission.py outputs_v2/submission_v2.csv
```

### Run Data-Free Test Suite
This suite does not read gated competition tracking, generated submissions, or match-derived caches:
```bash
python3 -B -m unittest \
  tests.test_v2_spec \
  tests.test_tuning_firewall \
  tests.test_docs_quotes \
  tests.test_v2_hardening -v
```

### Run Participant-Only Full Test Suite
Run this only in the local participant environment containing the gated competition tracking:
```bash
python3 -B -m unittest discover -s tests -p "test_*.py" -v
```

---

## 4. Test Suite Coverage

### A. Pipeline & Compliance Suite ([`tests/test_v2_pipeline.py`](tests/test_v2_pipeline.py) — 9 tests)
1. `test_01_gt_firewall`: Static grep ensuring `src/v2/` contains zero references to ground-truth files, loaders, or forbidden evaluation metrics.
2. `test_02_determinism`: Verifies baseline SHA-256 hashes and run-to-run bit-for-bit determinism for both configurations.
3. `test_03_label_vocabulary`: Verifies all 199,319 rows belong strictly to `{H, A, D, X}` for both configurations.
4. `test_04_block_boundaries`: Verifies `X` appears exclusively outside graded blocks (40,462 frames) and zero `X` inside.
5. `test_05_cache_firewall`: Verifies `src/v2/` never imports, loads, or references v1 cache or probability artifacts.
6. `test_06_official_validator`: Runs `evaluation/validate_submission.py` and asserts `OK` on both configurations.
7. `test_07_clean_clone_no_cache_reproducibility`: Clones code into an OS scratch directory with no caches and confirms bit-for-bit SHA-256 match.
8. `test_08_no_legacy_imports`: Asserts `src/v2/` imports nothing from `src.mtah_*` or `src.jev_*`.
9. `test_09_no_v1_cache_read`: Asserts `src/v2/` never references `cache_bar_ath_*`.

### B. Spec-Conformance Unit Suite ([`tests/test_v2_spec.py`](tests/test_v2_spec.py) — 11 tests)
1. `test_a_flip_dated_at_opp_start`: Turnover is back-dated to `opp_start`, not `T_CONFIRM`.
2. `test_b_candidate_resets_opponent_counter`: Current possessor resets opponent counter; no-candidate frames neither reset nor increment.
3. `test_c_unknown_ball_holds_possessor`: Unknown ball or no candidate holds possessor across gaps $> G_{\text{INTERP}}$.
4. `test_d_ball_gap_interpolation_boundary`: Gaps of exactly $G_{\text{INTERP}}$ are interpolated; $G_{\text{INTERP}} + 1$ stay unknown (tested on both active and `apriori`).
5. `test_e_margin_and_rtouch_boundaries`: Validates `MARGIN` and `R_TOUCH` boundaries (tested on both active and `apriori`).
6. `test_f_restart_backfill_and_hold_rule`: After dead-to-live transition, state resets; first candidate backfills; if none, hold.
7. `test_g_out_of_bounds_boundary_and_t_oob`: `OOB_MARGIN` and `T_OOB` persistence boundary conditions.
8. `test_h_collective_stop_boundaries_and_ball_skip`: `T_STOP - 1` vs `T_STOP`, `V_STOP` boundary, and uncomputable ball speed skip.
9. `test_i_cleanup_rules_and_pass_cap`: Cleanup rules (i), (ii), and 300-interleaved flicker fixed-point convergence.
10. `test_j_half_independence`: Possessor state does not carry over from Half 1 to Half 2.
11. `test_k_pre_kickoff_boundaries`: Pre-kickoff dead block spans `half_start..kickoff-1` for Half 2 and is empty for Half 1.

### C. Tuning Firewall & Quote Verification Suites (3 tests)
- [`tests/test_tuning_firewall.py`](tests/test_tuning_firewall.py) (2 tests): Ensures `tools/tuning/` contains zero BAR-ATH or v1 forbidden tokens and `src/v2/` never imports `tools/tuning/`.
- [`tests/test_docs_quotes.py`](tests/test_docs_quotes.py) (1 test): Runs [`tools/docs/check_quotes.py`](tools/docs/check_quotes.py) to verify every quote across project docs matches its cited source lines with zero uncited quotes.

### D. Data-Free Hardening Regression Suite
- [`tests/test_v2_hardening.py`](tests/test_v2_hardening.py): Verifies external-citation isolation, cache provenance and invalidation, corrupt/legacy cache rebuilding, and bare output-path handling without requiring competition assets.

---

## 5. External Tuning Protocol & Results ([`tools/tuning/`](tools/tuning/), Stage 3)

Tier B possession constants (`R_TOUCH`, `MARGIN`, `T_CONFIRM`, `G_INTERP`) were tuned on open-source broadcast tracking data from **SkillCorner** (zero overlap with `BAR-ATH`).

### 5.1 External Dataset Source & Execution
- **Source Repository:** [SkillCorner Open Data](https://github.com/SkillCorner/opendata.git) (`commit 4340d274572876239c154c90bc507a9b3250a656`, MIT License).
- **Data Scope:** 20 matches from the Australian A-League 2024/2025 (`competition_id: 61`, `season_id: 95`, `competition_edition_id: 870`).
- **Storage Location:** Kept outside the repository (e.g. `../ext_data/skillcorner` or `SKILLCORNER_DATA_DIR`).

To clone the external data and re-run the tuning harness:
```bash
git clone https://github.com/SkillCorner/opendata.git ../ext_data/skillcorner
git -C ../ext_data/skillcorner checkout 4340d274572876239c154c90bc507a9b3250a656
python -B tools/tuning/harness.py
```

The harness executes baseline $\theta_0$ evaluation, an 81-configuration grid search on the 11 tuning matches, top-candidate selection $\theta^*$ by mean `spell_f1_mean`, paired hold-out evaluation on 8 hold-out matches ($\Delta_j = \text{spell\_f1\_mean}(m_j, \theta^*) - \text{spell\_f1\_mean}(m_j, \theta_0)$), bootstrap CIs ($B = 10,000$, decision seed `42` and robustness seeds `0`, `1`), mechanized adoption ($\bar{\Delta} > 0$ and $CI_{\text{low}} > 0$), Amendment 2 diagnostics, and serialization into `tools/tuning/results/`.

### 5.2 Partition, Grid Search & Hold-Out Decision
- **Tuning Set (11 matches):** `1874553, 1886347, 1899585, 1925299, 1927964, 1959846, 1986691, 1996435, 1996436, 2006229, 2006363` — Baseline $\theta_0$ Mean Spell F1: **0.48954**; Top Candidate $\theta^*$ Mean Spell F1: **0.52153** ($+0.03199$).
- **Hold-Out Set (8 matches):** `2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461`
- **Excluded (1 match):** `1953632` (catalog status `not_started`)
- **Selected Candidate $\theta^*$:** $R_{\text{TOUCH}} = 2.5\text{ m}$, $\text{MARGIN} = 0.75\text{ m}$, $T_{\text{CONFIRM}} = 1.2\text{ s}$ (`12f @ 10fps`; `30f @ 25fps`), $G_{\text{INTERP}} = 1.2\text{ s}$ (`12f @ 10fps`; `30f @ 25fps`).

| Hold-Out Match ID | Baseline $\theta_0$ F1 | Candidate $\theta^*$ F1 | Paired Difference $\Delta_j$ |
| :---: | :---: | :---: | :---: |
| 2007448 | 0.51086 | 0.53140 | +0.02054 |
| 2007721 | 0.50061 | 0.53913 | +0.03852 |
| 2010085 | 0.49818 | 0.51799 | +0.01981 |
| 2011166 | 0.48784 | 0.50817 | +0.02032 |
| 2013725 | 0.47022 | 0.49024 | +0.02002 |
| 2015213 | 0.50387 | 0.48215 | -0.02171 |
| 2016236 | 0.46316 | 0.52703 | +0.06387 |
| 2017461 | 0.42200 | 0.45221 | +0.03021 |
| **Mean ($\bar{\Delta}$)** | **0.48209** | **0.50604** | **+0.02395** |

- **Mean Paired Difference ($\bar{\Delta}$):** $+0.023947$
- **Decision Bootstrap 95% CI ($B = 10,000$, Seed 42):** `[+0.007967, +0.039078]` (`Seed 0`: `[+0.007877, +0.039022]`; `Seed 1`: `[+0.008006, +0.038890]`).
- **Verdict:** $\bar{\Delta} > 0$ and $CI_{\text{low}} > 0$ — **ADOPTED** into `src/v2/` as `v2.1-external`.

### 5.3 Amendment 2 External Diagnostics (All 20 Matches)
- **Diagnostic (a) — Coverage & Sensor Stats:** 1,182,244 total discrete period frames (1,123,013 across 19 included matches); 632,293 live in-phase frames (599,455 across 19 included); pooled out-of-play share **46.52%** (549,951 / 1,182,244; 46.62% across 19 included); aggregate in-phase ball detection **92.94%** (587,649 / 632,293); candidate share under $\theta_0$ **52.29%** (330,638 / 632,293).
- **Diagnostic (b) — Dead-Ball Kinematics Agreement:**
  - *Frozen v2 constants converted to 10 fps (`tools/tuning/results/diagnostics_amendment_2b_frozen_constants.csv`):* Reference dead 46.52% (549,951 / 1,182,244), predicted dead 41.39% (489,298 / 1,182,244), precision **93.21%**, recall **82.93%**, F1 **87.77%**.
  - *Non-v2 constants (`tools/tuning/results/diagnostics_amendment_2b.csv`, `$V_{\text{STOP}} = 1.35\text{ m/s}$`, `$T_{\text{STOP}} = 15\text{ f}$`, `$V_{\text{BALL\_STOP}} = 1.5\text{ m/s}$`):* Predicted dead 41.97%, precision **92.37%**, recall **83.34%**, F1 **87.62%**. *(Provenance: introduced in commit `65b5830` ([`tools/tuning/PLAN.md:197`](tools/tuning/PLAN.md)); `1.35 m/s` matches `stoppage_speed_w75` in [`legacy_v1/src/mtah_model.py:355`](legacy_v1/src/mtah_model.py#L355), whereas `T_STOP = 1.5 s` has no recorded source; does not represent v2. Speed inputs also differ: 1 s centered displacement at 10 fps vs. BAR-ATH optical tracker speed at 25 fps.)*

---

## 6. Structural Diagnostics (`BAR-ATH`) & Technical Limitations

### Structural Diagnostics (Bug Detection Only)
*Measured directly on generated `outputs_v2/` CSVs and pipeline execution logs; not scored against ground truth:*

| Metric | `v2.1-external` (Primary) | `v2.0.1-apriori` (Baseline) |
|---|---:|---:|
| **Total Frames** | 199,319 | 199,319 |
| **Graded Frames (H1 / H2)** | 158,857 (79,467 / 79,390) | 158,857 (79,467 / 79,390) |
| **Ungraded Abstentions (`X`)** | 40,462 | 40,462 |
| **Dead Ball (`D`) (H1 / H2)** | 49,565 (22,321 / 27,244) | 49,565 (22,321 / 27,244) |
| **Live Home (`H`) (H1 / H2)** | 62,947 (32,762 / 30,185) | 64,600 (34,966 / 29,634) |
| **Live Away (`A`) (H1 / H2)** | 46,345 (24,384 / 21,961) | 44,692 (22,180 / 22,512) |
| **Possession Spells (H1 / H2)** | 274 (133 / 141) | 205 (104 / 101) |
| **Mean Spell Duration (H1 / H2)** | 17.2 s / 14.8 s | 22.0 s / 20.7 s |
| **Cleanup Iterations** | 2 passes | 2 passes |

### Technical Limitations & Validation Status
1. **Validation Status:** The v2/v2.1 pipeline has not been scored against private ground truth for BAR-ATH; no accuracy, macro-F1, or spell-F1 metric is reported for the match.
2. **Smart Tagging Excluded from Dead Ball:** Smart Tagging event intervals are excluded because, as documented in [`starter/load_smart_tagging.py:38-39`](starter/load_smart_tagging.py):
   > *"The tagging marks a ~5 s window around the restart rather than the instant, so the check below is containment, not equality."*
   and in [`starter/load_smart_tagging.py:77-78`](starter/load_smart_tagging.py):
   > *"the tagging marks a window around the restart, not the instant of it"*.
   Applying these windows directly would mislabel several seconds of live play following restarts.
3. **External Tuning Estimate:** The hold-out improvement (+0.023947 mean paired difference across 8 hold-out fixtures) is an empirical measurement on external broadcast tracking data, not a performance guarantee on BAR-ATH optical tracking. See [`CONSTANTS.md`](CONSTANTS.md) Section 5 for all boundary-effect and definition-discrepancy notes.
4. **Dead-Ball Disparity:** External tuning evaluated under an oracle phase-complement dead-ball mask (46.52% pooled out-of-play share across all 20 matches), whereas BAR-ATH evaluates under a physical heuristic dead-ball mask (31.20% dead-ball share across graded frames, 49,565 / 158,857 frames).
5. **Missing Ball Occlusion:** Ball coordinates are unavailable on 40.65% of graded frames in BAR-ATH (94,287 / 158,857 known; 64,570 unknown). During missing-ball intervals, the stoppage detector relies on player speed distribution.
6. **Collective Stoppage Contribution:** The collective stop rule accounts for 44,637 of the 49,291 union dead-ball frames (~90.6% of dead-ball frames; 28.1% of graded match time).
7. **Player Speed Units:** Documented in [`DATA.md:84-85`](DATA.md): *"position, and speed in m/s"*, consistent with empirical aggregates across graded frames (median 1.41 m/s, p90 3.85 m/s, max 9.43 m/s), despite vendor XML metadata template carrying `<Unit>kph</Unit>` ([`BAR-ATH_tactical_atd.xml:1136`](data/tactical/BAR-ATH_tactical_atd.xml)).

---

## 7. Repository Structure

```
the-outliers-mtah2027/
├── src/
│   └── v2/                        # Active deterministic pipeline
│       ├── constants.py           # Numerical thresholds and configurations
│       ├── extractor.py           # ATD raw text parser and primitive extraction
│       ├── ball_tracker.py        # Missing ball gap interpolation
│       ├── control_detector.py     # Proximity and margin evaluation
│       ├── dead_ball.py           # Boundary and collective stop masks
│       ├── state_machine.py       # Possession state machine
│       ├── cleanup.py             # Fixed-point cleanup
│       └── run_v2.py              # CLI entry point
├── tools/
│   ├── tuning/                    # External tuning on SkillCorner open data
│   │   ├── PLAN.md                # Pre-registered tuning protocol
│   │   ├── harness.py             # Tuning grid search and bootstrap evaluation
│   │   └── results/               # Serialized tuning tables and summaries
│   └── docs/
│       └── check_quotes.py        # Automated quote citation checker
├── tests/                         # Automated test suite (23 compliance, unit & quote tests)
├── outputs_v2/                    # Final pipeline outputs
│   ├── submission_v2_1_external.csv  # Primary submission (v2.1)
│   └── submission_v2.csv          # Baseline fallback (v2.0.1)
├── legacy_v1/                     # Retired v1 codebase (superseded, archived)
├── CONSTANTS.md                   # Constants specification and decision record
├── TECHNICAL_DOCUMENT.md          # Official submission technical document
└── README.md                      # Unified repository documentation
```

---

## 8. Legacy Version 1 Archive ([`legacy_v1/`](legacy_v1/), Retired & Superseded)

The `legacy_v1/` directory preserves the Version 1 codebase developed during the initial phase of the competition solely for auditability and git history continuity. Because `v1` fitted LightGBM models and decoder parameters on the public 5-minute ground-truth window (`evaluation/gt/01_public.csv`), it was retired following the rules clarification. **None of the files in `legacy_v1/` are imported, executed, or used by `src/v2/`.**

- **Archived Modules (`legacy_v1/src/`):**
  - `legacy_v1/src/mtah_features.py`: Feature engineering pipeline computing spatial, kinematic, and contextual tracking features.
  - `legacy_v1/src/mtah_model.py`: LightGBM training harness, 5-fold temporal block cross-validation, and motor-base probability predictor.
  - `legacy_v1/src/mtah_decoder.py`: Post-processing decoder applying dead-ball masking, deflection absorption, and label smoothing.
  - `legacy_v1/src/jev_system_one.py`: Deterministic, local NumPy routines for epistemic calibration and conflict arbitration (referred to in v1 as Jev AI / System One). It makes zero external network calls and uses no LLM or external AI service.
  - `legacy_v1/src/generate_3d_replay_html.py`, `legacy_v1/src/viewer.html`, `legacy_v1/src/three.min.js`: 3D match visualizer and WebGL assets.
- **Archived Scripts & Guides:** `legacy_v1/auto_process_competition.py`, `legacy_v1/TECHNICAL_DOCUMENT.md` (superseded by [`TECHNICAL_DOCUMENT.md`](TECHNICAL_DOCUMENT.md)), `legacy_v1/The_Outliers.md`, `legacy_v1/deliverables/`, `legacy_v1/TEAM_GUIDE_EN.md`, `legacy_v1/DOCUMENTO_TECNICO_ES.md`, and `legacy_v1/GUIA_EQUIPO_ES.md`.

---

## 9. External Data Attribution & License Notice

Tier B constants were tuned on open-source broadcast tracking data provided by **SkillCorner** under the MIT License ([SkillCorner Open Data](https://github.com/SkillCorner/opendata.git), commit `4340d274572876239c154c90bc507a9b3250a656`, covering 20 Australian A-League 2024/25 matches).

In accordance with SkillCorner's credit request (ext:README.md:20-21):
> *"Thus, if you use the data, we kindly ask that you credit SkillCorner and hope you'll notify us on [Twitter](https://twitter.com/skillcorner) so we can follow the great work being done with this data."*

We gratefully acknowledge SkillCorner and PySport for open-sourcing these tracking datasets for sports analytics research.
