# Technical Document: Deterministic Possession Tracking Pipeline (v2.1)

**Team:** The Outliers  
**Competition:** More than a Hack 2027 (Phase 1: Automated Ball Possession Tracking)  
**Target Fixture:** `BAR-ATH` (FC Barcelona vs. Athletic Club, 199,319 video frames @ 25 fps)  
**Primary Output:** `outputs_v2/submission_v2_1_external.csv` (SHA-256 `5b4c3db7ede46f497822604eda83d837836ea336a4da5dad2312492439c98c5a`)  
**Baseline Fallback:** `outputs_v2/submission_v2.csv` (SHA-256 `89ff92a62a6e571430c0ff06716f6cf1136c8367cbf3eb1c3be4dfcfb04ef6f2`)

---

## 1. Executive Summary

This document describes the v2.1 deterministic possession tracking architecture developed by Team The Outliers for Phase 1 of the competition. 

The pipeline operates offline on Metrica optical tracking data to predict frame-level possession (`H`, `A`, `D`, `X`). Following the competition rules, the methodology adheres to two fundamental constraints:
1. **Zero Ground-Truth Exposure:** No parameter, heuristic threshold, or algorithmic rule was trained on, tuned against, or selected using the private ground-truth possession labels or the competition video.
2. **Algorithmic Determinism:** The system is an offline state machine combining kinematic proximity filters, spatial margins, turnover confirmation timers, physical dead-ball boundary detectors, and fixed-point morphological cleanup.

All engineering constants were either specified *a priori* from domain kinematics or tuned on open-source broadcast tracking data (SkillCorner Open Data) under a pre-registered protocol.

---

## 2. Data Used and Excluded Assets

### 2.1 Assets Used
- **Automatic Tracking Data (ATD):** `data/tactical/BAR-ATH_tactical_atd.txt` and `BAR-ATH_tactical_atd.xml`, providing player and ball $(x, y)$ coordinates and speeds across 199,319 frames at 25 fps.
- **Pitch Dimensions:** $105.3\text{ m} \times 68.0\text{ m}$, as documented in (DATA.md:93-94):
  > *"The metadata reports 105 × 68 m; `games.py` carries 105.3 × 68.0, measured from the corrected tracking."*
- **Kickoff Timestamps:**
  - Half 1 Kickoff: Frame 11,185, documented in (FAQ.md:34-35):
    > *"first half, by contrast, is graded from the kickoff itself"*
  - Half 2 Kickoff: Frame 113,742, documented in (FAQ.md:33-34):
    > *"does not kick off until **113,742**"*
- **Graded Block Boundaries:** Half 1 (`11,185..90,651`, 79,467 frames) and Half 2 (`110,808..190,197`, 79,390 frames), summing to 158,857 graded frames. All 40,462 frames outside these blocks are assigned `X`.

### 2.2 Excluded Assets
- **Competition Video (`BAR-ATH_tactical.mp4`):** Viewed visually only to understand what counts as possession; not decoded, read, or processed by the pipeline or by any tuning code. No video file is part of the repository, and the pipeline's output was not checked against the video.
- **Smart Tagging (`BAR-ATH_smart_tagging.csv`):** Excluded from dead-ball masking because event timestamps mark broad intervals rather than exact restarts, as documented in (starter/load_smart_tagging.py:38-39):
  > *"The tagging marks a ~5 s window around the restart rather than the instant, so the check below is containment, not equality."*
- **Ground Truth (`evaluation/gt/`):** Never loaded, opened, or imported by the pipeline.
- **Legacy v1 Models:** Machine learning models (GBDT classifiers, neural embeddings) and probability caches from earlier iterations are excluded and archived in `legacy_v1/`.

---

## 3. Methodology & Architecture

The v2.1 pipeline processes each match half independently through six modular stages:

```
[Raw Optical Tracking (25 fps)]
               │
               ▼
┌──────────────────────────────┐
│  3.1 Kinematic Ball Interp   │ ── Linear interpolation across gaps <= G_INTERP (30 frames / 1.2 s)
└──────────────┬───────────────┘
               │
               ├─────────────────────────────────────────┐
               ▼                                         ▼
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  3.2 Control Candidates      │         │  3.4 Physical Dead-Ball Mask │
│  - d_H, d_A to ball          │         │  - Pre-kickoff dead window   │
│  - d <= R_TOUCH (2.5 m)      │         │  - Out-of-bounds (12 frames) │
│  - Margin >= MARGIN (0.75 m) │         │  - Collective Stop (p75<1.5) │
└──────────────┬───────────────┘         └──────────────┬───────────────┘
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│  3.3 & 3.5 Possession State Machine & Composition            │
│  - Active intervals bounded by dead-ball mask                │
│  - Restart segment backfilled by first valid candidate       │
│  - Turnover confirmed after T_CONFIRM (30 frames / 1.2 s)    │
│  - Confirmed turnover back-dated to opp_start                │
│  - Uncontested, loose, or occluded ball holds prior state    │
└──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  3.6 Fixed-Point Morphological Cleanup                       │
│  - Rule (i): Live flicker < 25 frames (1.0 s) -> D           │
│  - Rule (ii): Dead flicker < 12 frames (0.48 s) -> Team      │
│  - Iterated until mathematical convergence                   │
└──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  3.7 Output Generation                                       │
│  - Graded blocks: H, A, D                                    │
│  - Ungraded frames: X                                        │
│  - 199,319 total rows export                                 │
└──────────────────────────────────────────────────────────────┘
```

### Parameter Specification and Classification Tiers

Parameters are categorized into two tiers:
- **Tier F (Fixed):** Grounded in official documentation, pitch geometry, or video frame rate.
- **Tier B (Engineering):** Kinematic thresholds and temporal buffers.

| Parameter | Value (v2.1) | Value (v2.0.1) | Tier | Origin / Specification Source |
|---|:---:|:---:|:---:|---|
| `FPS` | `25` | `25` | **F** | (DATA.md:163-164): `round(seconds * 25)` is a video frame |
| `PITCH_LENGTH_M` | `105.3` | `105.3` | **F** | (DATA.md:93-94): measured from corrected tracking |
| `PITCH_WIDTH_M` | `68.0` | `68.0` | **F** | (DATA.md:93-94): canonical pitch width |
| `HALF_1_KICKOFF` | `11,185` | `11,185` | **F** | (FAQ.md:34-35): first half kickoff frame |
| `HALF_2_KICKOFF` | `113,742` | `113,742` | **F** | (FAQ.md:33-34): second half kickoff frame |
| `GRADED_BLOCKS` | `[(11185, 90651), (110808, 190197)]` | Same | **F** | (DATA.md:255-259): official evaluation ranges |
| `R_TOUCH` | `2.5 m` | `2.0 m` | **B** | Externally tuned on SkillCorner open data |
| `MARGIN` | `0.75 m` | `0.50 m` | **B** | Externally tuned on SkillCorner open data |
| `T_CONFIRM` | `30 frames (1.2 s)` | `50 frames (2.0 s)` | **B** | Externally tuned on SkillCorner open data |
| `G_INTERP` | `30 frames (1.2 s)` | `50 frames (2.0 s)` | **B** | Externally tuned on SkillCorner open data |
| `OOB_MARGIN` | `1.0 m` | `1.0 m` | **B** | Pitch touchline buffer (a priori) |
| `T_OOB` | `12 frames (0.48 s)` | `12 frames (0.48 s)` | **B** | Out-of-bounds persistence window (a priori) |
| `V_STOP` | `1.5 m/s` | `1.5 m/s` | **B** | Player collective walking threshold (a priori) |
| `V_BALL_STOP` | `1.0 m/s` | `1.0 m/s` | **B** | Ball stationary speed threshold (a priori) |
| `T_STOP` | `75 frames (3.0 s)` | `75 frames (3.0 s)` | **B** | Stoppage duration filter (a priori) |
| `MIN_LIVE_FLICKER` | `25 frames (1.0 s)` | `25 frames (1.0 s)` | **B** | Rule (i) cleanup duration (a priori) |
| `MAX_DEAD_FLICKER` | `12 frames (0.48 s)` | `12 frames (0.48 s)` | **B** | Rule (ii) cleanup duration (a priori) |

---

## 4. Provenance of Externally Tuned Constants

### 4.1 External Dataset Scope and Attribution
Tier B constants ($R_{\text{TOUCH}}$, $\text{MARGIN}$, $T_{\text{CONFIRM}}$, $G_{\text{INTERP}}$) were tuned on broadcast tracking data open-sourced by **SkillCorner** under the MIT License ([SkillCorner Open Data](https://github.com/SkillCorner/opendata.git), commit `4340d274572876239c154c90bc507a9b3250a656`).

The dataset comprises 20 matches from the Australian A-League 2024/2025 season. In accordance with SkillCorner's credit request (ext:README.md:20-21):
> *"Thus, if you use the data, we kindly ask that you credit SkillCorner and hope you'll notify us on [Twitter](https://twitter.com/skillcorner) so we can follow the great work being done with this data."*

### 4.2 Pre-Registered Protocol & Decision Rule
The tuning experiment was specified and locked in `tools/tuning/PLAN.md` prior to execution:
- **Partition:** 11 tuning fixtures, 8 hold-out evaluation fixtures, and 1 excluded fixture (`1953632`, catalog status `not_started`).
- **Grid:** 81 configurations over $R_{\text{TOUCH}} \in \{1.5, 2.0, 2.5\}\text{ m}$, $\text{MARGIN} \in \{0.25, 0.50, 0.75\}\text{ m}$, $T_{\text{CONFIRM}} \in \{1.2, 2.0, 2.8\}\text{ s}$, and $G_{\text{INTERP}} \in \{1.2, 2.0, 2.8\}\text{ s}$.
- **Objective:** Macro-averaged spell-level F1 evaluated against SkillCorner dynamic events within phase intervals.
- **Adoption Criterion:** Quoted from (tools/tuning/PLAN.md:175-177):
  > *"4. **All-or-Nothing Adoption Criterion:**"*
  > *- Candidate $\theta^*$ is adopted **as a whole** into `src/v2/` **IF AND ONLY IF**:*
  > *$$\bar{\Delta} > 0 \quad \text{AND} \quad CI_{\text{low}} > 0$$*

### 4.3 Empirical Findings
- Selected candidate $\theta^*$: $R_{\text{TOUCH}} = 2.5\text{ m}$, $\text{MARGIN} = 0.75\text{ m}$, $T_{\text{CONFIRM}} = 1.2\text{ s}$, $G_{\text{INTERP}} = 1.2\text{ s}$.
- Hold-out paired difference: $\bar{\Delta} = +0.023947$ (+2.39 percentage points).
- Bootstrap 95% Confidence Interval ($B = 10,000$ draws, decision seed 42): `[+0.007967, +0.039078]` ($CI_{\text{low}} > 0$).
- Robustness sensitivity checks: seed 0 yielded `[+0.007877, +0.039022]`; seed 1 yielded `[+0.008006, +0.038890]`.

### 4.4 Technical Limitations of External Tuning
1. **Grid Boundary Effect:** All four tuned parameters sit at the edge of the pre-registered search box ($R_{\text{TOUCH}}$ and $\text{MARGIN}$ at maximum; $T_{\text{CONFIRM}}$ and $G_{\text{INTERP}}$ at minimum). The unconstrained optimum on external data may lie outside the grid, but search expansion was not permitted under protocol rules.
2. **Label Definition Discrepancy:** SkillCorner assigns possession via a per-frame dictionary (`possession: dict with keys player_id and group which indicates which player/team (home or away team) are in possession`, ext:README.md:40-41). In contrast, the competition requires sustained controlled possession, stating in (CHALLENGE.md:55):
   > *"A change of possession occurs when a player from the opposing team gains **controlled** possession of the ball."*
   and in (CHALLENGE.md:76-80):
   > *"#### What does not count as a change of possession"*
   > *"- a simple touch of the ball without control"*
   > *"- a deflection of a pass that does not intercept it"*
   > *"- an aerial duel won without subsequent control of the ball by that team"*
   Consequently, external tuning may optimize for different transition responsiveness than competition ground truth demands.
3. **Dead-Ball and Sensor Differences:** External data evaluates under an oracle dead-ball mask derived from phases of play with 92.94% ball detection (587,649 / 632,293 frames, `tools/tuning/results/diagnostics_amendment_2a.csv`). BAR-ATH evaluates under a heuristic dead-ball mask with 59.35% ball detection across all graded frames (94,287 / 158,857 frames).
4. **Estimated Nature of Gain:** The hold-out improvement (+0.024 mean paired difference) is an empirical measurement on external broadcast tracking data, not a performance guarantee on the BAR-ATH match.

---

## 5. Compliance Statement

- **Data Inputs:** The v2/v2.1 codebase reads automatic tracking files (`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`). It does not open, read, or reference `data/tactical/BAR-ATH_tactical.mp4`, `data/tactical/BAR-ATH_smart_tagging.csv`, or any file in `evaluation/gt/`.
- **Ground-Truth Independence:** The pipeline contains zero neural weights, zero random seeds, and was never optimized against ground-truth possession labels.
- **Historical v1 Retirement:** Earlier iterations of this repository fitted models and parameters on the public 5-minute ground-truth window. Following the organizers' rule clarification that parameter selection on ground truth is prohibited, Version 1 was retired, superseded, and archived in `legacy_v1/`. Version 2.1 is independent of legacy v1 code.

---

## 6. Reproducibility & Verification

### Execution Instructions
The pipeline is executed from the repository root:
```bash
# Primary submission (v2.1-external):
python -B -m src.v2.run_v2

# Baseline fallback (v2.0.1-apriori):
python -B -m src.v2.run_v2 --config apriori

# Re-run with overwrite:
python -B -m src.v2.run_v2 --overwrite
```

### Bit-for-Bit Cryptographic Hashes
- `outputs_v2/submission_v2_1_external.csv`: `5b4c3db7ede46f497822604eda83d837836ea336a4da5dad2312492439c98c5a`
- `outputs_v2/submission_v2.csv`: `89ff92a62a6e571430c0ff06716f6cf1136c8367cbf3eb1c3be4dfcfb04ef6f2`

### Measured Runtime and Environment
- **Platform:** Windows, Python 3.12.0, NumPy 2.2.6.
- **Execution Speed:** ~2.75 seconds with cached primitives (`data/cache_v2_primitives.npz`); ~8.64 seconds for fresh parsing of the raw 199,319-line ATD text file.
- **Testing:** The automated test suite (`tests/`) contains 23 tests verifying ground-truth firewall integrity, pipeline determinism, official format validation, spec conformance, clean-clone reproducibility, and quote accuracy.

---

## 7. Known Structural Diagnostics & Limitations

1. **Absence of BAR-ATH Ground-Truth Metrics:** Because ground-truth data was not opened, no accuracy, macro-F1, or spell-F1 score is reported for BAR-ATH.
2. **Dead-Ball Distribution:** On BAR-ATH, predicted dead-ball (`D`) represents 31.20% of graded frames (49,565 / 158,857 frames), compared to a 46.52% pooled out-of-play share on external broadcast data.
3. **Dominance of Collective Stoppage:** Collective player deceleration accounts for 44,637 of the 49,291 union dead-ball frames (~90.6% of dead frames; 28.1% of graded frames).
4. **Ball Occlusion Gaps:** Ball tracking is absent on 40.65% of graded frames (94,287 known vs 64,570 missing). When the ball is untracked, control candidates cannot be declared, and the state machine holds prior possession until dead ball or re-detection.
5. **$G_{\text{INTERP}}$ Invariance on BAR-ATH:** On BAR-ATH, changing $G_{\text{INTERP}}$ between 30 frames (1.2 s) and 50 frames (2.0 s) produces zero label differences across all 158,857 graded frames, because all missing-ball gaps bridged by interpolation are $\le 30$ frames (1.2 s).

---

## 8. Appendix: Repository File Map

```
the-outliers-mtah2027/
├── src/
│   └── v2/                         # Active deterministic pipeline
│       ├── constants.py            # Tier F and Tier B parameters
│       ├── extraction.py           # ATD tracking parser & primitive builder
│       ├── interpolation.py        # Kinematic ball trajectory interpolation
│       ├── candidates.py           # Proximity & margin candidate filters
│       ├── dead_ball.py            # Boundary & collective stop masks
│       ├── state_machine.py        # Possession transition state machine
│       ├── cleanup.py              # Fixed-point morphological cleaner
│       └── run_v2.py               # CLI runner with overwrite protection
├── tools/
│   ├── tuning/                     # External tuning suite (SkillCorner data)
│   │   ├── PLAN.md                 # Frozen pre-registered protocol
│   │   ├── harness.py              # Grid search and bootstrap evaluator
│   │   ├── README.md               # External results & diagnostic tables
│   │   └── results/                # CSV and JSON result summaries
│   └── docs/
│       └── check_quotes.py         # Automated quote citation checker
├── tests/                          # Compliance, unit, and quote tests
├── outputs_v2/                     # Output submission CSVs
├── legacy_v1/                      # Retired, superseded v1 codebase
├── CONSTANTS.md                    # Detailed parameter rationales
├── README_v2.md                    # Extended technical manual
├── TECHNICAL_DOCUMENT.md           # Submission technical document
└── README.md                       # Top-level repository guide
```
