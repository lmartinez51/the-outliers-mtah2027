# Constants Registry & A Priori Rationale (v2 & v2.1 Pipeline)

This registry records all constants utilized in the deterministic, ground-truth-free v2 possession tracking pipeline (`src/v2/`).

In accordance with the organizers' rules clarification, **no constant in this table was trained, fitted, calibrated, or tuned on ground-truth possession data or on the provided match video**. Every value was fixed before v2 was first run on the match and was not selected by optimizing any metric. The values were reasoned from CHALLENGE.md and general physical considerations; the team had seen public-window statistics during the earlier v1 work.

---

## 1. Frozen Baseline Constants Table (v2.0.1-apriori)

| ID | Value | Tier | Origin / Official Citation | Written A Priori Rationale | Introduction Date |
|---|---|:---:|---|---|:---:|
| `FPS` | `25` | **F** | [DATA.md:163-164](DATA.md) ("`round(seconds * 25)` is a video frame") | Fixed sampling rate of tactical video and optical tracking. | 2026-11-07 |
| `PITCH_LENGTH` | `105.3 m` | **F** | `games.py:GAMES["BAR-ATH"]["pitch"]`, [DATA.md:93-94](DATA.md) ("The metadata reports 105 × 68 m; `games.py` carries 105.3 × 68.0, measured from the corrected tracking.") | Field length measured from the corrected tracking. | 2026-11-07 |
| `PITCH_WIDTH` | `68.0 m` | **F** | `games.py:GAMES["BAR-ATH"]["pitch"]`, [DATA.md:93-94](DATA.md) ("The metadata reports 105 × 68 m; `games.py` carries 105.3 × 68.0, measured from the corrected tracking.") | Field width measured from the corrected tracking. | 2026-11-07 |
| `HALF_1_START` | `11,185` | **F** | [DATA.md:256](DATA.md), [FAQ.md:30](FAQ.md) | Official first graded frame of First Half. | 2026-11-07 |
| `HALF_1_END` | `90,651` | **F** | [DATA.md:256](DATA.md), [FAQ.md:30](FAQ.md) | Official last graded frame of First Half. | 2026-11-07 |
| `HALF_1_KICKOFF` | `11,185` | **F** | [FAQ.md:34-35](FAQ.md) ("first half, by contrast, is graded from the kickoff itself") | Match kickoff instant for First Half. | 2026-11-07 |
| `HALF_2_START` | `110,808` | **F** | [DATA.md:258](DATA.md), [FAQ.md:30](FAQ.md) | Official first graded frame of Second Half. | 2026-11-07 |
| `HALF_2_END` | `190,197` | **F** | [DATA.md:258](DATA.md), [FAQ.md:30](FAQ.md) | Official last graded frame of Second Half. | 2026-11-07 |
| `HALF_2_KICKOFF` | `113,742` | **F** | [FAQ.md:33-34](FAQ.md) ("does not kick off until **113,742**") | Published Second Half kickoff frame. | 2026-11-07 |
| `N_VIDEO_FRAMES` | `199,319` | **F** | [DATA.md:246](DATA.md), `games.py` | Total video frames in `BAR-ATH`. | 2026-11-07 |
| `R_TOUCH` | `2.0 m` | **B** | Team decision, 2026-11-07, before the first v2 run | Maximum physical reach of a player plus automatic optical tracking position error ($\sim 0.5\text{--}1.0\text{ m}$). | 2026-11-07 |
| `MARGIN` | `0.5 m` | **B** | Team decision, 2026-11-07, before the first v2 run | Minimum separation required between the nearest player and opponent before declaring control candidate. | 2026-11-07 |
| `G_INTERP` | `50 frames` (2.0 s) | **B** | Team decision, 2026-11-07, before the first v2 run | Typical flight time of a pass; gaps $\le 2\text{ s}$ represent temporary tracking occlusions during active play. | 2026-11-07 |
| `T_CONFIRM` | `50 frames` (2.0 s) | **B** | [CHALLENGE.md:73](CHALLENGE.md) ("a player keeps the ball for a few seconds") | Minimum duration required for opponent to demonstrate controlled possession before confirming turnover. | 2026-11-07 |
| `OOB_MARGIN` | `1.0 m` | **B** | Team decision, 2026-11-07, before the first v2 run | Buffer beyond touchlines/by-lines to absorb homography projection inaccuracies and ball radius. | 2026-11-07 |
| `T_OOB` | `12 frames` (0.48 s) | **B** | Team decision, 2026-11-07, before the first v2 run | Persistence window to rule out single-frame sensor flutter or noisy boundary spikes. | 2026-11-07 |
| `SPEED_SMOOTH` | `25 frames` (1.0 s) | **B** | Team decision, 2026-11-07, before the first v2 run | Centered 1-second rolling average window to filter instantaneous kinematic tracking noise. | 2026-11-07 |
| `V_STOP` | `1.5 m/s` (5.4 km/h) | **B** | Team decision, 2026-11-07, before the first v2 run | Standard human walking pace applied to the 75th percentile of player speeds to identify collective match stoppages. | 2026-11-07 |
| `V_BALL_STOP` | `1.0 m/s` (3.6 km/h) | **B** | Team decision, 2026-11-07, before the first v2 run | Ball velocity threshold below which the ball is considered stationary / at rest. | 2026-11-07 |
| `T_STOP` | `75 frames` (3.0 s) | **B** | Team decision, 2026-11-07, before the first v2 run | Minimum stoppage duration required to declare dead ball from collective velocity drop. | 2026-11-07 |
| `MIN_LIVE_FLICKER` | `25 frames` (1.0 s) | **B** | Team decision, 2026-11-07, before the first v2 run | Cleanup threshold: live intervals shorter than 1.0 s bounded by dead balls are deemed spurious touches and converted to `D`. | 2026-11-07 |
| `MIN_DEAD_FLICKER` | `12 frames` (0.48 s) | **B** | Team decision, 2026-11-07, before the first v2 run | Cleanup threshold: dead intervals shorter than 0.48 s between identical possessors are deemed sensor artifacts and converted to live ball. | 2026-11-07 |

---

## 2. Externally Tuned Constants (v2.1-external)

Under the Stage 3 tuning protocol, Tier B possession constants were tuned on external tracking data from SkillCorner open data (MIT license), completely disjoint from the provided fixture (`BAR-ATH`).

- **Origin:** External tuning, SkillCorner open data, MIT license, 2026-11-09, commit `65b5830` (A-League 2024/2025; 11 tuning matches, 8 hold-out matches).
- **Adoption Criteria:** Paired difference bootstrap over 8 hold-out matches ($B = 10,000$ draws, seed 42): $\bar{\Delta} = +0.023947$, $95\%\text{ CI} = [+0.007967, +0.039078]$. Since $\bar{\Delta} > 0$ and $CI_{\text{low}} > 0$, the candidate configuration was **mechanically adopted**.

| Constant | Adopted Value (Physical) | Adopted Value (Frames @ 25 fps) | Previous A Priori Value | Written Empirical & Physical Rationale |
|---|:---:|:---:|:---:|---|
| `R_TOUCH` | `2.5 m` | N/A (meters) | `2.0 m` | Accommodates wider optical tracking sensor dispersion for lunging/extended player reaches. |
| `MARGIN` | `0.75 m` | N/A (meters) | `0.50 m` | Stronger contested-ball rejection threshold; reduces false-positive turnovers during 50-50 duels. |
| `T_CONFIRM` | `1.2 s` | `30 frames` | `50 frames` (2.0 s) | Faster turnover confirmation; matches typical duration of controlled second-touch receptions. |
| `G_INTERP` | `1.2 s` | `30 frames` | `50 frames` (2.0 s) | Shorter interpolation horizon avoids bridging across actual turnovers during loose ball episodes. |
| `MIN_LIVE_FLICKER` | `1.0 s` | `25 frames` | `25 frames` (1.0 s) | Kept *a priori* per Amendment 1 (cleanup rules suspended during external evaluation). |

---

## 3. Additional Algorithmic Conventions & Formulas

1. **Player Speed Percentile (`p75`):**
   - Percentile value: `0.75` (75th percentile of active, non-NaN player speeds).
   - Nearest-rank indexing formula implemented in `extractor.py:134`:
     ```python
     valid_speeds.sort()
     p75_player_speed[idx] = valid_speeds[int(0.75 * (len(valid_speeds) - 1))]
     ```
2. **Ball Velocity Uncomputability Skip:**
   - In `dead_ball.py:128-130`, when evaluating centered ball velocity over $[f - 12, f + 12]$, if either endpoint is missing / unknown, the ball condition is skipped (`ball_is_stopped = True`), allowing the stoppage detector to evaluate based on collective player speed alone.
3. **Pitch Dimensions Source:**
   - Dimensions used: $105.3\text{ m} \times 68.0\text{ m}$ from `games.py:GAMES["BAR-ATH"]["pitch"]`.
   - The XML metadata header reports canonical dimensions $105.0\text{ m} \times 68.0\text{ m}$ ([`BAR-ATH_tactical_atd.xml:68-69`](data/tactical/BAR-ATH_tactical_atd.xml)). As documented in [`DATA.md:93-94`](DATA.md): *"The metadata reports 105 × 68 m; `games.py` carries 105.3 × 68.0, measured from the corrected tracking."*

---

## 4. Classification Tiers

- **Tier F (Fixed):** Non-tunable constants determined strictly by competition definition, data geometry, or official documentation.
- **Tier B (Engineering Constants):** Parameters governing physical control candidates, turnover confirmation, and interpolation. Tuned strictly on external open tracking data.

---

## 5. Decision Record: Promotion of v2.1-external

- **Date:** 2026-11-10
- **Human Decision:** The team has decided that the primary submission for the competition is **`v2.1-external`** (`outputs_v2/submission_v2_1_external.csv`, SHA-256 `5b4c3db7ede46f497822604eda83d837836ea336a4da5dad2312492439c98c5a`), with the frozen a priori configuration **`v2.0.1-apriori`** (`outputs_v2/submission_v2.csv`, SHA-256 `89ff92a62a6e571430c0ff06716f6cf1136c8367cbf3eb1c3be4dfcfb04ef6f2`) preserved as the reproducible baseline fallback.
- **Pre-Registered Adoption Criterion:** Promotion was executed strictly in accordance with the pre-registered decision rule in [`tools/tuning/PLAN.md:175-178`](tools/tuning/PLAN.md):
  > *"4. **All-or-Nothing Adoption Criterion:**"*
  > *- Candidate $\theta^*$ is adopted **as a whole** into `src/v2/` **IF AND ONLY IF**:*
  > *$$\bar{\Delta} > 0 \quad \text{AND} \quad CI_{\text{low}} > 0$$*
  > *where $CI_{\text{low}}$ is the lower bound of the 95% bootstrap confidence interval computed with $B = 10,000$ draws (using `seed = 42` per Amendment 3).*
- **Empirical Validation Metrics (`tools/tuning/results/tuning_summary.json`):**
  - **Mean Paired Difference ($\bar{\Delta}$):** `+0.023947` (+2.39 percentage points of macro-averaged spell-level F1 over the 8 hold-out fixtures).
  - **95% Bootstrap Confidence Interval (`seed = 42`, Pre-Registered Decision Seed):** `[+0.007967, +0.039078]` (strictly positive, $CI_{\text{low}} > 0$).
  - **Robustness Sensitivity Checks ($B = 10,000$ draws):**
    - `seed = 0`: `[+0.007877, +0.039022]` ($CI_{\text{low}} > 0$)
    - `seed = 1`: `[+0.008006, +0.038890]` ($CI_{\text{low}} > 0$)
- **Technical Limitations & Ground-Truth Firewall Boundaries:**
  1. **Sensor & Platform Discrepancy:** External data is broadcast tracking (SkillCorner, 10 fps) derived from broadcast television video using single-camera pan-tilt-zoom tracking and off-screen kinematic extrapolation. The competition dataset is multi-camera high-resolution tactical optical tracking (Metrica FIFA/EPTS, 25 fps) with whole-pitch sensor coverage. Sensor noise profiles, occlusion handling, and coordinate precision differ between broadcast and multi-camera systems.
  2. **Temporal Resolution & Frame-Rate Conversion:** Constants were converted from physical durations (seconds) to discrete frames assuming continuous kinematic invariance ($T_{\text{frames}} = \text{round}(T_{\text{seconds}} \times \text{fps})$). However, discrete frame-rate aliasing exists: 10 fps cannot resolve event durations shorter than 0.10 s, whereas 25 fps resolves down to 0.04 s. Velocity derivatives and rolling filter responses differ under discrete downsampling.
  3. **Measured $G_{\text{INTERP}}$ Invariance on BAR-ATH:** On the challenge match BAR-ATH, changing $G_{\text{INTERP}}$ between 30 frames (1.2 s) and 50 frames (2.0 s) resulted in **exactly zero changes** across all 158,857 graded frames: exactly 73 frames (0.05% of graded match time) are interpolated under both thresholds because every bridged missing-ball gap in the tracking data happens to be $\le 30$ frames (1.2 s). Total known ball frames remain identically 94,287 / 158,857 (59.35%).
  4. **Proxy Objective vs Competition Ground Truth:** The external tuning objective maximized spell-level macro-F1 agreement against SkillCorner's automated tactical phases of play. While this provides a leak-free objective proxy for possession continuity, it does not replicate or expose the challenge organizers' private ground-truth labeling conventions.

