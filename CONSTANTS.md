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
- **Human Decision:** The team has decided that the primary submission for the competition is **`v2.1-external`** (`outputs_v2/submission_v2_1_external.csv`, SHA-256 `5b4c3db7ede46f497822604eda83d837836ea336a4da5dad2312492439c98c5a`), with the frozen a priori configuration **`v2.0.1-apriori`** (`outputs_v2/submission_v2.csv`, SHA-256 `89ff92a62a6e571430c0ff06716f6cf1136c8367cbf3eb1c3be4dfcfb04ef6f2`) preserved as the reproducible baseline fallback (available via `--config apriori`).
- **Pre-Registered Adoption Criterion:** Promotion was executed strictly in accordance with the pre-registered decision rule in [`tools/tuning/PLAN.md:175-178`](tools/tuning/PLAN.md):
  > *"4. **All-or-Nothing Adoption Criterion:**"*
  > *- Candidate $\theta^*$ is adopted **as a whole** into `src/v2/` **IF AND ONLY IF**:*
  > *$$\bar{\Delta} > 0 \quad \text{AND} \quad CI_{\text{low}} > 0$$*
  > *- If $\bar{\Delta} \le 0$ or $CI_{\text{low}} \le 0$, the external evidence does not decisively support a change. The frozen \*a priori\* baseline constants strictly remain, and no changes are made to `src/v2/constants.py`.*
- **Empirical Validation Metrics (`tools/tuning/results/tuning_summary.json`):**
  - **Mean Paired Difference ($\bar{\Delta}$):** `+0.023947` (+2.39 percentage points of macro-averaged spell-level F1 over the 8 hold-out fixtures).
  - **95% Bootstrap Confidence Interval (`seed = 42`, Pre-Registered Decision Seed):** `[+0.007967, +0.039078]` (strictly positive, $CI_{\text{low}} > 0$).
  - **Robustness Sensitivity Checks ($B = 10,000$ draws):**
    - `seed = 0`: `[+0.007877, +0.039022]` ($CI_{\text{low}} > 0$)
    - `seed = 1`: `[+0.008006, +0.038890]` ($CI_{\text{low}} > 0$)
- **Technical Limitations & Ground-Truth Firewall Boundaries:**
  1. **Tuned Constants at Boundary of Pre-Registered Grid:** All four tuned parameters in $\theta^*$ selected by the tuning protocol lie at the extreme edge of their pre-registered discrete search grids:
     - $R_{\text{TOUCH}} = 2.5\text{ m}$ is the grid maximum ($\{1.5, 2.0, 2.5\}\text{ m}$).
     - $\text{MARGIN} = 0.75\text{ m}$ is the grid maximum ($\{0.25, 0.50, 0.75\}\text{ m}$).
     - $T_{\text{CONFIRM}} = 1.2\text{ s}$ ($30\text{ frames}$) is the grid minimum ($\{1.2, 2.0, 2.8\}\text{ s}$).
     - $G_{\text{INTERP}} = 1.2\text{ s}$ ($30\text{ frames}$) is the grid minimum ($\{1.2, 2.0, 2.8\}\text{ s}$).
     The true unconstrained empirical optimum on external data may lie outside the pre-registered bounding box, but no search expansion was permitted under the frozen protocol.
  2. **Discrepancy Between External Touch Definition and Competition Control Definition:** SkillCorner tracking labels possession via a per-frame dictionary (`possession: dict with keys player_id and group which indicates which player/team (home or away team) are in possession`, SkillCorner repository at commit `4340d274572876239c154c90bc507a9b3250a656`, `README.md:40-41`). By contrast, the competition specification requires sustained **controlled possession**, explicitly stating in [`CHALLENGE.md:55`](CHALLENGE.md):
     > *"A change of possession occurs when a player from the opposing team gains \*\*controlled\*\* possession of the ball."*
     and in [`CHALLENGE.md:76-80`](CHALLENGE.md):
     > *"#### What does not count as a change of possession"*
     > *"- a simple touch of the ball without control"*
     > *"- a deflection of a pass that does not intercept it"*
     > *"- an aerial duel won without subsequent control of the ball by that team"*
     Consequently, external tuning may optimize for different transition responsiveness than the competition ground-truth requires.
  3. **Dead-Ball Mask and Ball Detection Disparity:** The external matches evaluate under an oracle dead-ball mask derived from SkillCorner phases of play with 92.94% ball detection (587,649 / 632,293 frames, computed from `tools/tuning/results/diagnostics_amendment_2a.csv`). By contrast, the competition match BAR-ATH evaluates under a heuristic dead-ball mask with only 59.35% ball detection across all graded frames (94,287 / 158,857 frames). Heuristic dead-ball errors and high ball occlusion on BAR-ATH may interact unpredictably with shorter confirmation windows.
  4. **Hold-Out Gain is an External Estimate, Not a BAR-ATH Guarantee:** The hold-out gain (+0.024 mean paired difference across the 8 hold-out fixtures; `tuning_summary.json:mean_paired_difference` = 0.023947) is an empirical measurement on external broadcast tracking data (SkillCorner repository at commit `4340d274572876239c154c90bc507a9b3250a656`, `README.md:11`: *"Broadcast tracking data is tracking data collected through computer vision and machine learning out of the broadcast video"*), not a measured performance guarantee on BAR-ATH optical tracking ([`DATA.md:74`](DATA.md): *"Metrica's FIFA/EPTS export, a metadata XML plus a raw text file"*).
  5. **Temporal Resolution & Frame-Rate Conversion:** Constants were converted between physical durations (seconds) and discrete frames ($T_{\text{frames}} = \text{round}(T_{\text{seconds}} \times \text{fps})$). Discrete frame-rate aliasing differs: 10 fps cannot resolve event durations shorter than 0.10 s, whereas 25 fps resolves down to 0.04 s.
  6. **Measured $G_{\text{INTERP}}$ Invariance on BAR-ATH:** On the challenge match BAR-ATH, changing $G_{\text{INTERP}}$ between 30 frames (1.2 s) and 50 frames (2.0 s) resulted in **exactly zero changes** across all 158,857 graded frames: exactly 73 frames (0.05% of graded match time) are interpolated under both thresholds because every bridged missing-ball gap in the tracking data happens to be $\le 30$ frames (1.2 s). Total known ball frames remain identically 94,287 / 158,857 (59.35%).

