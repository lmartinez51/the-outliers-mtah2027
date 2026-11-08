# Constants Registry & A Priori Rationale (v2 Pipeline)

This registry records all constants utilized in the deterministic, ground-truth-free v2 possession tracking pipeline (`src/v2/`).

In accordance with the organizers' rules clarification, **no constant in this table was trained, fitted, calibrated, or tuned on ground-truth possession data or on the provided match video**. Every value was fixed before v2 was first run on the match and was not selected by optimizing any metric. The values were reasoned from CHALLENGE.md and general physical considerations; the team had seen public-window statistics during the earlier v1 work.

---

## 1. Frozen Constants Table

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

## 2. Additional Algorithmic Conventions & Formulas

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

## 3. Classification Tiers

- **Tier F (Fixed):** Non-tunable constants determined strictly by competition definition, data geometry, or official documentation.
- **Tier B (A Priori Engineering Prior):** Values established a priori based on physical and biomechanical principles of football. In a future optional stage, these constants are designated as eligible candidates for hyperparameter tuning strictly on **external** tracking datasets.
