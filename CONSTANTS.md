# Constants Registry & A Priori Rationale (v2 Pipeline)

This registry records all constants utilized in the deterministic, ground-truth-free v2 possession tracking pipeline (`src/v2/`).

In accordance with the organizers' rules clarification, **no constant in this table was trained, fitted, calibrated, or tuned on ground-truth possession data or on the provided match video**. Every value was fixed a priori before running the pipeline on the full match, with written physical or domain rationales.

---

## 1. Frozen Constants Table

| ID | Value | Tier | Origin / Official Citation | Written A Priori Rationale | Introduction Date |
|---|---|:---:|---|---|:---:|
| `FPS` | `25` | **F** | [DATA.md:245](DATA.md) ("`round(seconds * 25)` is a video frame") | Fixed sampling rate of tactical video and optical tracking. | 2026-11-07 |
| `PITCH_LENGTH` | `105.3 m` | **F** | [DATA.md:93](DATA.md), `games.py:GAMES["BAR-ATH"]["pitch"]` | Field length measured from homography-corrected pitch coordinates. | 2026-11-07 |
| `PITCH_WIDTH` | `68.0 m` | **F** | [DATA.md:93](DATA.md), `games.py:GAMES["BAR-ATH"]["pitch"]` | Field width measured from homography-corrected pitch coordinates. | 2026-11-07 |
| `HALF_1_START` | `11,185` | **F** | [DATA.md:256](DATA.md), [FAQ.md:30](FAQ.md) | Official first graded frame of First Half. | 2026-11-07 |
| `HALF_1_END` | `90,651` | **F** | [DATA.md:256](DATA.md), [FAQ.md:30](FAQ.md) | Official last graded frame of First Half. | 2026-11-07 |
| `HALF_1_KICKOFF` | `11,185` | **F** | [FAQ.md:34-35](FAQ.md) ("first half ... graded from the kickoff itself") | Match kickoff instant for First Half. | 2026-11-07 |
| `HALF_2_START` | `110,808` | **F** | [DATA.md:258](DATA.md), [FAQ.md:30](FAQ.md) | Official first graded frame of Second Half. | 2026-11-07 |
| `HALF_2_END` | `190,197` | **F** | [DATA.md:258](DATA.md), [FAQ.md:30](FAQ.md) | Official last graded frame of Second Half. | 2026-11-07 |
| `HALF_2_KICKOFF` | `113,742` | **F** | [FAQ.md:33-34](FAQ.md) ("does not kick off until **113,742**") | Published Second Half kickoff frame. | 2026-11-07 |
| `N_VIDEO_FRAMES` | `199,319` | **F** | [DATA.md:246](DATA.md), `games.py` | Total video frames in `BAR-ATH`. | 2026-11-07 |
| `R_TOUCH` | `2.0 m` | **B** | Stage 2 Spec (approved a priori) | Maximum physical reach of a player plus automatic optical tracking position error ($\sim 0.5\text{--}1.0\text{ m}$). | 2026-11-07 |
| `MARGIN` | `0.5 m` | **B** | Stage 2 Spec (approved a priori) | Minimum separation required between the nearest player and opponent before declaring unambiguous control candidate. | 2026-11-07 |
| `G_INTERP` | `50 frames` (2.0 s) | **B** | Stage 2 Spec (approved a priori) | Typical flight time of a pass; gaps $\le 2\text{ s}$ represent temporary tracking occlusions during active play. | 2026-11-07 |
| `T_CONFIRM` | `50 frames` (2.0 s) | **B** | [CHALLENGE.md:73](CHALLENGE.md) ("a player keeps the ball for a few seconds") | Minimum duration required for opponent to demonstrate controlled possession before confirming turnover. | 2026-11-07 |
| `OOB_MARGIN` | `1.0 m` | **B** | Stage 2 Spec (approved a priori) | Buffer beyond touchlines/by-lines to absorb homography projection inaccuracies and ball radius. | 2026-11-07 |
| `T_OOB` | `12 frames` (0.48 s) | **B** | Stage 2 Spec (approved a priori) | Persistence window to rule out single-frame sensor flutter or noisy boundary spikes. | 2026-11-07 |
| `SPEED_SMOOTH` | `25 frames` (1.0 s) | **B** | Stage 2 Spec (approved a priori) | Centered 1-second rolling average window to filter instantaneous kinematic tracking noise. | 2026-11-07 |
| `V_STOP` | `1.5 m/s` (5.4 km/h) | **B** | Stage 2 Spec (approved a priori) | Standard human walking pace applied to the 75th percentile of player speeds to identify collective match stoppages. | 2026-11-07 |
| `V_BALL_STOP` | `1.0 m/s` (3.6 km/h) | **B** | Stage 2 Spec (approved a priori) | Ball velocity threshold below which the ball is considered stationary / at rest. | 2026-11-07 |
| `T_STOP` | `75 frames` (3.0 s) | **B** | Stage 2 Spec (approved a priori) | Minimum stoppage duration required to declare dead ball from collective velocity drop. | 2026-11-07 |
| `MIN_LIVE_FLICKER` | `25 frames` (1.0 s) | **B** | Stage 2 Spec (approved a priori) | Cleanup threshold: live intervals shorter than 1.0 s bounded by dead balls are deemed spurious touches and converted to `D`. | 2026-11-07 |
| `MIN_DEAD_FLICKER` | `12 frames` (0.48 s) | **B** | Stage 2 Spec (approved a priori) | Cleanup threshold: dead intervals shorter than 0.48 s between identical possessors are deemed sensor artifacts and converted to live ball. | 2026-11-07 |

---

## 2. Classification Tiers

- **Tier F (Fixed):** Non-tunable constants determined strictly by competition definition, data geometry, or official documentation.
- **Tier B (A Priori Engineering Prior):** Values established a priori based on physical and biomechanical principles of football. In a future optional stage, these constants are designated as eligible candidates for hyperparameter tuning strictly on **external** tracking datasets.
