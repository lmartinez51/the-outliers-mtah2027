# Pre-Registration Plan: External Tuning of Possession Constants (Stage 3, Revision A2)

## 1. Overview & Governing Rules

This document pre-registers the tuning protocol for the Tier B possession constants using external tracking data from the open-source SkillCorner repository.

### Governing Constraints (STAGE 3 Rules)
1. **GT and BAR-ATH Firewall (S1):** All tuning code lives strictly in `tools/tuning/`. It does not read, import, or reference BAR-ATH match assets, `data/tactical/`, `data/cache_*`, `outputs_v2/`, `evaluation/gt/`, or any v1 code. `evaluation/possession_metrics.py` is imported solely by `tools/tuning/` and invoked solely on external arrays.
2. **External Storage (S2):** External tracking and metadata reside strictly outside the git repository at `C:\Users\leolm\ext_data\skillcorner`. No frame-level external tracking data is committed to the repository.
3. **Pre-Registration Lock (S3):** All decisions documented herein are locked prior to running any grid evaluation. Execution proceeds only after explicit human approval via the passphrase: `APPROVED PLAN`.
4. **Parametrization Invariance (S4):** In Phase B, `src/v2/` is parametrized to accept constants as function arguments with current *a priori* values as defaults. Default execution on BAR-ATH must reproduce the baseline SHA-256 hash `89ff92a62a6e571430c0ff06716f6cf1136c8367cbf3eb1c3be4dfcfb04ef6f2`.

---

## 2. External Dataset, License & Fixture Overlap Check

### 2.1 External Dataset Source & License
- **Repository:** `https://github.com/SkillCorner/opendata.git`
- **Location:** `C:\Users\leolm\ext_data\skillcorner`
- **License Terms:** The repository root carries an MIT License (`LICENSE`: *"The MIT License / Copyright (c) 2020 SkillCorner"* covering *"this software and associated documentation files"*). The repository `README.md` states: *"This data has been open sourced in a joint initiative between SkillCorner and PySport... Thus, if you use the data, we kindly ask that you credit SkillCorner and hope you'll notify us on Twitter"*.
- **Dataset Contents:** 20 matches from the Australian A-League 2024/2025 (`competition_id: 61`, `season_id: 95`, `competition_edition_id: 870`).

### 2.2 Fixture Disjointness Verification
- **Challenge Reference Fixture:** FC Barcelona vs. Athletic Club (`BAR-ATH`), played 2024-08-24 in LaLiga EA Sports 2024-25, Matchday 2 (`DATA.md:1-4`, `games.py:1-40`).
- **SkillCorner Fixtures:** All 20 matches are Australian domestic A-League fixtures played between 2024-11-23 and 2025-05-17. Zero matches involve FC Barcelona, Athletic Club, or Spanish competitions. Overlap is strictly zero.

---

## 3. Match Partition (11 Tuning / 8 Hold-Out / 1 Excluded)

The matches are sorted in ascending order by integer Match ID. Match `1953632` is excluded because its top-level catalog status in `data/matches.json` is `"not_started"`.

- **Tuning Set (11 matches):** Used for grid evaluation and selection of candidate configuration $\theta^*$.
- **Hold-Out Set (8 matches):** Strictly reserved for the final paired adoption test.
- **Excluded (1 match):** `1953632` (catalog status `"not_started"`).

### Final Match Partition Table
| Index | Match ID | Date (UTC) | Competition | Home Team | Away Team | Partition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 1874553 | 2024-11-23T06:00:00Z | AUS A-League 2024/25 | Brisbane FC | Adelaide United | **Tuning** |
| 02 | 1886347 | 2024-11-30T04:00:00Z | AUS A-League 2024/25 | Auckland FC | Newcastle | **Tuning** |
| 03 | 1899585 | 2024-12-07T04:00:00Z | AUS A-League 2024/25 | Auckland FC | Wellington P FC | **Tuning** |
| 04 | 1925299 | 2024-12-21T06:00:00Z | AUS A-League 2024/25 | Brisbane FC | Perth Glory | **Tuning** |
| 05 | 1927964 | 2024-12-22T06:00:00Z | AUS A-League 2024/25 | Western Sydney | Wellington P FC | **Tuning** |
| -- | 1953632 | 2024-12-31T08:00:00Z | AUS A-League 2024/25 | CC Mariners | Melbourne City | **EXCLUDED** (catalog `"not_started"`) |
| 06 | 1959846 | 2025-01-10T08:35:00Z | AUS A-League 2024/25 | Melbourne V FC | Western United | **Tuning** |
| 07 | 1986691 | 2025-01-26T06:00:00Z | AUS A-League 2024/25 | Western Sydney | Auckland FC | **Tuning** |
| 08 | 1996435 | 2025-02-01T06:00:00Z | AUS A-League 2024/25 | Sydney FC | Adelaide United | **Tuning** |
| 09 | 1996436 | 2025-02-01T04:00:00Z | AUS A-League 2024/25 | Auckland FC | Macarthur FC | **Tuning** |
| 10 | 2006229 | 2025-03-07T08:35:00Z | AUS A-League 2024/25 | Melbourne City | Macarthur FC | **Tuning** |
| 11 | 2006363 | 2025-03-08T08:35:00Z | AUS A-League 2024/25 | Melbourne V FC | CC Mariners | **Tuning** |
| 12 | 2007448 | 2025-03-15T10:45:00Z | AUS A-League 2024/25 | Perth Glory | Brisbane FC | **Hold-Out** |
| 13 | 2007721 | 2025-03-16T04:00:00Z | AUS A-League 2024/25 | Auckland FC | CC Mariners | **Hold-Out** |
| 14 | 2010085 | 2025-04-05T06:00:00Z | AUS A-League 2024/25 | Melbourne City | CC Mariners | **Hold-Out** |
| 15 | 2011166 | 2025-04-12T05:00:00Z | AUS A-League 2024/25 | Wellington P FC | Melbourne V FC | **Hold-Out** |
| 16 | 2013725 | 2025-04-27T07:00:00Z | AUS A-League 2024/25 | Western United | Sydney FC | **Hold-Out** |
| 17 | 2015213 | 2025-05-03T08:00:00Z | AUS A-League 2024/25 | Western United | Auckland FC | **Hold-Out** |
| 18 | 2016236 | 2025-05-09T09:35:00Z | AUS A-League 2024/25 | Western United | Adelaide United | **Hold-Out** |
| 19 | 2017461 | 2025-05-17T09:35:00Z | AUS A-League 2024/25 | Melbourne V FC | Auckland FC | **Hold-Out** |

---

## 4. Frozen Baseline Constants $\theta_0$ & Reconciliation

### 4.1 Frozen Values from `src/v2/constants.py` and `CONSTANTS.md`
The baseline constants $\theta_0$ are defined in `src/v2/constants.py` and documented in `CONSTANTS.md`:
- `R_TOUCH = 2.0 m` (`src/v2/constants.py:26`, `CONSTANTS.md:23`)
- `MARGIN = 0.5 m` (`src/v2/constants.py:27`, `CONSTANTS.md:24`)
- `T_CONFIRM = 50 frames` (2.0 s at 25 fps) (`src/v2/constants.py:29`, `CONSTANTS.md:26`)
- `G_INTERP = 50 frames` (2.0 s at 25 fps) (`src/v2/constants.py:28`, `CONSTANTS.md:25`)
- `MIN_LIVE_FLICKER = 25 frames` (1.0 s at 25 fps) (`src/v2/constants.py:36`, `CONSTANTS.md:33`)

*(Note on Previous Plan: The values in Revision A of PLAN.md [1.50m, 0.75m, 0.28s, 0.60s, 0.40s] had no source in `src/v2/constants.py` or `CONSTANTS.md`. They were an unattributed assistant fabrication during Phase A drafting and have been completely replaced by the frozen baseline constants above.)*

### 4.2 Baseline Configuration $\theta_0$
$$\theta_0 = (R_{\text{TOUCH}}=2.0\text{ m}, \, \text{MARGIN}=0.5\text{ m}, \, T_{\text{CONFIRM}}=2.0\text{ s}, \, G_{\text{INTERP}}=2.0\text{ s}, \, \text{MIN\_LIVE\_FLICKER}=1.0\text{ s})$$

### 4.3 Mandatory Harness Assertion
The tuning harness asserts that converting $\theta_0$ to frames at 25 fps reproduces `src/v2/constants.py` exactly:
```python
assert theta0['R_TOUCH'] == constants.R_TOUCH
assert theta0['MARGIN'] == constants.MARGIN
assert round(theta0['T_CONFIRM'] * 25) == constants.T_CONFIRM
assert round(theta0['G_INTERP'] * 25) == constants.G_INTERP
assert round(theta0['MIN_LIVE_FLICKER'] * 25) == constants.MIN_LIVE_FLICKER
```

---

## 5. Pre-Registered Grid Specification

All time parameters are strictly multiples of $0.2\text{ s}$, ensuring that frame counts are exact integers at both 10 fps (SkillCorner) and 25 fps (BAR-ATH). The grid comprises $3^5 = 243$ configurations centered on $\theta_0$.

| Constant | Tested Grid (Physical Units) | Frames at 10 fps (SkillCorner) | Frames at 25 fps (BAR-ATH) | Frame Count Integrity |
| :--- | :---: | :---: | :---: | :---: |
| `R_TOUCH` | `[1.5, 2.0, 2.5]` m | N/A (meters) | N/A (meters) | Exact float |
| `MARGIN` | `[0.25, 0.5, 0.75]` m | N/A (meters) | N/A (meters) | Exact float |
| `T_CONFIRM` | `[1.2, 2.0, 2.8]` s | `[12, 20, 28]` frames | `[30, 50, 70]` frames | Exact integers ($v \times \text{fps} \in \mathbb{Z}$) |
| `G_INTERP` | `[1.2, 2.0, 2.8]` s | `[12, 20, 28]` frames | `[30, 50, 70]` frames | Exact integers ($v \times \text{fps} \in \mathbb{Z}$) |
| `MIN_LIVE_FLICKER` | `[0.6, 1.0, 1.4]` s | `[6, 10, 14]` frames | `[15, 25, 35]` frames | Exact integers ($v \times \text{fps} \in \mathbb{Z}$) |

`T_CONFIRM` counts consecutive candidate frames at both frame rates. The harness asserts that for all grid values $v$, `abs(v * fps - round(v * fps)) < 1e-9`.

---

## 6. Dead-Ball Mask & Stoppage Exclusion Rationale

### 6.1 Documentation Definition of Dead Ball
SkillCorner README explicitly defines phases of play:
> `* Phases of play are only defined when the ball is in play. When the ball is out of play there is no phase of play`
> `* Each in-possession phase directly corresponds to an out-of-possession phase.`

The harness defines the dead-ball mask as:
$$\text{Dead-Ball Mask} = \{f \in \text{Period} \mid f \notin [P_{\text{start}}, P_{\text{end}}] \text{ for all phases } P\}$$

### 6.2 Aggregate Gap Statistics across All 20 Matches
Over all 20 matches in `data/matches/`:
- **Total phase gaps inside periods:** 2,040 gaps
- **Duration Min:** 0.10 s (1 frame at 10 fps)
- **Duration Median:** 20.40 s (204 frames at 10 fps)
- **Duration p90:** 53.52 s (535 frames at 10 fps)
- **Duration Max:** 170.50 s (1,705 frames at 10 fps)
- **Share of Period Frames inside Phases:** $631,111 / 1,182,210 = 53.38\%$
- **Short gaps $\le 1.0\text{ s}$:** 37 instances out of 2,040 ($1.81\%$).
- **Rule for short gaps:** Because gaps $\le 1.0\text{ s}$ represent only 1.81% of transitions (minor annotation boundaries between contiguous tactical phases), the gap statistics confirm that 98.19% of gaps are substantial game stoppages. Therefore, no ad-hoc gap-bridging rule is justified, and the dead-ball gap threshold is locked *a priori* at $0.0\text{ s}$ (strict complement of phase intervals).

### 6.3 Reason Why Dead-Ball Constants Stay Excluded (Tier A)
The dead-ball constants (`V_STOP`, `V_BALL_STOP`, `T_STOP`, `T_OOB`, `OOB_MARGIN`) stay *a priori* and are excluded from tuning because:
- Speed-based physical stoppage detection cannot be transferred reliably between datasets: SkillCorner player speeds would need to be numerically differentiated from 10 fps positions that are partly extrapolated off-screen, whereas BAR-ATH tracking provides a native optical tracker speed field calibrated at 25 fps.
- In the evaluation harness, this documented phase-complement mask serves as an oracle dead-ball input so that tuning isolates and evaluates only the Tier B possession constants. This oracle dead-ball conditioning is recorded as a domain limitation.

---

## 7. Label Derivation & Initialization Rules

Ground-truth possession labels for SkillCorner frames are derived as follows:
1. **Live Segmentation:** Only frames within the documented live-ball intervals (inside phases of play) are evaluated.
2. **Team Possession Assignment:**
   - For frames with non-null `possession.group` in `tracking_extrapolated.jsonl`: `'home team'` $\to$ `'H'`, `'away team'` $\to$ `'A'`.
   - For frames where `possession.group` is `null` (passes in flight, loose balls during live play): forward-fill from the last possessing team within that live stretch, following CHALLENGE.md: *"possession stays with the last team that had it while the ball is live."*
3. **Stretch Initialization (Back-Fill):**
   - At the beginning of each contiguous live stretch prior to the first touch, the label is back-filled from the first non-null `possession.group` within that stretch.
   - Fallback: The phase's `team_in_possession_id` is used **only** if the entire contiguous live stretch contains no non-null `possession.group`.
   - **Empirical Verification across All 20 Matches:** Out of 2,080 contiguous live stretches, 2,048 ($98.46\%$) contain a non-null possession touch and are initialized via back-fill. Only 32 stretches ($1.54\%$) contain no recorded touch and require fallback to `team_in_possession_id`.

---

## 8. Pipeline Transfer, Coordinate System & Player Handling

### 8.1 Coordinate System Alignment
- **BAR-ATH:** Raw tracking coordinates are normalised to $[0, 1]$ (`DATA.md:87`: *"Coordinates are normalised to 0–1, not metres. Multiply by the pitch to get metres"*). `src/v2/extractor.py:97-98, 112-113` scales coordinates by pitch length ($105.3\text{ m}$) and width ($68.0\text{ m}$), placing the origin $(0, 0)$ at the corner flag ($x \in [0, 105.3]$, $y \in [0, 68.0]$).
- **SkillCorner:** Pitch origin $(0, 0)$ is at the center of the pitch (`README.md: "the center of the coordinates is at the center of the pitch"`), with $x \in [-52.5, +52.5\text{ m}]$ and $y \in [-34.0, +34.0\text{ m}]$.
- **Equivalence:** Euclidean distance between player and ball $d = \sqrt{(x_p - x_b)^2 + (y_p - y_b)^2}$ is translation-invariant. Both coordinate systems yield identical distance metrics in meters.

### 8.2 Off-Screen (Extrapolated) Players
In candidate possession detection, the harness evaluates all players with coordinates present in `player_data` (both detected and extrapolated), because BAR-ATH optical tracking provides full-pitch player tracks. Extrapolated players situated off-screen naturally have large distances to the ball and do not trigger false candidate touches. This is recorded as a domain limitation.

---

## 9. Search Protocol & Mechanized Adoption Rule

### 9.1 Search Protocol
1. Evaluate baseline configuration $\theta_0 = (2.0, 0.5, 2.0, 2.0, 1.0)$ on all 11 tuning matches.
2. Evaluate the 243-point grid on the 11 tuning matches in a single grid pass (no restarts, no adaptive searching).
3. Compute the mean ranking metric $\bar{S}_{\text{tune}}(\theta) = \frac{1}{11} \sum_{i=1}^{11} \text{spell\_f1\_mean}(m_i, \theta)$.
4. Select candidate configuration $\theta^*$ that maximizes $\bar{S}_{\text{tune}}$. If multiple configurations tie, choose the configuration closest to $\theta_0$ in normalized Euclidean distance.

### 9.2 Mechanized Adoption Rule on Hold-Out Matches (All-or-Nothing)
Evaluate candidate $\theta^*$ and baseline $\theta_0$ on the 8 hold-out matches:
1. For each hold-out match $j \in \{1, \dots, 8\}$, compute the paired difference:
   $$\Delta_j = \text{spell\_f1\_mean}(m_j, \theta^*) - \text{spell\_f1\_mean}(m_j, \theta_0)$$
2. Compute the mean paired difference:
   $$\bar{\Delta} = \frac{1}{8} \sum_{j=1}^8 \Delta_j$$
3. Perform an empirical bootstrap over the 8 paired differences $\Delta_j$ with $B = 10,000$ draws and fixed seed `seed = 42`. Compute the 95% bootstrap confidence interval $[CI_{\text{low}}, CI_{\text{high}}]$ using the 2.5th and 97.5th percentiles.
4. **All-or-Nothing Adoption Criterion:**
   - Candidate $\theta^*$ is adopted **as a whole** into `src/v2/` **IF AND ONLY IF**:
     $$\bar{\Delta} > 0 \quad \text{AND} \quad CI_{\text{low}} > 0$$
   - If $\bar{\Delta} \le 0$ or $CI_{\text{low}} \le 0$, the external evidence does not decisively support a change. The frozen *a priori* baseline constants strictly remain, and no changes are made to `src/v2/constants.py`.
