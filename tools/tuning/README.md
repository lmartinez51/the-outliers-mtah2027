# External Tuning Protocol & Results (Stage 3, SkillCorner Open Data)

This directory implements the external parameter tuning protocol for Tier B possession constants using open-source broadcast tracking data from SkillCorner.

---

## 1. External Dataset Source & Retrieval

- **Source Repository:** [SkillCorner Open Data](https://github.com/SkillCorner/opendata.git)
- **License:** MIT License (Copyright (c) 2020 SkillCorner)
- **Commit Hash:** `4340d274572876239c154c90bc507a9b3250a656`
- **Data Scope:** 20 matches from the Australian A-League 2024/2025 (`competition_id: 61`, `season_id: 95`, `competition_edition_id: 870`).
- **Storage Location:** Kept strictly outside the repository at `C:\Users\leolm\ext_data\skillcorner`.
- **Fixture Overlap:** 0% overlap with the challenge fixture (BAR-ATH, LaLiga 2024-25).

To clone and verify the external data:
```bash
git clone https://github.com/SkillCorner/opendata.git C:\Users\leolm\ext_data\skillcorner
cd C:\Users\leolm\ext_data\skillcorner
git checkout 4340d274572876239c154c90bc507a9b3250a656
```

---

## 2. Execution Instructions

Run the tuning harness from the repository root:
```bash
python -B tools/tuning/harness.py
```

The harness executes:
1. Baseline $\theta_0$ evaluation on the 11 tuning matches and 8 hold-out matches.
2. 81-configuration grid search over $(R_{\text{TOUCH}}, \text{MARGIN}, T_{\text{CONFIRM}}, G_{\text{INTERP}})$ on the 11 tuning matches.
3. Selection of top candidate $\theta^*$ by mean `spell_f1_mean` (with normalized distance tie-breaking).
4. Paired hold-out evaluation on the 8 hold-out matches:
   $$\Delta_j = \text{spell\_f1\_mean}(m_j, \theta^*) - \text{spell\_f1\_mean}(m_j, \theta_0)$$
5. Bootstrap confidence intervals ($B = 10,000$ draws) with decision seed `42` and robustness seeds `0` and `1`.
6. Mechanized adoption rule evaluation: adopt if $\bar{\Delta} > 0$ and $CI_{\text{low}} > 0$.
7. Amendment 2(a) and 2(b) diagnostics across all 20 matches.
8. Serialization of all tabular artifacts into `tools/tuning/results/`.

---

## 3. Results Summary

### 3.1 Partition & Baseline Performance

- **Tuning Set (11 matches):** `1874553, 1886347, 1899585, 1925299, 1927964, 1959846, 1986691, 1996435, 1996436, 2006229, 2006363`
  - Baseline $\theta_0$ Mean Spell F1: **0.48954**
- **Hold-Out Set (8 matches):** `2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461`
  - Baseline $\theta_0$ Mean Spell F1: **0.48209**
- **Excluded (1 match):** `1953632` (catalog status `"not_started"`)

### 3.2 81-Grid Search on Tuning Matches

- **Top Candidate Configuration $\theta^*$:**
  - $R_{\text{TOUCH}} = 2.5\text{ m}$ (was $2.0\text{ m}$)
  - $\text{MARGIN} = 0.75\text{ m}$ (was $0.50\text{ m}$)
  - $T_{\text{CONFIRM}} = 1.2\text{ s}$ ($12\text{ frames @ 10 fps}$; $30\text{ frames @ 25 fps}$; was $2.0\text{ s} / 50\text{ frames}$)
  - $G_{\text{INTERP}} = 1.2\text{ s}$ ($12\text{ frames @ 10 fps}$; $30\text{ frames @ 25 fps}$; was $2.0\text{ s} / 50\text{ frames}$)
  - Tuning Set Mean Spell F1: **0.52153** (gain of $+0.03199$ over baseline)

### 3.3 Hold-Out Paired Evaluation & Bootstrap Decision

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
- **Decision Bootstrap 95% CI ($B = 10,000$, Seed 42):** `[+0.007967, +0.039078]`
- **Robustness Check Seed 0:** `[+0.007877, +0.039022]`
- **Robustness Check Seed 1:** `[+0.008006, +0.038890]`
- **Mechanized Decision Rule:**
  - $\bar{\Delta} > 0$ (`True`) AND $CI_{\text{low}} > 0$ (`True`).
  - **Verdict:** **ADOPTED** into pipeline specification as `v2.1-external`.

---

## 4. Amendment 2 Diagnostics (All 20 Matches)

### 4.1 Diagnostic (a): Coverage & Sensor Statistics
- **Total Period Frames:** 1,182,210
- **Total Live Frames (Inside Phases):** 632,266
- **Aggregate Out-of-Play Share:** **46.52%** (549,944 / 1,182,210 frames)
- **Aggregate Ball-Detected Share (in-phase):** **92.94%** (587,605 / 632,266 frames)
- **Aggregate Candidate Share under $\theta_0$ (in-phase):** **52.29%** (330,627 / 632,266 frames)

### 4.2 Diagnostic (b): A Priori Dead-Ball Kinematics Agreement
Agreement between v2's *a priori* physical stoppage mechanisms (out-of-bounds with 1.0 m margin; collective stop at $V_{\text{STOP}} = 1.35\text{ m/s}$ and $T_{\text{STOP}} = 1.5\text{ s}$ using 1 s centered displacement) and the reference phase-complement dead-ball mask across all 20 matches:
- **Reference Dead Share:** 46.52%
- **Predicted Dead Share:** 41.98%
- **Frame-Level Precision (`D`):** **92.37%**
- **Frame-Level Recall (`D`):** **83.34%**
- **Frame-Level F1 (`D`):** **87.62%**
*(Note: Nothing in this stoppage detector was tuned. This evaluation serves strictly as an empirical validation of the physical stoppage heuristics.)*
