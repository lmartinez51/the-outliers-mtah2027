# External Tuning Protocol & Results (Stage 3, SkillCorner Open Data)

This directory implements the external parameter tuning protocol for Tier B possession constants using open-source broadcast tracking data from SkillCorner.

---

## 1. External Dataset Source & Retrieval

- **Source Repository:** [SkillCorner Open Data](https://github.com/SkillCorner/opendata.git)
- **License:** MIT License (Copyright (c) 2020 SkillCorner)
- **Commit Hash:** `4340d274572876239c154c90bc507a9b3250a656`
- **Data Scope:** 20 matches from the Australian A-League 2024/2025 (`competition_id: 61`, `season_id: 95`, `competition_edition_id: 870`).
- **Storage Location:** Kept strictly outside the repository (e.g. `../ext_data/skillcorner` adjacent to repo, or path specified via environment variable `SKILLCORNER_DATA_DIR`).
- **Fixture Overlap:** 0% overlap with the challenge fixture (BAR-ATH, LaLiga 2024-25).

To clone and verify the external data:
```bash
git clone https://github.com/SkillCorner/opendata.git ../ext_data/skillcorner
cd ../ext_data/skillcorner
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
- **Total Period Frames:** 1,182,244 discrete frames (1,123,013 across 19 included matches)
- **Total Live Frames (Inside Phases):** 632,293 discrete frames (599,455 across 19 included matches)
- **Pooled Out-of-Play Share:** **46.52%** (549,951 / 1,182,244 frames across all 20 matches; 46.62%, 523,558 / 1,123,013 frames across 19 included matches)
- **Aggregate Ball-Detected Share (in-phase):** **92.94%** (587,649 / 632,293 frames)
- **Aggregate Candidate Share under $\theta_0$ (in-phase):** **52.29%** (330,638 / 632,293 frames)

### 4.2 Diagnostic (b): A Priori Dead-Ball Kinematics Comparison
Evaluates agreement between physical stoppage mechanisms (out-of-bounds with 1.0 m margin; collective stop using 1 s centered displacement) and the reference phase-complement dead-ball mask across all 20 matches.

**Run with Frozen v2 Constants (`tools/tuning/results/diagnostics_amendment_2b_frozen_constants.csv`):**
Using exact v2 frozen constants converted to 10 fps ($V_{\text{STOP}} = 1.5\text{ m/s}$, $V_{\text{BALL\_STOP}} = 1.0\text{ m/s}$, $T_{\text{STOP}} = 30\text{ frames / } 3.0\text{ s}$, $T_{\text{OOB}} = 5\text{ frames / } 0.5\text{ s}$):
- **Reference Dead Share:** 46.52% (549,951 / 1,182,244 frames)
- **Predicted Dead Share:** 41.39% (489,298 / 1,182,244 frames)
- **Frame-Level Precision (`D`):** **93.21%** (456,070 / 489,298 frames)
- **Frame-Level Recall (`D`):** **82.93%** (456,070 / 549,951 frames)
- **Frame-Level F1 (`D`):** **87.77%**

**Run with Non-v2 Constants (`tools/tuning/results/diagnostics_amendment_2b.csv`):**
Using originally drafted parameters ($V_{\text{STOP}} = 1.35\text{ m/s}$, $T_{\text{STOP}} = 15\text{ frames / } 1.5\text{ s}$, $V_{\text{BALL\_STOP}} = 1.5\text{ m/s}$, $T_{\text{OOB}} = 5\text{ frames / } 0.5\text{ s}$):
- **Reference Dead Share:** 46.52% (549,951 / 1,182,244 frames)
- **Predicted Dead Share:** 41.97% (496,181 / 1,182,244 frames)
- **Frame-Level Precision (`D`):** **92.37%** (458,319 / 496,181 frames)
- **Frame-Level Recall (`D`):** **83.34%** (458,319 / 549,951 frames)
- **Frame-Level F1 (`D`):** **87.62%**

*(Provenance & Status: These non-v2 parameters were introduced in commit `65b5830` ([`tools/tuning/PLAN.md:197`](tools/tuning/PLAN.md)). The threshold $V_{\text{STOP}} = 1.35\text{ m/s}$ matches `stoppage_speed_w75` in the legacy v1 pipeline ([`src/mtah_model.py:355`](../../src/mtah_model.py#L355)) [VERIFIED]; the duration $T_{\text{STOP}} = 1.5\text{ s}$ has no recorded source in the codebase or commit history [UNKNOWN]. This run does NOT represent or provide evidence about the v2 pipeline, which uses $V_{\text{STOP}} = 1.5\text{ m/s}$ and $T_{\text{STOP}} = 3.0\text{ s}$ (75 frames at 25 fps). Nothing in either stoppage detector was tuned. Furthermore, speed inputs differ fundamentally between datasets: 1 s centered displacement of detected players' positions at 10 fps versus the calibrated BAR-ATH optical tracker speed field at 25 fps.)*
