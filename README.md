# More than a Hack 2027 — Team The Outliers

Official competition repository for **More than a Hack 2027 (Phase 1: Automated Ball Possession Tracking)**, organized by **[FC Barcelona](https://www.fcbarcelona.com)**, **[Metrica Sports](https://metrica-sports.com)**, and **[Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**.

- **Team Name:** `The Outliers`
- **Competition Track:** Phase 1 (Automated Ball Possession Tracking)
- **Target Match:** `BAR-ATH` (FC Barcelona vs. Athletic Club, 199,319 video frames @ 25 fps)
- **Primary Submission Deliverable:** `outputs_v2/submission_v2_1_external.csv` (SHA-256 `5b4c3db7ede46f497822604eda83d837836ea336a4da5dad2312492439c98c5a`)
- **Baseline Fallback:** `outputs_v2/submission_v2.csv` (SHA-256 `89ff92a62a6e571430c0ff06716f6cf1136c8367cbf3eb1c3be4dfcfb04ef6f2`, accessible via `--config apriori`)

---

## 1. Documentation Index

- [`TECHNICAL_DOCUMENT.md`](TECHNICAL_DOCUMENT.md): Official technical document for the v2.1 deterministic possession pipeline.
- [`README_v2.md`](README_v2.md): Detailed algorithmic specification, stage-by-stage architecture, and structural diagnostics for v2 / v2.1.
- [`CONSTANTS.md`](CONSTANTS.md): Parameter tables, classification tiers (Tier F fixed vs. Tier B engineering), physical rationales, and the formal Decision Record.
- [`legacy_v1/README.md`](legacy_v1/README.md): Archive documentation for the retired Version 1 codebase.

---

## 2. Approach & Architecture Summary

The repository provides a deterministic offline ball possession tracking pipeline (`src/v2/`):
1. **Kinematic Ball Gap Interpolation:** Bridges ball tracking dropouts $\le 30$ frames (1.2 s) via linear interpolation.
2. **Physical Control Candidates:** Identifies player proximity ($R_{\text{TOUCH}} \le 2.5\text{ m}$) with relative separation margin ($\text{MARGIN} \ge 0.75\text{ m}$).
3. **Turnover Confirmation State Machine:** Requires continuous opponent candidate persistence over $T_{\text{CONFIRM}} = 30$ frames (1.2 s) before committing a turnover, back-dating the handover to the initial touch. Loose balls, aerial trajectories, and occlusion intervals hold the active possessor.
4. **Physical Dead-Ball Masking:** Identifies non-live play using pitch boundary margins (out-of-bounds persistence $\ge 12$ frames beyond 1.0 m) and collective player speed thresholds (75th percentile player speed $< 1.5\text{ m/s}$ with ball speed $< 1.0\text{ m/s}$ over $\ge 75$ frames).
5. **Fixed-Point Morphological Cleanup:** Eliminates high-frequency transient flickers between dead-ball intervals and team possessions.

---

## 3. Quickstart & Execution

### Prerequisites
- Python 3.12 (`.venv/Scripts/python.exe`)
- NumPy 2.2.6

### Running the Pipeline
The runner refuses to overwrite existing destination files by default (`FileExistsError`).

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

### Validate Output Format
```bash
python -B evaluation/validate_submission.py outputs_v2/submission_v2_1_external.csv
```

### Run Test Suite
```bash
python -B -m unittest discover -s tests -p "test_*.py" -v
```

---

## 4. Repository Structure

```
the-outliers-mtah2027/
├── src/
│   └── v2/                        # Active deterministic pipeline
│       ├── constants.py           # Numerical thresholds and configurations
│       ├── extraction.py          # ATD raw text parser and primitive extraction
│       ├── interpolation.py       # Missing ball gap interpolation
│       ├── candidates.py          # Proximity and margin evaluation
│       ├── dead_ball.py           # Boundary and collective stop masking
│       ├── state_machine.py       # Possession state machine
│       ├── cleanup.py             # Fixed-point cleanup
│       └── run_v2.py              # CLI entry point
├── tools/
│   ├── tuning/                    # External tuning on SkillCorner open data
│   │   ├── PLAN.md                # Pre-registered tuning protocol
│   │   ├── harness.py             # Tuning grid search and bootstrap evaluation
│   │   ├── README.md              # Protocol documentation and results summary
│   │   └── results/               # Serialized tuning tables and summaries
│   └── docs/                      # Documentation validation utilities
├── tests/                         # Automated test suite (compliance and spec)
├── outputs_v2/                    # Final pipeline outputs
│   ├── submission_v2_1_external.csv  # Primary submission (v2.1)
│   └── submission_v2.csv          # Baseline fallback (v2.0.1)
├── legacy_v1/                     # Retired v1 codebase (superseded, archived)
├── CONSTANTS.md                   # Constants specification and decision record
├── README_v2.md                   # Technical documentation for v2
├── TECHNICAL_DOCUMENT.md          # Official submission technical document
└── README.md                      # Repository overview
```

---

## 5. Validation Status & Technical Limitations

1. **Validation Status:** The v2/v2.1 pipeline has not been scored against private ground truth for BAR-ATH; no accuracy, macro-F1, or spell-F1 metric is reported for the full match.
2. **External Tuning Estimate:** The hold-out improvement observed during external parameter tuning (+0.024 mean paired difference across 8 hold-out fixtures) is an empirical measurement on external broadcast tracking data, not a performance guarantee on BAR-ATH optical tracking.
3. **Dead-Ball Disparity:** External tuning evaluated under an oracle dead-ball mask derived from SkillCorner phases of play (pooled out-of-play share of 46.52% across all 20 matches, 549,951 / 1,182,244 frames), whereas BAR-ATH evaluated under a physical heuristic dead-ball mask (31.20% dead-ball share across graded frames, 49,565 / 158,857 frames).
4. **Missing Ball Occlusion:** Ball coordinates are unavailable on 40.65% of graded frames in BAR-ATH (94,287 / 158,857 known; 64,570 unknown). During missing-ball intervals, the stoppage detector relies on player speed distribution.
5. **Collective Stoppage Contribution:** The collective stop rule accounts for 44,637 of the 49,291 union dead-ball frames (~90.6% of dead-ball frames; 28.1% of graded match time).

---

## 6. External Data Attribution & License Notice

Tier B constants were tuned on open-source broadcast tracking data provided by **SkillCorner** under the MIT License ([SkillCorner Open Data](https://github.com/SkillCorner/opendata.git), commit `4340d274572876239c154c90bc507a9b3250a656`, covering 20 Australian A-League 2024/25 matches).

In accordance with SkillCorner's credit request (ext:README.md:20-21):
> *"Thus, if you use the data, we kindly ask that you credit SkillCorner and hope you'll notify us on [Twitter](https://twitter.com/skillcorner) so we can follow the great work being done with this data."*

We gratefully acknowledge SkillCorner and PySport for open-sourcing these tracking datasets for sports analytics research.
