# Legacy Version 1 Archive (Superseded)

> **Status notice (Nov 2026).** Earlier versions of this repository fitted models and parameters using the public 5-minute ground-truth window. The organizers' clarification of the rules (received in November) does not allow this. We are reworking the submission. Metrics shown in earlier documents are in-sample and must not be read as valid results, and statements of compliance with RULES.md in those documents are withdrawn until the rework is complete.

---

## 1. Archive Purpose & Status

This directory contains the legacy Version 1 codebase and documentation developed by Team The Outliers during the initial phase of the competition.

- **Fitted on Ground Truth:** The v1 pipeline utilized machine learning models (LightGBM gradient-boosted decision trees, heuristic feature extractors, and calibrated post-processing decoders) whose hyperparameters and training weights were selected using cross-validation over the public 5-minute ground-truth window (`evaluation/gt/01_public.csv`).
- **Rules Clarification & Supersedence:** Following the organizers' clarification that ground-truth data must not be used to select or fit submission parameters for the match, **Version 1 has been completely retired and superseded**.
- **Not Part of Submission:** None of the files in `legacy_v1/` are executed, imported, or utilized by the final submission pipeline (`src/v2/`). They are retained in the repository solely for auditability, scientific transparency, and git history continuity.

---

## 2. Inventory of Archived v1 Assets

### Core Pipeline Modules (`legacy_v1/src/`)
- `legacy_v1/src/mtah_features.py`: Feature engineering pipeline computing spatial, kinematic, and contextual tracking features.
- `legacy_v1/src/mtah_model.py`: LightGBM training harness, 5-fold temporal block cross-validation, and motor-base probability predictor.
- `legacy_v1/src/mtah_decoder.py`: Post-processing decoder applying dead-ball masking, deflection absorption, and label smoothing.
- `legacy_v1/src/jev_system_one.py`: LLM-assisted tactical arbitration module.
- `legacy_v1/src/generate_3d_replay_html.py`, `legacy_v1/src/viewer.html`, `legacy_v1/src/three.min.js`: 3D match visualizer and WebGL assets.

### Archived Scripts & Guides
- `legacy_v1/auto_process_competition.py`: Legacy end-to-end execution script.
- `legacy_v1/TECHNICAL_DOCUMENT.md`: Original v1 technical document (superseded by `TECHNICAL_DOCUMENT.md` at root).
- `legacy_v1/The_Outliers.md`, `legacy_v1/deliverables/`: Historical project summaries and guide copies.
- `legacy_v1/TEAM_GUIDE_EN.md`, `legacy_v1/DOCUMENTO_TECNICO_ES.md`, `legacy_v1/GUIA_EQUIPO_ES.md`: Legacy bilingual guides and technical references.

For the active, compliant, deterministic competition pipeline, refer to [`README.md`](../README.md) and [`README_v2.md`](../README_v2.md).
