> **Status notice (Nov 2026).** Earlier versions of this repository fitted models and parameters using the public 5-minute ground-truth window. The organizers' clarification of the rules (received in November) does not allow this. We are reworking the submission. Metrics shown in earlier documents are in-sample and must not be read as valid results, and statements of compliance with RULES.md in those documents are withdrawn until the rework is complete.

# Team Onboarding & Quickstart Guide (English)

**Project:** More than a Hack 2027 — FC Barcelona & Metrica Sports  
**Team:** `The Outliers`  
**Authors:** Team The Outliers, Ana Karen Arellano, and Steven Ibarra  
**Repository:** [`the-outliers-mtah2027`](https://github.com/lmartinez51/the-outliers-mtah2027)  
**Objective:** Enable any teammate or competition evaluator to reproduce the entire pipeline, validate the official submission, and explore the interactive 3D tactical viewer in **under 3 minutes**.

---

## 1. System Requirements

- **Operating System:** Windows, macOS, or Linux.
- **Python Version:** `3.9`, `3.10`, `3.11`, or `3.12`.
- **Hardware:** Standard consumer CPU (Intel Core i5 / AMD Ryzen 5 or Apple Silicon M1/M2/M3).  
  *Crucial note:* **NO GPU OR CUDA REQUIRED.** The entire pipeline operates with vectorized NumPy matrices and lightweight CPU Gradient Boosted Decision Trees.
- **Disk Space:** ~250 MB free disk space (compressed tactical zip is ~35 MB).

---

## 2. Data Preparation (Tactical Files Only)

Unpack only the tactical files into the `data/tactical/` folder:

```
the-outliers-mtah2027/
└── data/
    └── tactical/
        ├── BAR-ATH_tactical_atd.txt          <-- 25 fps kinematic player/ball tracking (63 MB)
        ├── BAR-ATH_tactical_atd.xml          <-- Pitch & player metadata (4 KB)
        └── BAR-ATH_tactical_smart_tagging.xml <-- Match events and tactical phases (2.8 MB)
```

> [!IMPORTANT]
> **THE 1.8 GB RAW `.mp4` VIDEOS ARE NOT REQUIRED.**  
> Our optical kinematic extractor derives velocities, accelerations, ball distances, and directional momentum vectors directly from optical tracking coordinates in ~17 seconds, eliminating the need to download or process gigabytes of video footage.

---

## 3. Quickstart Virtual Environment Setup

Open your terminal in the repository root:

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\activate
# Linux or macOS:
source .venv/bin/activate

# 3. Install required libraries
pip install -r requirements.txt
```

---

## 4. Run Everything in One Single Command

To execute the entire end-to-end pipeline (feature engineering, probabilistic model training, Viterbi sequence decoding, official validation, scoreboard evaluation, and 3D replay compilation):

```bash
python auto_process_competition.py
```

### What does this command execute automatically?
1. **Step 1 (Features):** Instant cache load (`<0.3s`) or streaming 199,319 frames (`~17s`) from raw tactical files.
2. **Step 2 (Probabilistic Models + Jev AI):** Fits the Occlusion-Robust Dead-Ball Detector (Head 1) and Mathematically Team-Symmetric Live Possession Model (Head 2) with deterministic Jev AI (`System One`) epistemic arbitration primitives.
3. **Step 3 (3-State Soft Viterbi Decoding):** Eliminates transient flickers and generates `submission.csv` (`199,319` frames).
4. **Step 4 (Official Validation & Leaderboard):** Executes `evaluation/validate_submission.py` and `evaluation/run_evaluation.py`, confirming **Rank #1** on the public scoreboard (`spellF1 = 0.9000`).
> ⚠️ *In-sample / out-of-fold figures: not a valid result. See the Status notice at the top of this document.*

5. **Step 5 (3D Replay Generation):** Compiles the standalone, zero-dependency `barca_3d_interactive_replay.html` viewer.

---

## 5. Master Orchestrator Command-Line Flags

| CLI Flag | Description |
|---|---|
| `python auto_process_competition.py --team-name "MyTeam"` | Sets your team name, writing `<team_name>.csv` to both the root and `evaluation/results/` as required by `SUBMISSION.md`. |
| `python auto_process_competition.py --rebuild-cache` | Forces full re-extraction and model refitting from raw `data/tactical/` files. |
| `python auto_process_competition.py --no-jev` | Runs the Motor Base ablation (without Jev AI epistemic primitives) to compare performance metrics. |
| `python auto_process_competition.py --3d-slice full --3d-stride 2` | Exports the full 90-minute match to the 3D viewer at 12.5 fps. |
| `python auto_process_competition.py --skip-3d` | Executes numerical processing and evaluation only, skipping 3D compilation. |

---

## 6. 3D Visualization: Which File to Use? (`barca_3d_interactive_replay.html` vs. `src/viewer.html`)

The repository includes two HTML files for visualization, each serving a distinct purpose:

| File | Project Role | When to Use? | Key Features |
|---|---|---|---|
| **`barca_3d_interactive_replay.html`** *(in root)* | **Standalone Production Replay** | **For teammates, competition evaluators, and judges.** | **100% Offline & Self-Contained.** Embedded Three.js library and 7,500 frames of match telemetry. No web server or internet connection required. Simply **double-click** to open. |
| **`src/viewer.html`** *(in `src/`)* | **Developer Source Template** | **For technical development and UI iterations.** | Dynamically loads telemetry from `src/match_payload.js`. Modify this file when editing CSS styling, 3D lighting, camera angles, or HUD controls, then recompile with the orchestrator. |

> [!TIP]
> **For 99% of team reviews and official evaluations:** Open **`barca_3d_interactive_replay.html`** in the repository root directly in your browser.

---

## 7. Match Time Simulation: How to Analyze Any Match Slice

By default, `barca_3d_interactive_replay.html` contains **7,500 frames (5 minutes, frames 11,185 to 18,684)**, matching the exact **official evaluation window with public Ground Truth**. This keeps the standalone file ultra-lightweight (~5 MB), enables zero-latency loading, and facilitates competition score verification.

If teammates wish to analyze **the 1st Half, 2nd Half, the full 90-minute match, or any custom minute window**, they can export any slice using these examples:

### 📌 Example 1: Full First Half (~45-53 min)
Exports from opening kickoff to half-time whistle (`frames 11,185 to 90,651`):
```bash
# Using direct generator (stride 2 = 12.5 fps recommended for optimal smoothness and file size):
python generate_3d_replay_html.py --half 1 --stride 2 --standalone

# Or via master orchestrator:
python auto_process_competition.py --3d-slice half1 --3d-stride 2
```

### 📌 Example 2: Full Second Half (~45-53 min)
Exports from second-half kickoff to full-time whistle (`frames 110,808 to 190,197`):
```bash
# Using direct generator:
python generate_3d_replay_html.py --half 2 --stride 2 --standalone

# Or via master orchestrator:
python auto_process_competition.py --3d-slice half2 --3d-stride 2
```

### 📌 Example 3: Full 90-Minute Match (All 199,319 frames)
Exports all recorded tracking data across both halves:
```bash
# Using direct generator:
python generate_3d_replay_html.py --full --stride 2 --standalone

# Or via master orchestrator:
python auto_process_competition.py --3d-slice full --3d-stride 2
```

### 📌 Example 4: Custom Minute Window (e.g., Minutes 20 to 30)
To zoom in on a specific phase of play, use the frame calculation formula:  
$$\text{frame} = \text{minute} \times 60 \text{ s} \times 25 \text{ fps}$$
- **Minute 20:** $20 \times 60 \times 25 = 30,000$
- **Minute 30:** $30 \times 60 \times 25 = 45,000$

```bash
# Export custom window at native 25 fps with custom output filename:
python generate_3d_replay_html.py --start-frame 30000 --end-frame 45000 --stride 1 --standalone --output replay_min20_to_30.html
```

> [!TIP]
> **Performance Guidance (`--stride`):**
> - `--stride 1` (25 fps native): Maximum kinematic fidelity, ideal for short tactical clips (5-10 min).
> - `--stride 2` (12.5 fps interpolated): Smooth playback cutting HTML file size and browser RAM consumption by 50%. Recommended for full halves or 90 minutes.

---

## 8. Interactive 3D Tactical Viewer Manual

To open the 3D replay, simply **double-click** or open `barca_3d_interactive_replay.html` in Google Chrome, Microsoft Edge, Mozilla Firefox, or Safari.

### 🎮 Keyboard Shortcuts
- **`Spacebar`**: Play / Pause match playback.
- **`M`**: Cycle through the 3 AI Brains:
  1. `🧠 OUR AI (+ Jev System One)`: Full pipeline with deflection absorption, pass protection, and throw-in arbitration.
  2. `⚙️ MOTOR BASE (Without Jev AI)`: Pure GBDT + Viterbi baseline without epistemic contest resolution.
  3. `🤖 NAIVE AI (Nearest Player)`: Baseline proximity heuristic illustrating frequent false flickering.
- **`H`**: Toggle **Clean Cinema Mode** (hides sidebars and telemetry HUD for an unobstructed pitch view).
- **`C`**: **Collapse / Expand Bottom Dock**, shrinking the panel vertically to only the primary narrative ticker to maximize pitch visibility.
- **`Left / Right Arrow Keys`**: Step backward / forward by 1 second.
- **`J` / `K`**: Jump to previous / next Key Moment (turnovers, goals, and Jev AI arbitrated contests).
- **`1`**: Broadcast TV Camera (dynamic zoom following ball action).
- **`2`**: Tactical Drone Camera (high diagonal angle).
- **`3`**: Top-Down 2D Birds-Eye Camera (coach tactical board).
- **`4`**: Free Orbit Camera (left-click + drag to rotate, scroll wheel to zoom).

### 🎨 Visual Elements on the 3D Pitch
- **🔵🔴 Blaugrana Players (White/Gold numbers):** FC Barcelona attacking / defending with real player names (Pedri, Lewandowski, Lamine Yamal, Cubarsí, Ter Stegen, etc.).
- **⚪🔴 Red & White Players (Cyan numbers):** Athletic Club defending / attacking with real player names (Nico Williams, Iñaki Williams, Sancet, Vivian, Padilla, etc.).
- **🟡 Dynamic Possession Beacon:** Electromagnetic 3D tether line connecting the active ball-carrier to the ball, anchored directly at boot/grass level.
- **🛡️ 3D Pass Shield Dome:** A golden aerodynamic canopy encasing the ball during high-speed passes ($v > 4.5\text{ m/s}$), visually demonstrating why our model resists switching possession to nearby defenders who cannot intercept the pass trajectory.
- **🟣 Violet Pulse (Jev AI):** Emits when the epistemic arbitrator resolves 50/50 physical duels or unassigned throw-in restarts.
- **🛑 Dead-Ball HUD Card:** Switches to an amber theme during set-pieces (`🛑 BALÓN MUERTO`), verifying that the engine never awards false possession to the restart taker prior to the dead-ball kick.

---

## 9. Official Submission Verification Checklist

Before uploading the final competition deliverable:
```bash
# Verify formatting integrity
python evaluation/validate_submission.py submission.csv

# Verify leaderboard metrics
python evaluation/run_evaluation.py
```

Expected terminal output:
- **Rows:** `199,319`
- **Integrity:** `OK submission.csv`
- **Spell F1:** `0.9000`
- **Macro F1:** `0.9772`
- **Possessions:** `15` (of 17 in the public ground truth, 100% spell precision)
> ⚠️ *In-sample / out-of-fold figures: not a valid result. See the Status notice at the top of this document.*

- **Abstentions on Graded Frames:** `0 frames (0.00%)`

For questions or suggestions, please contact Team The Outliers on the team channel!
