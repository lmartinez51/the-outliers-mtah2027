> **Status notice (Nov 2026).** Earlier versions of this repository fitted models and parameters using the public 5-minute ground-truth window. The organizers' clarification of the rules (received in November) does not allow this. We are reworking the submission. Metrics shown in earlier documents are in-sample and must not be read as valid results, and statements of compliance with RULES.md in those documents are withdrawn until the rework is complete.

# Technical Document: Factored 3-State Probabilistic Modeling with Deterministic Jev AI (`System One`) Epistemic Primitives for Automated Football Possession Tracking

**Competition:** More than a Hack 2027 (FC Barcelona & Metrica Sports)  
**Team Name:** `The Outliers`  
**Authors:** Team The Outliers, Ana Karen Arellano, and Steven Ibarra  
**Repository:** [`the-outliers-mtah2027`](https://github.com/lmartinez51/the-outliers-mtah2027)  
**Target Fixture:** `BAR-ATH` (FC Barcelona vs. Athletic Club)  
**Official Output File:** `The_Outliers.csv` / `submission.csv` (`MD5: f9507f09200e235665c5c988c6a1504e`)  
**Official Scoreboard Rank:** **Rank #1** (`spellF1 = 0.9000`, `macroF1 = 0.9772`, `15` possessions, `-0.8` frame latency)  

> ⚠️ *In-sample / out-of-fold figures: not a valid result. See the Status notice at the top of this document.*

---

## Executive Summary

Automated football possession tracking from optical tracking data (EPTS) presents four severe technical hurdles:
1. **Severe Sensor Occlusion:** The ball is missing or occluded on **40.7%** of frames in the full match ($81,048 / 199,319$ frames), causing standard heuristic nearest-player baselines to suffer catastrophic high-frequency flickering ($131$ false possessions, spellF1 $0.0378$).
2. **Double-Penalization of Stoppages (Dead Balls):** In traditional nearest-player setups, dead balls are forcibly misassigned to whichever player retrieves the ball for a restart, producing systematic possession bias.
3. **Smart Tagging Epistemic Ambiguity:** Human event tags provide rich tactical phases but lack team labels during set-piece restarts (all 26 Throw-Ins in `BAR-ATH` are tagged `Team = N/A`), and tag timestamps exhibit human logging lag ($\sim 1\text{--}3\text{ seconds}$) that conflicts with instantaneous tracking physics.
4. **Metric Alignment (`spellF1` vs. `macroF1`):** The primary competition ranking metric (`spellF1`) evaluates temporal IoU overlap on possession spells ($IoU \ge 0.3 \dots 0.7$), demanding sequence-level continuity and exact event boundary placement rather than naive frame-by-frame greedy voting.

To solve these challenges, we built a **Factored Two-Head Probabilistic Architecture** augmented by a **Deterministic Jev AI (`System One`) Epistemic Layer** (`src/jev_system_one.py`) and decoded via a **3-State Soft Viterbi Sequence Decoder** (`src/mtah_decoder.py`).

### Headline Benchmark Results (Official Public Ground Truth: 7,500 frames / 5 minutes)

| Model Pipeline | Rank | spellF1 (Primary) | macroF1 | Class D F1 | Possessions (GT=17) | Latency ($\Delta$ frames) | False Switches Avoided |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Our Engine + Jev AI (`System One`)** | **#1** | **0.9000** | **0.9772** | **0.9710** | **15 (GT=17, 100% precision)** | **-0.8** | **116 fewer than raw (-88.5%)** |
| Motor Base (No Jev AI, ablation) | — | 0.9125 | 0.9805 | 0.9710 | 15 (GT=17) | -0.75 | 116 fewer than raw (-88.5%) |
| `baseline_smooth125` | #3 | 0.4242 | 0.5812 | 0.1464 | 16 (-1) | -7.7 | — |
| `baseline_smooth50` | #4 | 0.3256 | 0.6207 | 0.3225 | 26 (+9) | +4.0 | — |
| `baseline_smooth25` | #5 | 0.2868 | 0.6200 | 0.3857 | 36 (+19) | +8.2 | — |
| `baseline_raw` (Nearest Player) | #6 | 0.0378 | 0.5972 | 0.4197 | 131 (+114) | -3.1 | Baseline (125 switches) |

> ⚠️ *In-sample / out-of-fold figures: not a valid result. See the Status notice at the top of this document.*

*Our system outperforms the best competition baseline by **+112.1% relative improvement** on `spellF1` ($0.9000$ vs. $0.4242$), with near-zero latency ($-0.8$ frames / $\sim 32\text{ ms}$) and zero spurious spells (15/15 predicted spells match a ground-truth possession). *Public-window figures are **in-sample** (the final model is fitted on those 7,500 labelled frames). The 5-Fold temporal-block OOF numbers are the honest generalisation estimate for the 95.3% of the graded match that is unseen private data.**

> ⚠️ *In-sample / out-of-fold figures: not a valid result. See the Status notice at the top of this document.*

---

## Section 1: The Approach (Architecture & Mathematical Foundations)

Our methodology decouples the possession problem into orthogonal kinematic, tactical, and temporal components:

```
+---------------------------------------------------------------------------------------------------+
| RAW INPUTS: BAR-ATH_tactical_atd.txt (25 fps Tracking) + BAR-ATH_tactical_smart_tagging.xml      |
+---------------------------------------------------------------------------------------------------+
                                                  |
                    [ Streaming Feature Extraction: src/mtah_features.py (16.5s CPU) ]
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
        +-----------------------------------+             +-----------------------------------+
        | HEAD 1: Dead-Ball Detector (GBDT) |             | HEAD 2: Live Possession Estimator |
        | P(Dead)                           |             | P(Home | Live)                    |
        | - 45% Ball-Dropout Augmentation   |             | - Strict Team-Swap Symmetry       |
        | - Speed slowdown ratios (w15/w75) |             | - 0.72 GBDT + 0.28 Smooth Prior   |
        | - Boundary & stoppage overrides   |             | - Immune to team color/slot bias  |
        +-----------------------------------+             +-----------------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
        +----------------------------------------------------------------------------------+
        | DETERMINISTIC JEV AI (SYSTEM ONE) EPISTEMIC LAYER (src/jev_system_one.py)        |
        | - Primitive 1: N/A Set-Piece Resolution (36 phases, 26 Throw-Ins, 26,500 frames) |
        | - Primitive 2: Tag-vs-Kinematics Epistemic Conflict Arbitration (8,087 frames)   |
        | - Primitive 3: Occlusion Pass-Bridge Damping (40.7% missing-ball frames)         |
        | - Primitive 4: Restart Kick-Point Boundary Sharpening                            |
        +----------------------------------------------------------------------------------+
                                                  |
                                                  v
        +----------------------------------------------------------------------------------+
        | 3-STATE SOFT VITERBI SEQUENCE DECODER (src/mtah_decoder.py)                      |
        | - Log-space dynamic programming with transition penalties (p_turnover = -4.2)    |
        | - Controlled possession filtering: Deflection filter (< 28 frames = 1.12s)       |
        | - Stoppage flicker filter (< 30 frames) & Micro-glitch filter (< 8 frames)       |
        | - Graded mask assignment: 'X' strictly on 40,462 non-graded frames (0% on graded)|
        +----------------------------------------------------------------------------------+
                                                  |
                                                  v
        +----------------------------------------------------------------------------------+
        | OFFICIAL DELIVERABLE: submission.csv (199,319 rows, MD5: f9507f09200e235665c5c9) |
        +----------------------------------------------------------------------------------+
```

### 1.1 Factored Probability Formulation
Rather than training an unconstrained 3-class classifier that entangles dead balls with live play, we factorize the state distribution:
$$P(y = \text{'D'}) = P_{\text{dead}}$$
$$P(y = \text{'H'}) = (1.0 - P_{\text{dead}}) \cdot P(\text{Home} \mid \text{Live})$$
$$P(y = \text{'A'}) = (1.0 - P_{\text{dead}}) \cdot (1.0 - P(\text{Home} \mid \text{Live}))$$

#### Head 1: Occlusion-Robust Dead-Ball Detector
Trained using `HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.05, min_samples_leaf=40, l2_regularization=2.0)`. To eliminate vulnerability to the 40.7% ball occlusion rate in unlabelled match periods, we inject **45% Ball-Dropout Data Augmentation**: during training on live frames, ball distance and speed features are synthetically masked out, forcing the model to learn collective player slowing signatures (ratio of rolling player speed $w_{15} / w_{75}$), pitch boundary dispersion, and referee whistle intervals, followed by full-match logit re-centering ($\Delta = -1.20$). Ball speed is computed as a uniform 25 Hz kinematic finite difference on every frame, so occluded and visible rows share one speed definition.

#### Head 2: Mathematically Team-Symmetric Live Possession Model
A critical flaw in standard supervised learning on single fixtures is team bias (the model memorizes that Home wore dark blue and occupied slots 0..23). We enforce strict mathematical symmetry using `HistGradientBoostingClassifier(max_depth=4, max_iter=120, learning_rate=0.05, min_samples_leaf=45, l2_regularization=2.5)` by:
1. **Team-Swap Training Augmentation:** Every training sample $(x, y)$ is duplicated as $(x_{\text{swap}}, 1 - y)$ where Home and Away distances, speeds, and centroid coordinates are mirrored across the pitch.
2. **Symmetrized Inference:**
   $$P(\text{Home} \mid \text{Live}) = \frac{1}{2} \left[ P_{\text{GBDT}}(x) + (1.0 - P_{\text{GBDT}}(x_{\text{swap}})) \right]$$
3. **Analytical Kinematic-Tactical Prior:** Blended at $72\%$ GBDT $+ 28\%$ Analytical Prior ($P_{\text{prior}}$), combining multi-second rolling proximity advantages ($w_{75}, w_{125}, w_{200}$), touch anchors, and smoothed Smart Tagging bias.

---

### 1.2 Deterministic Jev AI (`System One`) Epistemic Layer

The epistemic layer (`src/jev_system_one.py`) implements fast, deterministic, vectorized common-sense reasoning primitives that arbitrate ambiguity before sequence decoding:

#### Primitive 1: N/A Set-Piece & Throw-In Resolution
In `BAR-ATH_tactical_smart_tagging.xml`, 36 Smart Tagging intervals (including **all 26 Throw-Ins**) have `Team = N/A` or empty tags, covering **26,500 frames**. Standard systems lose possession context during these critical dead-to-live transitions.  
`resolve_na_set_pieces(ds)` inspects the post-restart kinematic evaluation window around the physical release frame $t_{\text{kick}} \in [s+25, \min(e_f, s+380)]$ (not the tag end, which can lie 35 s later); post-goal `Kick Off` events are resolved as the opposite of the preceding scoring team. By evaluating directional control anchors (`anchor_team`, `nearest_team_adv`, and forward kinematic trajectories), it deterministically resolves all 36 events:
- **18 intervals** resolved in favor of Home (FC Barcelona)
- **18 intervals** resolved in favor of Away (Athletic Club)  
This produces a robust prior $\text{bias} \in \{+1.0, -1.0\}$ during previously unassigned throw-in phases.

#### Primitive 2: Tag-vs-Kinematics Conflict Arbitration
Smart Tagging reflects human tactical logging, which carries an inherent 1 to 3 second response delay. When an unexpected turnover occurs, the Smart Tagging prior asserts that Team A is in possession while optical tracking proves that Team B has intercepted the ball and controlled it.  
Primitive 2 computes normalized Shannon entropy $H(p) = -\sum p_i \log_3 p_i$ and identifies epistemic conflict where $|\text{st\_team\_bias}| > 0.3$, the ball is visible, the opponent is within $1.45\text{m}$, and $|\text{nearest\_team\_adv}| > 2.0\text{m}$ opposes the tag. Across **8,087 graded conflict frames** (`8,115` full-match frames), the blending weight dynamically shifts from $0.72$ to $0.95$ toward instantaneous physical control, reducing turnover detection latency to **$-0.8$ frames**. Sustained short-window turnover evidence (rolling advantage plus last anchor opposing a non-transition tag) is arbitrated the same way.

#### Primitive 3: Occlusion Pass-Bridge Damping
During the 40.7% missing-ball frames, naive models drift toward 50/50 uncertainty or switch erratically. Primitive 3 checks if the anchor player immediately before occlusion and after occlusion belong to the same team ($\text{gap} \le 90$ frames). If so, it applies bounded log-odds reinforcement ($\Delta \text{logit} = \pm 1.2$), and across opposing anchors applies monotonic temporal interpolation ($w = 0.65$), bridging aerial passes and off-screen clearances without false turnovers.

#### Primitive 4: Restart Boundary Sharpening
Following a corner kick, free kick, or goal kick, dead-ball probabilities tend to linger after the ball has already been struck. Primitive 4 detects the initiation of movement ($\text{sp\_phase} \ge 0.85$ with $v_{\text{ball, w15}} > 5.5\text{ m/s}$ or $v_{\text{player, max}} > 4.2\text{ m/s}$) following a set piece and damps $P_{\text{dead}}$ ($\times 0.15$, capped at $0.20$), ensuring crisp, instantaneous transitions into live play. Two physical-continuity masks (`oob_to_kick_mask`, `post_goal_mask`) additionally keep the ball dead from the out-of-bounds frame until the restart kick, and through the post-goal celebration.

---

### 1.3 Strict Compliance with `RULES.md`

> ⚠️ *Compliance / validity statement withdrawn. See the Status notice at the top of this document.*

`RULES.md` mandates that all submissions must be reproducible, open, inspectable, and algorithmic:
- **Zero External API Calls:** `src/jev_system_one.py` contains **zero external network requests**, zero LLM inference APIs, and zero private credentials. It is a 100% self-contained, deterministic NumPy/pandas implementation.
- **Bit-for-Bit Determinism:** Running `auto_process_competition.py` generates the exact submission file with identical MD5 checksum (`f9507f09200e235665c5c988c6a1504e`).
- **Data Protection Compliance:** Raw tracking files (`data/tactical/`) and large derived coordinate exports (`src/match_payload.js`, `barca_3d_interactive_replay.html`, and `artifacts/`) are excluded from version control via `.gitignore`.

> ⚠️ *Compliance / validity statement withdrawn. See the Status notice at the top of this document.*

---

### 1.4 3-State Soft Viterbi Sequence Decoder

Frame-by-frame $\arg\max$ prediction generates intolerable flicker (over 125 false switches in 5 minutes). `src/mtah_decoder.py` applies a Hidden Markov Model sequence decoder operating in log-space:

$$\log A = \begin{bmatrix} 0.0 & p_{\text{turnover}} & p_{\text{live}\to\text{dead}} \\ p_{\text{turnover}} & 0.0 & p_{\text{live}\to\text{dead}} \\ p_{\text{dead}\to\text{live}} & p_{\text{dead}\to\text{live}} & 0.0 \end{bmatrix}$$

- **Calibrated Turnover Penalty ($p_{\text{turnover}} = -4.2$):** Suppresses spurious switches while allowing genuine interceptions to transition instantly.
- **Controlled Possession Filters:**
  - *Sandwiched Live Deflection:* If Team A has the ball, Team B touches or deflects it for $< 28\text{ frames}$ ($1.12\text{s}$), and Team A regains it ($A \to B \to A$), the touch is classified as a non-controlled deflection and absorbed into Team A's possession.
  - *Stoppage Flicker:* Brief live flickers inside dead balls ($D \to \text{Live } (< 30\text{ frames}) \to D$) are absorbed into $D$.
  - *Micro-Glitches:* Isolated single-frame anomalies ($< 8\text{ frames}$) are smoothed into their surrounding state.
- **Abstention Strategy (`X`):** Graded ground-truth frames penalize `X` as a miss. Therefore, we assign `X` strictly to the **40,462 non-graded frames** (halftime $90,652 \dots 110,807$, pre-kickoff $0 \dots 11,184$, and post-match $190,198 \dots 199,318$). On all 158,857 graded frames, our abstention rate is **0.00%**.

---

## Section 2: Assets Used and Assets Dropped

### 2.1 Assets Used
1. **Tactical ATD Tracking (`BAR-ATH_tactical_atd.txt` & `.xml`):**
   - 25 fps, 199,319 frames. Provides continuous pitch coordinates $(x, y)$ for 46 player slots and ball coordinates.
   - We utilized tactical coordinates for distance-to-ball, team centroids, relative player spread, and kinematic finite-difference velocities.
2. **Tactical Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`):**
   - High-level event intervals (Possession, Passes, Set Pieces, Throw-Ins).
   - Used for macro tactical conditioning, set-piece phase windows, and analytical prior generation.

### 2.2 Assets Dropped & Justification
1. **Broadcast Homography & Broadcast Tracking (`BAR-ATH_broadcast_atd.txt`, homography XML):**
   - *Reason for Dropping:* The broadcast feed contains only 164,488 frames, is unaligned with tactical time, features camera cuts, zoom changes, and commercial breaks, and suffers from a **34.2% invalid homography rate** during tight close-ups. Incorporating broadcast homography introduced temporal warping artifacts without improving tactical ball tracking.
2. **Raw MP4 Video Decoding (`BAR-ATH_tactical.mp4`):**
   - *Reason for Dropping:* Deep optical flow and 3D CNN video backbones require heavy GPU compute ($> 2\text{ hours}$ per match) and introduce non-deterministic CUDA rounding. Our streaming feature pipeline extracts equivalent kinematic signals (finite differences, rolling player spread, acceleration) in **16.5 seconds on CPU**, making raw video decoding completely redundant and unnecessarily cumbersome.

---

## Section 3: Possession Rules Implemented (`CHALLENGE.md` Alignment)

Our pipeline encodes football domain rules directly into the feature representation and sequence decoder:

1. **Loose Balls & Passes in Flight:**
   - When a pass travels between teammates, the ball is frequently closer to an opposing defender than to the kicker or receiver.
   - We implemented an **Active Pass Shield** feature: when the ball has high velocity ($v_{\text{ball}} \ge 4.5\text{ m/s}$) and moves along the vector connecting two teammates, possession is frozen to the passing team until the receiving touch occurs, completely eliminating "phantom pass interceptions."
2. **Non-Controlled Touches & Deflections:**
   - A goalkeeper parry, a defender blocking a cross, or a rebound off a shin does not constitute possession under Metrica & FC Barcelona guidelines.
   - We enforced the **28-frame ($1.12\text{s}$) deflection absorption rule**: an opposing touch must demonstrate sustained control ($\ge 28\text{ frames}$) to register as a turnover.
3. **Body-to-Body Contested Duels:**
   - In 50/50 duels where players from both teams are within $2.8\text{m}$ of the ball, Jev AI evaluates velocity vectors and relative distance differentials ($|\text{dist}_H - \text{dist}_A| \le 1.0\text{m}$) combined with the macro-phase prior, preventing artificial state flickering.
4. **Pre-Restart & Dead-Ball Stoppages:**
   - All set pieces (Corners, Free Kicks, Penalties, Throw-Ins) are enforced as Dead Ball ($D$) from the moment the ball goes out of bounds or play stops until the restart kick is executed ($v_{\text{ball}} > 3.0\text{ m/s}$).
   - The pre-2nd half kickoff block (frames $110,808 \dots 113,741$) is deterministically set to $D$.
5. **Halftime & Match Boundary Abstentions:**
   - Frames $0 \dots 11,184$ (pre-match), $90,652 \dots 110,807$ (halftime), and $190,198 \dots 199,318$ (post-match) are strictly assigned label `X`.

---

## Section 4: Side-by-Side Ablation Study & Limitations

### 4.1 Ablation Study & Cross-Validation Benchmarks

To demonstrate the concrete contribution of each architectural component, we evaluated the system under both **5-Fold Contiguous Temporal Block Out-of-Fold (OOF) Cross-Validation** (7,500 frames, 5 x 1-minute blocks) and the **Official 5-Minute Public Ground Truth Leaderboard**:

#### Table 4.1: Strictly Leak-Free 5-Fold Temporal Block OOF Cross-Validation
| Pipeline Configuration | Diagnostic Stage | macroF1 | spellF1 | Class H F1 | Class A F1 | Class D F1 | Live H/A Acc |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Motor Base (No Jev AI) | Raw CV Probabilities | 0.8884 | — | — | — | 0.8960 | 88.66% |
| **Motor + Jev AI (System One)** | **Raw CV Probabilities** | **0.8929** | — | — | — | **0.9015** | **89.11%** |
| Motor Base (No Jev AI) | Decoded OOF Benchmark | 0.9133 | 0.5846 | 0.9318 | 0.8915 | 0.9166 | 91.59% |
| **Motor + Jev AI (System One)** | **Decoded OOF Benchmark** | **0.9156** | **0.6714** | **0.9336** | **0.8965** | **0.9166** | **91.89%** |

> ⚠️ *In-sample / out-of-fold figures are not valid results and the compliance / validity statement is withdrawn. See the Status notice at the top of this document.*

*Jev AI delivers consistent net gains on unseen folds: `+0.0868` spellF1, `+0.0023` macroF1 (decoded), `+0.0044` macroF1 / `+0.0055` D F1 / `+0.45%` live accuracy (raw probabilities).*

#### Table 4.2: Official Public Ground Truth Scoreboard Ablation (7,500 frames)
| Configuration | spellF1 | macroF1 | Home F1 | Away F1 | Dead F1 | Possessions | Switches | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1. Naive Nearest Player (`baseline_raw`) | 0.0378 | 0.5972 | 0.7290 | 0.6430 | 0.4197 | 131 | 125 | -3.1f |
| 2. Rolling Median Smoothing (`baseline_smooth50`) | 0.3256 | 0.6207 | 0.7781 | 0.7615 | 0.3225 | 26 | 38 | +4.0f |
| 3. Factored Architecture + Viterbi (No Jev AI) | 0.9125 | 0.9805 | — | — | 0.9710 | 15 | — | -0.75f |
| **4. Full Engine + Jev AI System One (Final Submission)** | **0.9000** | **0.9772** | **0.9876** | **0.9731** | **0.9710** | **15** | **18** | **-0.8f** |

> ⚠️ *In-sample / out-of-fold figures: not a valid result. See the Status notice at the top of this document.*

#### Key Insights from Ablation:
- **Factored Modeling + Viterbi (+176% relative spellF1 over baseline_smooth50):** Moving from moving-average filtering to a factored HMM sequence decoder with controlled-possession filters is the single largest performance jump, lifting spellF1 from $0.3256$ to $0.9000$ (15 of 17 possessions, 100% precision).
- **Jev AI System One Epistemic Calibration:** Lifts out-of-fold spellF1 by `+0.0868` (0.5846 → 0.6714), macroF1 by `+0.0023`, raw-probability D F1 by `+0.0055` and live accuracy by `+0.45%`, arbitrates 8,087 graded tag-vs-physics conflict frames and resolves all 36 N/A set-piece phases (26,500 frames).
- **Honest caveat (in-sample public window):** there the Motor Base scores marginally higher than Motor + Jev (spellF1 `0.9125` vs. `0.9000`, macroF1 `0.9805` vs. `0.9772`) because Jev switches to Away `32` frames (1.28 s, frames 13,672..13,703) before the true turnover at 13,704. We deliberately did **not** tune a gate to that single event: the OOF folds (unseen data) favour Jev, and that is the criterion that matters for the private 95.3% of the match.

> ⚠️ *In-sample / out-of-fold figures: not a valid result. See the Status notice at the top of this document.*

#### 4.1.1 Version 2.0 Mathematical Upgrades (Before → After, identical production decoder)

| Metric | v1 (previous) | **v2 (current)** | Δ |
|---|:---:|:---:|:---:|
| 5-Fold OOF spellF1 | 0.3412 | **0.6714** | +0.3302 |
| 5-Fold OOF macroF1 | 0.8359 | **0.9156** | +0.0797 |
| 5-Fold OOF Class D F1 | 0.8076 | **0.9166** | +0.1090 |
| Public window spellF1 (in-sample) | 0.8353 | **0.9000** | +0.0647 |
| Public window macroF1 (in-sample) | 0.8896 | **0.9772** | +0.0876 |
| Public window Class D F1 (in-sample) | 0.7721 | **0.9710** | +0.1989 |
| Post-goal celebration frames labelled `D` (13,733 frames) | — | **100.0%** | — |
| Full-match graded split H / A / D | — | **31.8% / 21.8% / 46.4%** | 56.8 min live ball |

> ⚠️ *In-sample / out-of-fold figures: not a valid result. See the Status notice at the top of this document.*

Upgrades behind the gain:
1. **Uniform kinematic ball speed** (`src/mtah_features.py`): 25 Hz finite difference on all 199,319 frames. The reported ball speed was 100% `NaN` in frames `0..54,834`, which silently disabled every speed feature on the public window.
2. **Ball-dropout 35% → 45%** with rolling ball-speed windows (`w15/w35/w75`) zeroed on dropped live rows and realistic occlusion-gap sampling, so Head 1 cannot learn the shortcut `ball_speed_rolling == 0 ⇒ D`; logit re-centering `-1.20`.
3. **Proximity-attenuated `nearest_team_adv`**, so a ball rolling in empty space no longer manufactures control bias.
4. **N/A resolution at the physical release frame** `t_kick ∈ [s+25, min(e_f, s+380)]` (Throw-Ins), and post-goal `Kick Off` resolved as the opposite of the scoring team.
5. **`oob_to_kick_mask` and `post_goal_mask`**: out-of-bounds → restart-kick continuity and post-goal celebration are forced to `D`.
6. **Sustained tag-vs-physics conflict** (short-window turnover evidence, gated on occlusion `ball_gap_len >= 50` for the Home side) in addition to instantaneous touch conflict.
7. **Decoder** (`src/mtah_decoder.py`): deflection filter `22 → 28` frames (1.12 s), absorption of short flips next to `D` (`< 28` frames, `< 68` with an opposing tag bias), start-boundary fix. Spell precision on the OOF benchmark is 100% (11/11 predicted spells are true positives).

### 4.2 Known Limitations & Future Work
1. **Vertical (Z-Axis) Aerial Contests:** EPTS tracking provides 2D coordinates $(x, y)$. While our kinematic speed lift algorithm successfully models parabolic pass arcs, high aerial clearances and contested headers currently rely on 2D proximity. Integrating monocular camera homography to estimate true ball 3D elevation ($z$) would improve aerial duel arbitration.
2. **Multi-Camera Occlusion Recovery:** When the ball remains off-screen for $> 15\text{ seconds}$ during injuries or substitutions, our occlusion bridge freezes the last known state. Incorporating optical referee gesture recognition could further sharpen dead-ball boundaries during prolonged medical stoppages.

---

## Section 5: Reproduction Instructions

The entire pipeline is 100% automated, cross-platform, deterministic, and requires no GPU:

### 5.1 Environment Setup
```bash
# Clone repository
git clone https://github.com/lmartinez51/the-outliers-mtah2027.git
cd the-outliers-mtah2027

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install lightweight dependencies
pip install -r requirements.txt
```

### 5.2 Single-Command Master Execution
To execute the complete end-to-end pipeline (feature extraction, model training, Viterbi decoding, official validation, scoreboard evaluation, and 3D broadcast replay generation):

```bash
python auto_process_competition.py
```

### 5.3 Command-Line Options
```bash
# Force full re-extraction from raw tactical data:
python auto_process_competition.py --rebuild-cache

# Run with custom team name (copies to <team_name>.csv as required by SUBMISSION.md):
python auto_process_competition.py --team-name "BarcaAI"

# Run Motor Base ablation (without Jev AI):
python auto_process_competition.py --no-jev

# Export full-match 3D replay:
python auto_process_competition.py --3d-slice full --3d-stride 2
```

### 5.4 Official Submission Verification
```bash
python evaluation/validate_submission.py submission.csv
python evaluation/run_evaluation.py
```

### 5.5 Interactive 3D Tactical Broadcast Replay
Open `barca_3d_interactive_replay.html` directly in any web browser (Chrome, Edge, Firefox, Safari) to interactively explore the match in 3D WebGL with 3-way brain switching (`OUR` vs. `BASE` vs. `NAIVE`), Pass Shield visualization, and live Jev AI epistemic telemetry.

---
*Signed by:*  
**Team The Outliers**  
Lead Machine Learning Engineer & Researcher  
*More than a Hack 2027 — FC Barcelona & Metrica Sports*
