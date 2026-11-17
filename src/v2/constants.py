# -*- coding: utf-8 -*-
"""Deterministic pipeline constants (v2 and v2.1 specifications).

Tier F constants are fixed by official documentation.
Tier B constants comprise frozen a priori baselines (APRIORI dict) and
externally tuned values (promoted as defaults in v2.1) derived from
open-source SkillCorner tracking data under the MIT license.
"""

from typing import Dict, Union

# Tier F: Fixed by definition or official documentation
FPS: int = 25  # DATA.md:245 ("round(seconds * 25) is a video frame")
PITCH_LENGTH: float = 105.3  # games.py: GAMES["BAR-ATH"]["pitch"][0] and DATA.md:93
PITCH_WIDTH: float = 68.0    # games.py: GAMES["BAR-ATH"]["pitch"][1] and DATA.md:93

# Half definitions and published kickoff frames (FAQ.md:30-34, DATA.md:253-259)
HALF_1_START: int = 11185
HALF_1_END: int = 90651
HALF_1_KICKOFF: int = 11185

HALF_2_START: int = 110808
HALF_2_END: int = 190197
HALF_2_KICKOFF: int = 113742

N_VIDEO_FRAMES: int = 199319

# Tier B: Frozen a priori baseline constants (v2.0.1-apriori)
APRIORI: Dict[str, Union[float, int]] = {
    "R_TOUCH": 2.0,      # 2.0 m: automatic-tracking position error plus player reach
    "MARGIN": 0.5,       # 0.5 m: clear separation between teams before declaring a candidate
    "G_INTERP": 50,      # 50 frames (2.0 s @ 25 fps): typical flight time of a pass
    "T_CONFIRM": 50,     # 50 candidate frames: "a player keeps the ball for a few seconds"
}

# Tier B: Active default constants (promoted from external tuning, v2.1)
# Origin: external tuning, SkillCorner open data, MIT, 2026-11-09, commit 65b5830
# Tuned on 11 A-League matches; validated on 8 hold-out matches (bootstrap 95% CI: [+0.007967, +0.039078])
R_TOUCH: float = 2.5     # 2.5 m (was 2.0 m)
MARGIN: float = 0.75     # 0.75 m (was 0.5 m)
G_INTERP: int = 30       # 30 frames (1.2 s @ 25 fps, was 50 frames / 2.0 s)
T_CONFIRM: int = 30      # 30 frames (1.2 s @ 25 fps, was 50 frames / 2.0 s)

OOB_MARGIN: float = 1.0      # 1.0 m: tracking and homography error
T_OOB: int = 12              # 12 frames: rules out single-frame noise
SPEED_SMOOTH: int = 25       # 25 frames (1 s): standard 1 s smoothing window
V_STOP: float = 1.5          # 1.5 m/s: human walking pace applied to 75th percentile of player speeds
V_BALL_STOP: float = 1.0     # 1.0 m/s: ball essentially at rest
T_STOP: int = 75             # 75 frames (3 s): stoppage lasting at least a few seconds
V_MEDIAN_STOP: float = 1.0
V_P90_STOP: float = 2.0
FRACTION_LOW_STOP: float = 0.65
MIN_VALID_PLAYERS: int = 14
V_PLAYER_RESTART: float = 2.0
V_BALL_RESTART: float = 2.0
DECEL_DELTA: float = 0.75
T_EVENT_CONTEXT: int = 250
T_RESTART: int = 10
MIN_LIVE_FLICKER: int = 25   # 25 frames (1 s): shorter live segments between stoppages are not plausible plays
MIN_DEAD_FLICKER: int = 12   # 12 frames (0.5 s): shorter dead segments inside one possession are noise

# Standard label vocabulary
LABEL_H: str = "H"
LABEL_A: str = "A"
LABEL_D: str = "D"
LABEL_X: str = "X"
