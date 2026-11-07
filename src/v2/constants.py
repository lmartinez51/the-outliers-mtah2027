# -*- coding: utf-8 -*-
"""Deterministic pipeline constants (a priori specification, Stage 2).

All constants are frozen and approved a priori in Stage 2 specification.
No constants were selected, tuned, or calibrated on ground-truth possession data
or on the match video.
"""

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

# Tier B: A priori engineering judgments (frozen for v2)
R_TOUCH: float = 2.0         # 2.0 m: automatic-tracking position error plus player reach
MARGIN: float = 0.5          # 0.5 m: clear separation between teams before declaring a candidate
G_INTERP: int = 50           # 50 frames (2 s): typical flight time of a pass
T_CONFIRM: int = 50          # 50 candidate frames: CHALLENGE.md proxy: "a player keeps the ball for a few seconds"
OOB_MARGIN: float = 1.0      # 1.0 m: tracking and homography error
T_OOB: int = 12              # 12 frames: rules out single-frame noise
SPEED_SMOOTH: int = 25       # 25 frames (1 s): standard 1 s smoothing window
V_STOP: float = 1.5          # 1.5 m/s: human walking pace applied to 75th percentile of player speeds
V_BALL_STOP: float = 1.0     # 1.0 m/s: ball essentially at rest
T_STOP: int = 75             # 75 frames (3 s): stoppage lasting at least a few seconds
MIN_LIVE_FLICKER: int = 25   # 25 frames (1 s): shorter live segments between stoppages are not plausible plays
MIN_DEAD_FLICKER: int = 12   # 12 frames (0.5 s): shorter dead segments inside one possession are noise

# Standard label vocabulary
LABEL_H: str = "H"
LABEL_A: str = "A"
LABEL_D: str = "D"
LABEL_X: str = "X"
