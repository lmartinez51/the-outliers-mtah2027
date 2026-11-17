# -*- coding: utf-8 -*-
"""Load broad Smart Tagging set-piece windows as supporting evidence only."""

import os
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np


def load_set_piece_support(path: Optional[str], n_frames: int, fps: int) -> np.ndarray:
    """Return set-piece support; never interpret a broad window as dead directly."""
    support = np.zeros(n_frames, dtype=bool)
    if path is None:
        return support
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Smart Tagging file not found: {path}")
    root = ET.parse(path).getroot()
    for instance in root.iter("instance"):
        if (instance.findtext("code") or "").strip().lower() != "set pieces":
            continue
        start = max(0, int(round(float(instance.findtext("start")) * fps)))
        end = min(n_frames, int(round(float(instance.findtext("end")) * fps)))
        if end > start:
            support[start:end] = True
    return support
