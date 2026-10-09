"""
app/face/face_quality.py
------------------------
Quality gates for YuNet detections, shared by enrollment, evaluation and
(optionally) live recognition so that all three agree on what a "usable" face is.

YuNet row layout (15 values):
    0-3   x, y, w, h
    4-13  landmarks: right-eye, left-eye, nose, right-mouth, left-mouth
          ("right" = the subject's right, i.e. image-left)
    14    detection score

Why this exists: on extreme close-ups, profiles and partial faces YuNet can still
return a confident-looking box (an ear, half a face). Those frames produce garbage
embeddings that drag the enrolled mean away from the real identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    min_score: float = 0.80            # YuNet detection score
    min_face_px: int = 60              # min(box w, h) in pixels
    eye_dist_ratio: Tuple[float, float] = (0.30, 0.75)   # eye distance / box width
    max_roll_ratio: float = 0.50       # |dy between eyes| / eye distance


DEFAULT_CONFIG = QualityConfig()


def check_face(row: np.ndarray, cfg: QualityConfig = DEFAULT_CONFIG) -> Tuple[bool, str]:
    """Return (ok, reason). reason is "ok" or a short machine-readable code."""
    x, y, w, h = (float(v) for v in row[:4])
    score = float(row[14])
    pts = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
    r_eye, l_eye, nose, r_mouth, l_mouth = pts

    if score < cfg.min_score:
        return False, "low_score"
    if min(w, h) < cfg.min_face_px:
        return False, "too_small"

    eye_dist = float(np.linalg.norm(r_eye - l_eye))
    if eye_dist <= 1e-3 or w <= 1e-3:
        return False, "degenerate"
    lo, hi = cfg.eye_dist_ratio
    if not (lo < eye_dist / w < hi):
        return False, "bad_eye_ratio"             # partial face / ear / extreme profile
    if not (r_eye[0] < nose[0] < l_eye[0]):
        return False, "nose_outside_eyes"         # strong yaw
    if not (nose[1] > max(r_eye[1], l_eye[1]) and min(r_mouth[1], l_mouth[1]) > nose[1]):
        return False, "bad_vertical_order"        # eyes < nose < mouth violated
    if abs(float(r_eye[1] - l_eye[1])) > cfg.max_roll_ratio * eye_dist:
        return False, "extreme_roll"
    return True, "ok"
