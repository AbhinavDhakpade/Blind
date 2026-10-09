"""
app/tracking/iou_tracker.py
----------------------------
Lightweight IoU-based multi-object tracker.

Avoids heavy libraries (SORT, DeepSORT) that are impractical on Raspberry Pi 5.

Algorithm
---------
Each incoming Detection is matched to existing tracks via intersection-over-union.
Tracks accumulate a short bounding-box history which is used to compute a
relative area-growth rate (proxy for "approaching" in the absence of true depth).

MotionState
-----------
    APPROACHING   – bbox area growing  (object getting closer)
    RECEDING      – bbox area shrinking
    STATIONARY    – area stable
    UNKNOWN       – not enough history
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from app.detection.base import Detection

logger = logging.getLogger(__name__)


class MotionState(Enum):
    UNKNOWN     = auto()
    APPROACHING = auto()
    STATIONARY  = auto()
    RECEDING    = auto()


@dataclass
class TrackedObject:
    """A detection that has been assigned a persistent track ID."""
    track_id: int
    detection: Detection
    motion_state: MotionState = MotionState.UNKNOWN
    age: int = 1                # frames since first seen
    missed_frames: int = 0


# ---------------------------------------------------------------------------
# Internal track record (not part of public API)
# ---------------------------------------------------------------------------
@dataclass
class _Track:
    track_id: int
    detection: Detection
    age: int = 1
    missed_frames: int = 0
    area_history: deque = field(default_factory=lambda: deque(maxlen=8))

    def update(self, det: Detection) -> None:
        self.detection = det
        self.missed_frames = 0
        self.age += 1
        self.area_history.append(det.area)

    def compute_motion_state(self, threshold: float = 0.04) -> MotionState:
        history = list(self.area_history)
        if len(history) < 3:
            return MotionState.UNKNOWN
        # Linear regression slope of area vs time
        n = len(history)
        mean_t = (n - 1) / 2.0
        mean_a = sum(history) / n
        numerator = sum((i - mean_t) * (history[i] - mean_a) for i in range(n))
        denominator = sum((i - mean_t) ** 2 for i in range(n))
        if denominator == 0:
            return MotionState.STATIONARY
        slope = numerator / denominator
        # Normalise by mean area to get relative rate
        if mean_a == 0:
            return MotionState.STATIONARY
        relative_rate = slope / mean_a
        if relative_rate > threshold:
            return MotionState.APPROACHING
        if relative_rate < -threshold:
            return MotionState.RECEDING
        return MotionState.STATIONARY


class IoUTracker:
    """Assign persistent track IDs to detections using IoU matching."""

    def __init__(self, cfg) -> None:
        self._max_age: int = getattr(cfg, "max_age_frames", 8)
        self._min_iou: float = getattr(cfg, "min_iou", 0.30)
        self._approach_threshold: float = getattr(cfg, "approach_threshold", 0.04)
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0

    # ------------------------------------------------------------------
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Match *detections* to tracks; return tracked objects."""

        # Increment missed-frame counter for all existing tracks
        for track in self._tracks.values():
            track.missed_frames += 1

        unmatched_dets = list(detections)

        # Greedy IoU matching
        for track_id, track in list(self._tracks.items()):
            if not unmatched_dets:
                break
            best_iou = self._min_iou
            best_det: Optional[Detection] = None
            best_idx = -1
            for i, det in enumerate(unmatched_dets):
                if det.class_id != track.detection.class_id:
                    continue  # only match same class
                iou = _iou(track.detection.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
                    best_idx = i
            if best_det is not None:
                track.update(best_det)
                unmatched_dets.pop(best_idx)

        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            new_track = _Track(track_id=self._next_id, detection=det)
            new_track.area_history.append(det.area)
            self._tracks[self._next_id] = new_track
            self._next_id += 1

        # Cull stale tracks
        stale = [tid for tid, t in self._tracks.items() if t.missed_frames > self._max_age]
        for tid in stale:
            del self._tracks[tid]

        # Build output
        result: List[TrackedObject] = []
        for track in self._tracks.values():
            if track.missed_frames == 0:   # only yield active tracks
                motion = track.compute_motion_state(self._approach_threshold)
                result.append(
                    TrackedObject(
                        track_id=track.track_id,
                        detection=track.detection,
                        motion_state=motion,
                        age=track.age,
                        missed_frames=track.missed_frames,
                    )
                )
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union
