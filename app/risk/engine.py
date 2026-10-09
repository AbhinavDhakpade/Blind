"""
app/risk/engine.py
------------------
Risk calculation engine.

This module is deliberately isolated from detection, distance, and decision logic.
It consumes structured inputs and produces a structured RiskAssessment output.

Scoring factors (all configurable)
------------------------------------
1. Distance zone        – VERY_NEAR scores highest
2. Object class         – configurable per-class multiplier (vehicles score higher)
3. Motion state         – APPROACHING adds a large multiplier
4. Bounding-box position– objects centred in frame score higher
5. Detection confidence – low-confidence detections are discounted

The final risk level is determined by a numeric risk_score mapped to
{CLEAR, LOW, MEDIUM, HIGH, CRITICAL}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from app.detection.base import Detection
from app.distance.engine import DistanceEstimate, DistanceZone
from app.face.base import FaceResult
from app.tracking.iou_tracker import TrackedObject, MotionState

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    CLEAR    = 0
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

    def __lt__(self, other: "RiskLevel") -> bool:
        return self.value < other.value

    def __le__(self, other: "RiskLevel") -> bool:
        return self.value <= other.value


@dataclass
class RiskAssessment:
    """Structured output from the risk engine for one frame.

    Attributes
    ----------
    level:        Overall risk level for this frame.
    object_type:  Dominant hazardous object class name.
    distance_m:   Estimated distance to the dominant object (metres).
    zone:         DistanceZone of the dominant object.
    direction:    'left' | 'centre' | 'right' relative to frame.
    motion_state: MotionState of the dominant object.
    confidence:   Detection confidence of the dominant object.
    reason:       Short human-readable explanation.
    crowd:        True when ≥2 persons detected.
    face_results: Face recognition results for this frame.
    """
    level:        RiskLevel
    object_type:  str               = "none"
    distance_m:   Optional[float]   = None
    zone:         DistanceZone      = DistanceZone.FAR
    direction:    str               = "centre"
    motion_state: MotionState       = MotionState.UNKNOWN
    confidence:   float             = 0.0
    reason:       str               = "clear"
    crowd:        bool              = False
    face_results: List[FaceResult]  = None   # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.face_results is None:
            self.face_results = []


# Risk score boundaries (tunable)
_SCORE_THRESHOLDS = {
    RiskLevel.CRITICAL: 80.0,
    RiskLevel.HIGH:     55.0,
    RiskLevel.MEDIUM:   30.0,
    RiskLevel.LOW:      10.0,
}

_ZONE_SCORES = {
    DistanceZone.VERY_NEAR: 70.0,
    DistanceZone.NEAR:      40.0,
    DistanceZone.MEDIUM:    20.0,
    DistanceZone.FAR:        5.0,
}

_MOTION_MULTIPLIERS = {
    MotionState.APPROACHING: 1.8,
    MotionState.STATIONARY:  1.0,
    MotionState.RECEDING:    0.5,
    MotionState.UNKNOWN:     1.0,
}


class RiskEngine:
    """Compute a RiskAssessment from detections + distance + face results."""

    def __init__(self, cfg) -> None:
        raw_priority = getattr(cfg, "class_priority", {})
        # raw_priority may be a _Namespace (from YAML) or a plain dict
        if hasattr(raw_priority, "__dict__"):
            self._class_priority: dict = {
                k.replace("_", " "): v
                for k, v in raw_priority.__dict__.items()
            }
        else:
            self._class_priority = dict(raw_priority)
        self._centre_zone: float = getattr(cfg, "centre_zone_fraction", 0.5)
        self._min_confidence: float = getattr(cfg, "min_confidence", 0.40)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        tracked_objects: List[TrackedObject],
        distance_estimates: List[DistanceEstimate],
        face_results: List[FaceResult],
        frame_width: int,
    ) -> RiskAssessment:
        """Evaluate risk for a single frame and return a RiskAssessment."""

        # Filter by minimum confidence
        valid_tracked = [
            to for to in tracked_objects
            if to.detection.confidence >= self._min_confidence
        ]

        if not valid_tracked:
            # Still check for crowd via raw face results
            crowd = self._is_crowd(face_results)
            return RiskAssessment(
                level=RiskLevel.CLEAR,
                reason="no detections above confidence threshold",
                crowd=crowd,
                face_results=face_results,
            )

        # Build a distance lookup keyed on detection object identity
        dist_lookup = {id(de.detection): de for de in distance_estimates}

        # Score each tracked object
        scored: list[tuple[float, TrackedObject, DistanceEstimate]] = []
        for to in valid_tracked:
            de = dist_lookup.get(id(to.detection))
            if de is None:
                # Fallback: assume MEDIUM zone
                from app.distance.engine import DistanceEstimate as DE
                de = DE(
                    detection=to.detection,
                    distance_m=None,
                    zone=DistanceZone.MEDIUM,
                    source="unknown",
                )
            score = self._score(to, de, frame_width)
            scored.append((score, to, de))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_tracked, top_de = scored[0]

        level = self._score_to_level(top_score)
        direction = self._direction(top_tracked.detection, frame_width)
        crowd = self._is_crowd_from_tracked(valid_tracked) or self._is_crowd(face_results)

        reason = (
            f"{top_tracked.detection.class_name} at {top_de.zone.value} range "
            f"({top_tracked.motion_state.name.lower()}), "
            f"score={top_score:.1f}"
        )

        return RiskAssessment(
            level=level,
            object_type=top_tracked.detection.class_name,
            distance_m=top_de.distance_m,
            zone=top_de.zone,
            direction=direction,
            motion_state=top_tracked.motion_state,
            confidence=top_tracked.detection.confidence,
            reason=reason,
            crowd=crowd,
            face_results=face_results,
        )

    # ------------------------------------------------------------------
    def _score(
        self, tracked: TrackedObject, de: DistanceEstimate, frame_width: int
    ) -> float:
        base = _ZONE_SCORES.get(de.zone, 5.0)
        class_mult = float(self._class_priority.get(tracked.detection.class_name, 1.0))
        motion_mult = _MOTION_MULTIPLIERS.get(tracked.motion_state, 1.0)
        conf_mult = tracked.detection.confidence
        centre_mult = self._centre_multiplier(tracked.detection, frame_width)
        return base * class_mult * motion_mult * conf_mult * centre_mult

    # ------------------------------------------------------------------
    def _centre_multiplier(self, det: Detection, frame_width: int) -> float:
        """Objects near the horizontal centre of the frame score higher."""
        cx = det.center[0]
        centre_x = frame_width / 2.0
        zone_half = (self._centre_zone / 2.0) * frame_width
        distance_from_centre = abs(cx - centre_x)
        if distance_from_centre <= zone_half:
            return 1.3
        return 1.0

    # ------------------------------------------------------------------
    @staticmethod
    def _score_to_level(score: float) -> RiskLevel:
        if score >= _SCORE_THRESHOLDS[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        if score >= _SCORE_THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        if score >= _SCORE_THRESHOLDS[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        if score >= _SCORE_THRESHOLDS[RiskLevel.LOW]:
            return RiskLevel.LOW
        return RiskLevel.CLEAR

    # ------------------------------------------------------------------
    @staticmethod
    def _direction(det: Detection, frame_width: int) -> str:
        cx = det.center[0]
        left_bound = frame_width // 3
        right_bound = 2 * frame_width // 3
        if cx < left_bound:
            return "left"
        if cx > right_bound:
            return "right"
        return "centre"

    # ------------------------------------------------------------------
    @staticmethod
    def _is_crowd(face_results: List[FaceResult]) -> bool:
        from app.face.base import FaceIdentity
        return sum(
            1 for fr in face_results
            if fr.identity in (FaceIdentity.KNOWN, FaceIdentity.UNKNOWN)
        ) >= 2

    # ------------------------------------------------------------------
    @staticmethod
    def _is_crowd_from_tracked(tracked: List[TrackedObject]) -> bool:
        return sum(1 for to in tracked if to.detection.class_name == "person") >= 2
