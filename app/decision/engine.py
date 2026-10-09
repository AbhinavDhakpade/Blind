"""
app/decision/engine.py
----------------------
Priority-based decision logic.

Priority hierarchy
------------------
  P1 – Approaching object (haptic)       → CRITICAL/HIGH + APPROACHING
  P2 – Obstacle ahead (speech)           → MEDIUM/HIGH/CRITICAL stationary
  P3 – Crowd / person (face identify)    → crowd=True or person detected
  P0 – Clear path                         → no action

Only ONE action is returned per frame.  The AlertManager is responsible
for cooldown/debouncing.  The DecisionEngine is purely stateless logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from app.risk.engine import RiskAssessment, RiskLevel
from app.tracking.iou_tracker import MotionState
from app.face.base import FaceIdentity

logger = logging.getLogger(__name__)


class ActionType(Enum):
    NONE             = auto()
    HAPTIC_ALERT     = auto()   # P1
    SPEECH_OBSTACLE  = auto()   # P2
    IDENTIFY_PERSON  = auto()   # P3


@dataclass
class DecisionAction:
    """The action selected by the decision engine for this frame.

    Attributes
    ----------
    action_type:    What kind of alert (if any) to produce.
    risk_level:     Underlying risk level.
    object_type:    Name of the triggering object class.
    distance_m:     Estimated distance.
    direction:      Spatial direction string.
    person_name:    Identified person name (P3 KNOWN) or None.
    message:        Pre-formatted human-readable message for speech.
    """
    action_type:  ActionType
    risk_level:   RiskLevel       = RiskLevel.CLEAR
    object_type:  str             = "none"
    distance_m:   Optional[float] = None
    direction:    str             = "centre"
    person_name:  Optional[str]   = None
    message:      str             = ""


class DecisionEngine:
    """Translate a RiskAssessment into a single priority-ordered DecisionAction."""

    def __init__(self) -> None:
        pass  # stateless; no config needed beyond what RiskAssessment carries

    # ------------------------------------------------------------------
    def decide(self, assessment: RiskAssessment) -> DecisionAction:
        """Return the highest-priority action for this assessment."""

        # ── Priority 1: approaching object → haptic ──────────────────
        if self._is_approaching_hazard(assessment):
            return DecisionAction(
                action_type=ActionType.HAPTIC_ALERT,
                risk_level=assessment.level,
                object_type=assessment.object_type,
                distance_m=assessment.distance_m,
                direction=assessment.direction,
                message=f"{assessment.object_type} approaching",
            )

        # ── Priority 2: obstacle ahead → speech ───────────────────────
        if self._is_obstacle_ahead(assessment):
            return DecisionAction(
                action_type=ActionType.SPEECH_OBSTACLE,
                risk_level=assessment.level,
                object_type=assessment.object_type,
                distance_m=assessment.distance_m,
                direction=assessment.direction,
                message=f"Obstacle ahead, {assessment.object_type}",
            )

        # ── Priority 3: crowd / person → face identification ──────────
        if self._is_crowd_or_person(assessment):
            person_name, face_msg = self._identify_person(assessment)
            return DecisionAction(
                action_type=ActionType.IDENTIFY_PERSON,
                risk_level=assessment.level,
                object_type=assessment.object_type,
                distance_m=assessment.distance_m,
                person_name=person_name,
                message=face_msg,
            )

        # ── Clear path ─────────────────────────────────────────────────
        return DecisionAction(
            action_type=ActionType.NONE,
            risk_level=assessment.level,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _is_approaching_hazard(a: RiskAssessment) -> bool:
        return (
            a.level >= RiskLevel.MEDIUM
            and a.motion_state == MotionState.APPROACHING
        )

    @staticmethod
    def _is_obstacle_ahead(a: RiskAssessment) -> bool:
        return a.level >= RiskLevel.MEDIUM

    @staticmethod
    def _is_crowd_or_person(a: RiskAssessment) -> bool:
        return (
            a.crowd
            or a.object_type == "person"
            or any(
                fr.identity in (FaceIdentity.KNOWN, FaceIdentity.UNKNOWN)
                for fr in a.face_results
            )
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _identify_person(a: RiskAssessment) -> tuple[Optional[str], str]:
        known = [fr for fr in a.face_results if fr.identity == FaceIdentity.KNOWN]
        if known:
            name = known[0].name or "someone"
            return name, f"Hello {name}"

        has_unknown = any(
            fr.identity == FaceIdentity.UNKNOWN for fr in a.face_results
        )
        if has_unknown or a.crowd:
            return None, "Unknown person nearby"

        return None, "Person ahead"
