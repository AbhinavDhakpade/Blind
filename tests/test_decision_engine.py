"""
tests/test_decision_engine.py
------------------------------
Test the priority-based decision logic.
"""

import pytest
from app.decision.engine import DecisionEngine, ActionType
from app.risk.engine import RiskAssessment, RiskLevel
from app.face.base import FaceResult, FaceIdentity, NO_FACE_RESULT
from app.tracking.iou_tracker import MotionState
from app.distance.engine import DistanceZone


def _make_assessment(**kwargs):
    defaults = dict(
        level=RiskLevel.CLEAR,
        object_type="none",
        distance_m=None,
        zone=DistanceZone.FAR,
        direction="centre",
        motion_state=MotionState.STATIONARY,
        confidence=0.9,
        reason="test",
        crowd=False,
        face_results=[NO_FACE_RESULT],
    )
    defaults.update(kwargs)
    return RiskAssessment(**defaults)


engine = DecisionEngine()


def test_clear_path_no_action():
    a = _make_assessment(level=RiskLevel.CLEAR)
    action = engine.decide(a)
    assert action.action_type == ActionType.NONE


def test_approaching_object_triggers_haptic():
    a = _make_assessment(
        level=RiskLevel.HIGH,
        motion_state=MotionState.APPROACHING,
        object_type="car",
    )
    action = engine.decide(a)
    assert action.action_type == ActionType.HAPTIC_ALERT


def test_stationary_obstacle_triggers_speech():
    a = _make_assessment(
        level=RiskLevel.MEDIUM,
        motion_state=MotionState.STATIONARY,
        object_type="chair",
    )
    action = engine.decide(a)
    assert action.action_type == ActionType.SPEECH_OBSTACLE


def test_known_person_triggers_identify():
    known = FaceResult(identity=FaceIdentity.KNOWN, name="Alice", confidence=0.9)
    a = _make_assessment(
        level=RiskLevel.LOW,
        object_type="person",
        face_results=[known],
    )
    action = engine.decide(a)
    assert action.action_type == ActionType.IDENTIFY_PERSON
    assert action.person_name == "Alice"


def test_unknown_person_triggers_identify():
    unknown = FaceResult(identity=FaceIdentity.UNKNOWN, confidence=0.3)
    a = _make_assessment(
        level=RiskLevel.LOW,
        object_type="person",
        face_results=[unknown],
    )
    action = engine.decide(a)
    assert action.action_type == ActionType.IDENTIFY_PERSON
    assert action.person_name is None


def test_haptic_takes_priority_over_speech():
    """Approaching high-risk object should produce haptic, not speech."""
    a = _make_assessment(
        level=RiskLevel.CRITICAL,
        motion_state=MotionState.APPROACHING,
        object_type="car",
    )
    action = engine.decide(a)
    assert action.action_type == ActionType.HAPTIC_ALERT


def test_low_risk_receding_no_action():
    a = _make_assessment(
        level=RiskLevel.LOW,
        motion_state=MotionState.RECEDING,
        object_type="person",
    )
    action = engine.decide(a)
    # LOW receding → not "approaching" → check obstacle (need MEDIUM+)
    # LOW is below MEDIUM threshold for speech, and person is present → IDENTIFY
    assert action.action_type in (ActionType.IDENTIFY_PERSON, ActionType.NONE)
