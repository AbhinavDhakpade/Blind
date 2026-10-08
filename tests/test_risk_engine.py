"""
tests/test_risk_engine.py
-------------------------
Test risk scoring and level assignment.
"""

import pytest
from app.risk.engine import RiskEngine, RiskLevel
from app.detection.base import Detection
from app.distance.engine import DistanceEstimate, DistanceZone
from app.face.base import FaceResult, FaceIdentity, NO_FACE_RESULT
from app.tracking.iou_tracker import TrackedObject, MotionState


def _make_cfg():
    class RiskCfg:
        class_priority = {"person": 1.4, "car": 1.6}
        centre_zone_fraction = 0.5
        min_confidence = 0.40
    return RiskCfg()


def _make_tracked(cls="person", conf=0.9, bbox=(280, 150, 380, 350),
                  motion=MotionState.STATIONARY):
    det = Detection(class_id=0, class_name=cls, confidence=conf, bbox=bbox)
    return TrackedObject(track_id=1, detection=det, motion_state=motion)


def _make_de(det, zone=DistanceZone.NEAR, dist_m=1.2):
    return DistanceEstimate(detection=det, distance_m=dist_m, zone=zone, source="fused")


# ---------------------------------------------------------------------------

def test_clear_with_no_detections():
    engine = RiskEngine(_make_cfg())
    assessment = engine.evaluate([], [], [NO_FACE_RESULT], frame_width=640)
    assert assessment.level == RiskLevel.CLEAR


def test_far_object_is_low_or_clear():
    engine = RiskEngine(_make_cfg())
    tracked = _make_tracked()
    de = _make_de(tracked.detection, zone=DistanceZone.FAR, dist_m=5.0)
    assessment = engine.evaluate([tracked], [de], [NO_FACE_RESULT], frame_width=640)
    assert assessment.level in (RiskLevel.CLEAR, RiskLevel.LOW)


def test_near_person_is_at_least_medium():
    engine = RiskEngine(_make_cfg())
    tracked = _make_tracked(conf=0.9)
    de = _make_de(tracked.detection, zone=DistanceZone.NEAR, dist_m=1.2)
    assessment = engine.evaluate([tracked], [de], [NO_FACE_RESULT], frame_width=640)
    assert assessment.level >= RiskLevel.MEDIUM


def test_critical_approaching_car():
    engine = RiskEngine(_make_cfg())
    tracked = _make_tracked(cls="car", conf=0.95, motion=MotionState.APPROACHING)
    de = _make_de(tracked.detection, zone=DistanceZone.VERY_NEAR, dist_m=0.5)
    assessment = engine.evaluate([tracked], [de], [NO_FACE_RESULT], frame_width=640)
    assert assessment.level == RiskLevel.CRITICAL


def test_crowd_flag_set_for_two_persons():
    engine = RiskEngine(_make_cfg())
    t1 = _make_tracked(bbox=(100, 100, 200, 300))
    t2 = _make_tracked(bbox=(300, 100, 400, 300))
    t2.track_id = 2
    de1 = _make_de(t1.detection)
    de2 = _make_de(t2.detection)
    assessment = engine.evaluate([t1, t2], [de1, de2], [NO_FACE_RESULT], frame_width=640)
    assert assessment.crowd is True


def test_low_confidence_detection_filtered():
    engine = RiskEngine(_make_cfg())
    tracked = _make_tracked(conf=0.20)  # below min_confidence=0.40
    de = _make_de(tracked.detection, zone=DistanceZone.VERY_NEAR)
    assessment = engine.evaluate([tracked], [de], [NO_FACE_RESULT], frame_width=640)
    assert assessment.level == RiskLevel.CLEAR
