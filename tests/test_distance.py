"""
tests/test_distance.py
-----------------------
Test the hybrid distance estimation engine.
"""

import pytest
from app.distance.engine import DistanceEngine, DistanceZone
from app.detection.base import Detection
from app.sensors.base import SensorReading


def _make_cfg(very_near=0.8, near=1.5, medium=3.0, us_weight=0.75):
    class Cfg:
        very_near_m = very_near
        near_m = near
        medium_m = medium
        ultrasonic_weight = us_weight
    return Cfg()


def _make_det(x1=270, y1=200, x2=370, y2=400, cls="person"):
    return Detection(class_id=0, class_name=cls, confidence=0.9, bbox=(x1, y1, x2, y2))


def test_zone_very_near():
    """Sensor says 0.3 m → fused distance should be VERY_NEAR (<= 0.8 m).
    Use a large bbox (big area) so bbox estimate is also small."""
    engine = DistanceEngine(_make_cfg())
    # Large bbox → object occupies ~40% of frame → bbox_dist ≈ 1.5*sqrt(0.2/0.4) ≈ 1.06 m
    # Sensor = 0.3 m; fused = 0.75*0.3 + 0.25*1.06 ≈ 0.49 m → VERY_NEAR
    det = _make_det(x1=120, y1=60, x2=520, y2=420)
    reading = SensorReading(distance_m=0.3, valid=True)
    estimates = engine.calculate([det], reading, 640, 480)
    assert estimates[0].zone == DistanceZone.VERY_NEAR


def test_zone_near():
    """Sensor says 1.2 m → fused distance should be NEAR (0.8–1.5 m).
    Use a medium-small bbox so bbox estimate is a few metres."""
    engine = DistanceEngine(_make_cfg())
    # Bbox ~3% area → bbox_dist ≈ 1.5*sqrt(0.2/0.03) ≈ 3.9 m
    # Sensor = 1.2 m; fused = 0.75*1.2 + 0.25*3.9 ≈ 1.875 m → MEDIUM
    # Use a slightly larger bbox to bring fused closer to NEAR threshold
    # bbox ~8% area → bbox_dist ≈ 1.5*sqrt(0.2/0.08) ≈ 2.37 m
    # fused = 0.75*1.2 + 0.25*2.37 ≈ 1.49 m → NEAR (just inside boundary)
    det = _make_det(x1=220, y1=120, x2=440, y2=360)  # 220×240=52800 px, 52800/307200=17% area
    reading = SensorReading(distance_m=1.0, valid=True)
    estimates = engine.calculate([det], reading, 640, 480)
    # With 17% area: bbox_dist=1.5*sqrt(0.2/0.17)≈1.63; fused=0.75*1.0+0.25*1.63≈1.16 m → NEAR
    assert estimates[0].zone == DistanceZone.NEAR


def test_zone_far():
    engine = DistanceEngine(_make_cfg())
    # Tiny bbox (far away)
    det = _make_det(310, 230, 330, 250)
    reading = SensorReading(distance_m=5.0, valid=True)
    estimates = engine.calculate([det], reading, 640, 480)
    assert estimates[0].zone == DistanceZone.FAR


def test_invalid_sensor_falls_back_to_bbox():
    engine = DistanceEngine(_make_cfg())
    det = _make_det()
    reading = SensorReading(distance_m=None, valid=False, error="timeout")
    estimates = engine.calculate([det], reading, 640, 480)
    assert estimates[0].source == "bbox"
    assert estimates[0].distance_m is not None


def test_empty_detections():
    engine = DistanceEngine(_make_cfg())
    reading = SensorReading(distance_m=1.0, valid=True)
    estimates = engine.calculate([], reading, 640, 480)
    assert estimates == []
