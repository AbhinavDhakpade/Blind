"""
tests/test_tracker.py
----------------------
Test IoU-based object tracker.
"""

import pytest
from app.tracking.iou_tracker import IoUTracker, MotionState
from app.detection.base import Detection


class _Cfg:
    max_age_frames = 5
    min_iou = 0.30
    history_len = 6
    approach_threshold = 0.04


def _det(x1, y1, x2, y2, cls="person", conf=0.9):
    return Detection(class_id=0, class_name=cls, confidence=conf, bbox=(x1, y1, x2, y2))


def test_new_detection_creates_track():
    tracker = IoUTracker(_Cfg())
    dets = [_det(100, 100, 200, 300)]
    tracked = tracker.update(dets)
    assert len(tracked) == 1
    assert tracked[0].track_id == 0


def test_same_object_keeps_id():
    tracker = IoUTracker(_Cfg())
    dets1 = [_det(100, 100, 200, 300)]
    tracked1 = tracker.update(dets1)
    tid = tracked1[0].track_id

    # Slightly moved bbox (high IoU)
    dets2 = [_det(105, 105, 205, 305)]
    tracked2 = tracker.update(dets2)
    assert tracked2[0].track_id == tid


def test_disappearing_object_removed_after_max_age():
    tracker = IoUTracker(_Cfg())
    tracker.update([_det(100, 100, 200, 300)])
    # No detections for max_age+1 frames
    for _ in range(_Cfg.max_age_frames + 1):
        result = tracker.update([])
    assert result == []


def test_approaching_detected():
    tracker = IoUTracker(_Cfg())
    # Simulate growing bounding box over 6 frames
    for i in range(6):
        delta = i * 15
        tracker.update([_det(
            200 - delta, 150 - delta,
            440 + delta, 330 + delta
        )])
    # Last update
    result = tracker.update([_det(170, 120, 470, 360)])
    assert result[0].motion_state == MotionState.APPROACHING
