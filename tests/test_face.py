"""
tests/test_face.py
-------------------
Test face recognition: KNOWN, UNKNOWN, NO_FACE.
"""

import pytest
from app.face.mock_face_recognizer import MockFaceRecognizer
from app.face.base import FaceIdentity
import numpy as np


def _dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_mock_cycles_results():
    rec = MockFaceRecognizer()
    rec.load()

    results = []
    for _ in range(3):
        r = rec.process(_dummy_frame())
        results.append(r[0].identity)

    assert FaceIdentity.NO_FACE in results
    assert FaceIdentity.UNKNOWN in results
    assert FaceIdentity.KNOWN in results


def test_known_person_has_name():
    rec = MockFaceRecognizer()
    rec.load()
    # Advance to KNOWN result (index 2)
    rec._counter = 2
    result = rec.process(_dummy_frame())[0]
    assert result.identity == FaceIdentity.KNOWN
    assert result.name is not None


def test_no_face_result():
    rec = MockFaceRecognizer()
    rec.load()
    rec._counter = 0
    result = rec.process(_dummy_frame())[0]
    assert result.identity == FaceIdentity.NO_FACE
    assert result.name is None
