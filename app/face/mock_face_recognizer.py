"""
app/face/mock_face_recognizer.py
---------------------------------
Mock face recognizer for simulation / unit testing.
"""

from __future__ import annotations

import logging
import time
from typing import List

import numpy as np

from app.face.base import FaceRecognizerInterface, FaceResult, FaceIdentity

logger = logging.getLogger(__name__)


class MockFaceRecognizer(FaceRecognizerInterface):
    """Cycles through NO_FACE → UNKNOWN → KNOWN(person_001) for testing."""

    _CYCLE = [
        FaceResult(identity=FaceIdentity.NO_FACE),
        FaceResult(identity=FaceIdentity.UNKNOWN, confidence=0.30, bbox=(100, 80, 200, 200)),
        FaceResult(
            identity=FaceIdentity.KNOWN,
            name="person_001",
            confidence=0.82,
            bbox=(100, 80, 200, 200),
        ),
    ]

    def __init__(self, cfg_det=None, cfg_rec=None) -> None:
        self._counter = 0

    def load(self) -> None:
        logger.info("MockFaceRecognizer loaded (simulation).")

    def process(self, frame: np.ndarray) -> List[FaceResult]:
        result = self._CYCLE[self._counter % len(self._CYCLE)]
        self._counter += 1
        return [result]

    def unload(self) -> None:
        logger.info("MockFaceRecognizer unloaded.")
