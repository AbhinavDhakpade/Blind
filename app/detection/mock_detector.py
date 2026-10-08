"""
app/detection/mock_detector.py
------------------------------
Synthetic object detector for simulation / unit testing.

Returns a deterministic sequence of detections so the full pipeline
can be exercised without loading a real model.
"""

from __future__ import annotations

import logging
import math
import time
from typing import List

import numpy as np

from app.detection.base import ObjectDetectorInterface, Detection

logger = logging.getLogger(__name__)


class MockObjectDetector(ObjectDetectorInterface):
    """Returns synthetic detections that simulate an approaching person."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._call_count = 0
        self._start_time: float | None = None

    def load(self) -> None:
        self._start_time = time.monotonic()
        logger.info("MockObjectDetector loaded (simulation).")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._call_count += 1
        h, w = frame.shape[:2]

        elapsed = time.monotonic() - (self._start_time or 0.0)
        # Simulate a person that grows in bounding-box size (approaching)
        scale = 0.15 + 0.05 * abs(math.sin(elapsed * 0.5))

        box_w = int(w * scale)
        box_h = int(h * scale * 2)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.88,
                bbox=(x1, y1, x2, y2),
            )
        ]

    def unload(self) -> None:
        logger.info("MockObjectDetector unloaded.")
