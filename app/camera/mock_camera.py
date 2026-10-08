"""
app/camera/mock_camera.py
-------------------------
Mock camera backend for simulation / unit testing.

Generates synthetic BGR frames (solid colour + frame counter overlay)
so the full pipeline can be exercised without hardware.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from app.camera.base import CameraInterface, Frame

logger = logging.getLogger(__name__)


class MockCamera(CameraInterface):
    """Synthetic camera that produces test frames with no hardware."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._frame_id = 0
        self._initialized = False

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self._initialized = True
        logger.info(
            "MockCamera initialised: %dx%d @ %d fps (simulation)",
            self._cfg.width, self._cfg.height, self._cfg.fps,
        )

    # ------------------------------------------------------------------
    def capture(self) -> Frame:
        w, h = self._cfg.width, self._cfg.height
        # Alternate between two background colours to simulate scene change
        colour = (60, 100, 60) if self._frame_id % 2 == 0 else (60, 60, 100)
        image = np.full((h, w, 3), colour, dtype=np.uint8)

        # Synthesise a "person"-like rectangle in the centre
        cx, cy = w // 2, h // 2
        cv2.rectangle(image, (cx - 40, cy - 80), (cx + 40, cy + 80), (200, 150, 50), 2)
        cv2.putText(
            image,
            f"MOCK FRAME {self._frame_id}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        self._frame_id += 1
        # Honour target FPS to avoid spinning the CPU needlessly
        time.sleep(max(0.0, 1.0 / self._cfg.fps - 0.001))
        return Frame(image=image, timestamp=time.monotonic(), frame_id=self._frame_id)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self._initialized = False
        logger.info("MockCamera shut down.")
