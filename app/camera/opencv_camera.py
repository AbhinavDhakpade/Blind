"""
app/camera/opencv_camera.py
---------------------------
OpenCV VideoCapture backend.

Useful as a fallback on Raspberry Pi when using a USB webcam,
or for development/testing on a laptop.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from app.camera.base import CameraInterface, CameraError, Frame

logger = logging.getLogger(__name__)


class OpenCVCamera(CameraInterface):
    """Camera implementation backed by OpenCV's VideoCapture."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        device = getattr(self._cfg, "device_index", 0)
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise CameraError(f"OpenCV could not open camera device {device!r}.")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)

        logger.info(
            "OpenCV camera initialised: %dx%d @ %d fps (device=%s)",
            self._cfg.width, self._cfg.height, self._cfg.fps, device,
        )

    # ------------------------------------------------------------------
    def capture(self) -> Frame:
        if self._cap is None or not self._cap.isOpened():
            raise CameraError("Camera not initialised or already released.")
        ret, image = self._cap.read()
        if not ret or image is None:
            raise CameraError("Failed to read frame from OpenCV VideoCapture.")
        self._frame_id += 1
        return Frame(image=image, timestamp=time.monotonic(), frame_id=self._frame_id)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("OpenCV camera released.")
