"""
app/camera/picamera2_camera.py
------------------------------
PiCamera2 backend (Raspberry Pi 5 native stack).

Requires:
    sudo apt install -y python3-picamera2
    or: pip install picamera2   (when libcamera stack is present)
"""

from __future__ import annotations

import logging
import time

import numpy as np

from app.camera.base import CameraInterface, CameraError, Frame

logger = logging.getLogger(__name__)


class PiCamera2Camera(CameraInterface):
    """Camera implementation backed by the libcamera / picamera2 stack."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._cam = None
        self._frame_id = 0

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise CameraError(
                "picamera2 is not installed.  "
                "Run: sudo apt install -y python3-picamera2"
            ) from exc

        try:
            self._cam = Picamera2()
            config = self._cam.create_preview_configuration(
                main={
                    "size": (self._cfg.width, self._cfg.height),
                    "format": "BGR888",
                }
            )
            self._cam.configure(config)
            self._cam.set_controls({"FrameRate": float(self._cfg.fps)})
            self._cam.start()
            # Allow auto-exposure to settle
            time.sleep(1.0)
            logger.info(
                "PiCamera2 initialised: %dx%d @ %d fps",
                self._cfg.width, self._cfg.height, self._cfg.fps,
            )
        except Exception as exc:
            raise CameraError(f"Failed to initialise PiCamera2: {exc}") from exc

    # ------------------------------------------------------------------
    def capture(self) -> Frame:
        if self._cam is None:
            raise CameraError("Camera not initialised.")
        try:
            image: np.ndarray = self._cam.capture_array("main")
            self._frame_id += 1
            return Frame(image=image, timestamp=time.monotonic(), frame_id=self._frame_id)
        except Exception as exc:
            raise CameraError(f"Frame capture failed: {exc}") from exc

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
            except Exception:  # noqa: BLE001
                pass
            self._cam = None
            logger.info("PiCamera2 shut down.")
