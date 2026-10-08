"""
app/depth/onnx_depth.py
------------------------
Optional MiDaS-Small ONNX depth estimator.

WARNING: ~800 ms per frame on Raspberry Pi 5 CPU at 256×256.
Only enable if you have an accelerator or accept very low FPS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.depth.base import DepthEstimatorInterface
from app.detection.base import ModelLoadError

logger = logging.getLogger(__name__)


class ONNXDepthEstimator(DepthEstimatorInterface):

    def __init__(self, cfg) -> None:
        self._model_path = Path(cfg.path)
        self._input_size = tuple(getattr(cfg, "input_size", [256, 256]))
        self._session = None

    def load(self) -> None:
        if not self._model_path.exists():
            raise ModelLoadError(
                f"Depth model not found: {self._model_path}\n"
                "See models/depth/README.md."
            )
        try:
            import onnxruntime as ort  # type: ignore
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.warning(
                "Depth model loaded — NOTE: MiDaS-Small is ~800 ms/frame on Pi 5 CPU. "
                "Disable in config if performance is unacceptable."
            )
        except Exception as exc:
            raise ModelLoadError(f"Failed to load depth model: {exc}") from exc

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self._session is None:
            return None
        iw, ih = self._input_size
        rgb = cv2.cvtColor(cv2.resize(frame, (iw, ih)), cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: blob})[0][0]
        # Normalise to [0, 1]
        mn, mx = output.min(), output.max()
        if mx > mn:
            output = (output - mn) / (mx - mn)
        return output.astype(np.float32)

    def unload(self) -> None:
        self._session = None


class MockDepthEstimator(DepthEstimatorInterface):
    """Returns a gradient depth map for testing."""

    def __init__(self, cfg=None) -> None:
        pass

    def load(self) -> None:
        logger.info("MockDepthEstimator loaded.")

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        depth = np.linspace(0, 1, w, dtype=np.float32)
        return np.tile(depth, (h, 1))

    def unload(self) -> None:
        pass
