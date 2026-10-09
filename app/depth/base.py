"""
app/depth/base.py
-----------------
Optional monocular depth estimator interface.

IMPORTANT
---------
Monocular depth estimation on Raspberry Pi 5 is expensive.
MiDaS-Small at 256×256 takes ~800 ms per frame on Pi 5 CPU — far too slow
for real-time use as the primary depth source.

This module exists as an optional enhancement for offline analysis or
when a dedicated accelerator (Hailo-8 NPU) is available.

At runtime the depth estimator can be DISABLED in config:
    models.depth.enabled: false

When disabled, the DistanceEngine uses ultrasonic + bounding-box fusion only.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np


class DepthEstimatorInterface(abc.ABC):
    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return a normalised depth map (H×W float32) or None."""

    @abc.abstractmethod
    def unload(self) -> None: ...


class NullDepthEstimator(DepthEstimatorInterface):
    """No-op implementation used when depth is disabled."""

    def load(self) -> None:
        pass

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        return None

    def unload(self) -> None:
        pass
