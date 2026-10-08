"""
app/camera/base.py
------------------
Abstract camera interface and Frame dataclass.

All camera implementations must subclass CameraInterface.
The rest of the application only depends on this contract.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Frame:
    """A single captured camera frame.

    Attributes
    ----------
    image:
        BGR numpy array (H x W x 3), matching OpenCV convention.
    timestamp:
        Monotonic seconds at capture time.
    frame_id:
        Monotonically increasing frame counter.
    width, height:
        Convenience accessors.
    """
    image: np.ndarray
    timestamp: float
    frame_id: int
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self) -> None:
        self.height, self.width = self.image.shape[:2]


class CameraInterface(abc.ABC):
    """Abstract base class for all camera backends."""

    @abc.abstractmethod
    def initialize(self) -> None:
        """Open and configure the camera.  Raise CameraError on failure."""

    @abc.abstractmethod
    def capture(self) -> Frame:
        """Capture and return the next frame.  Raise CameraError on failure."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release camera resources gracefully."""

    # Context-manager support
    def __enter__(self) -> "CameraInterface":
        self.initialize()
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be captured."""
