"""
app/detection/base.py
---------------------
Defines the Detection dataclass and the ObjectDetectorInterface.

The rest of the application depends only on these types, never on YOLO
or any other inference backend directly.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class Detection:
    """A single object detection result.

    Attributes
    ----------
    class_id:   Integer class index.
    class_name: Human-readable class label (e.g. "person").
    confidence: Detection confidence in [0, 1].
    bbox:       Bounding box as (x1, y1, x2, y2) pixel coordinates.
    center:     (cx, cy) centre pixel.
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    center: Tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])


class ObjectDetectorInterface(abc.ABC):
    """Abstract base class for object detectors."""

    @abc.abstractmethod
    def load(self) -> None:
        """Load the model into memory.  Raise ModelLoadError on failure."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on *frame* (BGR H×W×3) and return detections."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Free model resources."""

    def __enter__(self) -> "ObjectDetectorInterface":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()


class ModelLoadError(RuntimeError):
    """Raised when a model file cannot be found or loaded."""
