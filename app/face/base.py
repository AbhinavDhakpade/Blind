"""
app/face/base.py
----------------
Face recognition interface contract.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


class FaceIdentity(Enum):
    """Classification outcome for a single detected face."""
    KNOWN   = auto()
    UNKNOWN = auto()
    NO_FACE = auto()


@dataclass
class FaceResult:
    """Result for one face found in a frame.

    Attributes
    ----------
    identity:   KNOWN / UNKNOWN / NO_FACE
    name:       Person label when identity is KNOWN, else None.
    confidence: Similarity score (cosine) in [0, 1]; None when NO_FACE.
    bbox:       (x1, y1, x2, y2) face bounding box; None when NO_FACE.
    """
    identity: FaceIdentity
    name: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[Tuple[int, int, int, int]] = None


# Singleton indicating no face was detected
NO_FACE_RESULT = FaceResult(identity=FaceIdentity.NO_FACE)


class FaceRecognizerInterface(abc.ABC):
    """Abstract base class for face recognition pipelines."""

    @abc.abstractmethod
    def load(self) -> None:
        """Load models and embedding database."""

    @abc.abstractmethod
    def process(self, frame: np.ndarray) -> List[FaceResult]:
        """Detect all faces in *frame* and return recognition results."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Free model resources."""

    def __enter__(self) -> "FaceRecognizerInterface":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
