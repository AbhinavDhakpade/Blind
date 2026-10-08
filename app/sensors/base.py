"""
app/sensors/base.py
-------------------
Ultrasonic sensor interface contract.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional


@dataclass
class SensorReading:
    """Result of a single ultrasonic distance measurement.

    Attributes
    ----------
    distance_m:
        Measured distance in metres.  ``None`` indicates a failed or
        timed-out measurement.
    valid:
        ``True`` when the measurement can be trusted.
    error:
        Human-readable error description when ``valid`` is ``False``.
    """
    distance_m: Optional[float]
    valid: bool
    error: str = ""


class UltrasonicSensorInterface(abc.ABC):
    """Abstract base class for all ultrasonic sensor backends."""

    @abc.abstractmethod
    def initialize(self) -> None:
        """Configure GPIO / hardware.  Raise SensorError on failure."""

    @abc.abstractmethod
    def measure(self) -> SensorReading:
        """Perform a distance measurement and return a SensorReading."""

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Release hardware resources."""

    def __enter__(self) -> "UltrasonicSensorInterface":
        self.initialize()
        return self

    def __exit__(self, *_) -> None:
        self.cleanup()


class SensorError(RuntimeError):
    """Raised on unrecoverable sensor failures."""
