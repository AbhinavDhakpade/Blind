"""
app/sensors/mock_sensor.py
--------------------------
Mock ultrasonic sensor for simulation / development.

Generates a configurable distance reading sequence so the risk engine
can be exercised without real hardware.
"""

from __future__ import annotations

import logging
import math
import time

from app.sensors.base import UltrasonicSensorInterface, SensorReading

logger = logging.getLogger(__name__)


class MockUltrasonicSensor(UltrasonicSensorInterface):
    """Generates a sinusoidally varying distance (0.5 m … 3.5 m) to simulate
    an object moving towards and away from the sensor."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._start_time: float | None = None
        self._period_s = 10.0   # full oscillation period
        self._min_m = 0.5
        self._max_m = 3.5
        self._fail_after: int | None = None  # set to N to simulate timeout
        self._call_count = 0

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self._start_time = time.monotonic()
        logger.info("MockUltrasonicSensor initialised (simulation).")

    # ------------------------------------------------------------------
    def measure(self) -> SensorReading:
        self._call_count += 1

        if self._fail_after is not None and self._call_count > self._fail_after:
            return SensorReading(
                distance_m=None,
                valid=False,
                error="Simulated sensor timeout.",
            )

        elapsed = time.monotonic() - (self._start_time or 0.0)
        phase = (elapsed % self._period_s) / self._period_s  # 0 … 1
        dist = self._min_m + (self._max_m - self._min_m) * 0.5 * (1.0 + math.sin(2 * math.pi * phase))
        return SensorReading(distance_m=round(dist, 3), valid=True)

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        logger.info("MockUltrasonicSensor cleaned up.")
