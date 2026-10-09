"""
app/sensors/gpio_sensor.py
--------------------------
HC-SR04 ultrasonic sensor via RPi.GPIO.

The HC-SR04 echo pin outputs 5 V — Raspberry Pi GPIO is 3.3 V tolerant.
Use a 1 kΩ + 2 kΩ voltage divider (or a logic-level converter) on the
ECHO line to avoid damaging the GPIO pin.

Wiring (with voltage divider on ECHO):
  TRIG → GPIO23 (direct, 3.3 V output is fine)
  ECHO → 1kΩ → GPIO24 → 2kΩ → GND
  VCC  → 5 V
  GND  → GND
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Optional

from app.sensors.base import UltrasonicSensorInterface, SensorReading, SensorError

logger = logging.getLogger(__name__)

_SPEED_OF_SOUND_MPS = 343.0  # m/s at ~20 °C


class HC_SR04Sensor(UltrasonicSensorInterface):
    """HC-SR04 driver using RPi.GPIO."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._trig: int = cfg.trigger_pin
        self._echo: int = cfg.echo_pin
        self._max_dist: float = getattr(cfg, "max_distance_m", 4.0)
        self._timeout: float = getattr(cfg, "timeout_s", 0.04)
        self._n_samples: int = getattr(cfg, "sample_count", 3)
        self._gpio = None

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore
        except ImportError as exc:
            raise SensorError(
                "RPi.GPIO is not installed or not running on a Raspberry Pi."
            ) from exc

        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self._trig, GPIO.OUT)
        GPIO.setup(self._echo, GPIO.IN)
        GPIO.output(self._trig, False)
        time.sleep(0.05)  # sensor stabilisation
        logger.info(
            "HC-SR04 initialised: TRIG=GPIO%d ECHO=GPIO%d", self._trig, self._echo
        )

    # ------------------------------------------------------------------
    def measure(self) -> SensorReading:
        readings: list[float] = []
        for _ in range(self._n_samples):
            dist = self._single_measurement()
            if dist is not None:
                readings.append(dist)

        if not readings:
            return SensorReading(
                distance_m=None,
                valid=False,
                error="All measurements timed out.",
            )

        median_dist = statistics.median(readings)
        if median_dist > self._max_dist:
            return SensorReading(
                distance_m=median_dist,
                valid=False,
                error=f"Distance {median_dist:.2f} m exceeds max {self._max_dist} m.",
            )
        return SensorReading(distance_m=median_dist, valid=True)

    # ------------------------------------------------------------------
    def _single_measurement(self) -> Optional[float]:
        GPIO = self._gpio
        if GPIO is None:
            return None

        # Send 10 µs trigger pulse
        GPIO.output(self._trig, True)
        time.sleep(0.00001)
        GPIO.output(self._trig, False)

        deadline = time.monotonic() + self._timeout

        # Wait for echo HIGH
        t_start = time.monotonic()
        while GPIO.input(self._echo) == 0:
            t_start = time.monotonic()
            if t_start > deadline:
                return None  # timeout

        # Wait for echo LOW
        t_end = time.monotonic()
        while GPIO.input(self._echo) == 1:
            t_end = time.monotonic()
            if t_end > deadline:
                return None  # timeout

        elapsed = t_end - t_start
        distance_m = (elapsed * _SPEED_OF_SOUND_MPS) / 2.0
        return distance_m

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        if self._gpio is not None:
            try:
                self._gpio.cleanup([self._trig, self._echo])
            except Exception:  # noqa: BLE001
                pass
            self._gpio = None
            logger.info("HC-SR04 GPIO cleaned up.")
