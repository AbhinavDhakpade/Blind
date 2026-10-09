"""
app/sensors/gpio_sensor.py
--------------------------
HC-SR04 ultrasonic sensor via RPi.GPIO API (``rpi-lgpio`` on Raspberry Pi 5).

The HC-SR04 echo pin outputs 5 V — Raspberry Pi GPIO is 3.3 V tolerant.
Use a 1 kΩ + 2 kΩ voltage divider (or a logic-level converter) on the
ECHO line to avoid damaging the GPIO pin.

Wiring (with voltage divider on ECHO):
  TRIG → GPIO23 (direct, 3.3 V output is fine)
  ECHO → 1kΩ → GPIO24 → 2kΩ → GND
  VCC  → 5 V
  GND  → GND

Threading model
---------------
Ranging takes tens of milliseconds and the sensor needs ≥ 60 ms between
pings.  Doing that inside the camera/inference loop would stall every frame,
so a background thread pings continuously and ``measure()`` just returns the
latest reading (marked invalid if it is older than ``stale_after_s``).
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from typing import Optional, Tuple

from app.sensors.base import UltrasonicSensorInterface, SensorReading, SensorError
from app.utils.gpio import import_gpio, setup_bcm, GPIOUnavailableError

logger = logging.getLogger(__name__)

_SPEED_OF_SOUND_MPS = 343.0  # m/s at ~20 °C

# The echo line goes high within ~1 ms of the trigger; if it has not after
# this long the sensor is not responding (unplugged / wiring fault).
_ECHO_START_TIMEOUT_S = 0.02


class HC_SR04Sensor(UltrasonicSensorInterface):
    """HC-SR04 driver with a background ranging thread."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._trig: int = cfg.trigger_pin
        self._echo: int = cfg.echo_pin
        self._max_dist: float = getattr(cfg, "max_distance_m", 4.0)
        self._timeout: float = getattr(cfg, "timeout_s", 0.04)
        self._n_samples: int = getattr(cfg, "sample_count", 3)
        self._ping_gap_s: float = getattr(cfg, "ping_interval_s", 0.06)
        self._stale_after_s: float = getattr(cfg, "stale_after_s", 0.6)
        self._gpio = None

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[SensorReading, float]] = None
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        try:
            GPIO = import_gpio()
            setup_bcm(GPIO)
        except GPIOUnavailableError as exc:
            raise SensorError(str(exc)) from exc

        self._gpio = GPIO
        GPIO.setup(self._trig, GPIO.OUT)
        GPIO.setup(self._echo, GPIO.IN)
        GPIO.output(self._trig, False)
        time.sleep(0.05)  # sensor stabilisation
        logger.info(
            "HC-SR04 initialised: TRIG=GPIO%d ECHO=GPIO%d", self._trig, self._echo
        )
        self._start_thread()

    # ------------------------------------------------------------------
    def _start_thread(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="hcsr04-ranging", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                reading = self._ranging_cycle()
            except Exception as exc:  # never let the thread die silently
                logger.error("Ultrasonic ranging error: %s", exc)
                reading = SensorReading(distance_m=None, valid=False, error=str(exc))
            with self._lock:
                self._latest = (reading, time.monotonic())

            if reading.valid:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures == 50:
                    logger.warning(
                        "Ultrasonic sensor: 50 consecutive failed cycles (%s). "
                        "Check wiring / voltage divider.", reading.error,
                    )

    # ------------------------------------------------------------------
    def _ranging_cycle(self) -> SensorReading:
        """Take ``sample_count`` pings (spaced ≥ ping gap) and return the median."""
        readings: list[float] = []
        for _ in range(self._n_samples):
            if self._stop.is_set():
                break
            dist = self._single_measurement()
            if dist is not None:
                readings.append(dist)
            # Let echoes die down; avoids reading our own previous ping.
            self._stop.wait(self._ping_gap_s)

        if not readings:
            return SensorReading(
                distance_m=None, valid=False, error="All measurements timed out."
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
    def measure(self) -> SensorReading:
        """Return the latest reading without blocking."""
        with self._lock:
            latest = self._latest
        if latest is None:
            return SensorReading(distance_m=None, valid=False, error="No reading yet.")
        reading, stamp = latest
        age = time.monotonic() - stamp
        if age > self._stale_after_s:
            return SensorReading(
                distance_m=None,
                valid=False,
                error=f"Sensor reading stale ({age:.2f} s old).",
            )
        return reading

    # ------------------------------------------------------------------
    def _single_measurement(self) -> Optional[float]:
        GPIO = self._gpio
        if GPIO is None:
            return None
        clock = time.perf_counter

        # 10 µs trigger pulse
        GPIO.output(self._trig, True)
        time.sleep(0.00001)
        GPIO.output(self._trig, False)

        # Wait for echo HIGH
        start_deadline = clock() + _ECHO_START_TIMEOUT_S
        t_start = clock()
        while GPIO.input(self._echo) == 0:
            t_start = clock()
            if t_start > start_deadline:
                return None

        # Wait for echo LOW (pulse width ∝ distance)
        end_deadline = t_start + self._timeout
        t_end = clock()
        while GPIO.input(self._echo) == 1:
            t_end = clock()
            if t_end > end_deadline:
                return None

        return ((t_end - t_start) * _SPEED_OF_SOUND_MPS) / 2.0

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._gpio is not None:
            try:
                self._gpio.cleanup([self._trig, self._echo])
            except Exception:  # noqa: BLE001
                pass
            self._gpio = None
            logger.info("HC-SR04 GPIO cleaned up.")
