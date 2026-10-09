"""
app/alerts/haptic.py
---------------------
Haptic (vibration motor) alert interface and implementations.

GPIO wiring
-----------
Vibration motor (typically 5 V with transistor/MOSFET driver):
  GPIO18 (PWM0) → Base/Gate of NPN transistor (e.g. 2N2222 / BC547)
  Collector → Motor –
  Emitter   → GND
  Motor +   → 5 V
  Add a flyback diode across the motor terminals.

Do NOT connect the motor directly to the GPIO pin — motors draw
far more current than GPIO can supply.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HapticAlertInterface(abc.ABC):
    @abc.abstractmethod
    def initialize(self) -> None: ...

    @abc.abstractmethod
    def vibrate(self, pattern: dict) -> None:
        """Execute a vibration pattern dict with keys: pulses, on_ms, off_ms."""

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def cleanup(self) -> None: ...


# ---------------------------------------------------------------------------
# GPIO implementation
# ---------------------------------------------------------------------------

class GPIOHapticAlert(HapticAlertInterface):
    """Vibration motor controller via RPi.GPIO PWM."""

    def __init__(self, cfg) -> None:
        self._pin: int = cfg.vibration_pin
        self._gpio = None
        self._pwm = None

    def initialize(self) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore
        except ImportError as exc:
            raise RuntimeError("RPi.GPIO not available.") from exc

        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self._pin, GPIO.OUT)
        GPIO.output(self._pin, False)
        self._pwm = GPIO.PWM(self._pin, 100)   # 100 Hz PWM
        self._pwm.start(0)
        logger.info("GPIOHapticAlert initialised on GPIO%d.", self._pin)

    def vibrate(self, pattern: dict) -> None:
        if self._pwm is None:
            return
        pulses = int(pattern.get("pulses", 1))
        on_ms  = float(pattern.get("on_ms",  200))
        off_ms = float(pattern.get("off_ms", 100))
        for _ in range(pulses):
            self._pwm.ChangeDutyCycle(90)
            time.sleep(on_ms / 1000.0)
            self._pwm.ChangeDutyCycle(0)
            if off_ms > 0:
                time.sleep(off_ms / 1000.0)

    def stop(self) -> None:
        if self._pwm is not None:
            self._pwm.ChangeDutyCycle(0)

    def cleanup(self) -> None:
        self.stop()
        if self._pwm is not None:
            self._pwm.stop()
        if self._gpio is not None:
            try:
                self._gpio.cleanup([self._pin])
            except Exception:  # noqa: BLE001
                pass
        logger.info("GPIOHapticAlert cleaned up.")


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------

class MockHapticAlert(HapticAlertInterface):
    """Logs vibration events instead of activating hardware."""

    def __init__(self) -> None:
        self.last_pattern: dict = {}

    def initialize(self) -> None:
        logger.info("MockHapticAlert initialised (simulation).")

    def vibrate(self, pattern: dict) -> None:
        self.last_pattern = pattern
        logger.info(
            "HAPTIC [MOCK] pulses=%s on=%sms off=%sms",
            pattern.get("pulses"), pattern.get("on_ms"), pattern.get("off_ms"),
        )

    def stop(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_haptic(cfg_hardware, cfg_alerts) -> HapticAlertInterface:
    backend = str(getattr(cfg_hardware, "gpio_backend", "mock")).lower()
    if backend == "gpio":
        return GPIOHapticAlert(cfg_hardware)
    return MockHapticAlert()
