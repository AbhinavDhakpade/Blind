"""
app/utils/gpio.py
-----------------
Single place that imports the GPIO library and explains failures.

Raspberry Pi 5 uses the RP1 I/O chip, which the classic ``RPi.GPIO`` package
does NOT support.  The drop-in replacement ``rpi-lgpio`` provides the same
``import RPi.GPIO as GPIO`` API on top of ``lgpio``, so the rest of the code
is unchanged.  Install it with apt (see SETUP.md), not pip alongside RPi.GPIO.
"""

from __future__ import annotations

from types import ModuleType


class GPIOUnavailableError(RuntimeError):
    """RPi.GPIO (or its rpi-lgpio replacement) cannot be used."""


_PI5_HINT = (
    "On Raspberry Pi 5 the classic RPi.GPIO package does not work.\n"
    "Fix:\n"
    "  sudo apt remove -y python3-rpi.gpio\n"
    "  sudo apt install -y python3-rpi-lgpio\n"
    "  pip uninstall -y RPi.GPIO        # if it was pip-installed in your venv\n"
    "and create the venv with: python3 -m venv --system-site-packages venv"
)


def import_gpio() -> ModuleType:
    """Import and return the ``RPi.GPIO`` module (real or rpi-lgpio)."""
    try:
        import RPi.GPIO as GPIO  # type: ignore
    except ImportError as exc:
        raise GPIOUnavailableError(
            "RPi.GPIO not importable (not on a Raspberry Pi, or not installed).\n"
            + _PI5_HINT
        ) from exc
    return GPIO


def setup_bcm(GPIO: ModuleType) -> None:
    """``GPIO.setmode(BCM)`` with a helpful error on unsupported platforms."""
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
    except RuntimeError as exc:
        raise GPIOUnavailableError(f"GPIO initialisation failed: {exc}\n{_PI5_HINT}") from exc
