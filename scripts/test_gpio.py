"""
scripts/test_gpio.py
---------------------
GPIO and vibration motor sanity test (Raspberry Pi only).
Pulses the vibration motor pin 3 times to confirm wiring.
"""

from __future__ import annotations
import sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from app.utils.config import load_config
from app.alerts.haptic import create_haptic


def main():
    cfg = load_config()
    print(f"GPIO backend: {cfg.hardware.gpio_backend}")
    print(f"Vibration pin: GPIO{cfg.hardware.vibration_pin}")

    haptic = create_haptic(cfg.hardware, cfg.alerts)
    haptic.initialize()

    patterns = [
        {"pulses": 1, "on_ms": 200, "off_ms": 0},
        {"pulses": 2, "on_ms": 150, "off_ms": 100},
        {"pulses": 3, "on_ms": 100, "off_ms": 80},
    ]

    for i, pat in enumerate(patterns):
        print(f"  Pattern {i+1}: {pat}")
        haptic.vibrate(pat)
        time.sleep(0.5)

    haptic.cleanup()
    print("GPIO test complete.")


if __name__ == "__main__":
    main()
