"""
scripts/test_ultrasonic.py
---------------------------
Quick ultrasonic sensor test.  Reads 20 measurements and reports statistics.
"""

from __future__ import annotations
import sys, time, statistics
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from app.utils.config import load_config
from app.sensors.factory import create_sensor


def main():
    cfg = load_config()
    print(f"Sensor backend: {cfg.sensor.backend}")

    with create_sensor(cfg.sensor) as sensor:
        readings = []
        for i in range(20):
            r = sensor.measure()
            status = f"{r.distance_m:.3f} m" if r.valid else f"INVALID ({r.error})"
            print(f"  {i+1:2d}: {status}")
            if r.valid and r.distance_m is not None:
                readings.append(r.distance_m)
            time.sleep(0.1)

    if readings:
        print(f"\nValid readings: {len(readings)}/20")
        print(f"Min={min(readings):.3f} m  Max={max(readings):.3f} m  "
              f"Median={statistics.median(readings):.3f} m")
    else:
        print("No valid readings obtained.")


if __name__ == "__main__":
    main()
