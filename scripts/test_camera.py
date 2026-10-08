"""
scripts/test_camera.py
-----------------------
Quick camera test.  Captures 30 frames and reports FPS.
"""

from __future__ import annotations
import sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from app.utils.config import load_config
from app.camera.factory import create_camera


def main():
    cfg = load_config()
    print(f"Camera backend: {cfg.camera.backend}")
    print(f"Resolution: {cfg.camera.width}x{cfg.camera.height} @ {cfg.camera.fps} fps")

    with create_camera(cfg.camera) as cam:
        t0 = time.monotonic()
        n = 30
        for i in range(n):
            frame = cam.capture()
            print(f"  Frame {frame.frame_id}: {frame.width}x{frame.height}  ts={frame.timestamp:.3f}")
        elapsed = time.monotonic() - t0
        print(f"\nCaptured {n} frames in {elapsed:.2f}s → {n/elapsed:.1f} FPS")


if __name__ == "__main__":
    main()
