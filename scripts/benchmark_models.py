"""
scripts/benchmark_models.py
----------------------------
Measure inference latency for all loaded models.
Reports min / mean / max over N iterations.

Usage
-----
    python scripts/benchmark_models.py [--iterations 50]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from app.utils.config import load_config


def benchmark(name: str, fn, iterations: int, warmup: int = 3):
    # Warm-up
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            pass

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            continue
        times.append((time.perf_counter() - t0) * 1000.0)

    if not times:
        print(f"  {name}: FAILED (no successful iterations)")
        return

    print(
        f"  {name:40s}  "
        f"min={min(times):6.1f}ms  "
        f"avg={sum(times)/len(times):6.1f}ms  "
        f"max={max(times):6.1f}ms  "
        f"({len(times)} runs)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config()
    n = args.iterations

    dummy_640 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_112 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    print(f"\n=== Vision BOB Model Benchmarks ({n} iterations) ===\n")

    # Object detection
    try:
        from app.detection.factory import create_detector
        det = create_detector(cfg.models.object_detection)
        det.load()
        benchmark("Object detection (YOLOv8n ONNX)", lambda: det.detect(dummy_640), n)
        det.unload()
    except Exception as e:
        print(f"  Object detection: SKIPPED ({e})")

    # Face recognition
    try:
        from app.face.factory import create_face_recognizer
        fr = create_face_recognizer(cfg.models.face_detection, cfg.models.face_recognition)
        fr.load()
        benchmark("Face recognition pipeline", lambda: fr.process(dummy_640), n)
        fr.unload()
    except Exception as e:
        print(f"  Face recognition: SKIPPED ({e})")

    # Depth (optional)
    try:
        from app.depth.factory import create_depth_estimator
        de = create_depth_estimator(cfg.models.depth)
        de.load()
        benchmark("Depth estimation (MiDaS-Small)", lambda: de.estimate(dummy_640), n)
        de.unload()
    except Exception as e:
        print(f"  Depth estimation: SKIPPED ({e})")

    print("\nNote: These benchmarks run on the current machine.")
    print("Raspberry Pi 5 performance will differ. See docs/PERFORMANCE.md.")


if __name__ == "__main__":
    main()
