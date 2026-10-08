"""
app/utils/perf.py
-----------------
Lightweight performance counter.

Usage
-----
    from app.utils.perf import PerfMonitor

    perf = PerfMonitor(cfg.performance)

    with perf.measure("detection"):
        detections = detector.detect(frame)

    perf.tick()          # call once per main loop iteration

    perf.report()        # prints aggregated stats every N seconds
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PerfMonitor:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._enabled = getattr(cfg, "log_fps", True) or getattr(cfg, "log_latency", True)
        self._interval = getattr(cfg, "stats_interval_s", 10)

        self._frame_times: deque[float] = deque(maxlen=60)
        self._latencies: dict[str, deque[float]] = {}
        self._last_report = time.monotonic()

    # ------------------------------------------------------------------
    def tick(self) -> None:
        """Call once per processed frame."""
        if not self._enabled:
            return
        self._frame_times.append(time.monotonic())
        now = time.monotonic()
        if now - self._last_report >= self._interval:
            self.report()
            self._last_report = now

    @contextmanager
    def measure(self, label: str):
        """Context manager that records wall-clock duration for *label*."""
        if not self._enabled:
            yield
            return
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if label not in self._latencies:
                self._latencies[label] = deque(maxlen=60)
            self._latencies[label].append(elapsed_ms)

    def report(self) -> None:
        """Log aggregated performance statistics."""
        if len(self._frame_times) >= 2:
            intervals = [
                self._frame_times[i] - self._frame_times[i - 1]
                for i in range(1, len(self._frame_times))
            ]
            avg_fps = 1.0 / (sum(intervals) / len(intervals)) if intervals else 0.0
            logger.info("[PERF] FPS: %.1f", avg_fps)

        for label, times in self._latencies.items():
            if times:
                avg = sum(times) / len(times)
                logger.info("[PERF] %-30s avg %.1f ms", label, avg)

        if getattr(self._cfg, "log_cpu", False):
            try:
                import psutil
                logger.info("[PERF] CPU: %.1f%%  MEM: %.1f MB",
                            psutil.cpu_percent(),
                            psutil.Process().memory_info().rss / 1e6)
            except ImportError:
                pass
