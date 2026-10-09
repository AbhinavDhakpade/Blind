"""
app/face/temporal.py
--------------------
Temporal confirmation for face recognition.

A name is only reported as KNOWN once it has been seen in at least
`min_hits` of the last `window` frames. Until then the face is reported
as UNKNOWN, so a single lucky frame can never trigger "Hello <name>".
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from typing import Deque, List, Set

from app.face.base import FaceIdentity, FaceResult


class FaceSmoother:
    def __init__(self, window: int = 5, min_hits: int = 3) -> None:
        self._min_hits = min_hits
        self._history: Deque[Set[str]] = deque(maxlen=window)

    def update(self, results: List[FaceResult]) -> List[FaceResult]:
        # Names recognised in this frame (each name counted once per frame).
        names_now = {
            r.name for r in results
            if r.identity == FaceIdentity.KNOWN and r.name
        }
        self._history.append(names_now)

        # Names seen often enough across the recent frames.
        counts = Counter(n for frame_names in self._history for n in frame_names)
        confirmed = {n for n, c in counts.items() if c >= self._min_hits}

        smoothed: List[FaceResult] = []
        for r in results:
            if r.identity == FaceIdentity.KNOWN and r.name not in confirmed:
                smoothed.append(replace(r, identity=FaceIdentity.UNKNOWN, name=None))
            else:
                smoothed.append(r)
        return smoothed