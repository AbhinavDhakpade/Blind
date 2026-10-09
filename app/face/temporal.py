"""
app/face/temporal.py
--------------------
Temporal confirmation for face recognition.

Call update() once per face-recognition run (not once per camera frame).

* A name is reported as KNOWN only after it was seen in at least `min_hits`
  of the last `window` runs.
* UNKNOWN is reported only after an unknown face was seen in at least
  `unknown_hits` of the last `window` runs.
* Until either is confirmed, the face is PENDING: it is left out of the
  result entirely. Previously a not-yet-confirmed known face was demoted to
  UNKNOWN, which caused a false "Unknown person nearby" at start-up.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, List, Set, Tuple

from app.face.base import FaceIdentity, FaceResult, NO_FACE_RESULT


class FaceSmoother:
    def __init__(self, window: int = 5, min_hits: int = 3, unknown_hits: int = 4) -> None:
        self._min_hits = min_hits
        self._unknown_hits = unknown_hits
        # Each entry: (names recognised this run, whether any unknown face was seen)
        self._history: Deque[Tuple[Set[str], bool]] = deque(maxlen=window)

    def reset(self) -> None:
        """Forget all history (e.g. when nobody has been in view for a while)."""
        self._history.clear()

    def update(self, results: List[FaceResult]) -> List[FaceResult]:
        names_now = {
            r.name for r in results
            if r.identity == FaceIdentity.KNOWN and r.name
        }
        unknown_now = any(r.identity == FaceIdentity.UNKNOWN for r in results)
        self._history.append((names_now, unknown_now))

        counts = Counter(n for names, _ in self._history for n in names)
        confirmed = {n for n, c in counts.items() if c >= self._min_hits}
        unknown_confirmed = (
            sum(1 for _, u in self._history if u) >= self._unknown_hits
        )

        out: List[FaceResult] = []
        for r in results:
            if r.identity == FaceIdentity.KNOWN:
                if r.name in confirmed:
                    out.append(r)
                # else: pending -> dropped
            elif r.identity == FaceIdentity.UNKNOWN:
                if unknown_confirmed:
                    out.append(r)
                # else: pending -> dropped
            else:
                out.append(r)          # "no face" results pass through unchanged
        return out or [NO_FACE_RESULT]
