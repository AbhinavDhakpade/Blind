"""
app/distance/engine.py
-----------------------
Hybrid distance estimation.

Strategy
--------
1. Ultrasonic reading – reliable metric distance but a single point in space.
2. Bounding-box area  – relative estimate; larger bbox → object is nearer.
3. Fusion             – weighted blend, configurable via ultrasonic_weight.

The ultrasonic sensor reports the distance to *whatever is directly in front*
of the sensor.  When multiple objects are detected, the sensor distance is
attributed to the detection whose bounding-box centre is closest to the image
centre (assuming the sensor is aligned with the camera optical axis).

DistanceZone
    FAR       : > medium_m
    MEDIUM    : near_m < d <= medium_m
    NEAR      : very_near_m < d <= near_m
    VERY_NEAR : d <= very_near_m
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np

from app.detection.base import Detection
from app.sensors.base import SensorReading

logger = logging.getLogger(__name__)


class DistanceZone(Enum):
    FAR       = "far"
    MEDIUM    = "medium"
    NEAR      = "near"
    VERY_NEAR = "very_near"


@dataclass
class DistanceEstimate:
    """Distance information for a single detection.

    Attributes
    ----------
    detection:      The associated Detection object.
    distance_m:     Estimated distance in metres (may be None).
    zone:           Categorical distance zone.
    source:         'ultrasonic', 'bbox', or 'fused'.
    """
    detection: Detection
    distance_m: Optional[float]
    zone: DistanceZone
    source: str


class DistanceEngine:
    """Combine ultrasonic + bounding-box information to estimate distances."""

    def __init__(self, cfg) -> None:
        self._very_near_m: float = getattr(cfg, "very_near_m", 0.80)
        self._near_m:      float = getattr(cfg, "near_m",      1.50)
        self._medium_m:    float = getattr(cfg, "medium_m",    3.00)
        self._us_weight:   float = getattr(cfg, "ultrasonic_weight", 0.75)

        # Calibration constant for bbox-based estimate.
        # Treat a detection occupying 20% of the frame area as being ~1.5 m away.
        self._bbox_ref_area_fraction = 0.20
        self._bbox_ref_distance_m    = 1.50

    # ------------------------------------------------------------------
    def calculate(
        self,
        detections: List[Detection],
        sensor_reading: SensorReading,
        frame_width: int,
        frame_height: int,
    ) -> List[DistanceEstimate]:
        """Produce a DistanceEstimate for each detection."""
        frame_area = max(1, frame_width * frame_height)
        frame_cx = frame_width // 2

        # Determine which detection is most likely targeted by the sensor
        # (centre-most object in the horizontal axis)
        if detections and sensor_reading.valid and sensor_reading.distance_m is not None:
            distances_from_centre = [abs(det.center[0] - frame_cx) for det in detections]
            sensor_target_idx = int(np.argmin(distances_from_centre))
        else:
            sensor_target_idx = -1

        results: List[DistanceEstimate] = []
        for i, det in enumerate(detections):
            # --- Bounding-box estimate ---
            area_fraction = det.area / frame_area
            if area_fraction > 0:
                bbox_dist = self._bbox_ref_distance_m * (
                    self._bbox_ref_area_fraction / area_fraction
                ) ** 0.5
            else:
                bbox_dist = self._medium_m  # fallback

            # --- Fusion ---
            if (
                sensor_reading.valid
                and sensor_reading.distance_m is not None
                and i == sensor_target_idx
            ):
                us_dist = sensor_reading.distance_m
                fused = (
                    self._us_weight * us_dist
                    + (1.0 - self._us_weight) * bbox_dist
                )
                dist_m = round(fused, 3)
                source = "fused"
            else:
                dist_m = round(bbox_dist, 3)
                source = "bbox"

            zone = self._classify_zone(dist_m)
            results.append(
                DistanceEstimate(
                    detection=det,
                    distance_m=dist_m,
                    zone=zone,
                    source=source,
                )
            )
        return results

    # ------------------------------------------------------------------
    def _classify_zone(self, distance_m: float) -> DistanceZone:
        if distance_m <= self._very_near_m:
            return DistanceZone.VERY_NEAR
        if distance_m <= self._near_m:
            return DistanceZone.NEAR
        if distance_m <= self._medium_m:
            return DistanceZone.MEDIUM
        return DistanceZone.FAR
