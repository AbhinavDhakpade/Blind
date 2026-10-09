"""
app/main.py
-----------
Vision BOB — main application entry point.

Usage
-----
    python app/main.py                    # use config/config.yaml
    python app/main.py --simulation       # force all backends to mock
    python app/main.py --config config/config.local.yaml
    python app/main.py --debug            # verbose logging + overlay
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.utils.config import load_config
from app.utils.logger import setup_logging
from app.utils.perf import PerfMonitor
from app.camera.factory import create_camera
from app.sensors.factory import create_sensor
from app.detection.factory import create_detector
from app.face.factory import create_face_recognizer
from app.depth.factory import create_depth_estimator
from app.distance.engine import DistanceEngine
from app.tracking.iou_tracker import IoUTracker
from app.risk.engine import RiskEngine
from app.decision.engine import DecisionEngine
from app.alerts.haptic import create_haptic
from app.alerts.speech import create_speech
from app.alerts.manager import AlertManager

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision BOB – Raspberry Pi 5 Assistive Vision System"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--simulation", action="store_true",
        help="Override all hardware backends with mock implementations",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging and visual overlay",
    )
    return parser.parse_args()


def _apply_simulation_overrides(cfg) -> None:
    """Force mock backends for every hardware component."""
    cfg.camera.backend            = "mock"
    cfg.sensor.backend            = "mock"
    cfg.hardware.gpio_backend     = "mock"
    cfg.alerts.speech_backend     = "mock"
    cfg.alerts.haptic_backend     = "mock"
    cfg.models.object_detection.backend   = "mock"
    cfg.models.face_detection.backend     = "mock"
    cfg.models.face_recognition.backend   = "mock"
    if hasattr(cfg.models, "depth"):
        cfg.models.depth.enabled = False


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    if args.simulation:
        _apply_simulation_overrides(cfg)
        logger.info("=== SIMULATION MODE ===")

    if args.debug:
        cfg.logging.level = "DEBUG"
        cfg.performance.debug_overlay = True

    setup_logging(cfg)
    logger.info("Vision BOB starting up.")

    # ------------------------------------------------------------------
    # Instantiate all components
    # ------------------------------------------------------------------
    camera      = create_camera(cfg.camera)
    sensor      = create_sensor(cfg.sensor)
    detector    = create_detector(cfg.models.object_detection)
    face_rec    = create_face_recognizer(
                      cfg.models.face_detection,
                      cfg.models.face_recognition,
                  )
    depth_est   = create_depth_estimator(cfg.models.depth)
    dist_engine = DistanceEngine(cfg.distance)
    tracker     = IoUTracker(cfg.tracking)
    risk_engine = RiskEngine(cfg.risk)
    decision    = DecisionEngine()
    haptic      = create_haptic(cfg.hardware, cfg.alerts)
    speech      = create_speech(cfg.alerts)
    alert_mgr   = AlertManager(cfg.alerts, haptic, speech)
    perf        = PerfMonitor(cfg.performance)

    # ------------------------------------------------------------------
    # Graceful shutdown handler
    # ------------------------------------------------------------------
    _running = [True]

    def _shutdown(sig, frame):  # noqa: ANN001
        logger.info("Shutdown signal received.")
        _running[0] = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ------------------------------------------------------------------
    # Initialise hardware
    # ------------------------------------------------------------------
    try:
        sensor.initialize()
        alert_mgr.initialize()
        camera.initialize()
    except Exception as exc:
        logger.critical("Initialisation failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Load models (may fail for optional models; handled gracefully)
    try:
        detector.load()
    except Exception as exc:
        logger.error("Object detector failed to load: %s", exc)
        logger.warning("Falling back to mock detector.")
        from app.detection.mock_detector import MockObjectDetector
        detector = MockObjectDetector(cfg.models.object_detection)
        detector.load()

    try:
        face_rec.load()
    except Exception as exc:
        logger.error("Face recognizer failed to load: %s", exc)
        logger.warning("Falling back to mock face recognizer.")
        from app.face.mock_face_recognizer import MockFaceRecognizer
        face_rec = MockFaceRecognizer()
        face_rec.load()

    try:
        depth_est.load()
    except Exception as exc:
        logger.warning("Depth estimator failed to load (non-fatal): %s", exc)
        from app.depth.base import NullDepthEstimator
        depth_est = NullDepthEstimator()

    logger.info("All components initialised.  Entering main loop.")

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------
    try:
        while _running[0]:
            loop_start = time.monotonic()

            # 1. Capture frame
            try:
                with perf.measure("camera.capture"):
                    frame = camera.capture()
            except Exception as exc:
                logger.error("Camera capture failed: %s", exc)
                time.sleep(0.1)
                continue

            # 2. Read ultrasonic sensor (non-blocking)
            try:
                with perf.measure("sensor.measure"):
                    sensor_reading = sensor.measure()
            except Exception as exc:
                logger.warning("Sensor read failed: %s", exc)
                from app.sensors.base import SensorReading
                sensor_reading = SensorReading(
                    distance_m=None, valid=False, error=str(exc)
                )

            # 3. Object detection
            with perf.measure("detection.detect"):
                detections = detector.detect(frame.image)

            # 4. Track detections (IoU-based)
            with perf.measure("tracking.update"):
                tracked_objects = tracker.update(detections)

            # 5. Face recognition (only when persons detected)
            person_detected = any(d.class_name == "person" for d in detections)
            if person_detected:
                with perf.measure("face.process"):
                    face_results = face_rec.process(frame.image)
            else:
                from app.face.base import NO_FACE_RESULT
                face_results = [NO_FACE_RESULT]

            # 6. Distance estimation
            with perf.measure("distance.calculate"):
                distance_estimates = dist_engine.calculate(
                    detections,
                    sensor_reading,
                    frame.width,
                    frame.height,
                )

            # 7. Risk evaluation
            with perf.measure("risk.evaluate"):
                risk = risk_engine.evaluate(
                    tracked_objects,
                    distance_estimates,
                    face_results,
                    frame.width,
                )

            # 8. Decision
            with perf.measure("decision.decide"):
                action = decision.decide(risk)

            # 9. Alert
            with perf.measure("alerts.handle"):
                alert_mgr.handle(action)

            # 10. Debug overlay (dev only)
            if getattr(cfg.performance, "debug_overlay", False):
                _draw_overlay(frame.image, tracked_objects, risk, action)

            perf.tick()

    except Exception as exc:
        logger.critical("Unhandled exception in main loop: %s", exc, exc_info=True)
    finally:
        logger.info("Shutting down components.")
        try:
            detector.unload()
            face_rec.unload()
            depth_est.unload()
            camera.shutdown()
            sensor.cleanup()
            alert_mgr.cleanup()
        except Exception as exc:
            logger.error("Error during shutdown: %s", exc)
        logger.info("Vision BOB shut down cleanly.")


# ---------------------------------------------------------------------------
# Optional debug overlay
# ---------------------------------------------------------------------------

def _draw_overlay(image, tracked_objects, risk, action) -> None:
    try:
        import cv2
        for to in tracked_objects:
            x1, y1, x2, y2 = to.detection.bbox
            colour = (0, 255, 0) if risk.level.value <= 1 else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            label = f"{to.detection.class_name} {to.motion_state.name[:3]}"
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
        cv2.putText(image,
                    f"Risk: {risk.level.name} | Action: {action.action_type.name}",
                    (5, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Vision BOB Debug", image)
        cv2.waitKey(1)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
