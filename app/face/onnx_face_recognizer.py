"""
app/face/onnx_face_recognizer.py
---------------------------------
Face recognition pipeline using:

  Detection : OpenCV YuNet (lightweight, ~1 ms on CPU)
  Embedding : MobileFaceNet (ONNX, ~10–15 ms on Pi 5)
  Matching  : cosine similarity against a pre-built .pkl embedding database

Inference chain
---------------
  BGR frame
    → YuNet face detector (OpenCV DNN)  →  face bounding boxes
    → align + crop (112×112)
    → MobileFaceNet ONNX                →  128-d embedding
    → cosine similarity vs. database    →  name / UNKNOWN
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from app.face.face_alignment import align_face

from app.face.base import FaceRecognizerInterface, FaceResult, FaceIdentity, NO_FACE_RESULT
from app.detection.base import ModelLoadError

logger = logging.getLogger(__name__)


class ONNXFaceRecognizer(FaceRecognizerInterface):

    def __init__(self, cfg_face_detection, cfg_face_recognition) -> None:
        self._det_cfg = cfg_face_detection
        self._rec_cfg = cfg_face_recognition

        self._det_model_path = Path(cfg_face_detection.path)
        self._rec_model_path = Path(cfg_face_recognition.path)
        self._embeddings_path = Path(cfg_face_recognition.embeddings_path)

        self._det_conf: float = getattr(cfg_face_detection, "confidence_threshold", 0.6)
        self._det_nms: float  = getattr(cfg_face_detection, "nms_threshold", 0.3)
        self._sim_threshold: float = getattr(cfg_face_recognition, "similarity_threshold", 0.55)
        self._input_size: tuple = tuple(getattr(cfg_face_recognition, "input_size", [112, 112]))

        self._face_detector = None     # cv2.FaceDetectorYN
        self._rec_session   = None     # onnxruntime session
        self._db: Dict[str, np.ndarray] = {}   # name → mean embedding

    # ------------------------------------------------------------------
    def load(self) -> None:
        self._load_detector()
        self._load_recogniser()
        self._load_database()

    # ------------------------------------------------------------------
    def _load_detector(self) -> None:
        if not self._det_model_path.exists():
            raise ModelLoadError(
                f"Face detection model not found: {self._det_model_path}\n"
                "See models/face_detection/README.md."
            )
        try:
            self._face_detector = cv2.FaceDetectorYN.create(
                str(self._det_model_path),
                "",
                (320, 320),
                self._det_conf,
                self._det_nms,
                top_k=10,
            )
            logger.info("YuNet face detector loaded: %s", self._det_model_path.name)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load YuNet detector: {exc}") from exc

    # ------------------------------------------------------------------
    def _load_recogniser(self) -> None:
        if not self._rec_model_path.exists():
            raise ModelLoadError(
                f"Face recognition model not found: {self._rec_model_path}\n"
                "See models/face_recognition/README.md."
            )
        try:
            import onnxruntime as ort  # type: ignore
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._rec_session = ort.InferenceSession(
                str(self._rec_model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info("MobileFaceNet ONNX recogniser loaded: %s", self._rec_model_path.name)
        except ImportError as exc:
            raise ModelLoadError("onnxruntime not installed.") from exc
        except Exception as exc:
            raise ModelLoadError(f"Failed to load face recogniser: {exc}") from exc

    # ------------------------------------------------------------------
    def _load_database(self) -> None:
        if not self._embeddings_path.exists():
            logger.warning(
                "Face embedding database not found at %s. "
                "All faces will be classified as UNKNOWN. "
                "Run: python scripts/enroll_faces.py",
                self._embeddings_path,
            )
            self._db = {}
            return
        with self._embeddings_path.open("rb") as fh:
            self._db = pickle.load(fh)
        logger.info("Loaded %d enrolled identities.", len(self._db))

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> List[FaceResult]:
        if self._face_detector is None:
            return [NO_FACE_RESULT]

        h, w = frame.shape[:2]
        self._face_detector.setInputSize((w, h))

        _, faces = self._face_detector.detect(frame)
        if faces is None or len(faces) == 0:
            return [NO_FACE_RESULT]

        results: List[FaceResult] = []
        for face_row in faces:
            bbox = self._parse_bbox(face_row, w, h)
            if self._rec_session is None or not self._db:
                results.append(FaceResult(
                    identity=FaceIdentity.UNKNOWN,
                    bbox=bbox,
                    confidence=None,
                ))
                continue

            crop = self._align_crop(frame, face_row, w, h)
            if crop is None:
                continue
            embedding = self._embed(crop)
            name, sim = self._match(embedding)
            decision = name if sim >= self._sim_threshold else "UNKNOWN"
            logger.info(
                "FACE: best=%s sim=%.4f thr=%.2f -> %s",
                name, sim, self._sim_threshold, decision,
            )
            if sim >= self._sim_threshold:
                results.append(FaceResult(
                    identity=FaceIdentity.KNOWN,
                    name=name,
                    confidence=float(sim),
                    bbox=bbox,
                ))
            else:
                results.append(FaceResult(
                    identity=FaceIdentity.UNKNOWN,
                    confidence=float(sim),
                    bbox=bbox,
                ))
        return results if results else [NO_FACE_RESULT]

    # ------------------------------------------------------------------
    def _parse_bbox(
        self, face_row: np.ndarray, img_w: int, img_h: int
    ) -> Tuple[int, int, int, int]:
        x, y, bw, bh = int(face_row[0]), int(face_row[1]), int(face_row[2]), int(face_row[3])
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + bw)
        y2 = min(img_h, y + bh)
        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    def _align_crop(
        self, frame: np.ndarray, face_row: np.ndarray, img_w: int, img_h: int
    ) -> Optional[np.ndarray]:
        """Align using the 5 YuNet landmarks (same as enrollment)."""
        iw, _ = self._input_size
        return align_face(frame, face_row, output_size=iw)

    # ------------------------------------------------------------------
    def _embed(self, crop: np.ndarray) -> np.ndarray:
        """Run MobileFaceNet and return L2-normalised 128-d embedding."""
        blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)

        input_name = self._rec_session.get_inputs()[0].name
        output = self._rec_session.run(None, {input_name: blob})[0][0]
        norm = np.linalg.norm(output)
        if norm > 0:
            output = output / norm
        return output

    # ------------------------------------------------------------------
    def _match(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Find the best-matching identity via cosine similarity."""
        best_name = "unknown"
        best_sim = -1.0
        for name, db_emb in self._db.items():
            sim = float(np.dot(embedding, db_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name, best_sim

    # ------------------------------------------------------------------
    def unload(self) -> None:
        self._face_detector = None
        self._rec_session = None
        logger.info("Face recognizer unloaded.")
