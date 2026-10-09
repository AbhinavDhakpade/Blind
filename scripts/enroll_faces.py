"""
scripts/enroll_faces.py
------------------------
Build the face embedding database from datasets/faces/known/<person>/*.jpg

Pipeline per image:  YuNet -> quality gate -> 5-point alignment -> MobileFaceNet
Each identity is stored as ONE mean, L2-normalised 512-d embedding
(dict[str, np.ndarray]) -- same pickle format as before, so the recognizer is unchanged.

New in this version
-------------------
* Quality gates (app/face/face_quality.py) reject bad detections (ears, half faces,
  profiles, tiny faces) instead of letting them pollute the mean embedding.
* A per-identity report shows how many images were kept and why others were rejected.
* Identities with fewer than --min-images accepted faces are NOT written (a mean of
  2-3 images is not trustworthy); use --force-small to override.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from app.face.face_alignment import align_face
from app.face.face_quality import QualityConfig, check_face

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enroll known faces into embedding database.")
    p.add_argument("--dataset", default=str(_ROOT / "datasets/faces/known"))
    p.add_argument("--output", default=str(_ROOT / "data/face_embeddings/embeddings.pkl"))
    p.add_argument("--model", default=str(_ROOT / "models/face_recognition/mobilefacenet.onnx"))
    p.add_argument("--detector", default=str(_ROOT / "models/face_detection/face_detection_yunet.onnx"))
    p.add_argument("--input-size", type=int, default=112)
    p.add_argument("--min-score", type=float, default=0.80, help="min YuNet score for enrollment")
    p.add_argument("--min-face-px", type=int, default=60)
    p.add_argument("--min-images", type=int, default=5,
                   help="skip identities with fewer accepted faces than this")
    p.add_argument("--force-small", action="store_true",
                   help="write identities even if below --min-images")
    return p.parse_args()


def load_onnx_session(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    path = Path(model_path)
    if not path.exists():
        logger.error("Model not found: %s", path)
        sys.exit(1)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def embed_face(session, face_crop: np.ndarray, input_size: int) -> np.ndarray:
    resized = cv2.resize(face_crop, (input_size, input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = (rgb - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, 0)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0][0]
    norm = np.linalg.norm(output)
    return output / norm if norm > 0 else output


def detect_largest_face(detector, image: np.ndarray):
    """Return the full YuNet detection row (box + 5 landmarks + score) of the largest face."""
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    areas = [row[2] * row[3] for row in faces]
    return faces[int(np.argmax(areas))]


def main() -> None:
    args = parse_args()
    qcfg = QualityConfig(min_score=args.min_score, min_face_px=args.min_face_px)

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        logger.error("Dataset directory not found: %s", dataset_path)
        sys.exit(1)

    det_path = Path(args.detector)
    if not det_path.exists():
        logger.error("Face detector model not found: %s", det_path)
        sys.exit(1)

    logger.info("Loading models...")
    # Low detector threshold on purpose: we apply our own (stricter) quality gate afterwards.
    face_detector = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), 0.6, 0.3, top_k=10)
    rec_session = load_onnx_session(args.model)

    person_dirs = sorted(d for d in dataset_path.iterdir() if d.is_dir())
    if not person_dirs:
        logger.error("No person subdirectories found in %s", dataset_path)
        sys.exit(1)

    db: Dict[str, np.ndarray] = {}
    summary = []

    for person_dir in person_dirs:
        name = person_dir.name
        image_files = sorted(f for f in person_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
        if not image_files:
            logger.warning("No images found for %s, skipping.", name)
            continue

        embeddings: List[np.ndarray] = []
        rejected: Counter = Counter()

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                rejected["unreadable"] += 1
                continue

            row = detect_largest_face(face_detector, img)
            if row is None:
                rejected["no_face"] += 1
                logger.info("  REJECT %-12s %s (no_face)", name, img_path.name)
                continue

            ok, reason = check_face(row, qcfg)
            if not ok:
                rejected[reason] += 1
                logger.info("  REJECT %-12s %s (%s, score=%.2f)", name, img_path.name, reason, float(row[14]))
                continue

            aligned = align_face(img, row, output_size=args.input_size)
            if aligned is None or aligned.size == 0:
                rejected["align_failed"] += 1
                logger.info("  REJECT %-12s %s (align_failed)", name, img_path.name)
                continue

            embeddings.append(embed_face(rec_session, aligned, args.input_size))

        n_ok, n_all = len(embeddings), len(image_files)
        reasons = ", ".join(f"{k}={v}" for k, v in rejected.most_common()) or "none"
        summary.append((name, n_ok, n_all, reasons))

        if n_ok == 0:
            logger.warning("%s: no usable faces -> NOT enrolled.", name)
            continue
        if n_ok < args.min_images and not args.force_small:
            logger.warning("%s: only %d usable faces (< %d) -> NOT enrolled. "
                           "Add more/better images or use --force-small.", name, n_ok, args.min_images)
            continue

        mean_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_emb)
        db[name] = mean_emb / norm if norm > 0 else mean_emb

    logger.info("---- Enrollment report ----")
    for name, n_ok, n_all, reasons in summary:
        logger.info("%-14s kept %2d / %2d   rejected: %s", name, n_ok, n_all, reasons)

    with output_path.open("wb") as fh:
        pickle.dump(db, fh)
    logger.info("Database saved to %s (%d identities).", output_path, len(db))


if __name__ == "__main__":
    main()
