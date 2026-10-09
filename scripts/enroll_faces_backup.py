"""
scripts/enroll_faces.py
------------------------
Build the face embedding database from images in datasets/faces/known/.

Usage
-----
    python scripts/enroll_faces.py
    python scripts/enroll_faces.py --dataset datasets/faces/known
                                   --output  data/face_embeddings/embeddings.pkl
                                   --model   models/face_recognition/mobilefacenet.onnx
                                   --detector models/face_detection/face_detection_yunet.onnx

Each person subdirectory name becomes the identity label.
Multiple images are averaged into one representative mean embedding.

Directory structure expected
----------------------------
    datasets/faces/known/
        person_001/
            image1.jpg
            image2.jpg
        person_002/
            photo1.png
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enroll known faces into embedding database.")
    p.add_argument("--dataset",  default=str(_ROOT / "datasets/faces/known"))
    p.add_argument("--output",   default=str(_ROOT / "data/face_embeddings/embeddings.pkl"))
    p.add_argument("--model",    default=str(_ROOT / "models/face_recognition/mobilefacenet.onnx"))
    p.add_argument("--detector", default=str(_ROOT / "models/face_detection/face_detection_yunet.onnx"))
    p.add_argument("--input-size", type=int, default=112)
    return p.parse_args()


def load_onnx_session(model_path: str):
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        logger.error("onnxruntime not installed.  Run: pip install onnxruntime")
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
    """Return the largest face bounding box or None."""
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    # Pick largest bbox
    areas = [row[2] * row[3] for row in faces]
    idx = int(np.argmax(areas))
    row = faces[idx]
    x, y, bw, bh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(w, x + bw); y2 = min(h, y + bh)
    return image[y1:y2, x1:x2]


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        logger.error("Dataset directory not found: %s", dataset_path)
        sys.exit(1)

    logger.info("Loading face detection model...")
    det_path = Path(args.detector)
    if det_path.exists():
        face_detector = cv2.FaceDetectorYN.create(
            str(det_path), "", (320, 320), 0.6, 0.3, top_k=10
        )
    else:
        logger.warning("Face detector model not found at %s; skipping face alignment.", det_path)
        face_detector = None

    logger.info("Loading face recognition model...")
    rec_session = load_onnx_session(args.model)

    db: Dict[str, np.ndarray] = {}

    person_dirs = sorted(d for d in dataset_path.iterdir() if d.is_dir())
    if not person_dirs:
        logger.error("No person subdirectories found in %s", dataset_path)
        sys.exit(1)

    for person_dir in person_dirs:
        name = person_dir.name
        image_files = [
            f for f in person_dir.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS
        ]
        if not image_files:
            logger.warning("No images found for %s, skipping.", name)
            continue

        embeddings: List[np.ndarray] = []
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Could not read %s, skipping.", img_path)
                continue

            face_crop = None
            if face_detector is not None:
                face_crop = detect_largest_face(face_detector, img)

            if face_crop is None or face_crop.size == 0:
                face_crop = img   # fall back to whole image

            emb = embed_face(rec_session, face_crop, args.input_size)
            embeddings.append(emb)
            logger.info("  Enrolled: %s / %s", name, img_path.name)

        if embeddings:
            mean_emb = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(mean_emb)
            db[name] = mean_emb / norm if norm > 0 else mean_emb
            logger.info("Enrolled %s: %d image(s) → mean embedding.", name, len(embeddings))

    with output_path.open("wb") as fh:
        pickle.dump(db, fh)

    logger.info("Database saved to %s (%d identities).", output_path, len(db))


if __name__ == "__main__":
    main()
