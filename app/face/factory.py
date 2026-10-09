"""app/face/factory.py"""
from __future__ import annotations
from app.face.base import FaceRecognizerInterface


def create_face_recognizer(cfg_det, cfg_rec) -> FaceRecognizerInterface:
    det_backend = str(getattr(cfg_det, "backend", "mock")).lower()
    rec_backend = str(getattr(cfg_rec, "backend", "mock")).lower()

    # Both must agree; mock wins if either side is mock
    if det_backend == "mock" or rec_backend == "mock":
        from app.face.mock_face_recognizer import MockFaceRecognizer
        return MockFaceRecognizer(cfg_det, cfg_rec)

    if det_backend == "yunet_opencv" and rec_backend == "onnx":
        from app.face.onnx_face_recognizer import ONNXFaceRecognizer
        return ONNXFaceRecognizer(cfg_det, cfg_rec)

    raise ValueError(
        f"Unknown face backend combination: det={det_backend!r}, rec={rec_backend!r}."
    )
