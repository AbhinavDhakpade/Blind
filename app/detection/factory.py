"""app/detection/factory.py"""
from __future__ import annotations
from app.detection.base import ObjectDetectorInterface


def create_detector(cfg) -> ObjectDetectorInterface:
    backend = str(getattr(cfg, "backend", "mock")).lower()

    if backend == "onnx":
        from app.detection.onnx_detector import ONNXObjectDetector
        return ONNXObjectDetector(cfg)

    if backend == "mock":
        from app.detection.mock_detector import MockObjectDetector
        return MockObjectDetector(cfg)

    raise ValueError(f"Unknown detector backend: {backend!r}. Valid: 'onnx', 'mock'.")
