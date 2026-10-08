"""app/detection/__init__.py"""
from app.detection.base import ObjectDetectorInterface, Detection
from app.detection.factory import create_detector

__all__ = ["ObjectDetectorInterface", "Detection", "create_detector"]
