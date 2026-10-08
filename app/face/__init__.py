"""app/face/__init__.py"""
from app.face.base import FaceRecognizerInterface, FaceResult, FaceIdentity
from app.face.factory import create_face_recognizer

__all__ = ["FaceRecognizerInterface", "FaceResult", "FaceIdentity", "create_face_recognizer"]
