"""
app/camera/factory.py
---------------------
Instantiate the correct CameraInterface based on config.
"""

from __future__ import annotations

from app.camera.base import CameraInterface


def create_camera(cfg) -> CameraInterface:
    """Factory function – returns the configured camera backend.

    Parameters
    ----------
    cfg:
        The ``camera`` sub-namespace from the application config.

    Returns
    -------
    CameraInterface
    """
    backend = str(getattr(cfg, "backend", "mock")).lower()

    if backend == "picamera2":
        from app.camera.picamera2_camera import PiCamera2Camera
        return PiCamera2Camera(cfg)

    if backend == "opencv":
        from app.camera.opencv_camera import OpenCVCamera
        return OpenCVCamera(cfg)

    if backend == "mock":
        from app.camera.mock_camera import MockCamera
        return MockCamera(cfg)

    raise ValueError(
        f"Unknown camera backend: {backend!r}.  "
        "Valid options: 'picamera2', 'opencv', 'mock'."
    )
