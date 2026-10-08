"""app/depth/factory.py"""
from __future__ import annotations
from app.depth.base import DepthEstimatorInterface, NullDepthEstimator


def create_depth_estimator(cfg) -> DepthEstimatorInterface:
    if not getattr(cfg, "enabled", False):
        return NullDepthEstimator()

    backend = str(getattr(cfg, "backend", "mock")).lower()

    if backend == "onnx":
        from app.depth.onnx_depth import ONNXDepthEstimator
        return ONNXDepthEstimator(cfg)

    if backend == "mock":
        from app.depth.mock_depth import MockDepthEstimator
        return MockDepthEstimator(cfg)

    raise ValueError(f"Unknown depth backend: {backend!r}.")
