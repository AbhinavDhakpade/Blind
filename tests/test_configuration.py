"""
tests/test_configuration.py
----------------------------
Verify the configuration loader works correctly.
"""

import os
import pytest
from pathlib import Path

from app.utils.config import load_config, _Namespace


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def test_config_loads():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, _Namespace)


def test_camera_defaults():
    cfg = load_config(CONFIG_PATH)
    assert cfg.camera.width == 640
    assert cfg.camera.height == 480
    assert cfg.camera.fps == 15


def test_distance_thresholds():
    cfg = load_config(CONFIG_PATH)
    assert cfg.distance.very_near_m < cfg.distance.near_m < cfg.distance.medium_m


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_env_override(monkeypatch):
    monkeypatch.setenv("VISION_BOB_CAMERA_FPS", "30")
    cfg = load_config(CONFIG_PATH)
    assert cfg.camera.fps == 30
