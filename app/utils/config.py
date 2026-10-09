"""
app/utils/config.py
-------------------
Centralised configuration loader.

Reads config/config.yaml (base) and optionally merges
config/config.local.yaml (local overrides, not committed).
Provides attribute-style access via dotted paths.

Usage
-----
    from app.utils.config import load_config, get_config

    cfg = load_config()           # call once at startup
    cam_w = cfg.camera.width
    pin   = cfg.hardware.vibration_pin
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _Namespace:
    """Recursive attribute-access wrapper around a dict."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _Namespace(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Namespace({self.__dict__!r})"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (base is not mutated)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CONFIG: _Namespace | None = None
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> _Namespace:
    """Load (and cache) the application configuration.

    Parameters
    ----------
    config_path:
        Optional explicit path to a YAML config file.
        Defaults to ``<project_root>/config/config.yaml``.
    """
    global _CONFIG

    base_path = Path(config_path) if config_path else _PROJECT_ROOT / "config" / "config.yaml"
    local_path = base_path.with_name("config.local.yaml")

    if not base_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {base_path}\n"
            "Run from the project root or pass an explicit config_path."
        )

    with base_path.open() as fh:
        data: dict = yaml.safe_load(fh) or {}

    if local_path.exists():
        with local_path.open() as fh:
            local_data: dict = yaml.safe_load(fh) or {}
        data = _deep_merge(data, local_data)

    # Allow environment-variable overrides for a small set of keys.
    # E.g.  VISION_BOB_CAMERA_BACKEND=mock
    _apply_env_overrides(data)

    _CONFIG = _Namespace(data)
    return _CONFIG


def get_config() -> _Namespace:
    """Return the cached config, loading defaults if never initialised."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def _apply_env_overrides(data: dict) -> None:
    """Apply VISION_BOB_* environment variables as flat key overrides."""
    prefix = "VISION_BOB_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        # VISION_BOB_CAMERA_BACKEND → ["camera", "backend"]
        parts = env_key[len(prefix):].lower().split("_")
        node = data
        for part in parts[:-1]:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                break
        else:
            leaf = parts[-1]
            if isinstance(node, dict) and leaf in node:
                original = node[leaf]
                # attempt type coercion
                try:
                    if isinstance(original, bool):
                        node[leaf] = env_val.lower() in ("1", "true", "yes")
                    elif isinstance(original, int):
                        node[leaf] = int(env_val)
                    elif isinstance(original, float):
                        node[leaf] = float(env_val)
                    else:
                        node[leaf] = env_val
                except ValueError:
                    node[leaf] = env_val
