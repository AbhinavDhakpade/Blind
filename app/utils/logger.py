"""
app/utils/logger.py
-------------------
Centralised logging setup.

Call ``setup_logging(cfg)`` once at application startup.
All modules then use the standard ``logging.getLogger(__name__)`` pattern.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.utils.config import _Namespace


def setup_logging(cfg: "_Namespace") -> None:
    """Configure root logger from configuration object."""
    log_cfg = cfg.logging

    level = getattr(logging, str(log_cfg.level).upper(), logging.INFO)
    log_path = Path(cfg.get("_project_root", ".")) / log_cfg.file if hasattr(log_cfg, "file") else None

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=getattr(log_cfg, "max_bytes", 5_242_880),
            backupCount=getattr(log_cfg, "backup_count", 3),
        )
        handlers.append(rotating)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)-35s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for h in handlers:
        h.setLevel(level)
        h.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any existing handlers (avoids duplicates on re-init)
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
