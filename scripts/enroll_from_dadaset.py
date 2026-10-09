"""
scripts/enroll_from_dadaset.py
-------------------------------
One-shot enrollment helper that copies images from the top-level Dadaset/
directory into datasets/faces/known/ and then runs the standard enrollment
pipeline (scripts/enroll_faces.py) to build the embedding database.

Usage
-----
    # From the project root (raspberry_pi_assistive_ai/)
    python scripts/enroll_from_dadaset.py

    # Override source path
    python scripts/enroll_from_dadaset.py --dadaset ../../Dadaset

    # Skip the copy step (images already in datasets/faces/known/)
    python scripts/enroll_from_dadaset.py --skip-copy

Directory mapping
-----------------
    <dadaset_root>/abhinav/    →  datasets/faces/known/abhinav/
    <dadaset_root>/siddhi/     →  datasets/faces/known/siddhi/
    <dadaset_root>/vaishnavi/  →  datasets/faces/known/vaishnavi/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]          # raspberry_pi_assistive_ai/
_WORKSPACE_ROOT = _ROOT.parent                        # Vision_BOB/
_KNOWN_DIR = _ROOT / "datasets" / "faces" / "known"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy Dadaset images and enroll them into the face database."
    )
    p.add_argument(
        "--dadaset",
        default=str(_WORKSPACE_ROOT / "Dadaset"),
        help="Path to the Dadaset root folder (default: ../../Dadaset relative to project root)",
    )
    p.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip copying images (use if already copied to datasets/faces/known/)",
    )
    return p.parse_args()


def copy_dadaset(dadaset_root: Path) -> None:
    """Copy each person subfolder from Dadaset into datasets/faces/known/."""
    if not dadaset_root.exists():
        logger.error("Dadaset path not found: %s", dadaset_root)
        sys.exit(1)

    person_dirs = [d for d in dadaset_root.iterdir() if d.is_dir()]
    if not person_dirs:
        logger.error("No person subdirectories found in %s", dadaset_root)
        sys.exit(1)

    _KNOWN_DIR.mkdir(parents=True, exist_ok=True)

    for src in person_dirs:
        dst = _KNOWN_DIR / src.name
        if dst.exists():
            logger.info("Destination already exists, overwriting: %s", dst)
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        img_count = sum(1 for f in dst.rglob("*") if f.suffix.lower() in _IMAGE_EXTS)
        logger.info("Copied %s → %s  (%d images)", src, dst, img_count)


def run_enrollment() -> None:
    """Delegate to the standard enroll_faces.py script."""
    enroll_script = _ROOT / "scripts" / "enroll_faces.py"
    import runpy
    sys.argv = ["enroll_faces.py"]   # reset argv so argparse in enroll_faces gets defaults
    logger.info("Running enrollment pipeline…")
    runpy.run_path(str(enroll_script), run_name="__main__")


def main() -> None:
    args = parse_args()

    if not args.skip_copy:
        logger.info("=== Step 1: Copy images from Dadaset ===")
        copy_dadaset(Path(args.dadaset))
    else:
        logger.info("Skipping copy step (--skip-copy).")

    logger.info("=== Step 2: Build face embedding database ===")
    run_enrollment()


if __name__ == "__main__":
    main()
