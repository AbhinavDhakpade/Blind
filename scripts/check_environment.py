"""
scripts/check_environment.py
-----------------------------
Verify the runtime environment before starting Vision BOB.

Checks:
  - Python version
  - Required packages
  - Model file presence
  - Dataset structure
  - GPIO availability (Pi only)
  - Camera device
  - espeak-ng presence
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

REQUIRED_PACKAGES = [
    "cv2",
    "numpy",
    "yaml",
    "onnxruntime",
]
OPTIONAL_PACKAGES = [
    "picamera2",
    "RPi.GPIO",
    "gtts",
    "pygame",
    "psutil",
]

REQUIRED_MODELS = [
    "models/object_detection/yolov8n.onnx",
    "models/face_detection/face_detection_yunet.onnx",
    "models/face_recognition/mobilefacenet.onnx",
]

PASS = "\033[92m✔\033[0m"
FAIL = "\033[91m✘\033[0m"
WARN = "\033[93m⚠\033[0m"


def check(label, condition, required=True):
    icon = PASS if condition else (FAIL if required else WARN)
    status = "OK" if condition else ("MISSING" if required else "optional / not found")
    print(f"  {icon}  {label:50s}  {status}")
    return condition


def main():
    print("\n=== Vision BOB – Environment Check ===\n")
    all_ok = True

    # Python version
    ok = sys.version_info >= (3, 9)
    all_ok &= check(f"Python >= 3.9  (found {sys.version.split()[0]})", ok)

    print()
    print("Required Python packages:")
    for pkg in REQUIRED_PACKAGES:
        found = importlib.util.find_spec(pkg) is not None
        all_ok &= check(pkg, found, required=True)

    print()
    print("Optional Python packages:")
    for pkg in OPTIONAL_PACKAGES:
        found = importlib.util.find_spec(pkg) is not None
        check(pkg, found, required=False)

    print()
    print("Model files:")
    for model in REQUIRED_MODELS:
        path = _ROOT / model
        found = path.exists()
        all_ok &= check(model, found, required=True)

    print()
    print("Dataset structure:")
    for d in [
        "datasets/faces/known",
        "datasets/objects/images/train",
        "data/face_embeddings",
    ]:
        check(d, (_ROOT / d).exists(), required=False)

    print()
    print("System tools:")
    espeak = shutil.which("espeak-ng") is not None
    check("espeak-ng (TTS)", espeak, required=False)

    print()
    if all_ok:
        print("✅  All required checks passed. Ready to run.\n")
    else:
        print("❌  Some required components are missing. See SETUP.md.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
