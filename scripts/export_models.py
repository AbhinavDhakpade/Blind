"""
scripts/export_models.py
------------------------
Run ONCE on a PC (not the Pi) to turn the .pt weights into the ONNX file
the app expects. Needs:  pip install ultralytics onnx onnxruntime

    python scripts/export_models.py              # stock COCO model (default)
    python scripts/export_models.py --custom     # 13-class vision-impaired model

COCO  -> models/object_detection/yolov8n.onnx
custom -> models/object_detection/vision_impaired.onnx
          (then set models.object_detection.path and class_names_path in
           config/config.local.yaml; see the printed hint)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models/object_detection/vision_impaired_yolov8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--custom", action="store_true", help="export best.pt (13 classes)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics onnx onnxruntime")

    weights = SRC / "weights" / ("best.pt" if args.custom else "yolov8n.pt")
    target = ROOT / "models/object_detection" / (
        "vision_impaired.onnx" if args.custom else "yolov8n.onnx"
    )
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    out = YOLO(str(weights)).export(format="onnx", imgsz=640, simplify=True, opset=12, dynamic=False)
    shutil.copy(out, target)
    print(f"Saved {target}")

    import onnxruntime as ort
    s = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    print("input :", s.get_inputs()[0].shape, " (expect [1, 3, 640, 640])")
    print("output:", s.get_outputs()[0].shape, " (expect [1, 4+classes, 8400])")

    if args.custom:
        print(
            "\nAdd to config/config.local.yaml:\n"
            "models:\n  object_detection:\n"
            "    path: models/object_detection/vision_impaired.onnx\n"
            "    class_names_path: models/object_detection/vision_impaired_yolov8/classes.txt"
        )


if __name__ == "__main__":
    main()
