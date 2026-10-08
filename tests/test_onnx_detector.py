"""Tests for ONNXObjectDetector decoding (no model file required for unit tests)."""
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app.detection.onnx_detector import ONNXObjectDetector

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "models/object_detection/yolov8n.onnx"


def _det(input_wh=(640, 640), conf=0.4):
    cfg = NS(path="x.onnx", confidence_threshold=conf, nms_threshold=0.45,
             input_size=list(input_wh), class_names_path="")
    d = ONNXObjectDetector(cfg)
    d._class_names = ["person", "bicycle", "car"]
    return d


def _tensor(boxes_cxcywh, cls, conf, n_classes=3):
    """Build a [1, 4+nc, N] YOLOv8-style output."""
    n = len(boxes_cxcywh)
    out = np.zeros((1, 4 + n_classes, n), dtype=np.float32)
    for i, (b, c, s) in enumerate(zip(boxes_cxcywh, cls, conf)):
        out[0, :4, i] = b
        out[0, 4 + c, i] = s
    return out


def test_letterbox_roundtrip_pixel_coords():
    """640x480 frame in a 640x640 model: box must map back to the right pixels."""
    d = _det((640, 640))
    frame = np.zeros((480, 640, 3), np.uint8)
    blob, scale, px, py = d._preprocess(frame)
    assert blob.shape == (1, 3, 640, 640)
    assert scale == pytest.approx(1.0) and px == 0 and py == 80
    # object at original (100,100)-(200,300) -> model space adds pad_y=80
    cx, cy, w, h = 150, 200 + 80, 100, 200
    out = _tensor([[cx, cy, w, h]], [0], [0.9])
    dets = d._postprocess(out, 640, 480, scale, px, py)
    assert len(dets) == 1
    assert dets[0].class_name == "person"
    x1, y1, x2, y2 = dets[0].bbox
    assert (x1, y1, x2, y2) == (100, 100, 200, 300)


def test_normalised_output_is_detected_and_scaled():
    d = _det((640, 640))
    frame = np.zeros((640, 640, 3), np.uint8)
    _, scale, px, py = d._preprocess(frame)
    out = _tensor([[0.5, 0.5, 0.25, 0.25]], [2], [0.8])
    dets = d._postprocess(out, 640, 640, scale, px, py)
    assert dets[0].bbox == (240, 240, 400, 400)
    assert dets[0].class_name == "car"


def test_nonsquare_model_input():
    """224x128 (HxW) style model: letterbox keeps aspect ratio."""
    d = _det((128, 224))
    frame = np.zeros((480, 640, 3), np.uint8)
    blob, scale, px, py = d._preprocess(frame)
    assert blob.shape == (1, 3, 224, 128)
    assert scale == pytest.approx(0.2)


def test_low_confidence_filtered_and_nms():
    d = _det((640, 640))
    out = _tensor([[100, 100, 50, 50], [102, 100, 50, 50], [400, 400, 50, 50]],
                  [0, 0, 1], [0.9, 0.8, 0.2])
    dets = d._postprocess(out, 640, 640, 1.0, 0, 0)
    assert len(dets) == 1            # duplicate suppressed, low-conf dropped


@pytest.mark.skipif(not BUNDLED.exists(), reason="model not installed")
def test_bundled_model_loads_and_runs_with_any_config_size():
    cfg = NS(path=str(BUNDLED), confidence_threshold=0.45, nms_threshold=0.45,
             input_size=[640, 640],
             class_names_path=str(ROOT / "models/object_detection/coco_classes.txt"))
    d = ONNXObjectDetector(cfg)
    d.load()                          # must adopt the model's own input size
    d.detect(np.zeros((480, 640, 3), np.uint8))
