"""
app/detection/onnx_detector.py
------------------------------
Object detector backed by ONNX Runtime.

Designed for YOLOv8n / YOLO11n exported to ONNX format.
The inference input is a square (640×640 by default) normalised
RGB tensor; the model outputs the YOLOv8 detection head format:
  shape [1, 84, 8400]  → 84 = 4 (box) + 80 (class scores)

This avoids any Ultralytics Python dependency at runtime — only
onnxruntime and opencv-python are required.

Model export (on a PC with GPU):
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=640, simplify=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from app.detection.base import ObjectDetectorInterface, Detection, ModelLoadError

logger = logging.getLogger(__name__)


class ONNXObjectDetector(ObjectDetectorInterface):
    """YOLOv8n / YOLO11n ONNX Runtime detector."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._model_path = Path(cfg.path)
        self._conf_threshold: float = getattr(cfg, "confidence_threshold", 0.45)
        self._nms_threshold: float = getattr(cfg, "nms_threshold", 0.45)
        self._input_size: tuple[int, int] = tuple(getattr(cfg, "input_size", [640, 640]))
        self._class_names: list[str] = []
        self._session = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self._model_path.exists():
            raise ModelLoadError(
                f"Object detection model not found: {self._model_path}\n"
                "See models/object_detection/README.md for export instructions."
            )

        # Load class names
        names_path = Path(getattr(self._cfg, "class_names_path", ""))
        if names_path.exists():
            self._class_names = names_path.read_text().strip().splitlines()
        else:
            # Fallback: COCO 80-class names
            self._class_names = _COCO_CLASSES

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ModelLoadError(
                "onnxruntime not installed.  Run: pip install onnxruntime"
            ) from exc

        # Prefer CPU on Raspberry Pi; suppress verbose ONNX logs
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX object detector loaded: %s", self._model_path.name)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._session is None:
            raise RuntimeError("Detector not loaded.  Call load() first.")

        input_h, input_w = self._input_size[1], self._input_size[0]
        orig_h, orig_w = frame.shape[:2]

        # Pre-process
        blob = self._preprocess(frame, input_w, input_h)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})

        # Post-process
        detections = self._postprocess(
            outputs[0], orig_w, orig_h, input_w, input_h
        )
        return detections

    # ------------------------------------------------------------------
    def _preprocess(
        self, frame: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Resize + normalise to [1, 3, H, W] float32 RGB tensor."""
        resized = cv2.resize(frame, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))          # HWC → CHW
        blob = np.expand_dims(blob, axis=0)           # add batch dim
        return blob

    # ------------------------------------------------------------------
    def _postprocess(
        self,
        output: np.ndarray,
        orig_w: int,
        orig_h: int,
        input_w: int,
        input_h: int,
    ) -> List[Detection]:
        """Parse YOLOv8 output tensor → list of Detection objects."""
        # output shape: [1, 84, 8400]
        predictions = output[0]          # [84, 8400]
        predictions = predictions.T      # [8400, 84]

        boxes_raw = predictions[:, :4]   # cx, cy, w, h (normalised)
        scores_raw = predictions[:, 4:]  # [8400, 80]

        class_ids = np.argmax(scores_raw, axis=1)
        confidences = scores_raw[np.arange(len(scores_raw)), class_ids]

        mask = confidences >= self._conf_threshold
        boxes_raw = boxes_raw[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes_raw) == 0:
            return []

        # Convert cx, cy, w, h → x1, y1, x2, y2 in original pixel coords
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        x1 = ((boxes_raw[:, 0] - boxes_raw[:, 2] / 2) * input_w * scale_x).astype(int)
        y1 = ((boxes_raw[:, 1] - boxes_raw[:, 3] / 2) * input_h * scale_y).astype(int)
        x2 = ((boxes_raw[:, 0] + boxes_raw[:, 2] / 2) * input_w * scale_x).astype(int)
        y2 = ((boxes_raw[:, 1] + boxes_raw[:, 3] / 2) * input_h * scale_y).astype(int)

        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # NMS
        bboxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            bboxes_xyxy,
            confidences.tolist(),
            self._conf_threshold,
            self._nms_threshold,
        )
        if indices is None or len(indices) == 0:
            return []

        indices = np.array(indices).flatten()
        detections: List[Detection] = []
        for idx in indices:
            cid = int(class_ids[idx])
            name = self._class_names[cid] if cid < len(self._class_names) else str(cid)
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=float(confidences[idx]),
                    bbox=(int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])),
                )
            )
        return detections

    # ------------------------------------------------------------------
    def unload(self) -> None:
        self._session = None
        logger.info("ONNX object detector unloaded.")


# ---------------------------------------------------------------------------
# COCO 80 class names fallback
# ---------------------------------------------------------------------------
_COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]
