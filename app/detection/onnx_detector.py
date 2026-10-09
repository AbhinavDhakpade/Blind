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
    """YOLOv8n / YOLO11n ONNX Runtime detector.

    Handles the details that commonly break YOLO ONNX deployments:
      * input size is read from the model itself (static models), falling
        back to ``input_size`` in config only for dynamic-shape models;
      * frames are letterboxed (aspect ratio preserved, grey padding);
      * output boxes are accepted either as pixels in model-input space
        (Ultralytics default) or as normalised 0-1 values (auto-detected);
      * boxes are mapped back through the letterbox to original pixels;
      * the number of class names is checked against the model output.
    """

    _PAD_VALUE = 114  # Ultralytics letterbox grey

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._model_path = Path(cfg.path)
        self._conf_threshold: float = getattr(cfg, "confidence_threshold", 0.45)
        self._nms_threshold: float = getattr(cfg, "nms_threshold", 0.45)
        # config input_size is [width, height]
        self._cfg_input_wh: tuple[int, int] = tuple(getattr(cfg, "input_size", [640, 640]))
        self._input_w: int = self._cfg_input_wh[0]
        self._input_h: int = self._cfg_input_wh[1]
        self._class_names: list[str] = []
        self._session = None
        self._input_name: str = "images"

    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self._model_path.exists():
            raise ModelLoadError(
                f"Object detection model not found: {self._model_path}\n"
                "See models/object_detection/README.md for export instructions."
            )

        names_cfg = getattr(self._cfg, "class_names_path", "")
        names_path = Path(names_cfg) if names_cfg else None
        if names_path is not None and names_path.exists():
            self._class_names = [
                ln.strip() for ln in names_path.read_text().splitlines() if ln.strip()
            ]
        else:
            self._class_names = list(_COCO_CLASSES)

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ModelLoadError(
                "onnxruntime not installed.  Run: pip install onnxruntime"
            ) from exc

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape  # [1, 3, H, W]; entries may be str/None if dynamic
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int):
            if (w, h) != self._cfg_input_wh:
                logger.warning(
                    "Config input_size %s differs from model's fixed input %dx%d (WxH); "
                    "using the model's size.", list(self._cfg_input_wh), w, h,
                )
            self._input_w, self._input_h = w, h
        # else: dynamic model -> keep config size

        # Validate class-name count against model output channels
        out_shape = self._session.get_outputs()[0].shape  # [1, 4+nc, N]
        if len(out_shape) == 3 and isinstance(out_shape[1], int):
            n_classes = out_shape[1] - 4
            if n_classes != len(self._class_names):
                raise ModelLoadError(
                    f"Model has {n_classes} classes but class names file has "
                    f"{len(self._class_names)}.  Fix class_names_path in config."
                )

        logger.info(
            "ONNX object detector loaded: %s (input %dx%d, %d classes)",
            self._model_path.name, self._input_w, self._input_h, len(self._class_names),
        )

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._session is None:
            raise RuntimeError("Detector not loaded.  Call load() first.")

        orig_h, orig_w = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: blob})
        return self._postprocess(outputs[0], orig_w, orig_h, scale, pad_x, pad_y)

    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray):
        """Letterbox to (input_h, input_w) and return blob, scale, pad_x, pad_y."""
        orig_h, orig_w = frame.shape[:2]
        scale = min(self._input_w / orig_w, self._input_h / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self._input_h, self._input_w, 3), self._PAD_VALUE, dtype=np.uint8)
        pad_x = (self._input_w - new_w) // 2
        pad_y = (self._input_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        return np.ascontiguousarray(blob), scale, pad_x, pad_y

    # ------------------------------------------------------------------
    def _postprocess(
        self,
        output: np.ndarray,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> List[Detection]:
        """Parse YOLOv8 output [1, 4+nc, N] -> Detection list in original pixels."""
        predictions = output[0].T                 # [N, 4+nc]
        boxes_raw = predictions[:, :4].astype(np.float32)   # cx, cy, w, h
        scores_raw = predictions[:, 4:]

        class_ids = np.argmax(scores_raw, axis=1)
        confidences = scores_raw[np.arange(len(scores_raw)), class_ids]

        mask = confidences >= self._conf_threshold
        boxes_raw = boxes_raw[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        if len(boxes_raw) == 0:
            return []

        # Ultralytics ONNX exports give pixels in model-input space.  Some
        # exports give 0-1 normalised values; detect that and scale up.
        if float(boxes_raw.max()) <= 1.5:
            boxes_raw[:, [0, 2]] *= self._input_w
            boxes_raw[:, [1, 3]] *= self._input_h

        cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        # model-input space -> original image space (undo pad then scale)
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale

        x1 = np.clip(x1, 0, orig_w).astype(int)
        y1 = np.clip(y1, 0, orig_h).astype(int)
        x2 = np.clip(x2, 0, orig_w).astype(int)
        y2 = np.clip(y2, 0, orig_h).astype(int)

        # OpenCV NMSBoxes expects [x, y, w, h]
        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, confidences.tolist(), self._conf_threshold, self._nms_threshold,
        )
        if indices is None or len(indices) == 0:
            return []

        detections: List[Detection] = []
        for idx in np.array(indices).flatten():
            if x2[idx] <= x1[idx] or y2[idx] <= y1[idx]:
                continue
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
