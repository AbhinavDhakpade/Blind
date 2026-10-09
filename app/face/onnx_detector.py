"""
app/detection/onnx_detector.py
------------------------------
Object detector backed by ONNX Runtime.

Designed for YOLOv8n / YOLO11n exported to ONNX format.
The inference input is a square (640x640 by default) normalised
RGB tensor; the model outputs the YOLOv8 detection head format:
  shape [1, 84, 8400]  -> 84 = 4 (box) + 80 (class scores)

This avoids any Ultralytics Python dependency at runtime - only
onnxruntime and opencv-python are required.

Model export (on a PC with GPU):
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=640, simplify=True)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from app.detection.base import ObjectDetectorInterface, Detection, ModelLoadError

logger = logging.getLogger(__name__)

# Log a sub-stage timing summary (DEBUG level) every N detect() calls.
_TIMING_EVERY = 30


class ONNXObjectDetector(ObjectDetectorInterface):
    """YOLOv8n / YOLO11n ONNX Runtime detector."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._model_path = Path(cfg.path)
        self._conf_threshold: float = getattr(cfg, "confidence_threshold", 0.45)
        self._nms_threshold: float = getattr(cfg, "nms_threshold", 0.45)
        self._input_size: tuple[int, int] = tuple(getattr(cfg, "input_size", [640, 640]))
        # Explicit thread count: default min(4, cores). Pi 5 has 4 cores.
        self._num_threads: int = int(
            getattr(cfg, "num_threads", 0) or min(4, os.cpu_count() or 4)
        )
        # Cap candidates passed to NMS so worst-case post-processing is bounded.
        self._max_candidates: int = int(getattr(cfg, "max_candidates", 100))
        self._class_names: list[str] = []
        self._session = None
        self._input_name: str = ""
        self._timing_n = 0
        self._timing_sum = [0.0, 0.0, 0.0]   # preprocess, inference, postprocess (ms)

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

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = self._num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info(
            "ONNX object detector loaded: %s (threads=%d)",
            self._model_path.name, self._num_threads,
        )

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._session is None:
            raise RuntimeError("Detector not loaded.  Call load() first.")

        input_h, input_w = self._input_size[1], self._input_size[0]
        orig_h, orig_w = frame.shape[:2]

        t0 = time.perf_counter()
        blob = self._preprocess(frame, input_w, input_h)
        t1 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: blob})
        t2 = time.perf_counter()
        detections = self._postprocess(outputs[0], orig_w, orig_h, input_w, input_h)
        t3 = time.perf_counter()

        self._record_timing((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
        return detections

    # ------------------------------------------------------------------
    def _record_timing(self, pre: float, inf: float, post: float) -> None:
        self._timing_sum[0] += pre
        self._timing_sum[1] += inf
        self._timing_sum[2] += post
        self._timing_n += 1
        if self._timing_n >= _TIMING_EVERY:
            n = self._timing_n
            logger.debug(
                "DETECT avg over %d calls: preprocess=%.1f ms  infer=%.1f ms  postprocess=%.1f ms",
                n, self._timing_sum[0] / n, self._timing_sum[1] / n, self._timing_sum[2] / n,
            )
            self._timing_n = 0
            self._timing_sum = [0.0, 0.0, 0.0]

    # ------------------------------------------------------------------
    def _preprocess(
        self, frame: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Resize + scale to a [1, 3, H, W] float32 RGB tensor (one native call)."""
        return cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(target_w, target_h),
            swapRB=True,
            crop=False,
        )

    # ------------------------------------------------------------------
    def _postprocess(
        self,
        output: np.ndarray,
        orig_w: int,
        orig_h: int,
        input_w: int,
        input_h: int,
    ) -> List[Detection]:
        """Parse YOLOv8 output tensor -> list of Detection objects."""
        # output shape: [1, 84, 8400]
        predictions = output[0].T         # [8400, 84]
        scores_raw = predictions[:, 4:]   # [8400, 80]

        class_ids = np.argmax(scores_raw, axis=1)
        confidences = scores_raw[np.arange(len(scores_raw)), class_ids]

        mask = confidences >= self._conf_threshold
        if not mask.any():
            return []
        boxes_raw = predictions[mask, :4]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Bound NMS cost: keep only the strongest candidates.
        if len(confidences) > self._max_candidates:
            top = np.argpartition(confidences, -self._max_candidates)[-self._max_candidates:]
            boxes_raw = boxes_raw[top]
            class_ids = class_ids[top]
            confidences = confidences[top]

        # Ultralytics ONNX exports give cx,cy,w,h in INPUT-PIXEL units (0..640).
        # Some other exports are normalised 0..1. Handle both.
        if float(boxes_raw.max()) <= 1.5:
            boxes_raw = boxes_raw * np.array(
                [input_w, input_h, input_w, input_h], dtype=np.float32
            )

        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]

        x1 = np.clip((cx - bw / 2) * scale_x, 0, orig_w).astype(int)
        y1 = np.clip((cy - bh / 2) * scale_y, 0, orig_h).astype(int)
        x2 = np.clip((cx + bw / 2) * scale_x, 0, orig_w).astype(int)
        y2 = np.clip((cy + bh / 2) * scale_y, 0, orig_h).astype(int)

        # cv2.dnn.NMSBoxes expects [x, y, w, h], NOT [x1, y1, x2, y2].
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
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
