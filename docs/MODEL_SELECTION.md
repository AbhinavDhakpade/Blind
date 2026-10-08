# Model Selection – Vision BOB

## Evaluation criteria

All models were evaluated against Raspberry Pi 5 constraints:

- CPU: Arm Cortex-A76 × 4, ~2.4 GHz, **no GPU, no NPU by default**
- RAM: 4–8 GB LPDDR4X
- Target latency: ≤ 300 ms per frame (≥ 3 FPS)
- Storage: SD card / SSD
- Inference runtime: ONNX Runtime CPU (preferred) or OpenCV DNN

---

## Decision Table

| Task | Selected Approach | Why | Expected Pi 5 Performance | Alternative Considered |
|------|-------------------|-----|--------------------------|------------------------|
| Object Detection | **YOLOv8n ONNX** (640×640) | Smallest YOLO model; 3.2 M params; ONNX export simplifies deployment; good accuracy on COCO 80 classes | ~150–250 ms/frame on CPU. Gives ~3–6 FPS which is acceptable for a walking-speed assistive system | YOLOv8s (too slow ~500 ms), MobileNet-SSD (lower accuracy), EfficientDet-Lite (marginal), YOLO11n (similar but less ONNX ecosystem maturity) |
| Face Detection | **YuNet (OpenCV DNN)** | Integrated into OpenCV ≥ 4.5.4; ~1–5 ms on CPU at 320×320; no additional pip dep; robust to partial faces | < 5 ms per frame | MTCNN (too slow, ~80 ms), BlazeFace (TFLite only, extra dep), RetinaFace (too heavy) |
| Face Recognition | **MobileFaceNet ONNX** (112×112) | 1 M params, ~15 ms per crop on Pi 5; 128-d embedding; cosine similarity matching in pure NumPy; no GPU required; well-documented ONNX exports available | ~10–20 ms per detected face | ArcFace ResNet-50 (too large, ~200 ms), FaceNet Inception (heavy), OpenCV LBPHFaceRecognizer (outdated, no embedding) |
| Depth / Distance | **Hybrid: ultrasonic + bbox-area fusion** | We already have a physical ultrasonic sensor providing metric distance to the nearest object. Bbox-area gives relative depth for other objects. No monocular depth model is needed for the primary use case | Ultrasonic < 10 ms; bbox math is negligible | MiDaS-Small ONNX — evaluated and rejected for primary use: ~800 ms/frame at 256×256 on Pi 5 CPU (verified by community benchmarks). Kept as optional enhancement |
| Risk Classification | **Rule-based scoring engine** (CPU) | Deterministic, zero inference latency, fully configurable. A learned classifier would add latency and require a training dataset without meaningful accuracy gain over a well-tuned rule engine for this structured input | < 1 ms | SVM / random forest on hand-crafted features — unnecessary complexity |
| Object Tracking | **IoU-based centroid tracker** | O(N²) IoU matching for N < 20 objects is negligible on CPU. Provides temporal bbox-area growth as approach proxy. Avoids Kalman-filter complexity of SORT and deep embedding cost of DeepSORT | < 2 ms | SORT (Kalman + IoU) — marginal improvement; DeepSORT (re-ID model, too expensive) |

---

## Why YOLOv8n and NOT a bigger model

YOLOv8n at 640×640 achieves mAP@50 = 52.9 on COCO.
The next variant, YOLOv8s (11.2 M params), achieves 64.0 but takes ~2× the
inference time (~400–500 ms on Pi 5), cutting effective FPS below 2.
For a walking-speed assistive system, 3–5 FPS with YOLOv8n is preferable to
1–2 FPS with YOLOv8s.

> mAP figures from Ultralytics official benchmarks (CPU, PyTorch, imgsz=640).
> Pi 5 ONNX Runtime may differ. Run `scripts/benchmark_models.py` for actual measurements.

---

## Why NOT monocular depth as primary distance source

MiDaS-Small (256×256, ONNX):
- Expected latency on Pi 5 CPU: **~600–900 ms** per frame
  (source: community benchmarks for ARM Cortex-A76, similar TFLite reports)
- Would reduce total pipeline FPS to < 1.5 FPS — unacceptable for real-time use
- Provides **relative** depth only, not metric distance

The HC-SR04 ultrasonic sensor provides accurate metric distance at < 10 ms.
The bounding-box area provides per-object relative distance.
Together they satisfy the system requirements without any neural depth model.

Monocular depth remains available as an **optional** feature for offline
analysis or future integration with an NPU (e.g. Hailo-8).

---

## Future optimisation paths

| Option | Benefit | Effort |
|--------|---------|--------|
| Hailo-8 NPU HAT | ~50× inference speedup | Medium |
| INT8 quantisation of YOLOv8n | ~30% speedup, minor accuracy drop | Low |
| NCNN backend | Marginal speedup over ONNX on ARM | Medium |
| Reduce input to 416×416 | ~35% faster, minor accuracy drop | Low |
| Face recognition every 3rd frame | ~2× pipeline speedup | Low |
