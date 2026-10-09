# Face Detection Model

## Expected file

```
models/face_detection/face_detection_yunet.onnx
```

## Model

**YuNet** — lightweight face detector integrated into OpenCV ≥ 4.5.4.

- Architecture: MobileNet-like, ~75 K parameters
- Input: variable size (set dynamically via `setInputSize()`)
- Output: face bounding boxes + 5 facial landmarks + confidence score
- Inference backend: OpenCV DNN (uses ONNX format)
- Latency on Pi 5: **< 5 ms** at 320×320

## Download

```bash
wget -O models/face_detection/face_detection_yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Or from the OpenCV model zoo:
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

## Usage

Loaded via `cv2.FaceDetectorYN.create(...)` — no extra Python packages required
beyond `opencv-python`.

## Is this model optional?

No. Face detection is required for face recognition. Without this model,
the system will log a warning and classify all faces as UNKNOWN.
