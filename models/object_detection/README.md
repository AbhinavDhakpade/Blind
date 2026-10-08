# Object Detection Model

## Expected file

```
models/object_detection/yolov8n.onnx
models/object_detection/coco_classes.txt   (optional; built-in fallback exists)
```

## Model format

ONNX format, opset 12 or higher.

Input:  `[1, 3, 640, 640]`  float32, normalised to [0, 1], RGB channel order.
Output: `[1, 84, 8400]`      YOLOv8 detection head format.
         84 = 4 (box: cx, cy, w, h normalised) + 80 (COCO class scores).

## How to export (on a GPU machine)

```bash
pip install ultralytics
python - <<'EOF'
from ultralytics import YOLO
model = YOLO("yolov8n.pt")   # downloads ~6 MB weights on first run
model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    opset=12,
    dynamic=False,
)
EOF
```

This produces `yolov8n.onnx`. Copy it to `models/object_detection/`.

## Custom training

If you want to detect a custom set of objects:

1. Prepare dataset in YOLO format (see `datasets/objects/README.md`).
2. Train on a GPU machine:
   ```bash
   yolo train data=datasets/objects/data.yaml model=yolov8n.pt epochs=100 imgsz=640
   ```
3. Export the trained model:
   ```bash
   yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 simplify=True
   ```
4. Copy `best.onnx` → `models/object_detection/yolov8n.onnx`
5. Update `models/object_detection/coco_classes.txt` with your custom class names.
6. Update `config/config.yaml` → `models.object_detection.class_names_path`.

## Do NOT train on Raspberry Pi

Training YOLOv8 requires a GPU. Running training on the Pi would take
many hours or days and is not recommended.

## Class names file format

Plain text, one class name per line, matching the model output index:

```
person
bicycle
car
...
```
