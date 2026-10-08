# Object Detection Dataset

## Purpose

This directory is a placeholder for a custom object detection dataset.

For most use cases, the default YOLOv8n model trained on COCO 80 classes
is sufficient. This dataset structure is only needed if you want to train
a custom model with additional or different classes.

## Directory Structure

```
datasets/objects/
├── images/
│   ├── train/      ← training images (.jpg, .png)
│   ├── val/        ← validation images
│   └── test/       ← test images (optional)
├── labels/
│   ├── train/      ← YOLO-format .txt annotations
│   ├── val/
│   └── test/
└── README.md       ← this file
```

## Annotation Format (YOLO)

One `.txt` file per image, same base name as the image.
Each line: `class_id cx cy width height`
(all values normalised 0–1 relative to image dimensions)

Example (`image001.txt`):
```
0 0.512 0.432 0.145 0.310
2 0.280 0.650 0.210 0.140
```

## Dataset Configuration File

Create `datasets/objects/data.yaml`:

```yaml
path: datasets/objects
train: images/train
val:   images/val
test:  images/test

nc: 80       # number of classes
names:
  0: person
  1: bicycle
  2: car
  # ... (full COCO list or your custom classes)
```

## How to train a custom model

Training must be performed on a GPU machine (not Raspberry Pi):

```bash
pip install ultralytics
yolo train \
    data=datasets/objects/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=vision_bob_custom
```

## Exporting the trained model

```bash
yolo export \
    model=runs/detect/vision_bob_custom/weights/best.pt \
    format=onnx \
    imgsz=640 \
    simplify=True \
    opset=12
```

Copy `best.onnx` → `models/object_detection/yolov8n.onnx`

## Do NOT place huge datasets in the ZIP

This directory should contain only the structure and this README.
Place actual images on the training machine or in cloud storage.
