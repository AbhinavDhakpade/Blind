# Vision BOB – Complete Raspberry Pi 5 Setup Guide

---

## Table of Contents

1. [Raspberry Pi OS Preparation](#1-raspberry-pi-os-preparation)
2. [Python Setup](#2-python-setup)
3. [Camera Setup](#3-camera-setup)
4. [GPIO Setup](#4-gpio-setup)
5. [Dependency Installation](#5-dependency-installation)
6. [Model Installation](#6-model-installation)
7. [Dataset Setup](#7-dataset-setup)
8. [Face Enrollment](#8-face-enrollment)
9. [Configuration](#9-configuration)
10. [Simulation Test](#10-simulation-test)
11. [Hardware Test](#11-hardware-test)
12. [Running the Application](#12-running-the-application)
13. [Troubleshooting](#13-troubleshooting)
14. [Performance Tuning](#14-performance-tuning)
15. [Auto-start as a Service](#15-auto-start-as-a-service)

---

## 1. Raspberry Pi OS Preparation

### Flash SD card / SSD

1. Download **Raspberry Pi OS Bookworm 64-bit** from https://www.raspberrypi.com/software/
2. Flash with **Raspberry Pi Imager** (recommended).
3. In Imager's Advanced Options:
   - Set hostname
   - Enable SSH
   - Configure Wi-Fi
4. Boot the Pi and connect via SSH or directly.

### Update system

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

---

## 2. Python Setup

Raspberry Pi OS Bookworm ships Python 3.11.

```bash
python3 --version   # should be 3.11+

sudo apt install -y python3-venv python3-pip python3-dev
```

---

## 3. Camera Setup

### Raspberry Pi Camera Module 3

```bash
# Enable camera in raspi-config
sudo raspi-config
# → Interface Options → Camera → Enable

# Install picamera2
sudo apt install -y python3-picamera2

# Test camera
libcamera-still -o test.jpg
```

If using a USB webcam instead, set `camera.backend: opencv` in config.

---

## 4. GPIO Setup

### Enable GPIO (Raspberry Pi 5)

The Pi 5 uses the RP1 I/O chip, which the classic `RPi.GPIO` package does
**not** support. Use `rpi-lgpio`, a drop-in replacement with the same
`import RPi.GPIO` API:

```bash
sudo apt remove -y python3-rpi.gpio     # conflicts with rpi-lgpio
sudo apt install -y python3-rpi-lgpio
```

Never `pip install RPi.GPIO` on a Pi 5. The virtual environment in step 5
must be created with `--system-site-packages` so it can see this package
and `picamera2`.

### HC-SR04 Voltage Divider Wiring

The HC-SR04 echo pin outputs **5 V**. Raspberry Pi GPIO is **3.3 V** maximum.

**Required voltage divider on ECHO:**

```
HC-SR04 ECHO ──┤ 1kΩ ├──── GPIO24 ──┤ 2kΩ ├──── GND
```

This reduces the 5 V echo signal to ~3.3 V.

### Vibration Motor Circuit

```
GPIO18 (PWM) → 10kΩ → NPN transistor base (e.g. 2N2222)
Collector → Vibration motor (–)
Motor (+) → 5V
Emitter → GND
Flyback diode (1N4001) across motor terminals (cathode to +)
```

Do NOT connect the motor directly to GPIO18. The GPIO pin cannot supply
enough current and the back-EMF will damage the SoC.

---

## 5. Dependency Installation

```bash
# Clone project
git clone <your-repo-url> vision_bob
cd vision_bob

# Create virtual environment
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install base dependencies
pip install -r requirements.txt

# Install Raspberry Pi–specific dependencies
pip install -r requirements-pi.txt

# Install dev dependencies (testing only)
pip install -r requirements-dev.txt
```

### ONNX Runtime for Raspberry Pi 5

ONNX Runtime for ARM64:

```bash
pip install onnxruntime
```

If the pre-built wheel is not available for your OS version:

```bash
pip install onnxruntime --extra-index-url https://pkgindex.arm.com
```

### espeak-ng (TTS)

```bash
sudo apt install -y espeak-ng
espeak-ng "Hello world"   # test
```

---

## 6. Model Installation

Models are **not included** in this repository (file-size constraints).
You must obtain or export them on a PC and copy them to the Pi.

### Object Detection: YOLOv8n ONNX

On a PC with GPU:

```bash
pip install ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, simplify=True, opset=12)
"
```

Copy `yolov8n.onnx` → `models/object_detection/yolov8n.onnx`

### Face Detection: YuNet (OpenCV)

Download from OpenCV Zoo:

```bash
wget -O models/face_detection/face_detection_yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

### Face Recognition: MobileFaceNet ONNX

Download from InsightFace or export from a MobileFaceNet PyTorch checkpoint:

```bash
# Option: download pre-converted ONNX from InsightFace model zoo
# See models/face_recognition/README.md for the exact URL
```

Copy `mobilefacenet.onnx` → `models/face_recognition/mobilefacenet.onnx`

### Verify models are present

```bash
python scripts/check_environment.py
```

---

## 7. Dataset Setup

### Object detection dataset (optional – for custom training)

```
datasets/objects/images/train/   ← training images
datasets/objects/images/val/     ← validation images
datasets/objects/labels/train/   ← YOLO-format .txt annotations
```

Training happens on a GPU machine, not on the Pi. See
[`datasets/objects/README.md`](datasets/objects/README.md).

### Face dataset

```
datasets/faces/known/
    person_001/
        image1.jpg
        image2.jpg
    person_002/
        photo1.png
```

Person names are directory names. Use any identifier (no real names required).

---

## 8. Face Enrollment

After placing face images:

```bash
python scripts/enroll_faces.py
```

This reads `datasets/faces/known/`, runs each image through the face detector
and recognition model, and saves `data/face_embeddings/embeddings.pkl`.

The application loads this database at startup. Re-run enroll_faces whenever
you add new people.

---

## 9. Configuration

```bash
cp config/config.yaml config/config.local.yaml
nano config/config.local.yaml
```

Key settings to review:

```yaml
camera:
  backend: "picamera2"   # or "opencv" for USB webcam

sensor:
  backend: "gpio"        # "mock" for testing without sensor

hardware:
  vibration_pin: 18      # BCM GPIO pin for motor
  gpio_backend: "gpio"   # "mock" for testing

alerts:
  speech_backend: "espeak"
  haptic_backend: "gpio"
```

---

## 10. Simulation Test

Run the full pipeline without any hardware:

```bash
python app/main.py --simulation
```

Expected output:
```
INFO  Vision BOB starting up.
INFO  MockCamera initialised: 640x480 @ 15 fps (simulation)
INFO  MockUltrasonicSensor initialised (simulation)
INFO  MockObjectDetector loaded (simulation)
INFO  All components initialised.  Entering main loop.
INFO  SPEECH [MOCK]: car approaching
INFO  HAPTIC [MOCK] pulses=3 on=250ms off=80ms
```

Press **Ctrl+C** to stop.

---

## 11. Hardware Test

### Camera

```bash
python scripts/test_camera.py
```

### Ultrasonic sensor

```bash
python scripts/test_ultrasonic.py
```

### Vibration motor

```bash
python scripts/test_gpio.py
```

### Full environment check

```bash
python scripts/check_environment.py
```

### Run tests

```bash
pytest tests/ -v
```

---

## 12. Running the Application

### Standard run

```bash
python app/main.py
```

### With debug overlay (shows bounding boxes)

```bash
python app/main.py --debug
```

### With custom config

```bash
python app/main.py --config config/config.local.yaml
```

---

## 13. Troubleshooting

### Camera not found

```
CameraError: OpenCV could not open camera device 0
```

- Check `raspi-config` → Interface Options → Camera is enabled
- Try `libcamera-still -o test.jpg`
- For USB webcam: verify `device_index: 0` in config

### Sensor timeout

```
All measurements timed out.
```

- Check wiring: TRIG → GPIO23, ECHO (voltage-divided) → GPIO24
- Check voltage divider values
- Check HC-SR04 is powered from 5 V

### Model not found

```
ModelLoadError: Object detection model not found
```

- Follow Section 6 to export/download the model
- Verify the path in config matches the actual file name

### espeak-ng not found

```bash
sudo apt install -y espeak-ng
```

### onnxruntime import error

```bash
pip install onnxruntime
```

### GPIO permission denied

```bash
sudo usermod -aG gpio $USER
# Log out and back in
```

---

## 14. Performance Tuning

For maximum frame rate on Raspberry Pi 5:

1. **Reduce resolution** in config: `640×480` is the recommended default.
2. **Frame skipping**: The main loop processes every frame by default.
   For resource-constrained scenarios, run face recognition only every Nth frame.
3. **Keep depth disabled**: `models.depth.enabled: false` (default).
   MiDaS-Small at 256×256 takes ~800 ms on Pi 5 CPU.
4. **Use a faster SD card** (A2-rated) or SSD via USB 3.
5. **Check thermal throttling**: `vcgencmd measure_temp` — add a heatsink
   if temp exceeds 80 °C.

Expected performance (see [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)):

| Component | Estimated latency (Pi 5 CPU) |
|-----------|------------------------------|
| Camera capture | < 10 ms |
| YOLOv8n ONNX 640×640 | 150–250 ms |
| YuNet face detection | ~5 ms |
| MobileFaceNet embed | ~15 ms |
| Total pipeline | ~200–300 ms |
| Effective FPS | ~3–5 fps |

> These are estimates. Run `scripts/benchmark_models.py` on the Pi for actual numbers.

---

## 15. Auto-start as a Service

Create `/etc/systemd/system/vision_bob.service`:

```ini
[Unit]
Description=Vision BOB Assistive Vision System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/vision_bob
ExecStart=/home/pi/vision_bob/venv/bin/python app/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable vision_bob
sudo systemctl start vision_bob
sudo journalctl -u vision_bob -f   # view logs
```
