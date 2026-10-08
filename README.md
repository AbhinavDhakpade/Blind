# Vision BOB – Raspberry Pi 5 Assistive Vision System

> **Safety disclaimer:** Vision BOB is an **experimental research prototype**
> intended to assist visually impaired users. It must **not** be relied upon
> as the sole safety mechanism. Always use additional safety precautions.

---

## Project Overview

Vision BOB is a real-time assistive vision system designed to run on a
**Raspberry Pi 5**. It continuously captures frames from a camera, combines
computer-vision inference with ultrasonic distance sensing, and produces
prioritised haptic and speech alerts to help a visually impaired user
navigate safely.

### What it does

| Priority | Trigger | Alert |
|----------|---------|-------|
| 1 (highest) | Object is **approaching** | Haptic vibration |
| 2 | Static **obstacle ahead** | Spoken warning |
| 3 | **Crowd or person** nearby | Speech: "Hello [name]" or "Unknown person" |
| — | Clear path | Silence |

---

## Architecture

```
Camera (PiCamera2 / OpenCV)
        +
Ultrasonic HC-SR04
        ↓
┌───────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5 Pipeline                    │
│                                                               │
│  Object Detection (YOLOv8n ONNX)                             │
│  ↓                                                            │
│  IoU Tracker → MotionState (APPROACHING / STATIONARY / …)    │
│  ↓                                                            │
│  Face Recognition (YuNet + MobileFaceNet ONNX)               │
│  ↓                                                            │
│  Distance Engine (ultrasonic + bbox fusion)                  │
│  ↓                                                            │
│  Risk Engine → RiskAssessment (CLEAR / LOW / … / CRITICAL)   │
│  ↓                                                            │
│  Decision Engine → Priority-ordered DecisionAction           │
│  ↓                                                            │
│  Alert Manager → HapticAlert / SpeechAlert                   │
└───────────────────────────────────────────────────────────────┘
        ↓
   Capture Next Frame
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full detail.

---

## Hardware Requirements

| Component | Notes |
|-----------|-------|
| Raspberry Pi 5 (4 GB or 8 GB) | Main compute |
| Raspberry Pi Camera Module 3 | libcamera / picamera2 |
| HC-SR04 Ultrasonic Sensor | Distance measurement |
| Vibration Motor (5 V) | Haptic feedback (with transistor driver) |
| Speaker / USB DAC | Speech output |
| NPN transistor (2N2222 / BC547) | Motor driver |
| 1 kΩ + 2 kΩ resistors | Echo voltage divider |
| Flyback diode (1N4001) | Motor protection |

### GPIO Wiring

| Component | Raspberry Pi Pin (BCM) | Purpose |
|-----------|----------------------|---------|
| HC-SR04 VCC | 5 V (Pin 2) | Power |
| HC-SR04 GND | GND (Pin 6) | Ground |
| HC-SR04 TRIG | GPIO 23 (Pin 16) | Trigger pulse |
| HC-SR04 ECHO → 1kΩ → GPIO24 → 2kΩ → GND | GPIO 24 (Pin 18) | Echo (voltage-divided) |
| Motor driver base | GPIO 18 (Pin 12) | PWM control |
| Motor driver GND | GND (Pin 20) | Ground |

> ⚠️ **ECHO is 5 V** — always use a voltage divider or level-shifter.
> Direct connection to GPIO will damage the Raspberry Pi.

---

## Software Requirements

- Raspberry Pi OS Bookworm (64-bit recommended)
- Python 3.11+
- See [`requirements.txt`](requirements.txt) and [`requirements-pi.txt`](requirements-pi.txt)

---

## Quick Start

### 1. Clone and set up

```bash
git clone <your-repo-url> vision_bob
cd vision_bob
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Simulation test (no hardware needed)

```bash
python app/main.py --simulation --debug
```

### 3. Check environment

```bash
python scripts/check_environment.py
```

### 4. Install models

See [`models/*/README.md`](models/) for download/export instructions.

### 5. Enroll known faces (optional)

```bash
# Place images in datasets/faces/known/<person_name>/
python scripts/enroll_faces.py
```

### 6. Run on Raspberry Pi

```bash
python app/main.py
```

---

## Dataset Preparation

- **Objects:** [`datasets/objects/README.md`](datasets/objects/README.md)
- **Faces:** [`datasets/faces/README.md`](datasets/faces/README.md)

---

## Full Setup

See [`SETUP.md`](SETUP.md) for the complete Raspberry Pi installation guide.

---

## Model Selection Rationale

See [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md).

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## License

MIT — see [`LICENSE`](LICENSE).
