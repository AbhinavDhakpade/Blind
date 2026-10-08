# Architecture – Vision BOB

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│                                                             │
│  PiCamera2Camera        HC-SR04UltrasonicSensor             │
│  (BGR numpy frame)      (distance_m + valid flag)           │
└──────────────────┬──────────────────────┬───────────────────┘
                   │                      │
                   ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    DETECTION LAYER                          │
│                                                             │
│  ONNXObjectDetector                                         │
│  YOLOv8n (640×640 ONNX Runtime CPU)                        │
│  → List[Detection(class_id, class_name, confidence, bbox)] │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRACKING LAYER                           │
│                                                             │
│  IoUTracker                                                 │
│  → List[TrackedObject(track_id, detection, motion_state,   │
│                        age, missed_frames)]                 │
│                                                             │
│  MotionState: APPROACHING | STATIONARY | RECEDING | UNKNOWN │
│  (computed from bbox-area growth rate over history window)  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │◄─────────────── Frame also goes to:
                   │
┌─────────────────────────────────────────────────────────────┐
│                 FACE RECOGNITION LAYER                      │
│  (only invoked when a "person" class is detected)           │
│                                                             │
│  YuNet (OpenCV DNN) → face bounding boxes                  │
│  MobileFaceNet (ONNX) → 128-d L2-normalised embedding      │
│  Cosine similarity → embeddings.pkl database               │
│  → List[FaceResult(KNOWN/UNKNOWN/NO_FACE, name, conf)]     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  DISTANCE ENGINE                            │
│                                                             │
│  Input:  detections + sensor_reading + frame dimensions    │
│  Method: weighted blend of:                                 │
│    ultrasonic_weight × US_distance                          │
│    + (1 - ultrasonic_weight) × bbox_area_estimate           │
│                                                             │
│  US sensor is attributed to the centre-most detection.      │
│  All other detections use bbox-only estimate.               │
│                                                             │
│  → List[DistanceEstimate(distance_m, zone, source)]        │
│    zone: FAR | MEDIUM | NEAR | VERY_NEAR                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     RISK ENGINE                             │
│                                                             │
│  Computes a numeric risk score per tracked object:          │
│    score = zone_base                                        │
│           × class_priority_multiplier                       │
│           × motion_multiplier                               │
│           × detection_confidence                            │
│           × centre_position_multiplier                      │
│                                                             │
│  The highest-scoring object determines the frame's level:  │
│    CLEAR < LOW < MEDIUM < HIGH < CRITICAL                   │
│                                                             │
│  → RiskAssessment(level, object_type, distance_m, zone,    │
│                   direction, motion_state, crowd, faces)    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   DECISION ENGINE                           │
│                                                             │
│  Stateless priority logic:                                  │
│                                                             │
│  P1: level >= MEDIUM AND motion == APPROACHING              │
│      → DecisionAction(HAPTIC_ALERT)                        │
│                                                             │
│  P2: level >= MEDIUM (stationary obstacle)                  │
│      → DecisionAction(SPEECH_OBSTACLE)                     │
│                                                             │
│  P3: crowd==True OR object=="person" OR known/unknown face  │
│      → DecisionAction(IDENTIFY_PERSON, name or None)       │
│                                                             │
│  else → DecisionAction(NONE) — clear path                  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    ALERT MANAGER                            │
│                                                             │
│  Applies per-action-type cooldown (configurable ms).        │
│                                                             │
│  HAPTIC_ALERT  → GPIOHapticAlert (PWM vibration motor)     │
│                  + brief speech for context                 │
│  SPEECH_OBSTACLE → EspeakSpeechAlert                       │
│  IDENTIFY_PERSON → EspeakSpeechAlert                       │
│                                                             │
│  Mock implementations available for all hardware.           │
└─────────────────────────────────────────────────────────────┘
                   │
                   ▼
            Capture Next Frame
```

---

## Component Responsibilities

### Camera (`app/camera/`)
- Abstracts PiCamera2, OpenCV VideoCapture, and Mock backends
- Returns `Frame(image: np.ndarray, timestamp, frame_id)`
- Manages camera lifecycle (init, capture, shutdown)

### Sensors (`app/sensors/`)
- HC-SR04 driver via the RPi.GPIO API (rpi-lgpio on Pi 5), ranging in a background thread; `measure()` returns the latest reading and never blocks
- Returns `SensorReading(distance_m, valid, error)`
- Median-of-N measurements for stability
- Mock sensor generates sinusoidal distance variation

### Detection (`app/detection/`)
- YOLOv8n ONNX Runtime inference
- Normalised pre-processing (640×640 RGB, /255)
- NMS via OpenCV
- Returns `Detection(class_id, class_name, confidence, bbox, center)`

### Tracking (`app/tracking/`)
- IoU-based greedy matching
- Tracks bbox-area history for approach detection
- Linear regression slope on area history → MotionState

### Face Recognition (`app/face/`)
- YuNet face detector (OpenCV DNN, < 5 ms)
- MobileFaceNet ONNX embedding extraction (~15 ms/face)
- Cosine similarity against pre-built `.pkl` database
- `FaceResult(KNOWN/UNKNOWN/NO_FACE, name, confidence, bbox)`

### Distance Engine (`app/distance/`)
- Fuses ultrasonic + bbox-area estimate
- Attributes sensor to centre-most detection
- Classifies into FAR / MEDIUM / NEAR / VERY_NEAR zones

### Risk Engine (`app/risk/`)
- Scores each tracked object with a multiplicative formula
- Produces `RiskAssessment` with level, direction, crowd flag, faces

### Decision Engine (`app/decision/`)
- Purely stateless priority logic
- No side effects; returns a single `DecisionAction`

### Alert Manager (`app/alerts/`)
- Routes actions to hardware
- Enforces cooldown per action type
- `AlertManager` is hardware-agnostic; receives injected implementations

---

## Data Flow Summary

```
Frame → Detections → Tracked Objects ─┐
                                      ├→ Risk → Decision → Alert
Sensor → SensorReading ──────────────┤
                                      │
Frame → Face Results ────────────────┘
```

---

## Safety Note

Vision BOB is an **experimental prototype**. It must not be the sole
safety system for navigation. Trained models may produce false positives
or false negatives. Environmental conditions (lighting, occlusion, sensor
angle) will affect reliability.
