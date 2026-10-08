# Performance – Vision BOB

> ⚠️ **NOT YET BENCHMARKED ON RASPBERRY PI 5**
>
> The figures below are estimates based on:
> - Community benchmarks for ARM Cortex-A76 (same CPU as Pi 5)
> - ONNX Runtime CPU inference reports
> - Ultralytics official benchmark tables
> - OpenCV DNN ARM benchmark reports
>
> **Do not treat these as verified measurements.**
> Run `scripts/benchmark_models.py` on the actual Raspberry Pi 5
> and update this file with real numbers.

---

## Estimated Per-Frame Latency (Pi 5 CPU, ONNX Runtime)

| Component | Estimated Latency | Basis |
|-----------|------------------|-------|
| Camera capture (PiCamera2) | 5–15 ms | picamera2 documentation |
| Ultrasonic sensor read | 5–25 ms | HC-SR04 datasheet (sample_count=3) |
| YOLOv8n ONNX (640×640) | 150–250 ms | Community ARM Cortex-A76 benchmarks |
| IoU tracker | < 2 ms | Python, N < 20 objects |
| YuNet face detection | 3–8 ms | OpenCV DNN ARM benchmarks |
| MobileFaceNet ONNX (112×112) | 10–20 ms | ARM inference reports |
| Distance fusion | < 1 ms | NumPy arithmetic |
| Risk engine | < 1 ms | Python rule logic |
| Decision engine | < 1 ms | Python logic |
| Alert (espeak subprocess) | async — no blocking | Runs in daemon thread |
| **Total pipeline (no face rec)** | **~160–280 ms** | Sum of above |
| **Total pipeline (with face rec)** | **~175–300 ms** | Sum of above |

**Effective FPS estimate:** ~3–6 FPS

This is acceptable for a pedestrian walking pace (~1.4 m/s).
At 3 FPS, the system updates every ~330 ms. At 1.4 m/s walking speed,
the user travels ~46 cm between frames — adequate for obstacle detection
with the ultrasonic sensor providing continuous metric distance.

---

## MiDaS-Small Depth (DISABLED by default)

| Model | Input Size | Estimated Pi 5 Latency |
|-------|-----------|----------------------|
| MiDaS-Small ONNX | 256×256 | ~600–900 ms |

**Conclusion:** Adding monocular depth would reduce total pipeline FPS to
< 1.5 FPS — unacceptable. Depth model is disabled by default.
Enable only if a hardware accelerator (Hailo-8 NPU) is available.

---

## Memory Usage Estimates

| Component | Estimated RAM |
|-----------|--------------|
| Raspberry Pi OS base | ~400 MB |
| Python + dependencies | ~200 MB |
| YOLOv8n ONNX | ~12 MB |
| YuNet ONNX | ~1 MB |
| MobileFaceNet ONNX | ~4 MB |
| Frame buffer (640×480×3) | ~1 MB |
| **Total estimated** | **~620 MB** |

Leaves ~3.4 GB free on a 4 GB Pi 5 — comfortable margin.

---

## Optimisation Roadmap

| Optimisation | Expected Improvement | Status |
|-------------|---------------------|--------|
| INT8 quantisation (YOLOv8n) | ~30% faster inference | Not implemented |
| Input resolution 416×416 | ~35% faster inference, minor accuracy drop | Try via config `input_size` |
| Face recognition every 3rd frame | ~2× pipeline speed | Not implemented |
| Hailo-8 NPU HAT | ~10–50× inference speedup | Hardware change |
| NCNN backend | ~5–15% vs ONNX on ARM | Alternative backend |

---

## How to Measure Real Performance

On the Raspberry Pi 5:

```bash
# Benchmark individual models
python scripts/benchmark_models.py --iterations 50

# Run full pipeline with performance logging
python app/main.py --debug

# Monitor CPU and temperature
watch -n 1 vcgencmd measure_temp
htop
```

The application logs aggregated FPS and per-component latency every 10 seconds
(configurable via `performance.stats_interval_s`).

---

## Results Template (fill in after Pi 5 testing)

| Metric | Target | Measured |
|--------|--------|----------|
| Object detection latency | < 300 ms | NOT BENCHMARKED |
| Face recognition latency | < 50 ms/face | NOT BENCHMARKED |
| Sensor latency | < 30 ms | NOT BENCHMARKED |
| Total pipeline latency | < 350 ms | NOT BENCHMARKED |
| Effective FPS | ≥ 3 FPS | NOT BENCHMARKED |
| CPU usage (steady state) | < 90% | NOT BENCHMARKED |
| RAM usage | < 1 GB | NOT BENCHMARKED |
| SoC temperature | < 75 °C | NOT BENCHMARKED |
