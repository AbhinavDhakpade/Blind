# Depth Estimation Model (Optional)

## Status

**DISABLED by default.** Set `models.depth.enabled: true` in config to enable.

## Why disabled?

MiDaS-Small at 256×256 runs at approximately **600–900 ms per frame** on
Raspberry Pi 5 CPU (ARM Cortex-A76, ONNX Runtime). This is too slow for the
primary real-time pipeline, which targets ≥ 3 FPS.

The system achieves adequate distance estimation using:
- HC-SR04 ultrasonic sensor (metric, < 10 ms)
- Bounding-box area estimation (relative, < 1 ms)

## When to enable

- You have a Hailo-8 NPU HAT (reduces latency to ~20 ms)
- You are running in offline post-processing mode
- You need fine-grained depth maps across the full scene

## Expected file

```
models/depth/midas_small.onnx
```

## Model

**MiDaS-Small** — lightweight monocular depth estimation.

- Input:  `[1, 3, H, W]`  float32, normalised to [0, 1], RGB
- Output: `[1, H, W]`     relative inverse depth map
- Not metric — the output is relative depth, not metres

## Download / Export

```bash
pip install torch torchvision timm

python - <<'EOF'
import torch
# MiDaS-Small
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

dummy = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    midas, dummy, "midas_small.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=12,
)
EOF
```

Copy `midas_small.onnx` → `models/depth/midas_small.onnx`.
