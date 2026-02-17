# Pong 2-Player — DMD Model + WebGPU

Neural diffusion model runs in the browser via ONNX Runtime Web + WebGPU.

## Setup (how we got here)

### 1. DMD model (toy-wm-private)

- **Train**: DMD 1-step distillation of pong2p teacher
  - Config: `configs/pong2p_dmd.yaml`
  - Checkpoint: `experiments/pong2p-dmd-1step/model.pt`
  - Architecture: d_model=320, n_heads=20, d_head=16, n_blocks=8

### 2. Export to ONNX

```bash
cd toy-wm-private
./run.sh python scripts/export_onnx.py \
  --config configs/pong2p_dmd.yaml \
  --checkpoint experiments/pong2p-dmd-1step/model.pt \
  --output static/onnx/pong2p_dmd.onnx
```

- Export uses `model.float()` for bf16→float32 (browser needs float32)
- KV-cache enabled for temporal context across frames

### 3. Web interface (onnx.html)

- **Constants** (must match model): `N_LAYERS=8`, `N_HEADS=20`, `D_HEAD=16`, `TOKS_PER_FRAME=65`, `N_WINDOW=30`
- **Model**: `pong2p_dmd.onnx` (~81MB)
- **Frame flip**: Model output is Y-flipped; we flip with `sy = H - 1 - floor(dy/scale)` when rendering to canvas
- **Providers**: WebGPU (faster) or WASM fallback. Use `?webgpu=1` for GPU.

### 4. Deploy

- Copy `pong2p_dmd.onnx` to `wendlerc.github.io/pong2p/`
- `.gitattributes`: `pong2p_dmd.onnx` excluded from LFS (81MB under GitHub 100MB limit)
- Live: https://wendlerc.github.io/pong2p/onnx.html

## Files

- `onnx.html` — Main demo (DMD model, WebGPU/WASM)
- `pong2p_dmd.onnx` — DMD 1-step ONNX model
- `client.html` — Classic Pong (no neural model)
- `debug.html` — Debug view
