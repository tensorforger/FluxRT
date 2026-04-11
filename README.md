# Flux Real-Time (FluxRT)

Real-time image editing pipeline powered by the **FLUX.2-klein-4B** model, optimized for consumer GPUs.

![Demo](./demos_gif/main_demo.gif)

On a single **NVIDIA RTX 5090**, FluxRT achieves:

| Metric                 | Value        |
| ---------------------- | ------------ |
| **Resolution**         | 512 × 512    |
| **Frame Rate**         | 25–50 FPS    |
| **End-to-End Latency** | ~0.3 seconds |

## How It Works

FluxRT combines multiple system-level and model-level optimizations to enable real-time inference.

### Spatial Cache

Spatial Cache is a custom KV-cache variant tailored for rectified flow models.

FLUX.2 models exhibit highly similar diffusion trajectories across adjacent frames. This temporal coherence allows reuse of intermediate computations between frames. Instead of recomputing all tokens, FluxRT selectively caches and reuses tokens from previous frames.

We initially applied caching to:

* Text tokens
* Reference image tokens

However, real-world video streams often contain static or slowly changing regions (e.g., backgrounds). FluxRT extends caching to these spatial regions, further reducing per-frame computation.

In practice, only **20–50% of tokens** need to be recomputed per frame.

This results in:

* Higher throughput (FPS)
* Lower latency
* Reduced GPU utilization

#### Implementation Details

* Keys and Values are cached **per token, per layer, per diffusion step**
* Cached values are reused directly in attention layers
* The model forward pass is patched to skip computation for cached tokens, including:

  * Feed-forward networks (FFN)
  * Linear projections
  * Query computation
  * Attention operations

#### Performance Comparison

Below is a comparison against the baseline (resolution: 576 × 320, 2 inference steps per frame, interpolation ×4):

| Dynamic Area | Baseline (No Cache)                                     | With Spatial Cache                                    |
| ------------ | ------------------------------------------------------- | ----------------------------------------------------- |
| Demo         | ![Spatial Cache OFF](./demos_gif/spatial_cache_off.gif) | ![Spatial Cache ON](./demos_gif/spatial_cache_on.gif) |
| 0–10%        | 20 FPS                                                  | 50 FPS                                                |
| 50%          | 20 FPS                                                  | 35 FPS                                                |
| 90–100%      | 20 FPS                                                  | 25 FPS                                                |

> The spatial update mask is shown in the corner:
> white pixels = recomputed, black pixels = reused.



### Real-Time Frame Interpolation

To ensure smooth visual transitions, FluxRT integrates real-time frame interpolation using the **RIFE** model.
It generates intermediate frames between model outputs.
Interpolation factor is configurable (see `interpolation_exp` in the config)
This significantly improves perceived motion smoothness without increasing core model latency.


### Multiprocessing & Shared Memory

FluxRT uses a multi-process architecture to decouple computation, I/O, and rendering:

* **Main Process**

  * Handles non-blocking input/output
  * Manages UI and user interaction

* **Inference Process**

  * Runs all models
  * Executes the generation loop sequentially

* **Output Scheduler Process**

  * Streams interpolated frames
  * Ensures smooth playback timing

To minimize overhead, inter-process communication uses **shared memory**, enabling near-zero-copy frame transfer and minimal latency.


### Model Compilation

All models are compiled using **TorchInductor** to maximize runtime performance.

> **Early Development Notice**
> This project is actively evolving. Some features may be unstable, and further optimizations are planned.
> Please report issues: [https://github.com/tensorforger/FluxRT/issues](https://github.com/tensorforger/FluxRT/issues)


## Minimal Example

```python
from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle
import cv2

def main():
    config_path = "configs/stream_processor_config.json"
    
    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()

    stream_processor.start()
    stream_processor.set_prompt(
        "Turn this image into cyberpunk night street scene, "
        "red and blue neon lamps, cinematic lighting, bokeh"
    )

    resolution = stream_processor.get_resolution()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = crop_maximal_rectangle(
            frame, resolution["height"], resolution["width"]
        )
        input_tensor.copy_from(resized_frame)

        processed_frame = output_tensor.to_numpy()
        cv2.imshow("Processed Stream", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    stream_processor.stop()

if __name__ == "__main__":
    main()
```

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA RTX 5090 or higher |
| **VRAM** | 32 GB+ |
| **RAM** | 64 GB recommended |
| **Python** | 3.12+ |
| **CUDA** | 12.8+ |

## Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/tensorforger/FluxRT
cd FluxRT
```

### 2. Install Python Dependencies

```bash
# Create and activate conda environment
conda create -n fluxrt python=3.12 pip -y
conda activate fluxrt

# Install PyTorch with CUDA support (adjust cu128 if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Download Required Models

Place models in the following structure:

```
FluxRT/
├── interpolation_model/
│   └── flownet.pkl                 # RIFE frame interpolation model
└── FLUX.2-klein-4B/
    ├── model_index.json
    ├── scheduler/
    │   └── scheduler_config.json
    ├── text_encoder/
    │   ├── config.json
    │   ├── generation_config.json
    │   ├── model-00001-of-00002.safetensors
    │   ├── model-00002-of-00002.safetensors
    │   └── model.safetensors.index.json
    ├── tokenizer/
    │   ├── added_tokens.json
    │   ├── chat_template.jinja
    │   ├── merges.txt
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   ├── tokenizer.json
    │   └── vocab.json
    ├── transformer/
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors
    └── vae/
        ├── config.json
        └── diffusion_pytorch_model.safetensors
```

#### Model Sources

- **RIFE Frame Interpolation** (`flownet.pkl`)  
   [Google Drive Download](https://drive.google.com/file/d/1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX/view)  
   *Backup*: [ECCV2022-RIFE Repository](https://github.com/hzwer/ECCV2022-RIFE)

- **FLUX.2-klein-4B Base Model**  
   [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

### 4. Run the OpenCV Demo *(Python-only)*

A minimal demo to validate the pipeline using OpenCV:

```bash
cd FluxRT
conda activate fluxrt
python scripts/run_cv2_demo.py
```

### 5. Run the Web UI Demo *(Experimental)*

>  **Under Development**: This feature may be unstable. Use with caution.

FluxRT includes a local Web UI for intuitive prompt editing and device selection. The frontend communicates with the backend via **WebSocket**.

#### Prerequisites
- [Node.js](https://nodejs.org/en/download) installed

#### Step-by-Step

1. **Build the frontend**
   ```bash
   cd FluxRT/web_ui
   npm install
   npm run build
   ```

2. **Install Web UI Python dependencies**
   ```bash
   cd FluxRT
   conda install uvicorn
   pip install -r requirements_web_ui.txt
   ```

3. **Start the backend (WebSocket server)**
   ```bash
   cd FluxRT
   conda activate fluxrt
   python scripts/run_websocket_server.py
   ```

4. **Start the frontend** *(in a separate terminal)*
   ```bash
   cd FluxRT/web_ui
   npm run start
   ```

5. **Open in browser**  
    [`http://localhost:3000`](http://localhost:3000)

##  Contributing

FluxRT is a research-oriented project under active development. We welcome:

-  Bug reports with reproduction steps
-  Feature requests and improvement suggestions
-  Code contributions via pull requests
