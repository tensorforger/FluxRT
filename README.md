# Flux Real-Time (FluxRT)

Real-time **FLUX.2** image editing pipeline optimized for consumer GPUs.

FluxRT enables low-latency live image transformation from **webcam**, **video files**, or other live inputs with **interactive prompt updates** and full **reference image conditioning** support.

On a single **NVIDIA RTX 5090**, FluxRT achieves:

| Metric                 | Value        |
|------------------------|--------------|
| **Resolution**         | 512 × 512    |
| **Frame Rate**         | 25–50 FPS    |
| **End-to-End Latency** | ~0.2 seconds |

![Main Demo](./assets/main_demo.gif)

## Real-Time Reference Image Workflows

FluxRT natively supports all FLUX.2 reference image features.

Example: a simple real-time AI fitting room using a clothing item image as reference input.

![Reference Demo](./assets/reference_demo.gif)


# Quick Start

### System Requirements

| Component | Requirement               |
| --------- | ------------------------- |
| GPU       | NVIDIA RTX 5090 or higher |
| VRAM      | 32 GB+                    |
| RAM       | 64 GB recommended         |
| Python    | 3.12+                     |
| CUDA      | 12.8+                     |


## 1. Clone the Repository

```bash
git clone https://github.com/tensorforger/FluxRT
cd FluxRT
````

## 2. Install Dependencies

```bash
# Create environment
conda create -n fluxrt python=3.12 pip -y
conda activate fluxrt

# Install PyTorch with CUDA support (adjust if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
pip install -r requirements.txt
pip install -e .
```

## 3. Download Required Models

### RIFE Frame Interpolation Model

Download `flownet.pkl`

* Google Drive
* Backup: [https://github.com/hzwer/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE)

### FLUX.2-klein-4B Base Model

Download from Hugging Face:

[https://huggingface.co/black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

Place models into the project folder.

<details>
<summary>Required directory structure</summary>

```text
FluxRT/
├── interpolation_model/
│   └── flownet.pkl
└── FLUX.2-klein-4B/
    ├── model_index.json
    ├── scheduler/
    ├── text_encoder/
    ├── tokenizer/
    ├── transformer/
    └── vae/
```

</details>


## 4. Run the Demos

### Gradio Demo

Interactive web UI with:

* live prompt editing
* webcam input
* local video processing

```bash
python scripts/run_gradio_demo.py
```

Then open:

```text
http://127.0.0.1:7860/
```

### OpenCV Demo

Minimal local demo:

```bash
python scripts/run_cv2_demo.py
```


### OpenCV Reference Image Demo

```bash
python scripts/run_cv2_reference_demo.py
```

# How It Works

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
| Demo         | ![Spatial Cache OFF](./assets/spatial_cache_off.gif)    | ![Spatial Cache ON](./assets/spatial_cache_on.gif)    |
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


# Integration


We'd be happy if you'd like to build something on top of this project. We've created a high-level API for easy integration.

Here is the minimal example:

```python
from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle
import cv2

config_path = "configs/stream_processor_config.json"

stream_processor = StreamProcessor(config_path)
input_tensor = stream_processor.get_input_tensor()
output_tensor = stream_processor.get_output_tensor()

stream_processor.start()
stream_processor.set_prompt(
    "Turn this image into cyberpunk night street scene, "
    "red and blue neon lamps, cinematic lighting"
)

resolution = stream_processor.get_resolution()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = crop_maximal_rectangle(
        frame,
        resolution["height"],
        resolution["width"]
    )

    input_tensor.copy_from(resized)

    output = output_tensor.to_numpy()
    cv2.imshow("FluxRT", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
stream_processor.stop()
```

#  Contributing

FluxRT is a research-oriented project under active development.
Please report any issues: [https://github.com/tensorforger/FluxRT/issues](https://github.com/tensorforger/FluxRT/issues).

Feature requests and improvement suggestions are welcome.

