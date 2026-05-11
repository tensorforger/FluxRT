# Flux Real-Time (FluxRT)

Real-time **FLUX.2** image editing pipeline optimized for consumer GPUs.

FluxRT enables low-latency live image transformation from **webcam**, **video files**, or other live inputs with **interactive prompt updates** and full **reference image conditioning** support.

Unlike offline image-to-image or video-to-video workflows, FluxRT supports:
- continuous streaming inference,
- live prompt changes,
- interactive reference-image conditioning,
- low-latency frame scheduling,
- spatially-aware KV reuse.

And unlike [Stream Diffusion](https://github.com/cumulo-autumn/streamdiffusion) it uses **Instruct image editing model** [FLUX.2-Klein](https://bfl.ai/models/flux-2-klein) that opens very precise control and refernce image conditioning.

The system is intended for:
- AI VTubing
- live stylization
- virtual try-on
- interactive art tools
- creative coding

With resolution `512 × 512`, FluxRT achieves:

| GPU                    | RTX 5090        |        RTX 4090 |
|------------------------|---------------- | --------------- |
| **Frame Rate**         | 20–40 FPS       | 15–30 FPS       |
| **End-to-End Latency** | ~0.2 seconds    | ~0.3 seconds    |

![Main Demo](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/main_demo.gif)

## Real-Time Reference Image Workflows

FluxRT natively supports all FLUX.2 reference image features.

Example: a simple real-time AI fitting room using a clothing item image as reference input.

![Reference Demo](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/reference_demo.gif)

Example 2: interactive paint-style app with iterative image updates.

![Interacive paint demo](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/paint_demo.gif)


# Quick Start

### System Requirements

|           | Minimal (int8 only) | Recommended         |
| --------- | ------------------- | ------------------- |
| GPU       | NVIDIA RTX 4090     | NVIDIA RTX 5090     |
| VRAM      | 20 GB               | 32 GB               |
| RAM       | 16 GB               | 32 GB               |


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

Ensure you have git lfs installed or run `git lfs install`

### RIFE Frame Interpolation Model

Original RIFE model lives here: [https://github.com/hzwer/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE)

But it is stored as a `pkl` file on the google drive.

So I have converted it to `safetensors` format and uploaded to [https://huggingface.co/TensorForger/RIFE-safetensors](https://huggingface.co/TensorForger/RIFE-safetensors)

You can simply clone the repo now:

```bash
cd FluxRT
git clone https://huggingface.co/TensorForger/RIFE-safetensors
```

### FLUX.2-klein-4B Model

Download from Hugging Face:

[https://huggingface.co/black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

```bash
cd FluxRT
git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
```

### Optional: int8 quantized FLUX.2-klein-4B

Download from Hugging Face:

[https://huggingface.co/aydin99/FLUX.2-klein-4B-int8](https://huggingface.co/aydin99/FLUX.2-klein-4B-int8)

This is required only if you want to use quantized `int8` inference.
Note that downloading unquantized `FLUX.2-klein-4B` model (previous point) is **still reqired**.

```bash
cd FluxRT
git clone https://huggingface.co/aydin99/FLUX.2-klein-4B-int8
```

<details>
<summary>Required directory structure</summary>

```text
FluxRT/
├── RIFE-safetensors/
│   └── flownet.safetensors
├── FLUX.2-klein-4B/
│   ├── model_index.json
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── transformer/
│   └── vae/
└── FLUX.2-klein-4B-int8/ (optional)
    ├── diffusion_pytorch_model.safetensors
    ├── text_encoder/
    ├── tokenizer/
    └── vae/
```

</details>


## 4. Run the Demos

To enable `int8` quantization you can either add falg `--int8` when running any script or set `enable_int8_quantization` to `true` in  the corresponding config.

### Gradio Demo

Interactive web UI with:

* live prompt editing
* live reference image swapping
* webcam input
* local video processing

```bash
python scripts/run_gradio_demo.py
```
Or to use `int8` quantization:

```bash
python scripts/run_gradio_demo.py --int8
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

### OpenCV Paint App

```bash
python scripts/run_cv2_paint.py
```

### Process local video with FluxRT
This script also supports CLI

```bash
python scripts/process_local_video.py --input input.mp4 --output out.mp4 --prompt "Turn this into oil on canvas art"
```

### Run performance benchmark

This will show throughput (FPS) with various dynamic area values and end-to-end latency.
The generated benchmark report will include the current configuration and hardware parameters.

```bash
python scripts/run_benchmark.py
```

You can check report generated on my machine in `benchmark_report.txt`

I would appreciate it if you could share your report in the [issues](https://github.com/tensorforger/FluxRT/issues), especially if it was generated on different hardware setup.

### How to use Lora

To enable lora:

1. Download lora weights to the repo root.
2. Add these lines in the corresponding config, in case of gradio demo it is `configs/config_with_reference.json` (example for [https://huggingface.co/Sawata97/flux2_4b_koni_animestyle](https://huggingface.co/Sawata97/flux2_4b_koni_animestyle)):

```json
"use_lora": true,
"lora_weights_path": "flux2_4b_koni_animestyle/Flux_klein_4b_anime_Koni.safetensors",
```

Loras work well with `--int8` flag too.
But note that there are still very few loras for `FLUX.2-Klein-4B` model.

# How It Works

FluxRT combines multiple system-level and model-level optimizations to enable real-time inference.

### Spatial KV Cache

Spatial KV Cache is a custom KV-cache variant tailored for rectified flow models.

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
| Demo         | ![Spatial Cache OFF](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/spatial_cache_off.gif)    | ![Spatial Cache ON](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/spatial_cache_on.gif)    |
| 0–10%        | 20 FPS                                                  | 50 FPS                                                |
| 50%          | 20 FPS                                                  | 30 FPS                                                |
| 90–100%      | 20 FPS                                                  | 20 FPS                                                |

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

