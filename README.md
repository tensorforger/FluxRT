# Flux Real-Time (FluxRT)

> Real-time image editing pipeline powered by the **FLUX.2-klein-4B** model, optimized for consumer GPUs.

![Video description](./demos/main_demo.mp4)

## Performance Highlights

On a single **NVIDIA RTX 5090**, FluxRT achieves:

| Metric | Value |
|--------|-------|
| Resolution | 512×512 |
| Frame Rate | ~20 FPS *(with interpolation)* |
| End-to-End Latency | ~0.3 seconds |

## How It Works

FluxRT combines several optimization techniques to enable real-time inference:

- **Seamless frame interpolation**: Efficient real-time interpolation reduces perceived latency
- **Multiprocessing & shared memory**: Parallelizes preprocessing, inference, and postprocessing
- **Model compilation**: All models are pre-compiled for faster execution

> **Early Development Notice**: This project is actively evolving. Some features may be unstable, and further optimizations are planned. Please [report issues](https://github.com/tensorforger/FluxRT/issues) if you encounter problems!


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
| **VRAM** | 32 GB or more |
| **RAM** | 64 GB recommended |
| **Python** | 3.12+ |
| **CUDA** | 12.8 (or adjust per your setup) |

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
