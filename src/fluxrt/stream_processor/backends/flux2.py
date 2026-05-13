import json

import cv2
import numpy as np
import torch
from PIL import Image
from accelerate import init_empty_weights
from diffusers.models import AutoencoderKLFlux2
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from transformers import AutoConfig, Qwen2TokenizerFast, Qwen3ForCausalLM

from fluxrt.stream_processor.pipeline import Flux2KleinPipeline
from fluxrt.stream_processor.transformer_flux2 import Flux2Transformer2DModel
from fluxrt.stream_processor.update_controller import UpdateController

from .base import BaseModelBackend


class Flux2Backend(BaseModelBackend):
    def load(self, config: dict, device: str) -> None:
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16

        if config.get("enable_int8_quantization", False):
            self._load_quantized(config, device)
        else:
            self._load_standard(config, device)

        h = config["resolution"]["height"]
        w = config["resolution"]["width"]

        reference_image_seq_len = None
        if config["use_reference_image"]:
            res = config["reference_image_resolution"]
            reference_image_seq_len = (res["width"] // 16) * (res["height"] // 16)

        self.update_controller = UpdateController(
            config,
            h,
            w,
            compression_ratio=16,
            reference_image_seq_len=reference_image_seq_len,
        )

        self.pipe = Flux2KleinPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            update_controller=self.update_controller,
            subprocess_config=config,
        )
        self.pipe.to(device)

        if config.get("use_lora", False):
            self.pipe.load_lora_weights(config.get("lora_weights_path", ""))

        self.reference_image = None
        self.prompt_embeds = None

    def _load_standard(self, config: dict, device: str) -> None:
        models_path = config["models_path"]
        dtype = self.dtype

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            f"{models_path}/scheduler", local_files_only=True, device=device
        )
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            f"{models_path}/transformer", local_files_only=True, device=device
        ).to(dtype)
        self.vae = AutoencoderKLFlux2.from_pretrained(
            f"{models_path}/vae", local_files_only=True, device=device
        ).to(dtype)
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            f"{models_path}/text_encoder", local_files_only=True
        ).to(device, dtype)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            f"{models_path}/tokenizer", local_files_only=True, device=device
        )

    def _load_quantized(self, config: dict, device: str) -> None:
        from optimum.quanto import requantize

        from fluxrt.stream_processor.quantized_flux2 import QuantizedFlux2Transformer2DModel

        models_path = config["models_path"]
        int8_models_path = config["int8_models_path"]
        dtype = self.dtype

        qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(
            int8_models_path, local_files_only=True
        )
        qtransformer.to(device=device, dtype=dtype)
        self.transformer = qtransformer._wrapped

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            f"{models_path}/scheduler", local_files_only=True, device=device
        )
        self.vae = AutoencoderKLFlux2.from_pretrained(
            f"{models_path}/vae", local_files_only=True, device=device
        ).to(device, dtype)

        cfg = AutoConfig.from_pretrained(
            f"{int8_models_path}/text_encoder", local_files_only=True
        )
        with init_empty_weights():
            text_encoder = Qwen3ForCausalLM(cfg)
        with open(f"{int8_models_path}/text_encoder/quanto_qmap.json") as f:
            qmap = json.load(f)
        state_dict = load_file(f"{int8_models_path}/text_encoder/model.safetensors")
        requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
        text_encoder.eval()
        text_encoder.to(device, dtype=dtype)
        self.text_encoder = text_encoder

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            f"{int8_models_path}/tokenizer", local_files_only=True
        )

    def compile(self) -> None:
        self.transformer = torch.compile(self.transformer)
        self.vae = torch.compile(self.vae)

    def encode_prompt(self, prompt: str) -> None:
        self.prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        self.update_controller.reset_cache()

    def reset_cache(self) -> None:
        self.update_controller.reset_cache()

    def set_reference_image(self, image: np.ndarray | None, resolution: dict) -> None:
        if image is not None:
            image = cv2.resize(image, (resolution["width"], resolution["height"]))
            self.reference_image = Image.fromarray(image)
        else:
            self.reference_image = Image.fromarray(
                np.zeros((resolution["height"], resolution["width"], 3), dtype=np.uint8)
            )
        self.update_controller.reset_cache()

    def set_mask(self, mask_tensor) -> None:
        self.update_controller.set_mask(mask_tensor)

    def process_frame(self, frame: np.ndarray, process_state: dict) -> np.ndarray:
        reference_list = [Image.fromarray(frame)]
        if self.config["use_reference_image"] and self.reference_image is not None:
            reference_list.append(self.reference_image)

        out = self.pipe(
            prompt_embeds=self.prompt_embeds,
            image=reference_list,
            height=self.config["resolution"]["height"],
            width=self.config["resolution"]["width"],
            guidance_scale=1.0,
            num_inference_steps=process_state["steps"],
            num_images_per_prompt=1,
            generator=torch.Generator(device=self.device).manual_seed(process_state["seed"]),
            output_type="np",
        )
        return (out.images[0] * 255).astype(np.uint8)
