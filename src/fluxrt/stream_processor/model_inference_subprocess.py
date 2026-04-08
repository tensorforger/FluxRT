import torch
import time
import os
import cv2
import numpy as np
from multiprocessing import Process, Value, Manager
from copy import deepcopy
from queue import Empty
from PIL import Image

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLFlux2
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from fluxrt.stream_processor.interpolation_model import IFNet
from fluxrt.stream_processor.transformer_flux2 import Flux2Transformer2DModel
from fluxrt.utils.shared_tensor import SharedTensor
from fluxrt.stream_processor.pipeline import Flux2KleinPipeline


class ModelInferenceSubprocess:
    def __init__(
        self,
        config: dict,
        input_shared_tensor_name: str,
        output_batch_shared_tensor_name: str,
        pack_is_ready,
        last_processing_time,
    ):
        self.running = Value("b", False)
        self.process = None
        self.config = config
        self.height = self.config["resolution"]["height"]
        self.width = self.config["resolution"]["width"]
        self.resolution = self.config["resolution"]
        self.prompt = self.config["default_prompt"]
        self.input_shared_tensor_name = input_shared_tensor_name
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time

        manager = Manager()
        self.command_queue = manager.Queue()
        self.shared_state = manager.dict()
        self.interpolation_exp = self.config.get("interpolation_exp", 1)

    def init_process_state(self):
        self.device = "cuda"
        self.process_state = {
            "prompt": self.config["default_prompt"],
            "steps": self.config["default_steps"],
            "seed": self.config["default_seed"],
        }

    def load_models(self):
        def convert(param):
            return {
                k.replace("module.", ""): v for k, v in param.items() if "module." in k
            }

        self.interpolation_model = IFNet()
        self.interpolation_model.load_state_dict(
            convert(torch.load("interpolation_model/flownet.pkl"))
        )
        self.interpolation_model.to(self.device, torch.float16)
        self.interpolation_model.eval()

        device = "cuda"
        dtype = torch.bfloat16

        models_path = self.config["models_path"]
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            f"{models_path}/scheduler", device=device
        )
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            f"{models_path}/transformer", device=device
        ).to(dtype)
        self.vae = AutoencoderKLFlux2.from_pretrained(
            f"{models_path}/vae", device=device
        ).to(dtype)
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            f"{models_path}/text_encoder"
        ).to(device, dtype)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            f"{models_path}/tokenizer", device=device
        )

        if self.config["compile_models"]:
            self.transformer = torch.compile(
                self.transformer,
                dynamic=False,
            )
            self.vae = torch.compile(
                self.vae,
                dynamic=False,
            )

            self.interpolation_model = torch.compile(
                self.interpolation_model,
            )

        self.pipe = Flux2KleinPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
        )
        self.pipe.to(device)

    def update_prompt_embeds(self, prompt):
        self.prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )

    def init_shared_tensors(self):
        h, w = self.resolution["height"], self.resolution["width"]

        self.input_shared_tensor = SharedTensor(
            (h, w, 3),
            name=self.input_shared_tensor_name,
        )

        # All interpolated then one original
        output_batch_size = 2**self.interpolation_exp
        self.output_batch_shared_tensor = SharedTensor(
            (output_batch_size, h, w, 3),
            name=self.output_batch_shared_tensor_name,
        )

    def process_init(self):
        """
        Initializes all resources required by the inference subprocess.
        """
        self.init_process_state()
        self.init_shared_tensors()
        self.load_models()
        self.update_prompt_embeds(self.process_state["prompt"])
        self.previous_frame = None

        if self.config["use_reference_image"]:
            image = cv2.imread(self.config["reference_image_path"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resolution = self.config["reference_image_resolution"]
            image = cv2.resize(image, (resolution["width"], resolution["height"]))
            self.reference_image = Image.fromarray(image)

    def start(self):
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self):
        self.running.value = False
        if self.process:
            self.process.join()

    def set_param(self, name: str, value) -> None:
        self.command_queue.put(("set_param", (name, value)))

    def update_process_state(self) -> None:
        """
        Called by the internal process
        """
        try:
            while True:
                cmd, payload = self.command_queue.get_nowait()
                if cmd == "set_param":
                    name, value = payload
                    self.process_state[name] = value
                    if name == "prompt":
                        self.update_prompt_embeds(value)

        except Empty:
            pass

    def receive_frame(self):
        """
        Reads frame from input shared memory, converts to RGB float16 GPU tensors.
        """
        frame = self.input_shared_tensor.to_numpy()
        frame_gpu = (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )
        return frame_gpu

    def interpolate_and_send_frames(self, frame):
        """
        Takes one new generated frame (torch tensor, RGB, on GPU, float16)
        Interpolates according to interpolation_exp times.
        Batches to [interpolated frames, new frame].
        Converts and sends this to shared memory.
        """
        if self.previous_frame is None:
            self.previous_frame = frame

        if self.interpolation_exp == 0:
            frames_out = frame
        else:
            frames = torch.cat([self.previous_frame, frame], dim=0)
            with torch.no_grad():
                for _ in range(self.interpolation_exp):
                    B = frames.size(0)
                    prevs = frames[:-1]
                    nexts = frames[1:]
                    mids = self.interpolation_model(torch.cat([prevs, nexts], dim=1))
                    H, W = frames.shape[2:]
                    new_frames = torch.empty(
                        2 * B - 1, 3, H, W, device=frames.device, dtype=frames.dtype
                    )
                    new_frames[0::2] = frames
                    new_frames[1::2] = mids
                    frames = new_frames
            frames_out = frames[1:]

        frames_cpu = (
            frames_out.mul(255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        self.output_batch_shared_tensor.copy_from(frames_cpu[..., ::-1])
        self.pack_is_ready.value = True
        self.previous_frame = frame

    def update_time(self, prev_time):
        now = time.time()
        processing_time = now - prev_time
        self.last_processing_time.value = processing_time
        print("fps", 1 / processing_time)
        return now

    def process_frame_with_pipeline(self, frame):
        """
        Takes frame as np uint8 RGB array
        Returns frame as np uint8 RGB array
        """
        input_frame = Image.fromarray(frame)

        reference_list = [input_frame]
        if self.config["use_reference_image"]:
            reference_list.append(self.reference_image)

        out_image = self.pipe(
            prompt_embeds=self.prompt_embeds,
            image=reference_list,
            height=self.resolution["height"],
            width=self.resolution["width"],
            guidance_scale=1.0,
            num_inference_steps=self.process_state["steps"],
            num_images_per_prompt=1,
            generator=torch.Generator(device=self.device).manual_seed(
                self.process_state["seed"]
            ),
        ).images[0]
        out_image = np.asarray(out_image)
        return out_image

    def convert_np_to_torch(self, frame):
        frame = (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )
        return frame

    def process_main(self):
        self.process_init()
        prev_time = time.time()
        while self.running.value:
            self.update_process_state()
            frame = self.input_shared_tensor.to_numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.process_frame_with_pipeline(frame)
            frame = self.convert_np_to_torch(frame)
            self.interpolate_and_send_frames(frame)
            prev_time = self.update_time(prev_time)
