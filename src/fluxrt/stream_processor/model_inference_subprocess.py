import time
from multiprocessing import Manager, Process, Value
from queue import Empty

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from fluxrt.stream_processor.backends.flux2 import Flux2Backend
from fluxrt.stream_processor.interpolation_model import IFNet
from fluxrt.stream_processor.liveportrait_postprocessor import LivePortraitPostProcessor
from fluxrt.utils.shared_tensor import SharedTensor

_BACKENDS = {
    "flux2": Flux2Backend,
}


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
        self.memory_reserved = Value("i", 0)
        self.process = None
        self.config = config
        self.height = config["resolution"]["height"]
        self.width = config["resolution"]["width"]
        self.resolution = config["resolution"]
        self.prompt = config["default_prompt"]
        self.logging = config.get("logging", True)
        self.input_shared_tensor_name = input_shared_tensor_name
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time
        self.interpolation_exp = config.get("interpolation_exp", 1)

        manager = Manager()
        self.command_queue = manager.Queue()
        self.shared_state = manager.dict()

    def enable_quantization(self):
        """
        Should be called before the subprocess is started.
        """
        self.config["enable_int8_quantization"] = True

    def _create_backend(self):
        name = self.config.get("backend", "flux2")
        cls = _BACKENDS.get(name)
        if cls is None:
            raise ValueError(f"Unknown backend: {name!r}. Available: {list(_BACKENDS)}")
        return cls()

    def init_process_state(self):
        self.device = "cuda"
        self.process_state = {
            "prompt": self.config["default_prompt"],
            "steps": self.config["default_steps"],
            "seed": self.config["default_seed"],
        }

    def load_models(self):
        self.interpolation_model = IFNet()
        self.interpolation_model.load_state_dict(
            load_file("RIFE-safetensors/flownet.safetensors")
        )
        self.interpolation_model.to("cuda", dtype=torch.float16)
        self.interpolation_model.eval()

        self.backend = self._create_backend()
        self.backend.load(self.config, self.device)

        if self.config.get("compile_models", False):
            self.backend.compile()
            self.interpolation_model = torch.compile(self.interpolation_model)

        self.lip_processor: LivePortraitPostProcessor | None = None
        self.lip_active = False
        lp_cfg = self.config.get("lip_transfer", {})
        if lp_cfg.get("enable", False):
            self.lip_processor = LivePortraitPostProcessor(models_dir=lp_cfg["models_dir"])

    def init_shared_tensors(self):
        h, w = self.resolution["height"], self.resolution["width"]
        self.input_shared_tensor = SharedTensor(
            (h, w, 3), name=self.input_shared_tensor_name
        )
        # All interpolated then one original
        output_batch_size = 2**self.interpolation_exp
        self.output_batch_shared_tensor = SharedTensor(
            (output_batch_size, h, w, 3), name=self.output_batch_shared_tensor_name
        )

    def process_init(self):
        """
        Initializes all resources required by the inference subprocess.
        """
        self.init_process_state()
        self.init_shared_tensors()
        self.load_models()
        self.backend.encode_prompt(self.process_state["prompt"])
        self.previous_frame = None

        if self.config.get("use_reference_image", False):
            resolution = self.config["reference_image_resolution"]
            image = cv2.imread(self.config.get("reference_image_path", ""))
            if image is None:
                image = np.zeros(
                    (resolution["height"], resolution["width"], 3), dtype=np.uint8
                )
                print(
                    "Warning: use_reference_image is set to true but no valid reference_image_path is provided."
                )
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.backend.set_reference_image(image, resolution)

        target_fps = self.config.get("target_fps", None)
        self.target_base_processing_time = None
        if target_fps is not None:
            target_base_fps = target_fps / (2**self.interpolation_exp)
            self.target_base_processing_time = 1 / target_base_fps

    def start(self):
        self.running.value = True
        self.process = Process(target=self.process_main, daemon=True)
        self.process.start()

    def stop(self):
        self.running.value = False
        if self.process:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()

    def set_param(self, name: str, value) -> None:
        self.command_queue.put(("set_param", (name, value)))

    def set_reference_image(self, image: np.ndarray | None) -> None:
        """
        Update the reference image on the fly.
        image: numpy uint8 RGB array
        Only valid when use_reference_image is true in config.
        """
        if not self.config.get("use_reference_image", False):
            raise ValueError(
                "set_reference_image called but use_reference_image is not enabled in the stream processor config"
            )
        self.command_queue.put(("set_reference_image", image))

    def set_mask(self, mask) -> None:
        """
        Update the mask on the fly.
        mask: numpy uint8 array of shape (h // compression_ratio, w // compression_ratio).
        Only valid when mask_calculation_method is set to manual in config.
        """
        if self.config.get("mask_calculation_method", "auto") != "manual":
            raise ValueError(
                "set_mask called but mask_calculation_method is not set to manual in the config"
            )
        self.command_queue.put(("set_mask", mask))

    def set_lip_transfer(self, enabled: bool) -> None:
        self.command_queue.put(("set_lip_transfer", enabled))

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
                        self.backend.encode_prompt(value)
                elif cmd == "set_reference_image":
                    resolution = self.config["reference_image_resolution"]
                    self.backend.set_reference_image(payload, resolution)
                elif cmd == "set_lip_transfer":
                    self.lip_active = payload
                elif cmd == "set_mask":
                    mask_tensor = (
                        torch.from_numpy(payload).unsqueeze(0).to(self.device)
                    )
                    self.backend.set_mask(mask_tensor)

        except Empty:
            pass

    def interpolate_frames(self, frame):
        """
        Takes one new generated frame (torch tensor, RGB, on GPU, float16)
        Interpolates according to interpolation_exp times.
        Batches to [interpolated frames, new frame].
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
        self.previous_frame = frame
        return frames_cpu[..., ::-1]

    def send_frames(self, frames):
        self.output_batch_shared_tensor.copy_from(frames)

    def sync_fps_and_send(self, prev_time, frames):
        now = time.time()
        processing_time = now - prev_time

        if self.target_base_processing_time is not None:
            sleep_time = max(0, self.target_base_processing_time - processing_time)
            time.sleep(sleep_time)
            now = time.time()

        processing_time = now - prev_time
        self.last_processing_time.value = processing_time
        self.send_frames(frames)
        self.pack_is_ready.value = True
        self.memory_reserved.value = torch.cuda.memory_reserved() // (1024 * 1024)

        if self.logging:
            print(
                f"base fps: {(1 / processing_time):.2f}, "
                f"interpolated fps: {(1 / processing_time * 2**self.interpolation_exp):.2f}"
            )
        return now

    def convert_np_to_torch(self, frame):
        return (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )

    def process_main(self):
        self.process_init()
        prev_time = time.time()
        while self.running.value:
            self.update_process_state()
            frame = self.input_shared_tensor.to_numpy()
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.backend.process_frame(original_frame, self.process_state)
            if self.lip_processor is not None and self.lip_active:
                frame = self.lip_processor.process(frame, original_frame)
            frame = self.convert_np_to_torch(frame)
            frames = self.interpolate_frames(frame)
            prev_time = self.sync_fps_and_send(prev_time, frames)
