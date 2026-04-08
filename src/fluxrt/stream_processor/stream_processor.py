import multiprocessing
from multiprocessing import Process, Value, Manager, Event
from fluxrt.utils import SharedTensor
from fluxrt.stream_processor.model_inference_subprocess import (
    ModelInferenceSubprocess,
)
from fluxrt.stream_processor.output_scheduler_subprocess import (
    OutputSchedulerSubprocess,
)
from queue import Empty
import json
import time


class StreamProcessor:
    def __init__(self, config_path: str):
        self.config = self.parse_config(config_path)
        self.resolution = self.config["resolution"]
        output_batch_size = 2 ** self.config["interpolation_exp"]

        self.input_shared_tensor = SharedTensor(
            (self.resolution["height"], self.resolution["width"], 3), create=True
        )
        self.output_shared_tensor = SharedTensor(
            (self.resolution["height"], self.resolution["width"], 3), create=True
        )
        self.output_batch_shared_tensor = SharedTensor(
            (output_batch_size, self.resolution["height"], self.resolution["width"], 3),
            create=True,
        )

        multiprocessing.set_start_method("spawn", force=True)

        self.pack_is_ready = Value("b", False)
        self.last_processing_time = Value("f", 0.0)

        self.model_inference_subprocess = ModelInferenceSubprocess(
            self.config,
            self.input_shared_tensor.name,
            self.output_batch_shared_tensor.name,
            self.pack_is_ready,
            self.last_processing_time,
        )

        self.output_scheduler_subprocess = OutputSchedulerSubprocess(
            self.config,
            self.output_batch_shared_tensor.name,
            self.output_shared_tensor.name,
            self.pack_is_ready,
            self.last_processing_time,
        )

    def parse_config(self, config_path: str) -> dict:
        with open(config_path, "r") as file:
            return json.load(file)

    def start(self) -> None:
        self.model_inference_subprocess.start()
        self.output_scheduler_subprocess.start()

    def get_input_tensor(self) -> SharedTensor:
        return self.input_shared_tensor

    def get_output_tensor(self) -> SharedTensor:
        return self.output_shared_tensor

    def stop(self) -> None:
        self.model_inference_subprocess.stop()
        self.output_scheduler_subprocess.stop()
        self.input_shared_tensor.close_and_unlink()
        self.output_shared_tensor.close_and_unlink()
        self.output_batch_shared_tensor.close_and_unlink()

    def set_prompt(self, prompt: str) -> None:
        self.model_inference_subprocess.set_param(name="prompt", value=prompt)

    def set_steps(self, steps: int) -> None:
        self.model_inference_subprocess.set_param(name="steps", value=steps)

    def set_seed(self, seed: int) -> None:
        self.model_inference_subprocess.set_param(name="seed", value=seed)

    def set_param(self, name: str, value) -> None:
        self.model_inference_subprocess.set_param(name=name, value=value)

    def get_resolution(self) -> dict:
        return self.resolution

    def get_input_shared_tensor_name(self) -> str:
        return self.input_shared_tensor.name

    def get_output_shared_tensor_name(self) -> str:
        return self.output_shared_tensor.name
