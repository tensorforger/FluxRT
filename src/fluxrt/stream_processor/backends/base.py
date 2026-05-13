from abc import ABC, abstractmethod
import numpy as np


class BaseModelBackend(ABC):
    @abstractmethod
    def load(self, config: dict, device: str) -> None: ...

    @abstractmethod
    def encode_prompt(self, prompt: str) -> None: ...

    @abstractmethod
    def process_frame(self, frame: np.ndarray, process_state: dict) -> np.ndarray: ...

    def compile(self) -> None:
        pass

    def reset_cache(self) -> None:
        pass

    def set_reference_image(self, image: np.ndarray | None, resolution: dict) -> None:
        pass

    def set_mask(self, mask_tensor) -> None:
        pass
