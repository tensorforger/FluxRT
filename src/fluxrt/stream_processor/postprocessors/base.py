from abc import ABC, abstractmethod
import numpy as np


class BasePostProcessor(ABC):
    @abstractmethod
    def process(self, source_rgb: np.ndarray, driving_rgb: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to source_rgb using driving_rgb as a reference signal.
        Both arrays are uint8 RGB, shape (H, W, 3).
        Returns processed uint8 RGB array.
        """
        ...
