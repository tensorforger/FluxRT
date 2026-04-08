import numpy as np
from multiprocessing import shared_memory

from typing import Union


class SharedTensor:
    def __init__(
        self,
        shape: np.shape,
        dtype: np.dtype = np.uint8,
        name: str = None,
        create: bool = False,
    ):
        """
        Args:
            shape: tuple-like shape of the array.
            dtype: numpy dtype or something convertible (e.g. np.float32).
            name: name of existing shared memory (required if create=False).
            create: if True, create a new SharedMemory block.
        """
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape) * self.dtype.itemsize)

        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=self.size)
            self.name = self.shm.name
        else:
            if name is None:
                raise ValueError("name must be provided when create=False")
            self.shm = shared_memory.SharedMemory(name=name)
            self.name = name

        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def copy_from(self, tensor: Union["SharedTensor", np.ndarray]) -> None:
        """
        Copy data from another tensor into this shared tensor.

        Args:
            tensor: numpy ndarray or other SharedTensor
        """
        if isinstance(tensor, SharedTensor):
            src = tensor.array
        else:
            src = np.asarray(tensor)

        if src.shape != self.shape:
            raise ValueError(
                f"Shape mismatch: source {src.shape} != target {self.shape}"
            )

        if src.dtype != self.dtype:
            src = src.astype(self.dtype, copy=False)

        np.copyto(self.array, src)

    def to_numpy(self) -> np.ndarray:
        """
        Copies shared tensor to regulat numpy ndarray

        Returns:
            A regular numpy ndarray that is a copy of the shared tensor's data.
        """
        return self.array.copy()

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()

    def close_and_unlink(self) -> None:
        self.close()
        self.unlink()
