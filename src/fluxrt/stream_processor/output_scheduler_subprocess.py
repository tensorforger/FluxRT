from multiprocessing import Process, Value
from fluxrt.utils.shared_tensor import SharedTensor
import time


class OutputSchedulerSubprocess:
    def __init__(
        self,
        config: dict,
        output_batch_shared_tensor_name: str,
        output_shared_tensor_name: str,
        pack_is_ready,
        last_processing_time,
    ):
        self.config = config
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.output_shared_tensor_name = output_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time

        self.running = Value("b", False)
        self.process = None

        self.interpolation_exp = self.config.get("interpolation_exp", 1)
        self.batch_size = 2**self.interpolation_exp

    def start(self) -> None:
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self) -> None:
        self.running.value = False
        if self.process:
            self.process.join()

    def process_init(self) -> None:
        """
        Called by the internal process
        """
        height = self.config["resolution"]["height"]
        width = self.config["resolution"]["width"]

        self.output_batch_shared_tensor = SharedTensor(
            (self.batch_size, height, width, 3),
            name=self.output_batch_shared_tensor_name,
        )
        self.output_shared_tensor = SharedTensor(
            (height, width, 3),
            name=self.output_shared_tensor_name,
        )

    def process_main(self) -> None:
        self.process_init()

        while self.running.value:
            if not self.pack_is_ready.value:
                continue

            proc_time = min(max(self.last_processing_time.value, 0.001), 1.0)
            sleep_interval = proc_time / self.batch_size

            for i in range(self.batch_size):
                self.output_shared_tensor.copy_from(
                    self.output_batch_shared_tensor.array[i]
                )
                if i < self.batch_size - 1:
                    time.sleep(sleep_interval)

            self.pack_is_ready.value = False
