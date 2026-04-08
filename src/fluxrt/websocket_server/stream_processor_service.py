from fastapi import WebSocket, WebSocketDisconnect
from fluxrt import StreamProcessor, SharedTensor, crop_maximal_rectangle
import numpy as np
import cv2


class StreamProcessorService:
    def __init__(self, config_path: str):
        self.stream_processor = StreamProcessor(config_path)
        self.stream_processor.start()

    def get_input_shared_tensor_name(self) -> str:
        return self.stream_processor.get_input_shared_tensor_name()

    def get_output_shared_tensor_name(self) -> str:
        return self.stream_processor.get_output_shared_tensor_name()

    def set_prompt(self, prompt: str) -> None:
        return self.stream_processor.set_prompt(prompt)

    def set_steps(self, steps: int) -> None:
        return self.stream_processor.set_steps(steps)

    def set_seed(self, seed: int) -> None:
        return self.stream_processor.set_seed(seed)

    def get_resolution(self) -> dict:
        return self.stream_processor.get_resolution()

    async def handle_websocket_stream(self, websocket: WebSocket) -> None:
        await websocket.accept()

        resolution = self.get_resolution()

        input_shared_tensor = SharedTensor(
            (resolution["height"], resolution["width"], 3),
            create=False,
            name=self.get_input_shared_tensor_name(),
        )

        output_shared_tensor = SharedTensor(
            (resolution["height"], resolution["width"], 3),
            create=False,
            name=self.get_output_shared_tensor_name(),
        )

        try:
            while True:
                message = await websocket.receive_bytes()
                data = np.frombuffer(message, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame = crop_maximal_rectangle(
                    frame,
                    target_height=resolution["height"],
                    target_width=resolution["width"],
                )

                input_shared_tensor.copy_from(frame)

                result = output_shared_tensor.to_numpy()

                _, jpeg = cv2.imencode(".jpg", result)
                await websocket.send_bytes(jpeg.tobytes())

        except WebSocketDisconnect:
            return

        except Exception as e:
            await websocket.close()
            raise e

    def get_latest_frame_jpeg(self) -> bytes:
        resolution = self.get_resolution()

        output_shared_tensor = SharedTensor(
            (resolution["height"], resolution["width"], 3),
            create=False,
            name=self.get_output_shared_tensor_name(),
        )
        result = output_shared_tensor.to_numpy()

        success, jpeg_data = cv2.imencode(".jpg", result)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")

        return jpeg_data.tobytes()
