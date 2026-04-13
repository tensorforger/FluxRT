import threading
import time

import cv2
import gradio as gr

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle

default_prompt = (
    "Turn this image into cyberpunk night street scene, "
    "red and blue neon lamps, cinematic lighting, bokeh"
)

stream_processor = None
input_tensor = None
output_tensor = None
resolution = None

stop_video_event = threading.Event()
processor_lock = threading.Lock()


def get_processor():
    global stream_processor, input_tensor, output_tensor, resolution

    if stream_processor is None:
        stream_processor = StreamProcessor("configs/stream_processor_config.json")
        stream_processor.start()
        stream_processor.set_prompt(default_prompt)

        input_tensor = stream_processor.get_input_tensor()
        output_tensor = stream_processor.get_output_tensor()
        resolution = stream_processor.get_resolution()

    return stream_processor, input_tensor, output_tensor, resolution


def to_bgr(frame):
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def to_rgb(frame):
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def process_frame(frame):
    _, input_tensor, output_tensor, resolution = get_processor()
    frame = crop_maximal_rectangle(frame, resolution["height"], resolution["width"])

    with processor_lock:
        input_tensor.copy_from(frame)
        processed = output_tensor.to_numpy()
    return frame, processed


def set_prompt(prompt: str):
    sp, _, _, _ = get_processor()
    sp.set_prompt(prompt)


def switch_mode(mode: str, request: gr.Request | None):
    if mode == "webcam":
        stop_video_event.set()
    elif mode == "local":
        stop_video_event.clear()

    webcam_visible = mode == "webcam"
    local_visible = mode == "local"
    return (
        gr.update(visible=webcam_visible),
        gr.update(visible=local_visible),
        None,
        None,
        gr.update(value=None),
        None,
        None,
    )


def process_webcam(frame):
    if frame is None:
        return None, None

    input_frame, processed = process_frame(to_bgr(frame))
    return to_rgb(input_frame), to_rgb(processed)


def process_local_video(video_path: str | None, request: gr.Request | None):
    if not video_path:
        return None, None

    stop_video_event.clear()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_time = 1.0 / fps

    try:
        while not stop_video_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            start = time.time()
            input_frame, processed = process_frame(frame)
            yield to_rgb(input_frame), to_rgb(processed)

            elapsed = time.time() - start
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
    finally:
        cap.release()


def main():
    get_processor()

    with gr.Blocks() as demo:
        mode = gr.Radio(
            choices=["webcam", "local"],
            value="webcam",
            label="Mode",
        )

        with gr.Column(visible=True) as webcam_panel:
            with gr.Row():
                webcam_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Input stream",
                )
                webcam_output = gr.Image(
                    streaming=True,
                    label="Processed stream",
                )

        with gr.Column(visible=False) as local_panel:
            video_file = gr.File(
                label="Choose local video",
                file_count="single",
                file_types=["video"],
                type="filepath",
            )
            with gr.Row():
                local_input = gr.Image(label="Input stream")
                local_output = gr.Image(streaming=True, label="Processed stream")

        prompt = gr.Textbox(
            value=default_prompt,
            label="Prompt",
            lines=3,
        )

        mode.change(
            switch_mode,
            inputs=mode,
            outputs=[
                webcam_panel,
                local_panel,
                webcam_input,
                webcam_output,
                video_file,
                local_input,
                local_output,
            ],
        )

        webcam_input.stream(
            process_webcam,
            inputs=webcam_input,
            outputs=[webcam_input, webcam_output],
            stream_every=0.04,
            concurrency_limit=1,
        )

        video_file.change(
            process_local_video,
            inputs=video_file,
            outputs=[local_input, local_output],
            concurrency_limit=1,
        )

        prompt.change(set_prompt, inputs=prompt, outputs=None)

    demo.queue(default_concurrency_limit=1).launch()


if __name__ == "__main__":
    main()
