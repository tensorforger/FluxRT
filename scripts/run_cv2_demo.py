from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle
import cv2

import time
import numpy as np


def main():
    config_path = "configs/stream_processor_config.json"

    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()
    stream_processor.start()

    resolution = stream_processor.get_resolution()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("assets/video.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(1000 / fps)

    c = 0

    # width = 576
    # height = 320
    width = 576
    height = 320
    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
    time.sleep(20)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        c += 1
        if c == 5:
            stream_processor.set_prompt(
                "Turn this image into oil on canvas art of the in style of Wassily Kandinsky. Complex abstract geometry, triangular patterns."
            )

        if c == 100:
            stream_processor.set_prompt(
                "Turn this image into cyberpunk night street scene, red and blue neon lamps, cinematic lighting, bokeh"
            )

        if c == 300:
            stream_processor.set_prompt(
                "Turn this image into art in style of Van Gogh's Starry Night artwork. Oil on canvas, smooth colors."
            )

        if c == 500:
            stream_processor.set_prompt("This man is wearing black classic suit")

        if c == 700:
            stream_processor.set_prompt(
                "Turn this image into the nature scene in the sunny steppe, make the natural light, golden hour, sunset"
            )

        resized_frame = crop_maximal_rectangle(
            frame, resolution["height"], resolution["width"]
        )
        input_tensor.copy_from(resized_frame)
        processed_frame = output_tensor.to_numpy()

        fps = (
            1 / stream_processor.model_inference_subprocess.last_processing_time.value
        ) * 4

        text = f"Spatial cache is OFF, {fps:.2f} FPS"

        org = (576 // 4 + 20, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 255)
        thickness = 1

        cv2.putText(
            processed_frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA
        )

        out.write(processed_frame)
        cv2.imshow("processed stream", processed_frame)

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    stream_processor.stop()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
