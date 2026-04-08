from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle
import cv2


def main():
    config_path = "configs/stream_processor_config.json"

    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()
    stream_processor.start()
    stream_processor.set_prompt(
        "Turn this image into cyberpunk night street scene, red and blue neon lamps, cinematic ligh, bokeh"
    )

    resolution = stream_processor.get_resolution()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("assets/video.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = crop_maximal_rectangle(
            frame, resolution["height"], resolution["width"]
        )
        input_tensor.copy_from(resized_frame)
        processed_frame = output_tensor.to_numpy()

        cv2.imshow("processed stream", processed_frame)

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    stream_processor.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
