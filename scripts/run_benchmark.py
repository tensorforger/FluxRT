import argparse

from fluxrt import StreamProcessor
from fluxrt.utils.scan_hardware import scan_hardware
import cv2
import json
import numpy as np
import time


def main():
    parser = argparse.ArgumentParser(description="Run FluxRT benchmark.")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args = parser.parse_args()

    config_path = "configs/benchmark_config.json"

    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()

    if args.int8:
        stream_processor.enable_quantization()
    stream_processor.start()

    resolution = stream_processor.get_resolution()

    print("Initializing...")
    while not stream_processor.is_ready():
        time.sleep(0.1)

    print("Warming up...")
    time.sleep(5)

    results = []

    frame = np.zeros((resolution["height"], resolution["width"], 3))
    aborted = False
    for dynamic_area in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        print(f"Testing with dynamic area: {dynamic_area * 100:.0f}%")
        start = time.time()
        c = 0
        sum_processing_time = 0.0
        sum_fps = 0.0
        while time.time() - start < 10:
            c += 1
            dynamic_width = int(resolution["width"] * dynamic_area)
            frame[:, 0:dynamic_width, :] = c * 16
            input_tensor.copy_from(frame)
            processed_frame = output_tensor.to_numpy()
            processing_time = stream_processor.get_last_processing_time()
            fps = 1.0 / processing_time
            fps *= 2 ** stream_processor.config.get("interpolation_exp", 0)
            sum_processing_time += processing_time
            sum_fps += fps

            cv2.imshow("Processed Stream", processed_frame)
            if cv2.waitKey(1000 // 25) & 0xFF == ord("q"):
                aborted = True
                break
        results.append((dynamic_area, sum_processing_time / c, sum_fps / c))
        if aborted:
            break

    print("Measuring end to end latency...")

    frame = np.zeros((resolution["height"], resolution["width"], 3))
    frame[:, : resolution["width"] // 2, :] = 255
    input_tensor.copy_from(frame)
    stream_processor.set_prompt("Repeat the image")
    for _ in range(100):
        processed_frame = output_tensor.to_numpy()
        cv2.imshow("Processed Stream", processed_frame)
        if cv2.waitKey(1000 // 25) & 0xFF == ord("q"):
            break
    frame[:, : resolution["width"] // 2 + 16, :] = 255
    start = time.time()
    input_tensor.copy_from(frame)
    while True:
        processed_frame = output_tensor.to_numpy()
        if np.any(processed_frame[:, resolution["width"] // 2 + 4 :, :] > 128):
            break
        cv2.imshow("Processed Stream", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    end_to_end_latency = time.time() - start

    cv2.destroyAllWindows()
    stream_processor.stop()

    print("\n####### Benchmark Report #######\n")
    print("Configuration:\n")
    print(json.dumps(stream_processor.config, indent=2, default=str))

    if args.int8:
        print("\nQuantization: int8 enabled through --int8 flag\n")

    print("\nHardware Information:\n")
    hardware_info = scan_hardware()
    print(json.dumps(hardware_info, indent=2, default=str))

    print("\nResults:\n")
    col_widths = (14, 20, 10)
    header = f"{'Dynamic Area':>{col_widths[0]}}  {'Processing Time (s)':>{col_widths[1]}}  {'FPS':>{col_widths[2]}}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for dynamic_area, processing_time, fps in results:
        print(
            f"{dynamic_area * 100:>{col_widths[0]}.0f}%"
            f"  {processing_time:>{col_widths[1]}.4f}"
            f"  {fps:>{col_widths[2]}.2f}"
        )
    print(separator)

    print(f"\nEnd-to-end latency: {end_to_end_latency:.4f} seconds\n")
    print("###### End of Report ######\n")


if __name__ == "__main__":
    main()
