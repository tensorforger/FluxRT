import argparse

from fluxrt import StreamProcessor
from fluxrt.utils.scan_hardware import scan_hardware
import cv2
import json
import numpy as np
import time
import torch


def main():
    parser = argparse.ArgumentParser(description="Run FluxRT benchmark.")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to benchmark.md instead of printing",
    )
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

    results = []

    print("Warming up...")
    time.sleep(5)

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
    latency_measured = True
    while True:
        processed_frame = output_tensor.to_numpy()
        if np.any(processed_frame[:, resolution["width"] // 2 + 4 :, :] > 240):
            break
        cv2.imshow("Processed Stream", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if time.time() - start > 10:
            latency_measured = False
            break
    end_to_end_latency = time.time() - start

    cv2.destroyAllWindows()
    reserved_memory = stream_processor.get_reserved_memory() / 1024
    stream_processor.stop()

    lines = []
    lines.append("# FluxRT Benchmark Report\n")

    lines.append("## Configuration\n")
    lines.append("```json")
    lines.append(json.dumps(stream_processor.config, indent=2, default=str))
    lines.append("```\n")

    if args.int8:
        lines.append("> **Quantization:** int8 enabled via `--int8` flag\n")

    lines.append("## Hardware Information\n")
    hardware_info = scan_hardware(stream_processor.config.get("device"))
    lines.append("```json")
    lines.append(json.dumps(hardware_info, indent=2, default=str))
    lines.append("```\n")

    lines.append("## Results\n")
    lines.append("| Dynamic Area | Processing Time (s) | FPS |")
    lines.append("|-------------:|--------------------:|----:|")
    for dynamic_area, processing_time, fps in results:
        lines.append(
            f"| {dynamic_area * 100:.0f}% | {processing_time:.4f} | {fps:.2f} |"
        )
    lines.append("")

    if latency_measured:
        lines.append(f"**End-to-end latency:** {end_to_end_latency:.4f} s\n")
    else:
        lines.append("**End-to-end latency:** measurement failed (timeout)\n")

    lines.append(f"**Reserved GPU memory:** {reserved_memory:.4f} GB")

    report = "\n".join(lines)

    if args.save:
        with open("benchmark.md", "w") as f:
            f.write(report)
        print("Report saved to benchmark.md")
    else:
        print(report)


if __name__ == "__main__":
    main()
