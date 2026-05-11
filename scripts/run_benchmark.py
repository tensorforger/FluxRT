import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from fluxrt import StreamProcessor
from fluxrt.utils.benchmark_report import (
    DEFAULT_DYNAMIC_AREAS,
    build_benchmark_settings,
    build_report_summary,
    interpolation_multiplier,
    parse_dynamic_areas,
    render_markdown_report,
    summarize_case,
    validate_output_path,
)
from fluxrt.utils.scan_hardware import scan_hardware


def make_frame(resolution: dict, dynamic_area: float, frame_index: int) -> np.ndarray:
    frame = np.zeros((resolution["height"], resolution["width"], 3), dtype=np.uint8)
    dynamic_width = int(resolution["width"] * dynamic_area)
    if dynamic_width > 0:
        frame[:, :dynamic_width, :] = (frame_index * 17) % 255
    return frame


def wait_for_ready(stream_processor: StreamProcessor, timeout: float) -> None:
    start = time.perf_counter()
    while not stream_processor.is_ready():
        failure = get_subprocess_failure(stream_processor)
        if failure:
            raise RuntimeError(failure)
        if time.perf_counter() - start > timeout:
            raise TimeoutError("startup readiness")
        time.sleep(0.05)


def wait_for_generated_frame(
    stream_processor: StreamProcessor, previous_count: int, timeout: float
) -> dict:
    start = time.perf_counter()
    while time.perf_counter() - start <= timeout:
        failure = get_subprocess_failure(stream_processor)
        if failure:
            raise RuntimeError(failure)
        stats = stream_processor.get_benchmark_stats()
        generated_frame_count = int(stats.get("generated_frame_count", 0))
        if generated_frame_count > previous_count:
            return stats
        time.sleep(0.005)
    raise TimeoutError("next generated frame")


def wait_for_memory_reset_ack(
    stream_processor: StreamProcessor, revision: int, timeout: float
) -> dict:
    start = time.perf_counter()
    while time.perf_counter() - start <= timeout:
        failure = get_subprocess_failure(stream_processor)
        if failure:
            raise RuntimeError(failure)
        stats = stream_processor.get_benchmark_stats()
        ack = int(stats.get("benchmark_memory_reset_revision", 0))
        if ack >= revision:
            return stats
        time.sleep(0.005)
    raise TimeoutError("memory reset acknowledgement")


def get_subprocess_failure(stream_processor: StreamProcessor) -> str | None:
    subprocesses = {
        "model_inference": stream_processor.model_inference_subprocess,
        "output_scheduler": stream_processor.output_scheduler_subprocess,
    }
    for name, subprocess in subprocesses.items():
        process = getattr(subprocess, "process", None)
        if process is None or process.pid is None:
            continue
        if process.exitcode is not None:
            return f"{name} subprocess exited with code {process.exitcode}"
    return None


def show_output_frame(output_tensor) -> bool:
    processed_frame = output_tensor.to_numpy()
    cv2.imshow("Processed Stream", processed_frame)
    return bool(cv2.waitKey(1) & 0xFF == ord("q"))


def run_generated_frame(
    stream_processor: StreamProcessor,
    input_tensor,
    output_tensor,
    resolution: dict,
    dynamic_area: float,
    frame_index: int,
    previous_count: int,
    timeout: float,
    show_window: bool,
) -> tuple[dict, bool]:
    input_tensor.copy_from(make_frame(resolution, dynamic_area, frame_index))
    stats = wait_for_generated_frame(stream_processor, previous_count, timeout)
    user_quit = show_output_frame(output_tensor) if show_window else False
    return stats, user_quit


def run_case(
    stream_processor: StreamProcessor,
    input_tensor,
    output_tensor,
    resolution: dict,
    dynamic_area: float,
    warmup_frames: int,
    frames: int | None,
    case_duration: float | None,
    timeout: float,
    show_window: bool,
    memory_reset_revision: int,
    multiplier: int,
) -> tuple[dict | None, int, str | None]:
    previous_count = int(
        stream_processor.get_benchmark_stats().get("generated_frame_count", 0)
    )

    for frame_index in range(warmup_frames):
        stats, user_quit = run_generated_frame(
            stream_processor,
            input_tensor,
            output_tensor,
            resolution,
            dynamic_area,
            frame_index,
            previous_count,
            timeout,
            show_window,
        )
        previous_count = int(stats.get("generated_frame_count", previous_count))
        if user_quit:
            return None, memory_reset_revision, "user_quit"

    memory_reset_revision += 1
    stream_processor.reset_benchmark_memory_stats(memory_reset_revision)
    wait_for_memory_reset_ack(stream_processor, memory_reset_revision, timeout)

    previous_count = int(
        stream_processor.get_benchmark_stats().get("generated_frame_count", 0)
    )
    processing_times = []
    last_stats = stream_processor.get_benchmark_stats()
    measurement_mode = "duration" if case_duration is not None else "frames"
    start = time.perf_counter()
    observed_samples = 0
    generated_frames = 0

    while True:
        if frames is not None and generated_frames >= frames:
            break
        if case_duration is not None and observed_samples > 0:
            if time.perf_counter() - start >= case_duration:
                break

        stats, user_quit = run_generated_frame(
            stream_processor,
            input_tensor,
            output_tensor,
            resolution,
            dynamic_area,
            warmup_frames + observed_samples,
            previous_count,
            timeout,
            show_window,
        )
        current_count = int(stats.get("generated_frame_count", previous_count))
        generated_frames += current_count - previous_count
        previous_count = current_count
        last_stats = stats
        processing_time = float(stats.get("last_processing_time", 0.0) or 0.0)
        if processing_time > 0:
            processing_times.append(processing_time)
        observed_samples += 1
        if user_quit:
            elapsed = time.perf_counter() - start
            case = summarize_case(
                dynamic_area,
                processing_times,
                elapsed,
                multiplier,
                last_stats.get("cuda_memory"),
                measurement_mode,
                case_duration,
                generated_frames,
                observed_samples,
            )
            return case, memory_reset_revision, "user_quit"

    elapsed = time.perf_counter() - start
    case = summarize_case(
        dynamic_area,
        processing_times,
        elapsed,
        multiplier,
        last_stats.get("cuda_memory"),
        measurement_mode,
        case_duration,
        generated_frames,
        observed_samples,
    )
    return case, memory_reset_revision, None


def measure_legacy_latency(
    stream_processor: StreamProcessor,
    input_tensor,
    output_tensor,
    resolution: dict,
    timeout: float,
    show_window: bool,
) -> dict:
    frame = np.zeros((resolution["height"], resolution["width"], 3), dtype=np.uint8)
    frame[:, : resolution["width"] // 2, :] = 255
    input_tensor.copy_from(frame)
    stream_processor.set_prompt("Repeat the image")

    for _ in range(100):
        processed_frame = output_tensor.to_numpy()
        if show_window:
            cv2.imshow("Processed Stream", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return {"legacy_end_to_end_latency_s": None, "status": "user_quit"}

    frame[:, : resolution["width"] // 2 + 16, :] = 255
    start = time.perf_counter()
    input_tensor.copy_from(frame)
    while time.perf_counter() - start <= timeout:
        processed_frame = output_tensor.to_numpy()
        if np.any(processed_frame[:, resolution["width"] // 2 + 4 :, :] > 128):
            return {
                "legacy_end_to_end_latency_s": time.perf_counter() - start,
                "status": "ok",
            }
        if show_window:
            cv2.imshow("Processed Stream", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return {"legacy_end_to_end_latency_s": None, "status": "user_quit"}

    return {"legacy_end_to_end_latency_s": None, "status": "timeout"}


def run_git(args: list[str]) -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def get_source_revision() -> dict:
    commit = run_git(["rev-parse", "--short", "HEAD"])
    dirty_output = run_git(["status", "--short", "--untracked-files=no"])
    return {
        "git_commit": commit or "unknown",
        "git_dirty": None if dirty_output is None else bool(dirty_output),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FluxRT benchmark.")
    parser.add_argument(
        "--config",
        default="configs/benchmark_config.json",
        help="Path to the stream processor config.",
    )
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization.")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable the OpenCV preview window. Recommended for shared reports.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional Markdown report path. Parent directory must already exist.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON report path. Parent directory must already exist.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Generated frames to measure per dynamic area. Default: 30 unless --case-duration is supplied.",
    )
    parser.add_argument(
        "--case-duration",
        type=float,
        default=None,
        help="Optional duration in seconds to measure each dynamic area.",
    )
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--dynamic-areas", default=DEFAULT_DYNAMIC_AREAS)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        dynamic_areas = parse_dynamic_areas(args.dynamic_areas)
        validate_output_path(args.output, "--output")
        validate_output_path(args.json_output, "--json-output")
    except ValueError as exc:
        parser.error(str(exc))

    if args.frames is not None and args.case_duration is not None:
        parser.error("--frames and --case-duration are mutually exclusive.")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be greater than 0.")
    if args.case_duration is not None and args.case_duration <= 0:
        parser.error("--case-duration must be greater than 0.")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be greater than or equal to 0.")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0.")

    frames = args.frames
    if frames is None and args.case_duration is None:
        frames = 30

    measurement_mode = "duration" if args.case_duration is not None else "frames"
    show_window = not args.no_window
    stream_processor = StreamProcessor(args.config)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()
    if args.int8:
        stream_processor.enable_quantization()

    settings = build_benchmark_settings(
        stream_processor.config,
        args.config,
        args.int8,
        dynamic_areas,
        args.warmup_frames,
        measurement_mode,
        frames,
        args.case_duration,
        args.timeout,
        show_window,
    )

    status = {
        "completed": False,
        "aborted": False,
        "abort_reason": "none",
        "cases_completed": 0,
        "cases_requested": len(dynamic_areas),
    }
    startup = {
        "startup_ready_s": None,
        "ready_vram": {"available": False},
    }
    cases = []
    legacy_latency = {"legacy_end_to_end_latency_s": None, "status": "not_run"}
    memory_reset_revision = 0

    try:
        print("Initializing...")
        startup_start = time.perf_counter()
        stream_processor.start()
        wait_for_ready(stream_processor, args.timeout)
        startup["startup_ready_s"] = time.perf_counter() - startup_start
        startup["ready_vram"] = stream_processor.get_benchmark_stats().get(
            "cuda_memory", {"available": False}
        )

        resolution = stream_processor.get_resolution()
        multiplier = interpolation_multiplier(stream_processor.config)

        for dynamic_area in dynamic_areas:
            print(f"Benchmarking dynamic area {dynamic_area * 100:.0f}%...")
            case, memory_reset_revision, abort_reason = run_case(
                stream_processor,
                input_tensor,
                output_tensor,
                resolution,
                dynamic_area,
                args.warmup_frames,
                frames,
                args.case_duration,
                args.timeout,
                show_window,
                memory_reset_revision,
                multiplier,
            )
            if case is not None:
                cases.append(case)
                status["cases_completed"] = len(cases)
                print(
                    f"  base_fps={case['base_fps']:.2f}, "
                    f"interpolated_fps={case['interpolated_fps']:.2f}, "
                    f"avg={case['avg_processing_ms']:.1f} ms"
                )
            if abort_reason:
                status["aborted"] = True
                status["abort_reason"] = abort_reason
                break

        if not status["aborted"]:
            print("Running legacy latency probe...")
            legacy_latency = measure_legacy_latency(
                stream_processor,
                input_tensor,
                output_tensor,
                resolution,
                args.timeout,
                show_window,
            )
            if legacy_latency.get("status") == "user_quit":
                status["aborted"] = True
                status["abort_reason"] = "user_quit"

    except TimeoutError as exc:
        status["aborted"] = True
        status["abort_reason"] = f"timeout: {exc}"
    except Exception as exc:
        status["aborted"] = True
        status["abort_reason"] = f"error: {type(exc).__name__}: {exc}"
    finally:
        if show_window:
            cv2.destroyAllWindows()
        stream_processor.stop(timeout=5.0)

    status["completed"] = (
        not status["aborted"] and status["cases_completed"] == status["cases_requested"]
    )
    report = {
        "status": status,
        "command": " ".join(sys.argv),
        "source": get_source_revision(),
        "settings": settings,
        "environment": scan_hardware(),
        "startup": startup,
        "cases": cases,
        "summary": build_report_summary(cases),
        "legacy_latency_probe": legacy_latency,
    }
    markdown = render_markdown_report(report)
    print()
    print(markdown)

    if args.output:
        Path(args.output).write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote Markdown report to: {args.output}")
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"Wrote JSON report to: {args.json_output}")


if __name__ == "__main__":
    main()
