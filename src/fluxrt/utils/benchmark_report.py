import os
from pathlib import Path


DEFAULT_DYNAMIC_AREAS = "0,0.1,0.25,0.5,0.75,0.9,1.0"


def parse_dynamic_areas(value: str) -> list[float]:
    dynamic_areas = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            dynamic_area = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid dynamic area '{item}'.") from exc
        if dynamic_area < 0.0 or dynamic_area > 1.0:
            raise ValueError(
                f"Invalid dynamic area '{item}'. Values must be between 0.0 and 1.0."
            )
        dynamic_areas.append(dynamic_area)

    if not dynamic_areas:
        raise ValueError("At least one dynamic area is required.")
    return dynamic_areas


def validate_output_path(path: str | None, label: str) -> None:
    if path is None:
        return
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"{label} points to a directory: {path}")
    parent = output_path.parent
    if parent and not parent.exists():
        raise ValueError(f"{label} parent directory does not exist: {parent}")


def interpolation_multiplier(config: dict) -> int:
    return 2 ** int(config.get("interpolation_exp", 0))


def sanitize_path(value) -> str:
    if value is None or value == "":
        return "unset"
    normalized = os.path.normpath(str(value))
    name = os.path.basename(normalized)
    return name or str(value)


def _resolution_text(config: dict) -> str:
    resolution = config.get("resolution", {})
    height = resolution.get("height", "unknown")
    width = resolution.get("width", "unknown")
    return f"{height}x{width}"


def build_benchmark_settings(
    config: dict,
    config_path: str,
    int8_cli_override: bool,
    dynamic_areas: list[float],
    warmup_frames: int,
    measurement_mode: str,
    frames: int | None,
    case_duration: float | None,
    timeout: float,
    window_enabled: bool,
) -> dict:
    multiplier = interpolation_multiplier(config)
    return {
        "config_path": sanitize_path(config_path),
        "prompt": config.get("default_prompt"),
        "steps": config.get("default_steps"),
        "seed": config.get("default_seed"),
        "resolution": _resolution_text(config),
        "dynamic_area_order": dynamic_areas,
        "warmup_frames_per_case": warmup_frames,
        "measurement_mode": measurement_mode,
        "measured_frames_per_case": frames if measurement_mode == "frames" else None,
        "case_duration_s": case_duration if measurement_mode == "duration" else None,
        "cache_reset_between_cases": False,
        "timeout_s": timeout,
        "window_enabled": window_enabled,
        "interpolation": {
            "interpolation_exp": config.get("interpolation_exp", 0),
            "interpolation_multiplier": multiplier,
        },
        "runtime_features": {
            "compile_models": config.get("compile_models"),
            "enable_spatial_cache": config.get("enable_spatial_cache"),
            "enable_int8_quantization": config.get("enable_int8_quantization"),
            "int8_cli_override": int8_cli_override,
            "use_reference_image": config.get("use_reference_image"),
            "reference_image_resolution": config.get("reference_image_resolution"),
            "use_lora": config.get("use_lora", False),
            "target_fps": config.get("target_fps"),
        },
        "paths": {
            "models_path": sanitize_path(config.get("models_path")),
            "int8_models_path": sanitize_path(config.get("int8_models_path")),
            "reference_image_path": sanitize_path(config.get("reference_image_path")),
            "lora_weights_path": sanitize_path(config.get("lora_weights_path")),
        },
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] + (
        sorted_values[upper_index] - sorted_values[lower_index]
    ) * fraction


def summarize_case(
    dynamic_area: float,
    processing_times_s: list[float],
    elapsed_s: float,
    multiplier: int,
    memory_stats: dict | None,
    measurement_mode: str,
    case_duration: float | None,
    generated_frames: int | None = None,
    observed_samples: int | None = None,
) -> dict:
    if generated_frames is None:
        generated_frames = len(processing_times_s)
    if observed_samples is None:
        observed_samples = len(processing_times_s)
    output_frames = generated_frames * multiplier
    processing_times_ms = [value * 1000.0 for value in processing_times_s]
    avg_processing_ms = (
        sum(processing_times_ms) / len(processing_times_ms)
        if processing_times_ms
        else None
    )
    base_fps = generated_frames / elapsed_s if elapsed_s > 0 else None
    interpolated_fps = base_fps * multiplier if base_fps is not None else None

    return {
        "dynamic_area": dynamic_area,
        "measurement_mode": measurement_mode,
        "case_duration_s": case_duration if measurement_mode == "duration" else None,
        "elapsed_s": elapsed_s,
        "observed_samples": observed_samples,
        "generated_frames": generated_frames,
        "output_frames": output_frames,
        "counter_gap_frames": max(generated_frames - observed_samples, 0),
        "avg_processing_ms": avg_processing_ms,
        "p50_processing_ms": percentile(processing_times_ms, 0.50),
        "p95_processing_ms": percentile(processing_times_ms, 0.95),
        "base_fps": base_fps,
        "interpolated_fps": interpolated_fps,
        "vram": memory_stats or {"available": False},
    }


def build_report_summary(cases: list[dict]) -> dict:
    base_values = [
        case["base_fps"] for case in cases if case.get("base_fps") is not None
    ]
    interpolated_values = [
        case["interpolated_fps"]
        for case in cases
        if case.get("interpolated_fps") is not None
    ]
    peak_allocated_values = [
        case.get("vram", {}).get("peak_allocated_mb")
        for case in cases
        if case.get("vram", {}).get("peak_allocated_mb") is not None
    ]
    peak_reserved_values = [
        case.get("vram", {}).get("peak_reserved_mb")
        for case in cases
        if case.get("vram", {}).get("peak_reserved_mb") is not None
    ]
    return {
        "scenario_average_base_fps": (
            sum(base_values) / len(base_values) if base_values else None
        ),
        "scenario_average_interpolated_fps": (
            sum(interpolated_values) / len(interpolated_values)
            if interpolated_values
            else None
        ),
        "max_peak_allocated_mb": (
            max(peak_allocated_values) if peak_allocated_values else None
        ),
        "max_peak_reserved_mb": (
            max(peak_reserved_values) if peak_reserved_values else None
        ),
    }


def _format_value(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def _render_key_values(values: dict) -> list[str]:
    return [f"- {key}: {_format_value(value)}" for key, value in values.items()]


def render_markdown_report(report: dict) -> str:
    settings = report["settings"]
    source = report.get("source", {})
    status = report.get("status", {})
    startup = report.get("startup", {})
    summary = report.get("summary", {})
    legacy_latency = report.get("legacy_latency_probe", {})
    lines = ["# FluxRT Benchmark Report", ""]

    lines.extend(
        [
            "## Status",
            f"- completed: {_format_value(status.get('completed'))}",
            f"- aborted: {_format_value(status.get('aborted'))}",
            f"- abort_reason: {_format_value(status.get('abort_reason'))}",
            f"- cases_completed: {_format_value(status.get('cases_completed'), 0)}",
            f"- cases_requested: {_format_value(status.get('cases_requested'), 0)}",
            f"- command: `{report.get('command', 'unknown')}`",
            f"- git_commit: {_format_value(source.get('git_commit'))}",
            f"- git_dirty: {_format_value(source.get('git_dirty'))}",
            "",
        ]
    )

    lines.extend(
        [
            "## Benchmark Settings",
            f"- config_path: `{settings['config_path']}`",
            f"- prompt: {settings.get('prompt')}",
            f"- steps: {_format_value(settings.get('steps'), 0)}",
            f"- seed: {_format_value(settings.get('seed'), 0)}",
            f"- resolution: {settings.get('resolution')}",
            f"- dynamic_area_order: {settings.get('dynamic_area_order')}",
            f"- warmup_frames_per_case: {_format_value(settings.get('warmup_frames_per_case'), 0)}",
            f"- measurement_mode: {settings.get('measurement_mode')}",
            f"- measured_frames_per_case: {_format_value(settings.get('measured_frames_per_case'), 0)}",
            f"- case_duration_s: {_format_value(settings.get('case_duration_s'))}",
            f"- cache_reset_between_cases: {_format_value(settings.get('cache_reset_between_cases'))}",
            f"- timeout_s: {_format_value(settings.get('timeout_s'))}",
            f"- window_enabled: {_format_value(settings.get('window_enabled'))}",
            "",
            "Runtime features:",
        ]
    )
    lines.extend(_render_key_values(settings["runtime_features"]))
    lines.extend(["", "Interpolation:"])
    lines.extend(_render_key_values(settings["interpolation"]))
    lines.extend(["", "Paths:"])
    lines.extend(_render_key_values(settings["paths"]))
    lines.append("")

    lines.extend(["## Benchmark Environment"])
    environment = report.get("environment", {})
    software = environment.get("software", {})
    lines.extend(
        [
            f"- platform: {_format_value(environment.get('platform'))}",
            f"- python: {_format_value(environment.get('python'))}",
            f"- torch: {_format_value(software.get('torch'))}",
            f"- torch_cuda: {_format_value(software.get('torch_cuda'))}",
            f"- cudnn: {_format_value(software.get('cudnn'))}",
            f"- cpu: {_format_value(environment.get('cpu'))}",
            f"- cpu_cores_logical: {_format_value(environment.get('cpu_cores_logical'), 0)}",
            f"- system_ram_gb: {_format_value(environment.get('system_ram_gb'))}",
            f"- nvidia_driver: {_format_value(environment.get('nvidia_driver'))}",
        ]
    )
    packages = software.get("packages", {})
    if packages:
        lines.extend(["", "Packages:"])
        lines.extend(_render_key_values(packages))
    gpus = environment.get("gpu")
    if isinstance(gpus, list):
        lines.extend(
            [
                "",
                "GPUs:",
                "",
                "| Index | Name | VRAM GB | CC | SMs |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for gpu in gpus:
            lines.append(
                "| "
                f"{_format_value(gpu.get('index'), 0)} | "
                f"{_format_value(gpu.get('name'))} | "
                f"{_format_value(gpu.get('vram_gb'))} | "
                f"{_format_value(gpu.get('cc'))} | "
                f"{_format_value(gpu.get('multi_processor_count'), 0)} |"
            )
    else:
        lines.extend(["", f"- gpu: {_format_value(gpus)}"])
    lines.append("")

    ready_vram = startup.get("ready_vram", {})
    lines.extend(
        [
            "## Startup / Ready",
            f"- startup_ready_s: {_format_value(startup.get('startup_ready_s'))}",
            f"- ready_allocated_mb: {_format_value(ready_vram.get('allocated_mb'))}",
            f"- ready_reserved_mb: {_format_value(ready_vram.get('reserved_mb'))}",
            f"- ready_peak_allocated_mb: {_format_value(ready_vram.get('peak_allocated_mb'))}",
            f"- ready_peak_reserved_mb: {_format_value(ready_vram.get('peak_reserved_mb'))}",
            "",
        ]
    )

    lines.extend(
        [
            "## Throughput",
            "",
            "| Dynamic | Observed | Generated | Output | Counter Gap | Avg ms | P50 ms | P95 ms | Base FPS | Interpolated FPS |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in report.get("cases", []):
        lines.append(
            "| "
            f"{_format_percent(case['dynamic_area'])} | "
            f"{case.get('observed_samples', case['generated_frames'])} | "
            f"{case['generated_frames']} | "
            f"{case['output_frames']} | "
            f"{case.get('counter_gap_frames', 0)} | "
            f"{_format_value(case.get('avg_processing_ms'))} | "
            f"{_format_value(case.get('p50_processing_ms'))} | "
            f"{_format_value(case.get('p95_processing_ms'))} | "
            f"{_format_value(case.get('base_fps'))} | "
            f"{_format_value(case.get('interpolated_fps'))} |"
        )
    lines.append("")

    lines.extend(
        [
            "## VRAM",
            "",
            "| Dynamic | Alloc MB | Reserved MB | Peak Alloc MB | Peak Reserved MB |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in report.get("cases", []):
        vram = case.get("vram", {})
        lines.append(
            "| "
            f"{_format_percent(case['dynamic_area'])} | "
            f"{_format_value(vram.get('allocated_mb'))} | "
            f"{_format_value(vram.get('reserved_mb'))} | "
            f"{_format_value(vram.get('peak_allocated_mb'))} | "
            f"{_format_value(vram.get('peak_reserved_mb'))} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Summary",
            f"- scenario_average_base_fps: {_format_value(summary.get('scenario_average_base_fps'))}",
            f"- scenario_average_interpolated_fps: {_format_value(summary.get('scenario_average_interpolated_fps'))}",
            f"- max_peak_allocated_mb: {_format_value(summary.get('max_peak_allocated_mb'))}",
            f"- max_peak_reserved_mb: {_format_value(summary.get('max_peak_reserved_mb'))}",
            "",
            "## Legacy Latency Probe",
            f"- legacy_end_to_end_latency_s: {_format_value(legacy_latency.get('legacy_end_to_end_latency_s'))}",
            f"- status: {_format_value(legacy_latency.get('status'))}",
            "",
        ]
    )

    return "\n".join(lines)
