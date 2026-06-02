import os
import platform
import subprocess
import torch


def _run(cmd):
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def get_cpu_name():
    system = platform.system()

    if system == "Windows":
        return _run(["wmic", "cpu", "get", "name"]).splitlines()[1].strip()

    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    return platform.processor() or "Unknown CPU"


def get_gpu_info(device='cuda'):
    gpus = []
    
    if device == 'cuda' and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1024**3, 2),
                    "cc": f"{props.major}.{props.minor}",
                }
            )
        return gpus

    elif device == 'xpu' and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(i)
            gpus.append(
                {
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1024**3, 2),
                    "cc": None,
                }
            )
        return gpus
    
    else:
        return "No supported GPU found"


def scan_hardware(device='cuda'):
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": get_cpu_name(),
        "cpu_cores_logical": os.cpu_count(),
        "gpu": get_gpu_info(device),
    }
    return info
