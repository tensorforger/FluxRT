import os
import platform
import subprocess
from importlib import metadata
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
        output = _run(["wmic", "cpu", "get", "name"])
        if output:
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if len(lines) > 1:
                return lines[1]
        return platform.processor() or "Unknown CPU"

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


def get_gpu_info():
    if not torch.cuda.is_available():
        return "No CUDA GPU"

    gpus = []

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        gpus.append(
            {
                "index": i,
                "name": props.name,
                "vram_gb": round(props.total_memory / 1024**3, 2),
                "vram_mb": round(props.total_memory / 1024**2, 2),
                "cc": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
        )

    return gpus


def get_system_ram_gb():
    system = platform.system()

    if system == "Windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / 1024**3, 2)
        except Exception:
            return None

    if system == "Darwin":
        output = _run(["sysctl", "-n", "hw.memsize"])
        if output:
            try:
                return round(int(output) / 1024**3, 2)
            except ValueError:
                pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * page_size / 1024**3, 2)
        except (ValueError, OSError):
            pass

    return None


def get_nvidia_driver_version():
    output = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if not output:
        return None
    return output.splitlines()[0].strip()


def get_package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def get_software_versions():
    packages = [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "optimum-quanto",
        "peft",
        "numpy",
        "opencv-python",
    ]
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": {name: get_package_version(name) for name in packages},
    }


def scan_hardware():
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "software": get_software_versions(),
        "cpu": get_cpu_name(),
        "cpu_cores_logical": os.cpu_count(),
        "system_ram_gb": get_system_ram_gb(),
        "nvidia_driver": get_nvidia_driver_version(),
        "gpu": get_gpu_info(),
    }
    return info
