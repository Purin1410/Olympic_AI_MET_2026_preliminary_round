"""Environment info for reproducibility, GPU monitoring and device selection."""
import json
import shutil
import platform
import subprocess

from common.config import log


def env_info(packages, path):
    """Write Python / library versions and GPU names to `path` (env.json)."""
    import importlib.metadata as md
    info = {"python": platform.python_version(), "packages": {}}
    for pk in packages:
        try:
            info["packages"][pk] = md.version(pk)
        except Exception:
            info["packages"][pk] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump(info, f, indent=1)
    log("env", info)
    return info


def start_gpu_monitor(interval_s, path):
    """Log GPU utilisation and memory every `interval_s` seconds to a CSV via nvidia-smi.

    Returns the process (call .terminate() at the end), or None when disabled / no GPU.
    """
    if not interval_s or not shutil.which("nvidia-smi"):
        return None
    f = open(path, "w")
    f.write("timestamp,index,util_gpu,mem_used_mib,mem_total_mib\n")
    f.flush()
    return subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
                             "--format=csv,noheader,nounits", "-l", str(interval_s)],
                            stdout=f, stderr=subprocess.DEVNULL)


def pick_device(name):
    """'auto' -> cuda if available, else cpu."""
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
