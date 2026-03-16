"""
Minimal accelerator helpers for autoresearch.

Keeps the repo small while making the runtime portable across CUDA, Ascend
NPU, MPS, and CPU.
"""

from contextlib import nullcontext

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def autodetect_device_type():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device(device_type):
    if device_type == "cuda":
        device = torch.device("cuda")
        torch.cuda.set_device(device)
        return device
    if device_type == "npu":
        device = torch.device("npu")
        torch.npu.set_device(device)
        return device
    return torch.device(device_type)


def seed_all(seed, device_type):
    torch.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(seed)
    elif device_type == "npu":
        torch.npu.manual_seed(seed)


def maybe_set_matmul_precision():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def make_autocast_context(device_type):
    if device_type in ["cuda", "npu"]:
        return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    return nullcontext()


def get_synchronize(device_type):
    if device_type == "cuda":
        return torch.cuda.synchronize
    if device_type == "npu":
        return torch.npu.synchronize
    return lambda: None


def get_max_memory_allocated(device_type):
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated
    if device_type == "npu":
        return torch.npu.max_memory_allocated
    return lambda: 0


def get_device_name(device_type):
    if device_type == "cuda":
        return torch.cuda.get_device_name(0)
    if device_type == "npu":
        return torch.npu.get_device_name(0)
    return device_type.upper()


def get_peak_flops(device_name):
    name = device_name.lower()
    table = (
        (["ascend", "910b"], 313e12),
        (["h200", "nvl"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800"], 756e12),
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["v100"], 125e12),
    )
    for needles, flops in table:
        if all(needle in name for needle in needles):
            return flops
    return float("inf")
