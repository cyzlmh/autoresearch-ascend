"""
Minimal Flash Attention compatibility shim for autoresearch.

Uses Flash Attention 3 kernels on CUDA when available, and falls back to
PyTorch SDPA everywhere else, including Ascend NPU.
"""

import os

import torch
import torch.nn.functional as F

try:
    from kernels import get_kernel
except ImportError:
    get_kernel = None


def _load_fa3():
    if get_kernel is None or not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
        repo = "varunneal/flash-attention-3" if (major, minor) == (9, 0) else "kernels-community/flash-attn3"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        return get_kernel(repo).flash_attn_interface
    except Exception:
        return None


_fa3 = _load_fa3()
HAS_FA3 = _fa3 is not None


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with optional sliding-window masking.
    Inputs are in (B, H, T, D) format.
    """
    tq = q.size(2)
    tk = k.size(2)
    window = window_size[0]

    if (window < 0 or window >= tq) and tq == tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    device = q.device
    row_idx = (tk - tq) + torch.arange(tq, device=device).unsqueeze(1)
    col_idx = torch.arange(tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)
