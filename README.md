# autoresearch-ascend

Ascend-focused fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

This repository keeps the original small, single-file training workflow, but adapts runtime behavior so it can run on Huawei Ascend (`torch_npu` / CANN) environments.

## Upstream baseline

- Base project: `karpathy/autoresearch`
- Core loop preserved: fixed 5-minute training budget, `train.py` as the main research surface

## What was modified for Ascend

| File | Change |
| --- | --- |
| `accelerator.py` | Added backend abstraction for `cuda` / `npu` / `mps` / `cpu` (device selection, seed, autocast, sync, memory metric, peak FLOPS lookup). |
| `flash_attention.py` | Added Flash Attention compatibility shim: FA3 on CUDA when available, SDPA fallback for non-CUDA (including Ascend). |
| `train.py` | Reworked runtime calls to use accelerator helpers instead of CUDA-only APIs; non-CUDA-safe `torch.compile` handling; NPU-compatible memory/sync reporting. |
| `prepare.py` | Added dataset download endpoint controls (`HF_ENDPOINT`, `AUTORESEARCH_DATA_BASE_URL`) and retry behavior improvements. |
| `pyproject.toml` | Project metadata updated for Ascend fork and dependencies aligned with this runtime layout. |
| `README.md` | Rewritten to document Ascend-specific behavior and usage. |

## Ascend environment requirements

- Python `>=3.10`
- PyTorch installed for your Ascend environment
- `torch_npu` + CANN already configured and validated
- Dependencies from `pyproject.toml` installed into the same environment

Example dependency install (inside your active Ascend env):

```bash
pip install matplotlib numpy pandas pyarrow requests rustbpe tiktoken kernels
```

## Quick start

```bash
# 1) Prepare dataset + tokenizer
python prepare.py

# 2) Run one 5-minute training experiment
python train.py
```

## Dataset download configuration

`prepare.py` supports these optional env vars:

- `AUTORESEARCH_CACHE_DIR=/path/to/cache`
- `HF_ENDPOINT=https://hf-mirror.com`
- `AUTORESEARCH_DATA_BASE_URL=https://<host>/datasets/karpathy/climbmix-400b-shuffle/resolve/main`

If `AUTORESEARCH_DATA_BASE_URL` is set, it is used directly.
Otherwise, `HF_ENDPOINT` is used as endpoint prefix.

## Ascend tuning knobs (manual edits)

For this fork, `DEVICE_BATCH_SIZE` is intentionally a manual knob in `train.py`.

If you hit OOM on 910B-class NPUs, edit these in `train.py` and rerun:

- `DEVICE_BATCH_SIZE` (first knob to lower)
- `WINDOW_PATTERN` (often use `"L"` on Ascend)
- `DEPTH`
- `TOTAL_BATCH_SIZE` (must remain divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`)

## Scope and non-goals

- Goal: keep a minimal fork that runs the autoresearch loop on Ascend.
- Non-goal: rebuild full `nanochat`-style multi-script infrastructure.

## License

MIT
