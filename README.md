# autoresearch-ascend

Ascend-only fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

This repo keeps the original autoresearch workflow (`program.md` drives the agent, agent edits `train.py`) and adapts runtime code for Huawei Ascend (`torch_npu` / CANN).

## 3-file core

The research loop is intentionally centered on three files:

- `prepare.py` (data prep + tokenizer + fixed evaluation)
- `train.py` (model/optimizer/training loop; the file agents iterate on)
- `program.md` (human-authored research instructions)

## What changed for Ascend

Compared with upstream `karpathy/autoresearch`, this fork makes these practical changes:

1. `train.py` is Ascend-only runtime:
   - uses `torch.npu` device setup, seed, sync, and memory stats
   - uses BF16 autocast on NPU
   - keeps `torch.compile` disabled for Ascend compatibility
2. `train.py` attention path uses in-file SDPA (windowed causal masking) instead of CUDA FlashAttention kernels.
3. `prepare.py` supports mirror-friendly dataset download configuration:
   - `HF_ENDPOINT`
   - `AUTORESEARCH_DATA_BASE_URL`
   - `AUTORESEARCH_CACHE_DIR`
4. Removed extra compatibility helper modules; logic is kept in core files.

## Requirements

- Python >= 3.10
- Ascend environment with `torch_npu` + CANN working
- Python deps (in the same environment):

```bash
pip install matplotlib numpy pandas pyarrow requests rustbpe tiktoken
```

## Quick start

```bash
# 1) Prepare dataset + tokenizer
python prepare.py

# 2) Run one 5-minute training experiment
python train.py
```

## Download endpoint options

Use these only if default Hugging Face access is blocked/slow:

- `AUTORESEARCH_CACHE_DIR=/path/to/cache`
- `HF_ENDPOINT=https://hf-mirror.com`
- `AUTORESEARCH_DATA_BASE_URL=https://<host>/datasets/karpathy/climbmix-400b-shuffle/resolve/main`

`AUTORESEARCH_DATA_BASE_URL` has highest priority when set.

## NPU tuning knobs (manual)

Edit directly in `train.py`:

- `DEVICE_BATCH_SIZE` (first OOM knob)
- `WINDOW_PATTERN` (`"L"` is often safer/faster on Ascend)
- `DEPTH`
- `TOTAL_BATCH_SIZE` (must stay divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`)

## License

MIT
