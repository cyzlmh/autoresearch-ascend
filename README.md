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

## One-night result (2026-03-16)

Run branch: `autoresearch/mar16b`  
Total experiments: `43`

- Baseline `val_bpb`: `1.3843`
- Best `val_bpb`: `1.1583` (`1.158260` run), about `16.3%` lower than baseline
- Peak memory: `9.1 GB` (from `50.1 GB`, about `82%` reduction)

Best configuration from the run:

```python
ASPECT_RATIO = 56
HEAD_DIM = 128
WINDOW_PATTERN = "L"
DEPTH = 6
DEVICE_BATCH_SIZE = 16
TOTAL_BATCH_SIZE = 2**17
WEIGHT_DECAY = 0.5
WARMDOWN_RATIO = 0.1
WARMUP_RATIO = 0.0
MATRIX_LR = 0.02
EMBEDDING_LR = 0.5
SCALAR_LR = 0.5
```

Key findings:

1. `WINDOW_PATTERN="L"` performed better than sliding-window patterns on this Ascend setup.
2. Smaller device batch size with more optimizer updates in fixed time improved convergence.
3. `DEPTH=6` gave a better speed/quality tradeoff under the 5-minute budget.
4. Higher weight decay (`0.5`) regularized well for this regime.
5. Short warmdown (`0.1`) helped keep more training time near peak LR.
6. Lower `MATRIX_LR` (`0.02`) and `EMBEDDING_LR` (`0.5`) improved stability.
7. `ASPECT_RATIO=56` was the best capacity/speed balance among tested values.

## License

MIT
