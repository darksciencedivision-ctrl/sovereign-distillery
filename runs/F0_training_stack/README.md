# F.0 — Training-stack / `sm_120` proof

## Result

**PASS — MEASURED on 2026-08-20 UTC.**

The repository-local Python 3.12 environment loaded `HuggingFaceTB/SmolLM2-135M` in bitsandbytes NF4 on the RTX 5060 Ti, returned finite CUDA logits, and completed a LoRA loss/backward/AdamW optimizer step.

| Gate | Result |
|---|---|
| F0-A CUDA/PyTorch | PASS — CUDA available; compute capability 12.0; `sm_120` present in the PyTorch build |
| F0-B bitsandbytes | PASS — `Linear4bit` produced finite GPU output |
| F0-C 4-bit model load | PASS — 210 `Linear4bit` modules, NF4 + bf16 compute + double quantization |
| F0-D forward | PASS — 14 input tokens, finite logits on `cuda:0` |
| F0-E training step | PASS — finite loss 8.469595, nonzero LoRA gradient norm 1.032419, optimizer step completed |

Peak PyTorch allocation was 258,847,232 bytes (246.86 MiB). This proves the selected kernel and QLoRA path on `sm_120`; it does **not** establish that a 1–3B seed fits the current free-VRAM envelope.

## Evidence

- `attempt-001/result.json` — gate results and measurements
- `attempt-001/hardware.json` — D1 hardware/software capture
- `attempt-001/environment.json` — host, RAM, disks, Python, and package versions
- `attempt-001/commands.txt` — exact environment and execution commands
- `attempt-001/stdout.txt` / `attempt-001/stderr.txt` — concise raw execution record

The model cache is external at `D:\_codex_tmp\sovereign_distillery_hf` and is not committed. The test model revision is pinned in `result.json`.

## Method basis

The stack follows the official [PyTorch local installation guidance](https://pytorch.org/get-started/locally/), [bitsandbytes installation matrix](https://huggingface.co/docs/bitsandbytes/installation), and [Transformers NF4/QLoRA guidance](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
