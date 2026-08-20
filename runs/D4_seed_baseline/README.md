# D4 — Seed candidate baseline

## Result

**PASS — all three exact-revision base checkpoints loaded in NF4 and completed a candidate-specific LoRA loss/backward/optimizer step.**

The available evaluation suite was not complete, so D4 used `D4-lite-v1`: eight fixed, deterministic, machine-checked tasks covering instruction following, structured JSON, code, math, reasoning, science, context retrieval, and tool-call formatting. No model judge or subjective score was used.

| ID | Exact score | Passed categories | Mean output tok/s | Median task latency | Model footprint | Peak QLoRA allocation | QLoRA step |
|---|---:|---|---:|---:|---:|---:|---|
| S001 OLMo 2 1B | 1/8 | structured output | 11.111 | 2.762 s | 1.359 GB | 2.341 GB | PASS |
| S002 Qwen3 1.7B Base | **4/8** | instruction, structured, science, tool format | 4.592 | 3.952 s | **1.327 GB** | **2.123 GB** | PASS |
| S003 SmolLM3 3B Base | **4/8** | structured, math, science, context | 5.217 | 3.097 s | 1.932 GB | 2.716 GB | PASS |

PyTorch allocation is not total system VRAM. `nvidia-smi` reported 7,663–7,716 MiB total system VRAM used after the short QLoRA steps, leaving little display-card headroom. These smoke steps prove the selected short-context path, not a production sequence length or batch size.

## Evidence

Each `S00N/` directory contains:

- `result.json` — load, task summary, latency/throughput, VRAM, and QLoRA measurements
- `tasks.jsonl` — generated output and deterministic validator result for every task
- `environment.json` — host, Python, CUDA, package, RAM, and pre-run VRAM capture
- `stdout.txt` / `stderr.txt` — concise raw execution record

Model weights remain in the external untracked cache `D:\_codex_tmp\sovereign_distillery_hf`.

## Limitation

`D4-lite-v1` is a reproducible comparative smoke baseline, not the future frozen Distillery evaluation suite. It is sufficient for this shortlist recommendation but not for checkpoint promotion or Distillery v1 Route B acceptance.
