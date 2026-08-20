# F.1 — Canonical-first teacher throughput

## Result

**PASS — MEASURED on 2026-08-20 UTC.**

The benchmark selected `T001 dolphin3:8b`, the smallest teacher in the corrected canonical parameter order. The exact 8B Q4_K_M model ran through Ollama 0.32.14 with runtime identity checked after every request.

| Measure | Result |
|---|---:|
| Cold load | 12.368 s |
| Warm measured trials | 15 (5 fixed prompts × 3 repeats) |
| Measured output tokens | 2,505 |
| Generation tok/s mean | **72.947** |
| Generation tok/s median | **72.475** |
| Minimum / maximum | 71.585 / 74.764 |
| Standard deviation | 0.931 tok/s |
| Variance | 0.867 (tok/s)² |
| Prompt-processing mean | 2,142.362 tok/s |
| VRAM before load | 2,086 MiB used / 5,806 MiB free |
| Peak observed VRAM | 7,171 MiB used |
| Observed model delta | 5,085 MiB |
| Ollama processor placement | 100% GPU |

## Derived corpus planning estimates

`DERIVED` using `target_output_tokens / 72.946535 measured tok/s`:

| Target | Estimate |
|---:|---:|
| 100k output tokens | 1,370.87 s / 22.85 min |
| 1M output tokens | 13,708.67 s / 3.81 h |
| 5M output tokens | 68,543.35 s / 19.04 h |

These estimates assume constant warm single-stream throughput. They exclude failures, restarts, prompt growth, thermal drift, concurrency effects, and scheduling overhead.

## Evidence

- `result.json` — summary statistics, resource observations, and derived estimates
- `prompts.json` — fixed workload and generation configuration
- `trials.jsonl` — all 15 raw measured trials
- `environment.json` — host, Ollama version, baseline RAM, and VRAM
- `stdout.txt` / `stderr.txt` — concise execution record
- `attempt-001/` — rejected first attempt contaminated by an unrelated concurrent Ollama model

F.1 proves technical execution only. `dolphin3:8b` remains `PENDING` rather than `ELIGIBLE` because its license class has not passed the primary-text policy audit.
