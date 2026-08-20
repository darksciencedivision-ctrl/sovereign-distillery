# Evidence Register

Repository evidence artifacts supporting the thesis's measured claims.

| # | Claim | Required artifact | Status |
|---|---|---|---|
| 1 | Hardware profile (VRAM free, sm_120, driver) | `runs/hardware_profile.json` | **Present — captured 2026-08-19 on the operator machine** |
| 2 | Teacher registry (43 entries, 39 teachers, ordering, roles) | `registry/teachers.json`, `registry/validation.json` | **MEASURED + independently validated — refreshed 2026-08-20** |
| 3 | Source tools exist | `tools/d1_characterize.py`, `tools/d2_registry.py` | **Present in v1.1 package** |
| 4 | Generation throughput on canonical T001 | `runs/F1_teacher_throughput/result.json`, `runs/F1_teacher_throughput/trials.jsonl` | **MEASURED — 15 trials; mean 72.947 tok/s; median 72.475 tok/s** |
| 5 | `sm_120` 4-bit QLoRA toolchain viability | `runs/F0_training_stack/attempt-001/result.json` | **MEASURED — PASS; NF4 load, CUDA forward, LoRA backward, and optimizer step succeeded** |
| 6 | D3 seed candidate compatibility and deterministic baseline | `runs/D4_seed_baseline/summary.json`, per-candidate results/tasks | **MEASURED — all candidate NF4/QLoRA paths PASS; S002 and S003 score 4/8** |

Items 1–3 are committed with the initial v1.1 baseline, promoting M-1 through M-5 to `MEASURED`. Item 5 supersedes the thesis-era unverified toolchain claim while preserving the original baseline capture, which correctly recorded that stack as absent on 2026-08-19.
