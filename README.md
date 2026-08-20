# Sovereign Distillery

**Evidence-Grounded Design Thesis and Engineering Baseline v1.1**  
for the Sovereign Research Workspace

[![Status](https://img.shields.io/badge/status-provisional-yellow)]()
[![Evidence](https://img.shields.io/badge/evidence-measured-blue)]()
[![Blocker](https://img.shields.io/badge/OQ--002-CLOSED-brightgreen)]()

---

## What this is

The Sovereign Distillery is the model-forging subsystem of the Sovereign Research Workspace. It processes a local library of open-weight language models as teachers, extracts useful capability deltas, and progressively trains a persistent Sovereign student lineage.

This repository contains the canonical **evidence-grounded thesis and engineering baseline** (v1.1), plus subsequent measured experimental evidence. Prior documents specified against modelled estimates. The thesis baseline is preserved by the annotated `thesis-v1.1` tag.

## Key measured facts

| Fact | Value |
|------|-------|
| GPU | NVIDIA RTX 5060 Ti (Blackwell, `sm_120`) |
| VRAM free at capture | **5.26 GiB** (of 7.96 GiB nominal) |
| Training stack at thesis capture | **Absent** (PyTorch / bitsandbytes not installed) |
| Library size | **43 entries / 39 generative teachers / 768.4 GiB** |
| Smallest generative model | **8B** |
| Local 1–3B seed candidates | **None** |

## Measured experimental status

| Phase | Status | Evidence |
|---|---|---|
| **F.0 — `sm_120` 4-bit training path** | **PASS** — NF4 load, CUDA forward, LoRA backward, optimizer step | [`runs/F0_training_stack/`](runs/F0_training_stack/) |
| **F.1 — teacher throughput** | Pending | Evidence directory will be created by F.1 |

## Documents

| Path | Description |
|------|-------------|
| [`THESIS.md`](THESIS.md) | Full evidence-grounded design thesis (canonical) |
| [`docs/SHIP_CHECKLIST.md`](docs/SHIP_CHECKLIST.md) | Pre-commit and visibility checklist |
| [`docs/OQ_CROSSWALK.md`](docs/OQ_CROSSWALK.md) | Open-question status + phase namespace mapping |
| [`docs/EVIDENCE_REGISTER.md`](docs/EVIDENCE_REGISTER.md) | Evidence artifacts, status, and remaining measurements |
| [`registry/VALIDATION.md`](registry/VALIDATION.md) | Independently recomputed D2 totals and canonical order |
| [`tools/`](tools/) | D1 / D2 / D3 measurement instruments |

## Current status

- **OQ-002 (library inventory)** — CLOSED.
- **D2 registry validation** — PASS: 43 entries, 39 teachers, 768.406 GiB; canonical and execution fields verified.
- **OQ-001 (seed)** — Bootstrap path preferred; external 1–3B candidates required.
- **OQ-003 (v1 acceptance)** — CLOSED (Route A = pipeline proof, Route B = transfer proof; v1 requires Route B).
- **F.0 toolchain gate** — PASS on PyTorch 2.13.0+cu130 / bitsandbytes 0.50.1; the 1–3B capacity boundary remains unmeasured.

## Immediate next steps

1. Measure real generation throughput on the smallest executable canonical teacher.
2. Validate registry order/tier metadata against the raw artifact.
3. Shortlist and baseline external 1–3B seed candidates.
4. Build and freeze the full evaluation suite.

## Visibility warning

**Keep this repository PRIVATE** while `LICENSE` is all-rights-reserved.  
Publishing grants view-and-fork rights under GitHub’s terms regardless of the licence text. Resolve OQ-007 before changing visibility.

## Authority

The human operator controls objectives, scope, promotions, strategic direction, and final acceptance. This repository plans and recommends. It does not decide.

## License

Internal engineering baseline. Distribution intent (OQ-007) remains open and gates downstream licensing analysis.
