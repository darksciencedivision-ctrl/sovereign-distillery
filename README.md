# Sovereign Distillery

**Evidence-Grounded Design Thesis and Engineering Baseline v1.1**  
for the Sovereign Research Workspace

[![Status](https://img.shields.io/badge/status-provisional-yellow)]()
[![Evidence](https://img.shields.io/badge/evidence-measured-blue)]()
[![Blocker](https://img.shields.io/badge/OQ--002-CLOSED-brightgreen)]()

---

## What this is

The Sovereign Distillery is the model-forging subsystem of the Sovereign Research Workspace. It processes a local library of open-weight language models as teachers, extracts useful capability deltas, and progressively trains a persistent Sovereign student lineage.

This repository contains the **first evidence-grounded thesis and engineering baseline** (v1.1). Prior documents specified against modelled estimates. This one specifies against measurements taken on the target machine on 19 August 2026.

## Key measured facts

| Fact | Value |
|------|-------|
| GPU | NVIDIA RTX 5060 Ti (Blackwell, `sm_120`) |
| VRAM free at capture | **5.26 GiB** (of 7.96 GiB nominal) |
| Training stack | **Absent** (PyTorch / bitsandbytes not installed) |
| Library size | **43 entries / 39 generative teachers / 768.4 GiB** |
| Smallest generative model | **8B** |
| Local 1–3B seed candidates | **None** |

## Documents

| Path | Description |
|------|-------------|
| [`THESIS.md`](THESIS.md) | Full evidence-grounded design thesis (canonical) |
| [`docs/SHIP_CHECKLIST.md`](docs/SHIP_CHECKLIST.md) | Pre-commit and visibility checklist |
| [`docs/OQ_CROSSWALK.md`](docs/OQ_CROSSWALK.md) | Open-question status + phase namespace mapping |
| [`docs/EVIDENCE_REGISTER.md`](docs/EVIDENCE_REGISTER.md) | Evidence artifacts, status, and remaining measurements |
| [`tools/`](tools/) | D1 / D2 / D3 measurement instruments |

## Current status

- **OQ-002 (library inventory)** — CLOSED.
- **OQ-001 (seed)** — Bootstrap path preferred; external 1–3B candidates required.
- **OQ-003 (v1 acceptance)** — CLOSED (Route A = pipeline proof, Route B = transfer proof; v1 requires Route B).
- Largest remaining unknown: whether `bitsandbytes` 4-bit executes on `sm_120` on this card.

## Immediate next steps

1. Install and verify the training stack (`sm_120` 4-bit forward pass).
2. Measure real generation throughput on one T1 teacher.
3. Acquire and baseline external 1–3B seed candidates.
4. Build and freeze the evaluation suite.

## Visibility warning

**Keep this repository PRIVATE** while `LICENSE` is all-rights-reserved.  
Publishing grants view-and-fork rights under GitHub’s terms regardless of the licence text. Resolve OQ-007 before changing visibility.

## Authority

The human operator controls objectives, scope, promotions, strategic direction, and final acceptance. This repository plans and recommends. It does not decide.

## License

Internal engineering baseline. Distribution intent (OQ-007) remains open and gates downstream licensing analysis.
