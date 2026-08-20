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
| **F.1 — teacher throughput** | **PASS** — T001 mean 72.947 tok/s across 15 trials | [`runs/F1_teacher_throughput/`](runs/F1_teacher_throughput/) |

## Documents

| Path | Description |
|------|-------------|
| [`THESIS.md`](THESIS.md) | Full evidence-grounded design thesis (canonical) |
| [`docs/SHIP_CHECKLIST.md`](docs/SHIP_CHECKLIST.md) | Pre-commit and visibility checklist |
| [`docs/OQ_CROSSWALK.md`](docs/OQ_CROSSWALK.md) | Open-question status + phase namespace mapping |
| [`docs/EVIDENCE_REGISTER.md`](docs/EVIDENCE_REGISTER.md) | Evidence artifacts, status, and remaining measurements |
| [`registry/VALIDATION.md`](registry/VALIDATION.md) | Independently recomputed D2 totals and canonical order |
| [`docs/seed/CANDIDATE_SHORTLIST.md`](docs/seed/CANDIDATE_SHORTLIST.md) | D3 exact-revision seed candidates and primary-license audit |
| [`tools/`](tools/) | D1 / D2 / D3 measurement instruments |

## Current status

- **OQ-002 (library inventory)** — CLOSED.
- **D2 registry validation** — PASS: 43 entries, 39 teachers, 768.406 GiB; canonical and execution fields verified.
- **OQ-001 (seed)** — Bootstrap path preferred; external 1–3B candidates required.
- **OQ-003 (v1 acceptance)** — CLOSED (Route A = pipeline proof, Route B = transfer proof; v1 requires Route B).
- **F.0 toolchain gate** — PASS on PyTorch 2.13.0+cu130 / bitsandbytes 0.50.1; the 1–3B capacity boundary remains unmeasured.
- **D3 seed shortlist** — COMPLETE with three Apache-2.0 base checkpoints; `SOV-SEED` remains operator-owned.
- **F.1 throughput** — PASS for canonical `T001 dolphin3:8b`; technical execution measured, license eligibility still pending.

## Immediate next steps

1. Baseline the three D3 seed candidates under the same deterministic D4 suite.
2. Produce an evidence-backed bootstrap-seed recommendation for operator review.
3. Audit primary licenses before any teacher becomes `ELIGIBLE`.
4. Build and freeze the full evaluation suite.

## Visibility warning

**Keep this repository PRIVATE** while `LICENSE` is all-rights-reserved.  
Publishing grants view-and-fork rights under GitHub’s terms regardless of the licence text. Resolve OQ-007 before changing visibility.

## Authority

The human operator controls objectives, scope, promotions, strategic direction, and final acceptance. This repository plans and recommends. It does not decide.

## License

Internal engineering baseline. Distribution intent (OQ-007) remains open and gates downstream licensing analysis.
