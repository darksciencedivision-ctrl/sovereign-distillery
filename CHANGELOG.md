# Changelog

## [Unreleased]

### Measured

- **F.0 PASS** — PyTorch 2.13.0+cu130 sees the RTX 5060 Ti as `sm_120`; bitsandbytes 0.50.1 loaded an NF4 model and completed CUDA forward, LoRA backward, and an AdamW optimizer step.
- **D2 PASS** — Recomputed 43 entries / 39 generative teachers / 768.406 GiB and verified canonical smallest-to-largest order.
- **F.1 PASS** — Canonical T001 measured at 72.947 mean tok/s over 15 trials with 5,085 MiB observed VRAM delta.

### Added

- Reproducible F.0 harness and structured evidence under `runs/F0_training_stack/`.
- D2 registry validator and schema-v2 canonical/execution queue fields.
- D3 shortlist of three exact-revision 1–3B base checkpoints with primary-license evidence.
- Reproducible F.1 harness, rejected contention attempt, raw trials, and measured registry execution overlay.

### Corrected

- Classified the 461M multimodal projector as `ARTIFACT` instead of a generative teacher.
- Replaced the ambiguous registry `teacher_count` with distinct entry, teacher, and non-teacher totals.

## [1.1.0] — 2026-08-19

### Fixed (from THESIS-REVIEW-v1.1 / PATCH-NOTES)

- **TR-1** — `.gitignore` no longer excludes evidence JSON under `runs/`.
- **TR-2** — Restored INV-8, INV-9, INV-10, INV-13, INV-15 to the invariant table.
- **TR-3** — Restored RD-2 (remediation route) and RD-3 (criticality tag + gate) in Capability Ledger and promotion rule.
- **TR-4** — Committed `tools/d1_characterize.py`, `tools/d2_registry.py`, `tools/d3_throughput.py`, `tools/README.md`.
- **TR-5** — Phase namespace canonicalised to `F.*`; dual D3/F.3 reference removed.
- **TR-6** — EF-3 tier verdicts aligned with §6 (T3/T4 remain in canonical queue).
- **TR-7 / TR-8** — Ship checklist records private-repo and inventory-disclosure requirements.
- **TR-9** — `docs-check` GitHub Actions workflow added (link resolution, cited-path existence, no weights/secrets).

### Added

- `docs/SHIP_CHECKLIST.md`
- `docs/OQ_CROSSWALK.md`
- `docs/EVIDENCE_REGISTER.md`
- `.github/workflows/docs-check.yml`

## [1.0.0] — 2026-08-19

### Added

- First evidence-grounded design thesis.
- Measured hardware profile (RTX 5060 Ti, 5.26 GiB free VRAM, `sm_120`).
- Complete library enumeration (43 entries / 39 generative teachers / 768.4 GiB).
- Tiered feasibility frontier (T1–T4).
- Dual-gate promotion rule (T + D).
- Route A / Route B acceptance criteria.
- Seed strategy Option C (bootstrap → native migration).
- Generation-depth and base-family registry fields.
- Distillery Knowledge Lineage scoped as structured experiment record.

### Closed

- OQ-002 (library inventory).
- OQ-003 (v1 acceptance).
- OQ-010 (ordering rationale).
