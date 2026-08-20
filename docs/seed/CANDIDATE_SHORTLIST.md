# D3 Seed Candidate Shortlist

**Status: SHORTLIST COMPLETE. `SOV-SEED: OPERATOR_DECISION_REQUIRED`.**

No seed is selected by this document. The three exact base checkpoints satisfy the 1–3B bootstrap envelope and carry primary-source Apache 2.0 declarations. D4 subsequently measured all three candidate-specific NF4 loads and short QLoRA steps as PASS; see `SEED_RECOMMENDATION.md`.

| ID | Exact base checkpoint | Parameters | Architecture | Native context | Primary license | D3 local QLoRA posture |
|---|---|---:|---|---:|---|---|
| S001 | [`allenai/OLMo-2-0425-1B`](https://huggingface.co/allenai/OLMo-2-0425-1B/tree/a1847dff35000b4271fa70afc5db10fd29fedbdf) | 1.485B | OLMo 2 | 4,096 | Apache-2.0, exact-revision model card | D4 short QLoRA step PASS |
| S002 | [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base/tree/ea980cb0a6c2ae4b936e82123acc929f1cec04c1) | 1.721B | Qwen3 | 32,768 | Apache-2.0, exact-revision LICENSE | D4 short QLoRA step PASS |
| S003 | [`HuggingFaceTB/SmolLM3-3B-Base`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base/tree/d78a42f79198603e614095753484a04c10c2b940) | 3.075B | SmolLM3 | 65,536 | Apache-2.0, exact-revision model card | D4 short QLoRA step PASS; least headroom |

## Primary-license audit

- **S001 — VERIFIED_PRIMARY_MODEL_CARD / FACT.** The exact revision declares both code and model under Apache 2.0. That repository has no standalone license file; the audited model-card SHA-256 is recorded in `candidate-seeds.json`.
- **S002 — VERIFIED_PRIMARY_LICENSE_FILE / FACT.** The exact revision contains the full standard [Apache 2.0 license](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/LICENSE); its SHA-256 is recorded.
- **S003 — VERIFIED_PRIMARY_MODEL_CARD / FACT.** The exact revision's license section links Apache 2.0. That repository has no standalone license file; the audited model-card SHA-256 is recorded.

All three are classified `PERMISSIVE` for shortlist purposes. This is technical license provenance, not legal advice. Exact-revision files govern over family-level summaries.

## Candidate tradeoffs

### S001 — OLMo 2 1B

Best transparency and expected memory margin. Ai2 publishes unusually complete training artifacts, which aligns with Distillery's evidence discipline. Its 4,096-token native context and English focus are meaningful limitations.

### S002 — Qwen3 1.7B Base

Best paper balance: moderate size, 32K native context, multilingual pretraining, standalone permissive license, and mature Transformers/quantization support. It is larger than S001 and must earn its recommendation through measured D4 results.

### S003 — SmolLM3 3B Base

Capacity and context upper bound. Its open blueprint is attractive, but the estimated QLoRA working set overlaps the workstation's measured free-VRAM envelope. Failure to complete the candidate-specific 4-bit test will classify it as future-compute rather than silently removing it.

## Hardware estimates

The original NF4 working-set ranges in `candidate-seeds.json` remain `ESTIMATE`. D4 adds measured short-step peak allocations beside them. Those short steps prove compatibility but not production sequence length, batch size, or sustained training stability.

## Decision boundary

D4 ran the same deterministic baseline on every candidate and recommends S002 for review. Canonical selection remains with the human operator.
