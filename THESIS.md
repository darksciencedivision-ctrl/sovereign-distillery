# Sovereign Distillery

### An Evidence-Grounded Architecture for Progressive Local Model Distillation

**A Design Thesis and Engineering Baseline for the Sovereign Research Workspace**

| | |
|---|---|
| **Document** | Thesis & Engineering Baseline v1.1 |
| **Date** | 19 August 2026 |
| **Status** | First evidence-grounded plan. Prior documents specified against estimates; this one specifies against measurements. |
| **Authority** | The human operator controls objectives, scope, promotions, strategic direction, and final acceptance. This document plans and recommends. It does not decide. |
| **Workspace** | `D:\Sovereign Distillery` |
| **Parent system** | Sovereign Research Workspace |

---

## Abstract

The Sovereign Distillery is proposed as the fourth major subsystem of the Sovereign Research Workspace. Its purpose is to treat a local library of open-weight language models as teachers, extract useful capability deltas, and progressively train a persistent Sovereign student lineage that can eventually serve as a native intelligence layer under Sovereign control.

Prior design documents reasoned from modelled hardware envelopes and an unknown library. Two instruments executed on the target machine on 19 August 2026 change the foundation of the project:

- The library contains **43 registry entries (39 generative teachers) totalling 768.4 GiB**. The smallest generative model is **8B**. There is no teacher smaller than any realistic student, and no local candidate in the 1–3B seed range.
- The machine exposes **8151 MiB nominal VRAM of which only 5383 MiB (5.26 GiB) was free at capture**. The training stack (PyTorch, bitsandbytes) is absent. Generation is therefore the binding constraint; training remains unproven until the stack is installed and an `sm_120` 4-bit forward pass is verified.

These measurements retire the project’s oldest blocker (library inventory) and replace it with sharper, more tractable constraints. The architecture survives; the plan does not. The operative teacher queue collapses from “the whole library, smallest to largest” into a four-tier feasibility frontier in which only five teachers are comfortably viable on current hardware. The seed must be acquired externally. Synthetic-derived teachers elevate diversity and generation-depth tracking from good practice to requirement. Same-base families reopen limited weight-merging opportunities while introducing measurement-correlation hazards.

This thesis integrates operator intent, prior design dialogue, two rounds of adversarial validation, and the first measured baseline into a single coherent engineering foundation. It defines invariants, a dual-gate promotion rule that protects cumulative capability retention, a two-route acceptance criterion, a tiered execution plan, and the concrete decisions the operator must still make.

---

## 1. Introduction

### 1.1 Problem statement

A multi-model research workspace that permanently depends on heterogeneous external models remains strategically incomplete. The Sovereign Distillery exists to convert that dependency into a controlled, provenance-aware lineage of models that the workspace itself produces, owns operationally, and can improve.

The original operator directive was clear:

> Process the local model library from smallest to largest. Treat each model as a teacher. Extract what is useful. Train a persistent Sovereign model that improves with each stage. Ultimately produce a Sovereign-owned LLM that can be wrapped, quantized, and deployed as the native intelligence layer of the system.

That directive is retained. What has changed is the empirical ground on which it must be executed.

### 1.2 Method

This document is the product of successive adversarial refinement:

1. Operator statement of intent.
2. Design dialogue that elaborated mechanisms and identified silent scope additions.
3. Independent validation report that surfaced contradictions, unsupported claims, and missing instruments.
4. Integration review that corrected defects in the first integrated baseline.
5. **Measured baseline** (hardware characterisation + library enumeration) that supersedes every prior hardware and library section.

Claims are labelled `MEASURED`, `DERIVED`, `ESTIMATE`, `ASSUMPTION`, or `UNVERIFIED`. Where a conclusion is structurally robust even if an estimate is wrong by a factor, that robustness is stated explicitly.

### 1.3 Contributions

- The first measured hardware profile and complete teacher registry for the project.
- A tiered feasibility frontier that replaces “process the whole library” with a schedulable plan.
- A dual-gate promotion rule (step tolerance **T** + drift floor **D**) that protects net retained capability.
- A seed strategy (bootstrap from external permissive base, later native migration) that resolves the ownership-versus-derivative contradiction without requiring from-scratch pretraining on consumer hardware.
- Explicit separation of strategic objective from release acceptance, and of pipeline proof (Route A) from transfer proof (Route B).

---

## 2. Measured Baseline

### 2.1 Hardware profile

Source: `tools/d1_characterize.py`, executed 2026-08-19 on `desktop-03ptabh`.

| Property | Value | Label |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti | `MEASURED` |
| VRAM nominal | 8151 MiB (7.96 GiB) | `MEASURED` |
| **VRAM free at capture** | **5383 MiB (5.26 GiB)** | `MEASURED` |
| VRAM consumed at rest | 2768 MiB (2.70 GiB) | `DERIVED` |
| Compute capability | 12.0 (`sm_120`, Blackwell) | `MEASURED` |
| Driver | 610.74 | `MEASURED` |
| PyTorch | **Absent** | `MEASURED` |
| bitsandbytes | **Absent** (blocked by torch) | `MEASURED` |

**Consequence.** Every prior VRAM calculation used the nominal 8 GiB figure. Against 5.26 GiB free:

| Configuration | Against 8 GiB | Against 5.26 GiB |
|---|---|---|
| QLoRA 7B (bs1, seq512, no ckpt) | Marginal | **Not possible** |
| QLoRA 3B (est.) | Comfortable | Feasible, tight |
| QLoRA 1B (est.) | Comfortable | Comfortable |
| Teacher + student co-resident | Not possible | Not possible |

The realistic local training envelope is therefore a **1B–3B student under QLoRA**. 7B is no longer an experiment to attempt; it is out of reach unless the 2.70 GiB at-rest consumption can be substantially reclaimed. A five-minute re-probe with Ollama and GPU applications closed remains outstanding.

**The largest single unknown in the project is no longer conceptual.** It is whether `bitsandbytes` can execute a 4-bit forward pass on `sm_120` on this specific card. That question has never been asked because the software required to ask it is not installed.

### 2.2 Model library

Source: `tools/d2_registry.py`, executed 2026-08-19.

| Metric | Value |
|---|---|
| Registry entries | **43** |
| Generative teachers | **39** |
| Non-teachers (embedding / artifact) | **4** |
| Total on disk | **768.4 GiB** |
| Smallest generative model | **8B** (`dolphin-llama3:8b`, 4.34 GiB) |
| Largest | 128B (`mistral-medium-3.5`, 74.73 GiB) |
| Dominant quantization | Q4_K_M (32 of 43) |
| Runtime | Ollama (all entries) |

Parameter classes present: 8B×4 · 12B×1 · 14–15B×4 · ~22–24B×5 · ~26–32B×15 · 35–36B×2 · ~69–70B×3 · 109–128B×3.

This is a serious inference and orchestration library. It was not assembled as distillation source material for a small student on a small card. That mismatch is the origin of the findings below.

---

## 3. Findings That Reorder the Project

### EF-1 — There is no teacher smaller than the student

`MEASURED`. The smallest generative model is 8B. The student, under any reading of the measured hardware, is 1B–3B. Teacher #1 is already 3–8× the student’s parameter count.

The original “smallest-to-largest” premise assumed cheap early teachers for pipeline validation and a possible foundational curriculum. That floor does not exist. The capability delta against the seed will be large and positive from the first teacher. Contradiction C-3 (empty early deltas under a competent seed) is retired. The “cheap debugging teacher” justification is also retired.

The ordering survives for a **third** reason: it is now a **generation-cost ordering**. Larger teachers cost dramatically more wall-clock to generate from; processing in ascending size therefore remains the correct risk and cost ordering.

### EF-2 — There is no seed candidate in the library

`DERIVED`. The library contains nothing between a 567M embedding model and 8B. The Sovereign seed cannot be sourced locally. Phase-A bootstrap requires acquiring 1–3B base candidates from outside the current library. This is a concrete, closeable gap that was invisible until the inventory existed.

### EF-3 — Generation, not training, is the binding constraint

`ESTIMATE`, structurally robust. With 5.26 GiB usable VRAM, a teacher’s corpus-generation throughput collapses once its weights exceed resident capacity. Modelled against an 8M-token targeted corpus:

| Tier | Teachers | Est. generation time | Verdict |
|---|---:|---|---|
| **T1 — VIABLE** | 5 | ≤ 4 days | Run these |
| **T2 — COSTLY** | 16 | 5–21 days | Justify each individually |
| **T3 — DEFERRED** | 12 | 22–90 days | Remains in the canonical queue; deferred until compute, corpus budget, or throughput changes |
| **T4 — INFEASIBLE** | 6 | 100–170 days | Remains in the canonical queue; requires a different compute path or a materially smaller generation budget |

Eighteen of thirty-nine teachers are not runnable at corpus scale on this machine. The precise day counts are estimates and will be wrong; the existence and approximate location of the frontier are structural.

### EF-4 — The training stack does not exist; the generation stack does

`MEASURED`. PyTorch is absent. Ollama is operational and serving 43 models. The Distillery can begin generating corpora today and cannot train anything today. The `sm_120` / bitsandbytes question remains unmeasured.

### EF-5 — A significant fraction of the library is already synthetic-derived

`DERIVED` from model identities. Multiple `deepseek-r1` variants are distillations of R1 into other bases; Dolphin models are heavily instruction-tuned on synthetic corpora. Training Sovereign on their outputs makes Sovereign a third-generation synthetic artifact for that material. This elevates the non-synthetic corpus fraction from good practice to requirement and makes teacher-lineage (`generation_depth`) a mandatory registry field.

### EF-6 — Same-base families are present

`DERIVED`, requires confirmation. Several R1 distills sit beside their own base models (Qwen2.5-14B, Qwen3-8B, Llama-3.3-70B classes). Where confirmed, this is the only circumstance in which compatible weight merging is legitimately available. The same fact is a measurement hazard: capability deltas across a same-base group are not independent.

---

## 4. Architecture

### 4.1 Position in the workspace

| Layer | Responsibility | Relationship to Distillery |
|---|---|---|
| Sovereign | Orchestration, memory, governance, long-horizon state | Owns Distillery workflow state, lineage registry, promotion decisions |
| Debate Table | Adversarial evaluation, critique | Advisory signal only; never the sole promotion gate |
| Multi-Model App | Heterogeneous execution, parallel interrogation | Extraction laboratory |
| **Sovereign Distillery** | Model construction, distillation, training, evaluation, quantization | Owns the model lifecycle and the evolving lineage |

The Distillery is operable **standalone** (INV-14). Sibling subsystems are optional sensors and critics, not hard dependencies.

### 4.2 Invariants

| ID | Invariant |
|---|---|
| INV-1 | Checkpoint is never overwritten. Promotion changes a pointer, never a file. |
| INV-2 | Every training example carries provenance recorded at generation time. |
| INV-3 | No checkpoint is promoted without passing the frozen evaluation suite. |
| INV-3b | Promotion requires **both** step tolerance **T** (vs parent) **and** drift floor **D** (vs historical best / governed floor). Breach of D requires explicit recorded operator override. |
| INV-4 | Distillation is offline and black-box by default. Teacher and student are never required to be co-resident. |
| INV-5 | Canonical master and quantized deployment artifacts are separate objects. |
| INV-6 | Evaluation suite is frozen before use and versioned separately from models. |
| INV-7 | Held-out evaluation data is never used for corpus generation. |
| INV-8 | Eligible teachers are examined smallest-to-largest. |
| INV-9 | A teacher with no useful capability delta may be skipped; `SKIPPED_NO_DELTA` is a valid outcome. |
| INV-10 | Teacher ordering does not require single-teacher-only training data; replay is permitted. |
| INV-11 | Parameter / architecture scale does not increase automatically with each teacher. |
| INV-12 | Scale may increase only when capacity evidence and available compute both permit it. |
| INV-13 | Cross-family weight transfer is never assumed; compatibility must be demonstrated. |
| INV-14 | Distillery is operable standalone. |
| INV-15 | Canonical promotion authority remains with the human operator. |

### 4.3 Core objects

- **Teacher** — Local model used as source of behaviour. Never modified. Registry fields include `base_family`, `derived_from`, `generation_depth`, `license_class`, content hash, role (`TEACHER` / `INFRASTRUCTURE` / `ARTIFACT` / `INELIGIBLE`).
- **Sovereign checkpoint** — Immutable student artifact `SOV-D<NNN>`. Full lineage record.
- **Capability Ledger** — Per-capability current score, historical best, governed floor, source, trend, and `criticality` (`CRITICAL` / `STANDARD` / `EXPERIMENTAL`). The criticality tag makes the promotion condition “no critical regression” machine-checkable (RD-3). Supplies the data for the drift-floor and criticality gates.
- **Corpus shard** — Immutable generated examples carrying full provenance.
- **Candidate state machine** — PLANNED → … → PROMOTABLE → PROMOTED → PRODUCTION, with QUARANTINED and REGRESSION_DETECTED as first-class states. Training completion is not promotion.
- **Distillery Knowledge Lineage** — Structured experiment record (lab notebook) owned by the Distillery. Answers “what did we try, what happened, what did it cost.” Not a learned policy engine at current sample size.
- **Deployment artifact** — Quantized, hardware-targeted derivative of a promoted master. Terminal; never a parent.

### 4.4 Canonical lifecycle (per teacher)

```
 1. INGEST        register; architecture, tokenizer, license, hash, lineage fields
 2. GATE          license class → permitted / restricted / rejected
 3. CHARACTERIZE  measure on frozen eval suite
 4. DIFFERENTIAL  teacher vs current checkpoint → capability delta
 5. DECIDE        empty delta → SKIP (valid); present delta → proceed
 6. CURRICULUM    prompt set targeting only the delta
 7. GENERATE      teacher outputs; provenance stamped per example
 8. FILTER        verification; reject unverifiable / incorrect; record rejection rate
 9. MIX           new shard + replay + non-synthetic fraction
10. TRAIN         QLoRA/LoRA from current parent → immutable candidate
11. EVALUATE      full suite + regression band + drift-floor check
12. ADJUDICATE    optional Debate Table pass — advisory only
13. PROMOTE       operator decision under dual-gate rule (T and D)
14. RECORD        lineage + Knowledge Lineage entry; corpus sealed
```

Empty delta at step 5 is a correct outcome, not a failure.

### 4.5 Promotion gate — dual threshold

A per-step tolerance alone cannot protect cumulative capability retention. Twenty promotions each losing two points within tolerance can destroy a capability while remaining individually legal. The project’s own most important metric — net retained capability — therefore requires a second gate:

1. **Step tolerance T** — maximum regression versus the immediate parent.
2. **Drift floor D** — maximum regression versus historical best / governed floor for that capability across the lineage.

A candidate that passes T but breaches D is not promotable without explicit recorded operator override. The Capability Ledger already stores the necessary data; only the gate was previously missing. Thresholds M (improvement margin), T, and D are operator-set and currently unset; they must be derived from measured run-to-run variance on the frozen suite.

**Criticality (RD-3).** A capability tagged `CRITICAL` may not regress against its parent by more than `max(T/2, measured_variance)`, independently of T and D. This is the fourth condition of the accepted promotion rule; the tag is the data it evaluates against.

**Remediation route (RD-2).** A candidate whose declared purpose is `REMEDIATION` satisfies the improvement condition by measured **restoration toward the governed floor** for its remediation target, not by a new capability gain. Without this, the `REMEDIATION` state has no legal exit.

### 4.6 Mechanism clarity

| Mechanism | Status for v1 |
|---|---|
| Offline black-box sequence-level distillation | **Primary.** Tokenizer- and architecture-agnostic. Fits the hardware. |
| White-box / logit-level KD | Research track. Memory cost and cross-tokenizer maturity make it unsuitable for v1. |
| Cross-family weight transplantation | Out of scope. No established method. |
| Same-family weight merging | Conditionally available where `base_family` confirmation exists (EF-6). |

Operator language “cannibalize” maps cleanly onto black-box sequence-level distillation. The specification states this explicitly.

---

## 5. Seed Strategy

### 5.1 Option C — Bootstrap, then native migration

The original framing of the seed problem as three mutually exclusive options was too narrow. The preferred path sequences them:

- **Phase B — Bootstrap.** Acquire a permissive open-weight 1–3B base from outside the library. Prove the pipeline. Accumulate datasets, evaluation suite, capability maps, training experience, and Knowledge Lineage entries as portable assets.
- **Phase C — Native migration.** Once pipeline maturity is demonstrated, migrate to a Sovereign-controlled architecture. The corpora, eval suite, and capability ledger are architecture-independent and transfer.

Sovereignty becomes a trajectory with named phases rather than a binary property claimed up front. The migration trigger remains open: demonstrated pipeline maturity, not enthusiasm.

### 5.2 Immediate consequence of EF-2

No local candidate exists. Seed acquisition is a prerequisite for F.3. Selection criteria: permissive license first (the seed’s terms bind every downstream checkpoint permanently), local trainability under the measured envelope, tokenizer suitability, and migration potential.

---

## 6. Teacher Tiering and the Feasibility Frontier

Every prior document treated the whole library as the input queue. Measurement says five of thirty-nine teachers are comfortably viable and eighteen are not viable at all on this machine at planned corpus scale.

| Tier | Meaning | Action |
|---|---|---|
| T1 | Viable (≤ ~4 days estimated generation) | Run these |
| T2 | Costly (5–21 days) | Justify each individually |
| T3 | Deferred (22–90 days) | Not on this hardware at this corpus size |
| T4 | Infeasible (100–170 days) | Not on this hardware |

The ordering remains ascending size within the runnable set. The precise day counts are estimates and will be replaced by measured throughput on one T1 teacher before any multi-teacher commitment is made.

**DR-1 (decision required):** Adopt the tiered queue as the operative teacher plan, replacing “process the whole library.”

---

## 7. Evaluation and Acceptance

### 7.1 Evaluation suite (must precede all training)

| Band | Content | Scoring |
|---|---|---|
| Deterministic | Executable code tests; checkable math; exact-match structured extraction; programmatic instruction-following constraints | Exact / pass-rate. No model in the loop. |
| Regression | Every capability previously demonstrated. Grows monotonically. | Delta vs parent + drift-floor check vs historical best / governed floor. |
| Held-out | Private. Never used for generation. | Same instruments. Anti-contamination control. |
| Advisory | Debate Table adversarial comparison | Recorded. Not gating. |

The suite is frozen and versioned before the first teacher is processed. Thresholds are derived from measured run-to-run variance, not intuition.

### 7.2 Acceptance — two routes

**Route A — Pipeline / decision proof** (valid intermediate milestone, not v1 complete):

Ingestion, characterization, differential evaluation, evidence-backed SKIP or ELIGIBLE disposition, provenance and lineage record, resumability under interruption.

**Route B — Transfer proof** (**required for v1**):

All of Route A plus curriculum, generation with provenance, filtering, mix, training, full evaluation including drift-floor, operator promotion decision, complete lineage, and a quantized deployment artifact that runs under the Sovereign harness. At least one successful interrupted-and-resumed run.

A Distillery that can only decide to skip has not been proven. Route B exercises the training machinery.

---

## 8. Risk Register (revised against measurement)

| ID | Risk | Severity | Change | Mitigation |
|---|---|---|---|---|
| R-8 | `sm_120` / bitsandbytes unavailable | **CRITICAL** | ↑↑ — never tested; stack absent | F.0 gate; escalate on failure |
| R-9 | Wall-clock infeasibility of generation | HIGH | ↑ and quantified | Tiering; measured throughput; corpus budget |
| R-3 | Synthetic-data collapse | HIGH | ↑ — teachers themselves synthetic-derived | Generation-depth tracking; non-synthetic floor |
| R-4 | Student capacity ceiling | HIGH | Certain — every teacher 3–40× student | Per-capability bars; never “match the teacher” |
| R-1 | Catastrophic forgetting / cumulative drift | HIGH | — | Replay + dual-gate (T and D) |
| R-5 | License contamination of lineage | HIGH | — | Provenance + exclusion query tested before real data |
| R-10 | Storage exhaustion | HIGH | ↑ quantified (768 GiB already) | Retention policy; DR-5 on ~470 GiB of T3/T4 |
| R-12 | Seed licence propagates permanently | HIGH | NEW | Permissive-first seed selection |
| R-13 | Same-base teachers produce correlated measurements | Medium | NEW (EF-6) | `base_family` / `derived_from` fields |
| R-14 | System RAM insufficient for offloaded teachers | HIGH | NEW — RAM never captured | Capture in F.0 |
| R-15 | Reclaimable VRAM never measured | Low | NEW | Re-probe with GPU apps closed |

---

## 9. Decisions Required from the Operator

| ID | Decision | Recommendation |
|---|---|---|
| **DR-1** | Adopt tiered queue (T1–T4) as operative plan | Adopt |
| **DR-2** | Approve acquisition of 1–3B seed candidates from outside the library | Approve — nothing local qualifies |
| **DR-3** | Corpus token budget per teacher | 2M for T1; revisit after measured throughput |
| **DR-4** | Confirm two-runtime split (Ollama generate / PyTorch train) | Confirm |
| **DR-5** | T3/T4 teachers — deferred-pending-compute or removed | Defer, cold-store (~470 GiB) |
| **DR-6** | Non-synthetic corpus floor | Set a value before the first run |

---

## 10. Implementation Sequence

Nothing in the later phases should be built before steps 1 and 2 return.

1. **F.0 — Install and verify the training stack.** Until a successful bitsandbytes 4-bit forward pass on `sm_120` is reported, the local training plan is unproven.
2. **F.1 — Measure real generation throughput** on one T1 teacher. Replaces all generation-time estimates.
3. **F.2 — Primary license audit** of the runnable set; assign `license_class`; record `base_family` / `derived_from`.
4. **F.3 — Acquire and baseline 1–3B seed candidates** (shortlist, do not freeze until evaluation suite exists).
5. **F.4 — Build and freeze the evaluation suite.** Highest-value remaining design artifact. Derive M/T/D from measured variance.
6. **F.5 — Provenance, lineage, Knowledge Lineage skeleton, exclusion query.** Cannot be retrofitted cleanly later.
7. **F.6 — End-to-end on the cheapest T1 teacher.** Goal is proven pipeline, not a better model. SKIP is a legitimate informative outcome.
8. **F.7–F.11 — Iterate T1, then selected T2; promote under dual-gate rule; produce deployment artifact.**

---

## 11. Open Questions — Current Status

| ID | Question | Status |
|---|---|---|
| OQ-001 | Seed path | Phase A (bootstrap) resolved in principle; candidate selection blocked by EF-2 until external acquisition |
| OQ-002 | Teacher inventory | **CLOSED.** 43 entries, 39 teachers, 768.4 GiB |
| OQ-003 | v1 acceptance | **CLOSED.** Route A / Route B |
| OQ-004 | Promotion thresholds M / T / D | Open — derive from measured variance |
| OQ-005 | Training-mix ratios | Open — non-synthetic floor required before first run |
| OQ-006 | Long-term compute envelope | Sharpened: 18 teachers, ~470 GiB currently infeasible |
| OQ-007 | Distribution intent | Open — gates license-audit severity |
| OQ-008 | Native architecture migration timing | Open — pipeline maturity, not enthusiasm |
| OQ-009 | Sibling repository interface contracts | Open — no repository inspected |
| OQ-010 | Smallest-to-largest rationale | **RESOLVED** — generation-cost ordering (stronger than prior risk-ordering justification); curriculum superiority remains HYP-1 |

---

## 12. Conclusion

The Sovereign Distillery is a real, buildable subsystem. The core concept has survived successive adversarial review and the first confrontation with measured reality.

What did not survive was a set of assumptions that were never more than modelled: that 8 GiB of VRAM was available, that the library contained small teachers suitable for cheap pipeline validation, that a seed could be selected locally, and that the entire library could be treated as a uniform input queue. Measurement retired those assumptions cleanly and replaced them with a concrete, tiered, schedulable plan.

The architecture absorbs the new facts. The dual-gate promotion rule protects the metric the project itself names as most important. The two-route acceptance criterion gives the project a terminal state. The Knowledge Lineage records what the Distillery learns about how models should be built. Option C turns sovereignty into a trajectory rather than a premature claim.

One hard technical unknown remains larger than any remaining design question: whether the training stack can execute a 4-bit forward pass on this specific Blackwell card. That is a one-day measurement. Everything downstream depends on the answer.

The specification is no longer ahead of the evidence. The evidence has caught up, and it has made the next steps smaller, sharper, and executable.

*This is a thesis and an engineering baseline. It is not an approval. Acceptance belongs to the operator.*

---

## Appendix A — Terminology

| Term | Definition |
|---|---|
| **Tier (T1–T4)** | Teacher classification by estimated corpus-generation feasibility on measured hardware |
| **Generation feasibility frontier** | Model size above which corpus generation ceases to be schedulable on this machine |
| **Governed floor** | Per-capability regression floor that moves only by recorded operator override |
| **Generation depth** | Distance in synthetic-derivation hops from a non-synthetic base |
| **Base family** | Group of models sharing a base checkpoint |
| **Role** | `TEACHER` / `INFRASTRUCTURE` / `ARTIFACT` / `INELIGIBLE` |
| **Two-runtime split** | Ollama for teacher generation, PyTorch for student training |
| **Route A / Route B** | Pipeline proof vs transfer proof; v1 requires Route B |
| **Drift floor D** | Second promotion gate protecting cumulative capability retention |

## Appendix B — Document lineage

| Document | Standing |
|---|---|
| Operator statements | Authoritative objectives preserved |
| Design dialogue | Engineering value retained; silent additions made explicit |
| Validation & Specification Report | Findings F-1..F-8 stand except where measurement supersedes |
| Integration Review (IR-1..IR-10) | All adopted |
| Canonical Report v2.0 | Architecture stands; hardware and library sections superseded by this thesis |
| **This thesis** | Current evidence-grounded baseline |

## Appendix C — Evidence log (selected)

| ID | Claim | Label |
|---|---|---|
| M-1 | RTX 5060 Ti, 8151 MiB nominal, 5383 MiB free, sm_120, driver 610.74 | `MEASURED` |
| M-2 | PyTorch / bitsandbytes absent | `MEASURED` |
| M-3 | 43 entries, 768.4 GiB, 8B–128B | `MEASURED` |
| M-4 | Smallest generative model 8B | `DERIVED` |
| M-5 | 2768 MiB consumed at rest | `DERIVED` |
| E-9/E-10 | Cross-tokenizer KD maturity and measured student/teacher gap | Primary literature |

---

*End of thesis.*
