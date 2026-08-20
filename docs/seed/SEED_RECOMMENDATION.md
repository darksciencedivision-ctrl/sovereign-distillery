# Seed Recommendation

**Recommendation: S002 — `Qwen/Qwen3-1.7B-Base`.**  
**Canonical status: `SOV-SEED: OPERATOR_DECISION_REQUIRED`.**

## FACTS

- All three candidates are exact-revision base/pretrained checkpoints in the 1–3B envelope.
- Their primary model repositories declare Apache 2.0; S002 includes a standalone exact-revision license file.
- F.0 proved the Windows `sm_120` bitsandbytes NF4 + LoRA path before candidate testing.
- The canonical Sovereign architecture is not redefined by this bootstrap recommendation.

## MEASUREMENTS

| ID | D4-lite exact score | Mean output tok/s | Model footprint | Peak QLoRA allocation | Candidate QLoRA step |
|---|---:|---:|---:|---:|---|
| S001 OLMo 2 1B | 1/8 | 11.111 | 1.359 GB | 2.341 GB | PASS |
| **S002 Qwen3 1.7B Base** | **4/8** | 4.592 | **1.327 GB** | **2.123 GB** | **PASS** |
| S003 SmolLM3 3B Base | **4/8** | 5.217 | 1.932 GB | 2.716 GB | PASS |

S002 passed exact instruction following, structured JSON, science multiple choice, and tool-call JSON. S003 passed structured JSON, math, science, and context retrieval. S001 passed structured JSON only. Full raw outputs are under `runs/D4_seed_baseline/`.

## DERIVED RESULTS

- S002 ties the highest deterministic score while requiring 593 MB less peak PyTorch allocation than S003.
- S002 supplies 32,768 native context tokens versus S001's 4,096, without S003's 3B upper-bound memory exposure.
- S001 is the throughput leader in this tiny generation workload but its 1/8 exact score is a weaker starting behavioral baseline.

## ASSUMPTIONS

- The eight-task `D4-lite-v1` smoke suite is directionally useful for bootstrap comparison even though it is not the future frozen evaluation suite.
- Short-context single-step QLoRA success predicts toolchain compatibility, not sustained training at production context or batch size.
- A base checkpoint's low instruction score is expected; Distillery must create Sovereign behavior rather than inherit an instruct policy.

## TRADEOFFS

- **S001** maximizes transparency, speed, and expected training margin but currently supplies the weakest measured behavioral baseline and shortest context.
- **S002** offers the strongest balance of measured task behavior, memory, context, license clarity, and ecosystem support.
- **S003** provides more capacity and long context, but its score only ties S002 while consuming more memory and leaving less room for realistic training sequences.

## RECOMMENDATION

Recommend **S002 `Qwen/Qwen3-1.7B-Base` at revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`** as the Distillery v1 bootstrap seed candidate for operator approval.

This is an `INTERPRETATION` grounded in D4 measurements. It is not a silent promotion, does not create a canonical checkpoint, and does not authorize a training run.

## UNKNOWNS

- Production-context QLoRA memory and sustained-step stability are unmeasured.
- The complete frozen Distillery evaluation suite is not implemented.
- No seed candidate has been measured after Sovereign-specific curriculum training.
- Dataset licensing, corpus composition, and long-run thermal/throughput behavior remain open.
