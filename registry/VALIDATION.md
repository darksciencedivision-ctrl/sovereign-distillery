# D2 Teacher Registry Validation

**Status: PASS — DERIVED from the measured registry artifact captured 2026-08-20 UTC.**

`tools/d2_validate.py` independently recomputed counts, storage, canonical order, roles, execution tiers, and required queue fields from `registry/teachers.json`.

| Measure | Validated value |
|---|---:|
| Total entries | 43 |
| Generative teachers | 39 |
| Non-teachers | 4 (3 infrastructure, 1 multimodal projector artifact) |
| Total storage | 768.406 GiB |
| Canonical first teacher | `T001` — `dolphin3:8b`, 8B, Q4_K_M, 4.583 GiB |
| Canonical last teacher | `T039` — `mistral-medium-3.5:latest`, 127.704B, Q4_K_M, 74.731 GiB |
| T1 / T2 / T3 / T4 | 5 / 16 / 12 / 6 |
| Deferred-compute teachers | 18 |
| License class UNKNOWN | 43 entries |

## Corrections from the v1 registry artifact

- The old artifact used `teacher_count: 43` for all entries. Schema v2 separates `entry_count: 43`, `teacher_count: 39`, and `non_teacher_count: 4`.
- The old artifact's record order was not canonical smallest-to-largest. Schema v2 assigns consecutive `canonical_order` values to generative teachers only.
- A 461M-parameter multimodal projector (`mmproj-F32.gguf`) is now correctly classified as `ARTIFACT`, not a teacher.
- Every teacher now carries `canonical_order`, `execution_tier`, and `execution_status`. Tiers remain execution overlays and do not remove deferred teachers from the canonical queue.

The thesis tag preserves the original v1.1 document. This validation supersedes its registry-order presentation without rewriting the tagged baseline.
