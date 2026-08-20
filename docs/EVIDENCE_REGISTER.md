# Evidence Register

Repository evidence artifacts supporting the thesis's measured claims.

| # | Claim | Required artifact | Status |
|---|---|---|---|
| 1 | Hardware profile (VRAM free, sm_120, driver) | `runs/hardware_profile.json` | **Present — captured 2026-08-19 on the operator machine** |
| 2 | Teacher registry (43 entries, ordering, roles) | `registry/teachers.json` | **Present — captured 2026-08-19 on the operator machine** |
| 3 | Source tools exist | `tools/d1_characterize.py`, `tools/d2_registry.py` | **Present in v1.1 package** |
| 4 | Generation throughput on a T1 teacher | `runs/generation_throughput.json` | Pending — requires `d3_throughput.py` + live run |

Items 1–3 are committed with the initial v1.1 baseline, promoting M-1 through M-5 to `MEASURED`. Re-run D1 after installing the training stack to resolve the separate `sm_120` / bitsandbytes compatibility question; the baseline capture correctly records that stack as absent.
