# Ship Checklist

## Pre-commit

- [ ] All TR-1 … TR-9 fixes summarized in `CHANGELOG.md` applied
- [ ] `tools/d1_characterize.py`, `tools/d2_registry.py` (and optionally `d3_throughput.py`) present
- [ ] Internal Markdown links resolve (`docs-check` workflow)
- [ ] Cited repository paths (`tools/…`, `registry/…`, …) exist
- [ ] No weights (`.gguf`, `.safetensors`, `.pt`, `.pth`) or secrets (`.env`, keys) committed
- [ ] **Keep this repository PRIVATE while `LICENSE` is all-rights-reserved.**  
      Publishing grants view-and-fork rights under GitHub’s terms regardless of the licence text.  
      Resolve OQ-007 before changing visibility. (TR-7 / TR-8)
- [ ] Before making public, review `registry/teachers.json` for local-path and inventory disclosure; consider a path-redacted public variant.

## Evidence refresh

The initial v1.1 commit includes `runs/hardware_profile.json` and `registry/teachers.json`, promoting M-1 through M-5 to `MEASURED`. Once the training stack is installed, re-run D1 and add the refreshed profile plus the first D3 throughput capture:

```
git add runs/hardware_profile.json runs/generation_throughput.json
```

This resolves the open toolchain-compatibility measurement and closes evidence-register item 4.

## Post-commit

- [ ] `docs-check` workflow green on `main`
- [ ] Operator has signed DR-1 … DR-6 (or explicitly deferred them)
- [ ] F.0 (training-stack verification) scheduled
