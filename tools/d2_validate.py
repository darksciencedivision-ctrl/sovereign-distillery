#!/usr/bin/env python3
"""Independently validate D2 registry totals, roles, and canonical order."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.registry.read_text(encoding="utf-8"))
    entries = payload["teachers"]
    teachers = [
        row
        for row in entries
        if row.get("role") == "TEACHER" and row.get("disposition") != "SUPERSEDED"
    ]
    non_teachers = [row for row in entries if row not in teachers]
    failures: list[str] = []

    expected_order = sorted(
        teachers,
        key=lambda row: (row.get("parameter_count") or 0, row.get("disk_size_bytes") or 0),
    )
    actual_positions = [row.get("canonical_order") for row in teachers]
    if actual_positions != list(range(1, len(teachers) + 1)):
        failures.append("canonical_order is not consecutive from 1")
    if [row["display_name"] for row in teachers] != [row["display_name"] for row in expected_order]:
        failures.append("teacher list is not canonical smallest-to-largest order")
    required = ("canonical_order", "execution_tier", "execution_status")
    for row in teachers:
        missing = [field for field in required if row.get(field) is None]
        if missing:
            failures.append(f"{row.get('model_id')} missing required fields: {', '.join(missing)}")
    if any(row.get("canonical_order") is not None for row in non_teachers):
        failures.append("a non-teacher has canonical_order")

    total_bytes = sum(int(row.get("disk_size_bytes") or 0) for row in entries)
    tier_counts = Counter(row.get("execution_tier") for row in teachers)
    role_counts = Counter(row.get("role") for row in entries)
    quantization_counts = Counter(row.get("quantization") or "UNKNOWN" for row in teachers)
    architecture_counts = Counter(row.get("architecture") or "UNKNOWN" for row in teachers)
    runtime_counts = Counter(row.get("source_runtime") or "UNKNOWN" for row in entries)
    license_unknown_count = sum(row.get("license_class") == "UNKNOWN" for row in entries)
    deferred_count = sum(row.get("execution_status") == "DEFERRED_COMPUTE" for row in teachers)

    cross_checks = {
        "entry_count_matches": payload.get("entry_count") == len(entries),
        "teacher_count_matches": payload.get("teacher_count") == len(teachers),
        "non_teacher_count_matches": payload.get("non_teacher_count") == len(non_teachers),
        "total_storage_matches": payload.get("total_storage_bytes") == total_bytes,
    }
    failures.extend(name for name, passed in cross_checks.items() if not passed)

    result = {
        "schema": "sovereign-distillery/teacher_registry_validation/v1",
        "status": "PASS" if not failures else "FAIL",
        "classification": "DERIVED_FROM_MEASURED_ARTIFACT",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(args.registry),
        "registry_generated_at": payload.get("generated_at"),
        "entry_count": len(entries),
        "generative_teacher_count": len(teachers),
        "non_teacher_count": len(non_teachers),
        "total_storage_bytes": total_bytes,
        "total_storage_gib": round(total_bytes / 1024**3, 3),
        "smallest_teacher": {
            key: teachers[0].get(key)
            for key in (
                "model_id",
                "display_name",
                "parameter_count",
                "quantization",
                "disk_size_gib",
                "canonical_order",
            )
        },
        "largest_teacher": {
            key: teachers[-1].get(key)
            for key in (
                "model_id",
                "display_name",
                "parameter_count",
                "quantization",
                "disk_size_gib",
                "canonical_order",
            )
        },
        "tier_counts": dict(sorted((str(key), value) for key, value in tier_counts.items())),
        "deferred_compute_count": deferred_count,
        "license_unknown_count": license_unknown_count,
        "role_counts": dict(sorted(role_counts.items())),
        "quantization_counts": dict(sorted(quantization_counts.items())),
        "architecture_counts": dict(sorted(architecture_counts.items())),
        "runtime_counts": dict(sorted(runtime_counts.items())),
        "cross_checks": cross_checks,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
