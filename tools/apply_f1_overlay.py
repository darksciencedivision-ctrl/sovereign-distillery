#!/usr/bin/env python3
"""Apply a validated F.1 measurement as a registry execution overlay."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--corpus-tokens", type=int, default=8_000_000)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("phase") != "F.1":
        raise SystemExit("only a passing F.1 result may update the execution overlay")
    teacher_result = result["teacher"]
    matches = [
        row
        for row in registry["teachers"]
        if row.get("model_id") == teacher_result["teacher_id"]
        and row.get("display_name") == teacher_result["model"]
        and row.get("canonical_order") == teacher_result["canonical_order"]
    ]
    if len(matches) != 1:
        raise SystemExit("F.1 teacher identity does not uniquely match the registry")
    teacher = matches[0]
    rate = float(result["statistics"]["generation_tokens_per_second"]["mean"])
    teacher["measured_tok_per_sec"] = rate
    teacher["measured_gen_days"] = args.corpus_tokens / rate / 86400
    teacher["execution_status"] = "MEASURED_EXECUTABLE"
    teacher["execution_tier"] = "T1-VIABLE"
    teacher["tier"] = "T1-VIABLE"
    teacher["execution_evidence"] = "runs/F1_teacher_throughput/result.json"
    teacher["disposition_history"].append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "F1_THROUGHPUT_MEASURED",
            "execution_status": "MEASURED_EXECUTABLE",
            "measured_tok_per_sec": rate,
            "note": "Technical executability only; disposition remains PENDING until license audit."
        }
    )
    args.registry.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(
        f"Applied F.1 overlay to {teacher['model_id']} {teacher['display_name']}: "
        f"{rate:.3f} tok/s, disposition={teacher['disposition']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
