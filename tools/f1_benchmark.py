#!/usr/bin/env python3
"""Run the reproducible Sovereign Distillery F.1 Ollama throughput benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


HOST = "http://127.0.0.1:11434"
PROMPTS = [
    {
        "id": "short_answer",
        "text": "In one sentence, explain the difference between latency and throughput.",
    },
    {
        "id": "long_form",
        "text": "Explain how gradient checkpointing reduces memory during neural-network training. Include its compute tradeoff and one practical limitation.",
    },
    {
        "id": "code",
        "text": "Write a Python function that merges two sorted integer lists in O(n) time. Include a docstring and three assert-based tests.",
    },
    {
        "id": "structured",
        "text": "Return only valid JSON with keys name, version, and deps, where deps is an array of exactly three strings.",
    },
    {
        "id": "reasoning",
        "text": "A train leaves at 14:05 traveling 80 km/h. Another leaves the same station at 14:35 traveling 120 km/h on the same route. At what time does the second train catch the first? Show the calculation.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def post_generate(model: str, prompt: str, max_tokens: int, seed: int) -> dict[str, object]:
    request = urllib.request.Request(
        HOST + "/api/generate",
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0,
                    "seed": seed,
                    "num_ctx": 4096,
                },
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = json.loads(response.read().decode("utf-8"))
    payload["client_wall_seconds"] = time.perf_counter() - started
    return payload


def nvidia_memory() -> dict[str, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    total, used, free = [int(value.strip()) for value in completed.stdout.strip().split(",")]
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def ollama_version() -> str:
    completed = subprocess.run(
        ["ollama", "--version"], capture_output=True, text=True, check=True, timeout=30
    )
    return (completed.stdout or completed.stderr).strip()


def ollama_ps() -> str:
    completed = subprocess.run(
        ["ollama", "ps"], capture_output=True, text=True, check=True, timeout=30
    )
    return completed.stdout.strip()


def resident_models() -> list[str]:
    with urllib.request.urlopen(HOST + "/api/ps", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [str(model.get("name")) for model in payload.get("models", [])]


def unload(model: str) -> None:
    subprocess.run(
        ["ollama", "stop", model], capture_output=True, text=True, check=False, timeout=60
    )


def measured_trial(
    model: str,
    prompt_id: str,
    prompt: str,
    repeat: int,
    max_tokens: int,
    seed: int,
) -> dict[str, object]:
    import psutil

    ram_before = psutil.virtual_memory()
    vram_before = nvidia_memory()
    payload = post_generate(model, prompt, max_tokens, seed)
    residents = resident_models()
    vram_after = nvidia_memory()
    ram_after = psutil.virtual_memory()
    eval_tokens = int(payload.get("eval_count", 0))
    eval_duration_ns = int(payload.get("eval_duration", 0))
    prompt_tokens = int(payload.get("prompt_eval_count", 0))
    prompt_duration_ns = int(payload.get("prompt_eval_duration", 0))
    return {
        "timestamp_utc": utc_now(),
        "prompt_id": prompt_id,
        "repeat": repeat,
        "seed": seed,
        "max_output_tokens": max_tokens,
        "output_tokens": eval_tokens,
        "prompt_tokens": prompt_tokens,
        "generation_tokens_per_second": (
            eval_tokens / (eval_duration_ns / 1e9) if eval_duration_ns else None
        ),
        "prompt_tokens_per_second": (
            prompt_tokens / (prompt_duration_ns / 1e9) if prompt_duration_ns else None
        ),
        "load_seconds": int(payload.get("load_duration", 0)) / 1e9,
        "prompt_eval_seconds": prompt_duration_ns / 1e9,
        "generation_seconds": eval_duration_ns / 1e9,
        "total_server_seconds": int(payload.get("total_duration", 0)) / 1e9,
        "client_wall_seconds": payload["client_wall_seconds"],
        "done_reason": payload.get("done_reason"),
        "resident_models_after": residents,
        "vram_before_mib": vram_before,
        "vram_after_mib": vram_after,
        "ram_available_before_bytes": ram_before.available,
        "ram_available_after_bytes": ram_after.available,
    }


def duration_projection(tokens: int, tokens_per_second: float) -> dict[str, object]:
    seconds = tokens / tokens_per_second
    return {
        "target_output_tokens": tokens,
        "formula": "target_output_tokens / measured_mean_generation_tokens_per_second",
        "estimated_seconds": seconds,
        "estimated_hours": seconds / 3600,
        "estimated_days": seconds / 86400,
        "classification": "DERIVED",
        "assumption": (
            "Mean warm single-stream throughput remains constant; this excludes failures, "
            "restarts, prompt growth, thermal drift, and long-run scheduling overhead."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher-id", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def note(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        stdout_lines.append(line)
        print(line, flush=True)

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    matching = [
        row
        for row in registry["teachers"]
        if row.get("model_id") == args.teacher_id and row.get("display_name") == args.model
    ]
    if len(matching) != 1:
        raise SystemExit("teacher/model pair is not unique in the supplied registry")
    teacher = matching[0]
    if teacher.get("canonical_order") != 1:
        raise SystemExit("F.1 requires canonical teacher order 1")

    import psutil

    baseline_memory = psutil.virtual_memory()
    environment = {
        "captured_at": utc_now(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": os.path.abspath(os.sys.executable),
        "ollama_version": ollama_version(),
        "ollama_host": HOST,
        "system_ram_total_bytes": baseline_memory.total,
        "system_ram_available_bytes": baseline_memory.available,
        "vram_before_unload_mib": nvidia_memory(),
    }
    write_json(args.out_dir / "environment.json", environment)
    write_json(
        args.out_dir / "prompts.json",
        {
            "temperature": 0,
            "num_ctx": 4096,
            "max_output_tokens": args.max_tokens,
            "repeats_per_prompt": args.repeats,
            "prompts": PROMPTS,
        },
    )

    trials: list[dict[str, object]] = []
    result: dict[str, object] = {
        "schema": "sovereign-distillery/f1-result/v1",
        "phase": "F.1",
        "status": "FAIL",
        "classification": "MEASURED",
        "teacher": {
            "teacher_id": teacher["model_id"],
            "canonical_order": teacher["canonical_order"],
            "model": teacher["display_name"],
            "parameter_count": teacher["parameter_count"],
            "parameter_basis": teacher["parameter_basis"],
            "quantization": teacher["quantization"],
            "architecture": teacher["architecture"],
            "runtime": teacher["source_runtime"],
            "context_length": teacher["context_length"],
            "execution_tier": teacher["execution_tier"],
        },
        "warmup": {},
        "statistics": {},
        "resource_measurements": {},
        "derived_generation_estimates": [],
        "limitations": [],
        "failures": [],
        "evidence_paths": [
            str(args.out_dir / name)
            for name in (
                "README.md",
                "result.json",
                "prompts.json",
                "trials.jsonl",
                "environment.json",
                "stdout.txt",
                "stderr.txt",
                "commands.txt",
            )
        ],
        "timestamp_utc": utc_now(),
    }

    exit_code = 1
    try:
        note(f"Unloading {args.model} before the cold-load capture")
        unload(args.model)
        environment["vram_after_unload_mib"] = nvidia_memory()
        cold = measured_trial(
            args.model,
            "cold_load_probe",
            "State the word ready and briefly identify yourself.",
            0,
            32,
            40,
        )
        if cold["resident_models_after"] != [args.model]:
            raise RuntimeError(
                f"runtime contamination after cold load: {cold['resident_models_after']}"
            )
        result["cold_load"] = cold
        note(
            f"Cold load: {cold['load_seconds']:.3f}s; "
            f"generation {cold['generation_tokens_per_second']:.3f} tok/s"
        )

        note("Running one unmeasured warm-up generation")
        warmup = measured_trial(
            args.model,
            "warmup",
            "Explain why fixed prompts improve benchmark reproducibility.",
            0,
            128,
            40,
        )
        if warmup["resident_models_after"] != [args.model]:
            raise RuntimeError(
                f"runtime contamination after warmup: {warmup['resident_models_after']}"
            )
        result["warmup"] = warmup

        note(f"Running {len(PROMPTS) * args.repeats} measured warm trials")
        for repeat in range(1, args.repeats + 1):
            for prompt in PROMPTS:
                trial = measured_trial(
                    args.model,
                    prompt["id"],
                    prompt["text"],
                    repeat,
                    args.max_tokens,
                    40 + repeat,
                )
                trials.append(trial)
                if trial["resident_models_after"] != [args.model]:
                    raise RuntimeError(
                        f"runtime contamination in measured trial: {trial['resident_models_after']}"
                    )
                note(
                    f"trial={len(trials):02d} prompt={prompt['id']} repeat={repeat} "
                    f"output={trial['output_tokens']} "
                    f"tok/s={trial['generation_tokens_per_second']:.3f}"
                )

        rates = [float(trial["generation_tokens_per_second"]) for trial in trials]
        prompt_rates = [float(trial["prompt_tokens_per_second"]) for trial in trials]
        mean_rate = statistics.mean(rates)
        statistics_record = {
            "trial_count": len(trials),
            "total_output_tokens": sum(int(trial["output_tokens"]) for trial in trials),
            "generation_tokens_per_second": {
                "mean": mean_rate,
                "median": statistics.median(rates),
                "minimum": min(rates),
                "maximum": max(rates),
                "standard_deviation": statistics.stdev(rates),
                "variance": statistics.variance(rates),
            },
            "prompt_tokens_per_second": {
                "mean": statistics.mean(prompt_rates),
                "median": statistics.median(prompt_rates),
            },
            "client_wall_seconds": {
                "total": sum(float(trial["client_wall_seconds"]) for trial in trials),
                "mean": statistics.mean(float(trial["client_wall_seconds"]) for trial in trials),
            },
        }
        result["statistics"] = statistics_record
        all_samples = [cold, warmup, *trials]
        peak_used = max(
            int(sample["vram_after_mib"]["used_mib"]) for sample in all_samples
        )
        baseline_used = int(environment["vram_after_unload_mib"]["used_mib"])
        result["resource_measurements"] = {
            "vram_total_mib": int(environment["vram_after_unload_mib"]["total_mib"]),
            "vram_used_after_unload_mib": baseline_used,
            "peak_observed_system_vram_used_mib": peak_used,
            "peak_observed_model_delta_mib": peak_used - baseline_used,
            "peak_observation_method": "nvidia-smi sampled immediately after each blocking generation",
            "ollama_ps_after_trials": ollama_ps(),
            "minimum_system_ram_available_bytes": min(
                int(sample["ram_available_after_bytes"]) for sample in all_samples
            ),
        }
        result["derived_generation_estimates"] = [
            duration_projection(tokens, mean_rate) for tokens in (100_000, 1_000_000, 5_000_000)
        ]
        result["limitations"] = [
            "This is deterministic single-stream throughput, not concurrent aggregate throughput.",
            "nvidia-smi was sampled after each blocking request; transient within-request peaks may be higher.",
            "Long-run projections assume constant mean warm throughput and exclude failures or scheduling overhead.",
            "The teacher license remains UNKNOWN; F.1 proves technical executability, not training-data eligibility.",
        ]
        result["status"] = "PASS"
        result["timestamp_utc"] = utc_now()
        note(
            f"F.1 PASS: mean={mean_rate:.3f} median={statistics.median(rates):.3f} "
            f"stdev={statistics.stdev(rates):.3f} tok/s"
        )
        exit_code = 0
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        result["failures"] = [{"category": "OTHER", "message": failure}]
        stderr_lines.append(traceback.format_exc())
        note(f"F.1 FAIL: {failure}")
    finally:
        write_json(args.out_dir / "environment.json", environment)
        write_json(args.out_dir / "result.json", result)
        (args.out_dir / "trials.jsonl").write_text(
            "".join(json.dumps(trial) + "\n" for trial in trials), encoding="utf-8"
        )
        (args.out_dir / "stdout.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
        (args.out_dir / "stderr.txt").write_text("\n".join(stderr_lines), encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
