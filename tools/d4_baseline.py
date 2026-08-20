#!/usr/bin/env python3
"""Deterministic D4 baseline and candidate-specific QLoRA smoke test."""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


TASKS = [
    {
        "id": "instruction_exact",
        "category": "instruction_following",
        "prompt": "Output exactly SOVEREIGN and nothing else.\nAnswer:",
        "max_new_tokens": 12,
        "validator": "exact_sovereign",
    },
    {
        "id": "structured_json",
        "category": "structured_output",
        "prompt": "Return only JSON with string key status set to ok and integer key count set to 3.\nJSON:",
        "max_new_tokens": 32,
        "validator": "status_count_json",
    },
    {
        "id": "code_square",
        "category": "code",
        "prompt": "Complete only the Python expression after return so this function squares every integer.\ndef square(n):\n    return ",
        "max_new_tokens": 24,
        "validator": "square_expression",
    },
    {
        "id": "math_multiply",
        "category": "math",
        "prompt": "Compute 37 * 19. Return only the integer.\nAnswer:",
        "max_new_tokens": 16,
        "validator": "integer_703",
    },
    {
        "id": "reasoning_train",
        "category": "reasoning",
        "prompt": "A train leaves at 14:05 at 80 km/h. A second leaves the same station at 14:35 at 120 km/h on the same route. Return only the catch-up time in HH:MM.\nAnswer:",
        "max_new_tokens": 24,
        "validator": "time_1535",
    },
    {
        "id": "science_phase_change",
        "category": "science",
        "prompt": "Which process directly changes a liquid into a gas? A) condensation B) evaporation C) freezing D) deposition. Return only the letter.\nAnswer:",
        "max_new_tokens": 8,
        "validator": "choice_b",
    },
    {
        "id": "context_retrieval",
        "category": "context_behavior",
        "prompt": (
            "Read the record and return only its passphrase. Record: project=Distillery; "
            "owner=operator; passphrase=cedar-47; status=experimental. "
            + "irrelevant telemetry stable; " * 180
            + "Question: What is the passphrase?\nAnswer:"
        ),
        "max_new_tokens": 20,
        "validator": "cedar_47",
    },
    {
        "id": "tool_call_json",
        "category": "tool_formatting",
        "prompt": "Return only this tool call as JSON: tool is lookup and args contains integer id 7.\nJSON:",
        "max_new_tokens": 40,
        "validator": "tool_json",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


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


def first_json(text: str) -> object:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no valid JSON value found")


def validate_output(kind: str, text: str) -> tuple[bool, str]:
    stripped = text.strip()
    try:
        if kind == "exact_sovereign":
            return stripped == "SOVEREIGN", f"exact={stripped == 'SOVEREIGN'}"
        if kind == "status_count_json":
            value = first_json(stripped)
            passed = isinstance(value, dict) and value == {"status": "ok", "count": 3}
            return passed, f"parsed={value!r}"
        if kind == "square_expression":
            expression = stripped.splitlines()[0].strip()
            parsed = ast.parse(expression, mode="eval")
            code = compile(parsed, "<candidate-expression>", "eval")
            passed = all(eval(code, {"__builtins__": {}}, {"n": n}) == n * n for n in (-3, 0, 5))
            return passed, f"expression={expression!r}"
        if kind == "integer_703":
            match = re.search(r"-?\d+", stripped.replace(",", ""))
            value = int(match.group()) if match else None
            return value == 703, f"first_integer={value!r}"
        if kind == "time_1535":
            match = re.search(r"\b\d{1,2}:\d{2}\b", stripped)
            value = match.group() if match else None
            return value in ("15:35", "3:35"), f"first_time={value!r}"
        if kind == "choice_b":
            match = re.search(r"\b([A-D])\b", stripped.upper())
            value = match.group(1) if match else None
            return value == "B", f"first_choice={value!r}"
        if kind == "cedar_47":
            normalized = stripped.lower().replace("_", "-")
            return normalized.startswith("cedar-47"), f"prefix={normalized[:24]!r}"
        if kind == "tool_json":
            value = first_json(stripped)
            passed = (
                isinstance(value, dict)
                and value.get("tool") == "lookup"
                and value.get("args") == {"id": 7}
            )
            return passed, f"parsed={value!r}"
    except Exception as error:
        return False, f"validator_error={type(error).__name__}: {error}"
    return False, "unknown validator"


def environment_record() -> dict[str, object]:
    import importlib.metadata
    import psutil
    import torch

    packages = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "peft", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    memory = psutil.virtual_memory()
    return {
        "captured_at": utc_now(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "packages": packages,
        "cuda_compiled_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "system_ram_total_bytes": memory.total,
        "system_ram_available_bytes": memory.available,
        "vram_at_start_mib": nvidia_memory(),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    shortlist = json.loads(args.shortlist.read_text(encoding="utf-8"))
    matches = [row for row in shortlist["candidates"] if row["id"] == args.candidate_id]
    if len(matches) != 1:
        raise SystemExit("candidate ID does not uniquely match the shortlist")
    candidate = matches[0]
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    task_results: list[dict[str, object]] = []

    def note(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        stdout_lines.append(line)
        print(line, flush=True)

    environment = environment_record()
    write_json(args.out_dir / "environment.json", environment)
    result: dict[str, object] = {
        "schema": "sovereign-distillery/d4-candidate-baseline/v1",
        "phase": "D4",
        "status": "FAIL",
        "classification": "MEASURED",
        "candidate": {
            key: candidate[key]
            for key in (
                "id",
                "exact_model",
                "revision",
                "checkpoint_type",
                "parameter_count",
                "architecture",
                "context_window_tokens",
            )
        },
        "quantization": {
            "method": "bitsandbytes NF4",
            "compute_dtype": "bfloat16",
            "double_quantization": True,
        },
        "model_load": {"passed": False},
        "task_summary": {},
        "performance": {},
        "qlora_smoke": {"passed": False},
        "limitations": [
            "D4-lite is an eight-task deterministic smoke baseline, not the future frozen Distillery evaluation suite.",
            "Base checkpoints are intentionally tested without inherited instruction tuning; low instruction scores are expected and are not a load failure.",
            "Generated task outputs are machine-validated; no model judge or subjective score is used.",
        ],
        "failures": [],
        "evidence_paths": [
            str(args.out_dir / name)
            for name in ("result.json", "tasks.jsonl", "environment.json", "stdout.txt", "stderr.txt")
        ],
        "timestamp_utc": utc_now(),
    }
    exit_code = 1
    try:
        import bitsandbytes as bnb
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        note(f"Loading {candidate['exact_model']} at {candidate['revision']} in NF4")
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            candidate["exact_model"], revision=candidate["revision"], cache_dir=args.cache_dir
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            candidate["exact_model"],
            revision=candidate["revision"],
            cache_dir=args.cache_dir,
            quantization_config=quantization,
            device_map={"": 0},
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        four_bit_layers = sum(1 for module in model.modules() if isinstance(module, bnb.nn.Linear4bit))
        if not four_bit_layers:
            raise RuntimeError("no bitsandbytes Linear4bit modules after quantized load")
        result["model_load"] = {
            "passed": True,
            "seconds": load_seconds,
            "bitsandbytes_linear4bit_layers": four_bit_layers,
            "model_memory_footprint_bytes": model.get_memory_footprint(),
            "peak_pytorch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_pytorch_reserved_bytes": torch.cuda.max_memory_reserved(),
            "system_vram_after_load_mib": nvidia_memory(),
            "device": str(next(model.parameters()).device),
        }
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        torch.cuda.reset_peak_memory_stats()

        note(f"Running {len(TASKS)} deterministic tasks")
        for task in TASKS:
            encoded = tokenizer(task["prompt"], return_tensors="pt", truncation=True, max_length=2048)
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            input_tokens = int(encoded["input_ids"].shape[-1])
            started = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=task["max_new_tokens"],
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            new_tokens = generated[0, input_tokens:]
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            output_tokens = int(new_tokens.numel())
            passed, validator_detail = validate_output(task["validator"], output_text)
            task_record = {
                "id": task["id"],
                "category": task["category"],
                "validator": task["validator"],
                "passed": passed,
                "validator_detail": validator_detail,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_seconds": elapsed,
                "output_tokens_per_second": output_tokens / elapsed if elapsed else None,
                "output_text": output_text,
            }
            task_results.append(task_record)
            note(
                f"task={task['id']} passed={passed} output_tokens={output_tokens} "
                f"tok/s={task_record['output_tokens_per_second']:.3f}"
            )

        rates = [float(row["output_tokens_per_second"]) for row in task_results]
        latencies = [float(row["latency_seconds"]) for row in task_results]
        category_scores = {
            row["category"]: {"passed": bool(row["passed"]), "score": int(bool(row["passed"]))}
            for row in task_results
        }
        passed_count = sum(bool(row["passed"]) for row in task_results)
        result["task_summary"] = {
            "suite": "D4-lite-v1",
            "task_count": len(task_results),
            "passed_count": passed_count,
            "exact_score": passed_count / len(task_results),
            "category_scores": category_scores,
        }
        result["performance"] = {
            "total_input_tokens": sum(int(row["input_tokens"]) for row in task_results),
            "total_output_tokens": sum(int(row["output_tokens"]) for row in task_results),
            "output_tokens_per_second_mean": statistics.mean(rates),
            "output_tokens_per_second_median": statistics.median(rates),
            "latency_seconds_mean": statistics.mean(latencies),
            "latency_seconds_median": statistics.median(latencies),
            "peak_eval_pytorch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_eval_pytorch_reserved_bytes": torch.cuda.max_memory_reserved(),
            "system_vram_after_eval_mib": nvidia_memory(),
        }

        note("Running candidate-specific LoRA loss/backward/optimizer smoke step")
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        suffixes = {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, bnb.nn.Linear4bit)
        }
        targets = [name for name in ("q_proj", "v_proj") if name in suffixes]
        if len(targets) < 2:
            targets = sorted(name for name in suffixes if name != "lm_head")[:2]
        lora = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lora)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        train_inputs = tokenizer(
            "Sovereign Distillery candidate-specific QLoRA verification.", return_tensors="pt"
        )
        train_inputs = {key: value.to("cuda") for key, value in train_inputs.items()}
        labels = train_inputs["input_ids"].clone()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        train_started = time.perf_counter()
        train_output = model(**train_inputs, labels=labels)
        loss = train_output.loss
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite training loss: {loss_value}")
        loss.backward()
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("candidate backward pass produced no LoRA gradients")
        gradient_norm = float(
            torch.sqrt(sum(torch.sum(gradient.detach().float() ** 2) for gradient in gradients)).item()
        )
        if not math.isfinite(gradient_norm) or gradient_norm == 0:
            raise RuntimeError(f"invalid candidate gradient norm: {gradient_norm}")
        optimizer.step()
        torch.cuda.synchronize()
        result["qlora_smoke"] = {
            "passed": True,
            "target_modules": targets,
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
            "loss": loss_value,
            "gradient_norm": gradient_norm,
            "seconds": time.perf_counter() - train_started,
            "peak_pytorch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_pytorch_reserved_bytes": torch.cuda.max_memory_reserved(),
            "system_vram_after_step_mib": nvidia_memory(),
        }
        result["status"] = "PASS"
        result["timestamp_utc"] = utc_now()
        note(
            f"D4 PASS for {candidate['id']}: deterministic score={passed_count}/{len(TASKS)}, "
            f"QLoRA step passed"
        )
        exit_code = 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        result["failures"] = [{"message": message}]
        stderr_lines.append(traceback.format_exc())
        note(f"D4 FAIL for {candidate['id']}: {message}")
    finally:
        (args.out_dir / "tasks.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in task_results), encoding="utf-8"
        )
        write_json(args.out_dir / "result.json", result)
        (args.out_dir / "stdout.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
        (args.out_dir / "stderr.txt").write_text("\n".join(stderr_lines), encoding="utf-8")
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
