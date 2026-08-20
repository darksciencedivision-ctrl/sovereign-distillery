#!/usr/bin/env python3
"""Execute and record the Sovereign Distillery F.0 QLoRA toolchain gate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "peft",
    "datasets",
    "trl",
    "safetensors",
    "tokenizers",
    "psutil",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def nvidia_query() -> dict[str, object]:
    fields = (
        "name,driver_version,memory.total,memory.used,memory.free,"
        "pci.bus_id,compute_cap,pstate,temperature.gpu"
    )
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    values = [part.strip() for part in completed.stdout.strip().split(",")]
    keys = [
        "name",
        "driver_version",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "pci_bus_id",
        "compute_capability_driver",
        "performance_state",
        "temperature_c",
    ]
    result = dict(zip(keys, values, strict=True))
    result["cuda_version_driver"] = None
    summary = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=True, timeout=30
    ).stdout
    marker = "CUDA UMD Version:"
    if marker in summary:
        result["cuda_version_driver"] = summary.split(marker, 1)[1].split()[0]
    return result


def host_environment() -> dict[str, object]:
    import psutil

    memory = psutil.virtual_memory()
    disks = {}
    for root in ("C:\\", "D:\\"):
        usage = psutil.disk_usage(root)
        disks[root] = {
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "used_bytes": usage.used,
        }
    return {
        "captured_at": utc_now(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "venv_path": sys.prefix if sys.prefix != sys.base_prefix else None,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "system_ram_total_bytes": memory.total,
        "system_ram_available_bytes": memory.available,
        "disks": disks,
        "packages": package_versions(),
    }


def classify_failure(message: str) -> str:
    lowered = message.lower()
    categories = (
        ("out of memory", "MEMORY"),
        ("bitsandbytes", "BITSANDBYTES"),
        ("quant", "QUANTIZATION"),
        ("peft", "PEFT"),
        ("kernel", "KERNEL"),
        ("cuda", "CUDA"),
        ("torch", "PYTORCH"),
    )
    return next((category for marker, category in categories if marker in lowered), "OTHER")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def note(message: str) -> None:
        timestamped = f"[{utc_now()}] {message}"
        stdout_lines.append(timestamped)
        print(timestamped, flush=True)

    result: dict[str, object] = {
        "schema": "sovereign-distillery/f0-result/v1",
        "phase": "F.0",
        "status": "BLOCKED",
        "gpu": {},
        "software": package_versions(),
        "test_model": {"model_id": args.model_id},
        "quantization": {
            "method": "bitsandbytes NF4",
            "load_in_4bit": True,
            "compute_dtype": "bfloat16",
            "double_quantization": True,
        },
        "forward_pass": {"passed": False},
        "training_step": {"passed": False},
        "peak_vram_bytes": 0,
        "limitations": [],
        "failures": [],
        "evidence_paths": [
            str(args.out_dir / "result.json"),
            str(args.out_dir / "environment.json"),
            str(args.out_dir / "hardware.json"),
            str(args.out_dir / "commands.txt"),
            str(args.out_dir / "stdout.txt"),
            str(args.out_dir / "stderr.txt"),
        ],
        "timestamp_utc": utc_now(),
    }

    environment = host_environment()
    write_json(args.out_dir / "environment.json", environment)

    exit_code = 1
    try:
        import bitsandbytes as bnb
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        note("F0-A: checking CUDA and PyTorch device visibility")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable through PyTorch")
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        sm_tag = f"sm_{properties.major}{properties.minor}"
        arch_list = torch.cuda.get_arch_list()
        gpu = nvidia_query()
        gpu.update(
            {
                "torch_device_name": torch.cuda.get_device_name(device),
                "torch_compute_capability": list(torch.cuda.get_device_capability(device)),
                "torch_sm_tag": sm_tag,
                "torch_arch_list": arch_list,
                "arch_supported_by_torch": sm_tag in arch_list,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "total_vram_bytes_torch": properties.total_memory,
            }
        )
        result["gpu"] = gpu
        if not gpu["arch_supported_by_torch"]:
            raise RuntimeError(f"{sm_tag} is absent from PyTorch architecture list")

        note("F0-B: checking bitsandbytes CUDA backend with Linear4bit")
        torch.cuda.empty_cache()
        linear = bnb.nn.Linear4bit(256, 256, compute_dtype=torch.bfloat16).cuda()
        sample = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
        linear_output = linear(sample)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(linear_output).all().item()):
            raise RuntimeError("bitsandbytes Linear4bit returned non-finite output")
        del linear, sample, linear_output
        torch.cuda.empty_cache()

        note(f"F0-C: loading {args.model_id} in NF4 on CUDA")
        torch.cuda.reset_peak_memory_stats(device)
        load_started = time.perf_counter()
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            cache_dir=args.cache_dir,
            quantization_config=quantization,
            device_map={"": device},
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        bnb_layers = sum(1 for module in model.modules() if isinstance(module, bnb.nn.Linear4bit))
        if bnb_layers == 0:
            raise RuntimeError("4-bit load completed without any bitsandbytes Linear4bit layers")
        result["test_model"].update(
            {
                "revision": getattr(model.config, "_commit_hash", None),
                "architecture": model.config.model_type,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "load_seconds": load_seconds,
                "bitsandbytes_linear4bit_layers": bnb_layers,
                "model_memory_footprint_bytes": model.get_memory_footprint(),
                "cache_dir": str(args.cache_dir),
            }
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(
            "Sovereign Distillery verifies one reproducible gradient-bearing step.",
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}

        note("F0-D: executing tokenized forward pass on CUDA")
        forward_started = time.perf_counter()
        with torch.no_grad():
            forward_output = model(**encoded)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - forward_started
        logits = forward_output.logits
        forward_finite = bool(torch.isfinite(logits).all().item())
        result["forward_pass"] = {
            "passed": forward_finite,
            "input_tokens": int(encoded["input_ids"].numel()),
            "logits_shape": list(logits.shape),
            "logits_finite": forward_finite,
            "seconds": forward_seconds,
            "device": str(logits.device),
        }
        if not forward_finite or logits.device.type != "cuda":
            raise RuntimeError("forward pass did not return finite CUDA logits")
        del forward_output, logits

        note("F0-E: attaching LoRA and executing loss/backward/optimizer step")
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        trainable_count = sum(parameter.numel() for parameter in trainable)
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        labels = encoded["input_ids"].clone()
        train_started = time.perf_counter()
        training_output = model(**encoded, labels=labels)
        loss = training_output.loss
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"training loss is not finite: {loss_value}")
        loss.backward()
        gradient_tensors = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if not gradient_tensors:
            raise RuntimeError("backward completed without LoRA gradients")
        gradient_norm = float(
            torch.sqrt(sum(torch.sum(gradient.detach().float() ** 2) for gradient in gradient_tensors)).item()
        )
        if not math.isfinite(gradient_norm) or gradient_norm == 0.0:
            raise RuntimeError(f"invalid LoRA gradient norm: {gradient_norm}")
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - train_started
        result["training_step"] = {
            "passed": True,
            "loss": loss_value,
            "gradient_norm": gradient_norm,
            "trainable_parameter_count": trainable_count,
            "optimizer": "torch.optim.AdamW",
            "learning_rate": 1e-4,
            "seconds": training_seconds,
            "device": str(next(model.parameters()).device),
        }
        result["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        result["status"] = "PASS"
        result["limitations"] = [
            "F.0 uses a 135M test model; it proves the kernel and QLoRA path, not feasibility of a 1–3B seed.",
            "The Windows display workload reduces free VRAM and may change between runs.",
            "datasets and trl were intentionally not installed because this gate does not require them.",
        ]
        note("F.0 PASS: 4-bit load, CUDA forward, LoRA backward, and optimizer step succeeded")
        exit_code = 0
    except Exception as error:
        failure_text = f"{type(error).__name__}: {error}"
        stderr_lines.append(traceback.format_exc())
        result["status"] = "FAIL"
        result["failures"] = [
            {
                "category": classify_failure(failure_text),
                "message": failure_text,
            }
        ]
        note(f"F.0 FAIL: {failure_text}")
    finally:
        result["timestamp_utc"] = utc_now()
        write_json(args.out_dir / "result.json", result)
        (args.out_dir / "stdout.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
        (args.out_dir / "stderr.txt").write_text("\n".join(stderr_lines), encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
