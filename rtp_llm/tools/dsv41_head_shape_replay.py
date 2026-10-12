"""Measure full-vocabulary and TP8 head GEMMs on identical saved final norms."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from rtp_llm.tools.dsv41_trace_compare import matrix, tensor_difference, trace_identity
from rtp_llm.utils.safetensors_header import safetensors_file_identity
from safetensors import safe_open


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tensor_digest(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def require_bytes(left, right, label):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(f"{label}: shape or dtype changed")
    if not torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
    ):
        raise ValueError(f"{label}: bytes changed")


def matmul_settings():
    return {
        "float32_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }


def load_cases(native_dir, reference_dir, profile):
    cases = []
    for prompt in range(2):
        start = 0
        for step in range(4):
            native_path = native_dir / f"native-prompt{prompt}-step{step}.pt"
            reference_path = reference_dir / f"reference-prompt{prompt}-step{step}.pt"
            native = torch.load(native_path, map_location="cpu", weights_only=True)
            reference = torch.load(
                reference_path, map_location="cpu", weights_only=True
            )
            identity = trace_identity(native["metadata"])
            if (
                identity != trace_identity(reference["metadata"])
                or identity["start_pos"] != start
                or native["metadata"]["precision_profile"] != profile
            ):
                raise ValueError("Head replay requires the admitted composed history")
            before, after = native["tensors"], reference["tensors"]
            hidden = matrix(before["final_norm"])
            require_bytes(hidden, matrix(after["final_norm"]), "final norm")
            require_bytes(hidden, matrix(before["hidden"]), "native hidden")
            if hidden.shape != (len(identity["input_ids"]), 5120):
                raise ValueError("Head replay requires complete V4.1 final norms")
            if hidden.dtype != torch.bfloat16:
                raise ValueError("Head replay requires the saved BF16 boundary")
            start += len(identity["input_ids"])
            cases.append(
                {
                    "prompt_index": prompt,
                    "step": step,
                    "identity": identity,
                    "native_path": str(native_path),
                    "reference_path": str(reference_path),
                    "hidden": hidden[-1:].clone(),
                    "saved_native_fp32": matrix(before["logits_fp32"]),
                    "saved_reference_fp32": matrix(after["logits_fp32"]),
                    "saved_native_bf16": matrix(before["logits_bf16_head"]),
                    "saved_reference_bf16_local": matrix(
                        after["local_logits_bf16_head"]
                    ),
                }
            )
    return cases


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("native_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Head replay requires fresh artifacts")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Head replay requires one leased SM120 GPU")
    torch.set_num_threads(1)
    root = cli.checkpoint.resolve()
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    head_path = root / index["head.weight"]
    file_identity = safetensors_file_identity(head_path.stat())
    with safe_open(head_path, framework="pt", device="cpu") as handle:
        head_cpu = handle.get_tensor("head.weight")
    if head_cpu.shape != (129280, 5120) or head_cpu.dtype != torch.bfloat16:
        raise ValueError("Unexpected V4.1 output head")
    weight_hash = tensor_digest(head_cpu)
    settings = matmul_settings()
    if settings["float32_precision"] != "highest" or settings["allow_tf32"]:
        raise ValueError("Held FP32 reproduction requires unchanged IEEE matmul")
    cases = load_cases(cli.native_dir, cli.reference_dir, cli.profile)
    input_hashes = {
        row["native_path"]: digest(Path(row["native_path"])) for row in cases
    }
    input_hashes.update(
        {row["reference_path"]: digest(Path(row["reference_path"])) for row in cases}
    )
    input_hashes[str(index_path)] = digest(index_path)
    head_bf16 = head_cpu.cuda()
    head_fp32 = head_bf16.float()
    # Separate allocations match the released rank-owned head geometry.
    shards_fp32 = [part.clone() for part in head_fp32.chunk(8, dim=0)]
    records, raw = [], []
    for case in cases:
        hidden = case["hidden"].cuda()
        first = torch.nn.functional.linear(hidden.float(), head_fp32)
        sharded = torch.cat(
            [
                torch.nn.functional.linear(hidden.float(), weight)
                for weight in shards_fp32
            ],
            dim=-1,
        )
        restored = torch.nn.functional.linear(hidden.float(), head_fp32)
        bf16 = torch.nn.functional.linear(hidden, head_bf16).float()
        bf16_sharded = torch.cat(
            [
                torch.nn.functional.linear(hidden, weight.bfloat16()).float()
                for weight in shards_fp32
            ],
            dim=-1,
        )
        require_bytes(hidden.cpu(), case["hidden"], "held input")
        row = dict(case)
        row.update(
            {
                "full_fp32": first.cpu(),
                "sharded_fp32": sharded.cpu(),
                "restored_fp32": restored.cpu(),
                "full_bf16": bf16.cpu(),
                "sharded_bf16": bf16_sharded.cpu(),
            }
        )
        if any(
            not torch.isfinite(value).all()
            for value in row.values()
            if isinstance(value, torch.Tensor)
        ):
            raise ValueError("Nonfinite held head output")
        comparisons = {
            "native_fp32_reproduction": ("saved_native_fp32", "full_fp32"),
            "reference_fp32_reproduction": ("saved_reference_fp32", "sharded_fp32"),
            "native_bf16_reproduction": ("saved_native_bf16", "full_bf16"),
            "full_restoration": ("full_fp32", "restored_fp32"),
            "fp32_shape_difference": ("full_fp32", "sharded_fp32"),
            "bf16_shape_difference": ("full_bf16", "sharded_bf16"),
        }
        metrics = {
            label: tensor_difference(row[left], row[right])
            for label, (left, right) in comparisons.items()
        }
        metrics["reference_local_bf16_reproduction"] = tensor_difference(
            row["saved_reference_bf16_local"], row["sharded_bf16"][:, :16160]
        )
        records.append(
            {
                "prompt_index": case["prompt_index"],
                "step": case["step"],
                "identity": case["identity"],
                "input_sha256": tensor_digest(case["hidden"]),
                "metrics": metrics,
            }
        )
        raw.append(row)
    require_bytes(head_bf16.cpu(), head_cpu, "NAS BF16 head")
    require_bytes(head_fp32.cpu(), head_cpu.float(), "FP32 head")
    for rank, weight in enumerate(shards_fp32):
        require_bytes(
            weight.cpu(),
            head_cpu[rank * 16160 : (rank + 1) * 16160].float(),
            f"owned shard {rank}",
        )
    if (
        matmul_settings() != settings
        or safetensors_file_identity(head_path.stat()) != file_identity
        or any(digest(Path(name)) != value for name, value in input_hashes.items())
    ):
        raise ValueError("Admitted head replay inputs or settings changed")
    with cli.tensor_artifact.open("xb") as stream:
        torch.save({"cases": raw, "profile": cli.profile}, stream)
    controls = [
        metric["bitwise_equal"]
        for row in records
        for label, metric in row["metrics"].items()
        if label.endswith("_reproduction") or label == "full_restoration"
    ]
    report = {
        "records": records,
        "case_count": len(records),
        "all_saved_heads_reproduced": all(controls),
        "immutable_head_shards": 8,
        "immutable_inputs": 8,
        "profile": cli.profile,
        "weight_name": "head.weight",
        "weight_shape": list(head_cpu.shape),
        "weight_dtype": str(head_cpu.dtype),
        "weight_sha256": weight_hash,
        "weight_file": str(head_path),
        "weight_file_identity": file_identity,
        "matmul_settings": settings,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "peak_torch_bytes": torch.cuda.max_memory_allocated(),
        "input_sha256": input_hashes,
        "script_sha256": digest(Path(__file__)),
        "tensor_artifact_sha256": digest(cli.tensor_artifact),
        "numerical_promotion": False,
    }
    with cli.report.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "case_count",
                    "all_saved_heads_reproduced",
                    "peak_torch_bytes",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
