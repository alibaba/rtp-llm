"""Compare matching-prefix V4.1 reference/native diagnostic snapshots on CPU."""

import argparse
import json
import re
from pathlib import Path

import torch


def trace_identity(metadata):
    ids = metadata["input_ids"]
    if len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Trace requires one nonempty sequence of integer tokens")
    if "start_pos" in metadata:
        start = metadata["start_pos"]
    else:
        spans = metadata["cache_spans"]
        if len(spans) != 1 or len(spans[0]) != 6 or spans[0][5] != len(ids):
            raise ValueError("Native trace requires one matching cache span")
        start = spans[0][4]
    if type(start) is not int or start < 0:
        raise ValueError("Trace requires a nonnegative integer start position")
    return {"input_ids": ids, "start_pos": start}


def matrix(value):
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or not value.numel() or not value.is_floating_point():
        raise ValueError("Trace tensor must be a nonempty floating-point matrix")
    if value.device.type != "cpu" or not torch.isfinite(value).all():
        raise ValueError("Trace tensor must contain finite CPU values")
    return value.contiguous()


def tensor_difference(reference, candidate):
    reference, candidate = matrix(reference), matrix(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("Trace tensor shapes disagree")
    left, right = reference.double().flatten(), candidate.double().flatten()
    error = right - left
    left_norm, right_norm = left.norm().item(), right.norm().item()
    error_norm = error.norm().item()
    identical = reference.dtype == candidate.dtype and torch.equal(
        reference.view(torch.uint8), candidate.view(torch.uint8)
    )
    return {
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "bitwise_equal": identical,
        "different_values": int((left != right).sum()),
        "max_abs": error.abs().max().item(),
        "mean_abs": error.abs().mean().item(),
        "rms_abs": error.square().mean().sqrt().item(),
        "relative_l2": error_norm / left_norm if left_norm else None,
        "cosine": (
            float(torch.dot(left, right)) / (left_norm * right_norm)
            if left_norm and right_norm
            else None
        ),
    }


def boundaries(tensors):
    names = sorted(name for name in tensors if name.startswith("layer"))
    pairs = [re.fullmatch(r"layer(\d+)\.ffn\.(input|output)", name) for name in names]
    if not pairs or any(match is None for match in pairs):
        raise ValueError("Trace has missing or unrecognized layer boundaries")
    layers = {int(match[1]) for match in pairs}
    expected = [
        f"layer{layer:02d}.ffn.{direction}"
        for layer in range(len(layers))
        for direction in ("input", "output")
    ]
    if names != expected or "final_norm" not in tensors:
        raise ValueError("Trace must include every layer input/output and final norm")
    return [*names, "final_norm"]


def top_logits(logits):
    logits = matrix(logits)
    if logits.shape[0] != 1:
        raise ValueError("Trace logits must describe one generation step")
    values, ids = logits[0].float().topk(min(10, logits.shape[1]))
    return {"ids": ids.tolist(), "values": values.tolist()}


def compare_traces(reference, candidate):
    identity = trace_identity(reference["metadata"])
    if identity != trace_identity(candidate["metadata"]):
        raise ValueError("Trace input tokens or start positions disagree")
    left, right = reference["tensors"], candidate["tensors"]
    names = boundaries(left)
    if names != boundaries(right):
        raise ValueError("Trace layer boundaries disagree")
    for name in names:
        if matrix(left[name]).shape[0] != len(identity["input_ids"]):
            raise ValueError("Trace boundary rows disagree with input tokens")
    differences = {name: tensor_difference(left[name], right[name]) for name in names}
    logits = tensor_difference(left["logits_fp32"], right["logits_fp32"])
    reference_top = top_logits(left["logits_fp32"])
    candidate_top = top_logits(right["logits_fp32"])
    result = {
        "identity": identity,
        "first_bitwise_difference": next(
            (name for name in names if not differences[name]["bitwise_equal"]), None
        ),
        "boundaries": differences,
        "logits_fp32": logits,
        "reference_top10": reference_top,
        "candidate_top10": candidate_top,
        "greedy_equal": reference_top["ids"][0] == candidate_top["ids"][0],
    }
    # A local head shard cannot establish global greedy agreement. Report its
    # range and compare only values for vocabulary IDs contained in that shard.
    head_comparisons = {}
    for label, tensors, head_name in (
        ("reference", left, "local_logits_bf16_head"),
        ("candidate", right, "logits_bf16_head"),
    ):
        if head_name not in tensors:
            continue
        head = matrix(tensors[head_name])
        count = head.shape[1]
        fp32 = matrix(tensors["logits_fp32"])[:, :count]
        watched = sorted(
            token
            for token in set(reference_top["ids"] + candidate_top["ids"])
            if token < count
        )
        head_comparisons[label] = {
            "vocabulary_range": [0, count],
            "difference": tensor_difference(fp32, head),
            "top10": top_logits(head),
            "watched_ids": watched,
            "fp32": fp32[0, watched].tolist(),
            "bf16": head[0, watched].tolist(),
        }
    result["head_precision"] = head_comparisons
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    traces = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in (args.reference, args.candidate)
    ]
    result = compare_traces(*traces)
    result["files"] = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
    }
    with args.report.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "identity",
                    "first_bitwise_difference",
                    "greedy_equal",
                    "logits_fp32",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
