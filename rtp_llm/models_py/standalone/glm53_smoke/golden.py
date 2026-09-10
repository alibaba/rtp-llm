"""Immutable per-rank golden files and explicit numeric regression metrics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def weight_manifest(weights):
    rows = {}
    groups = [("global", weights.global_weights)] + [
        (f"layer{i}", group) for i, group in enumerate(weights.weights)
    ]
    for prefix, group in groups:
        for key, value in sorted(group.items()):
            if isinstance(value, torch.Tensor):
                rows[f"{prefix}/{key}"] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": tensor_sha256(value),
                }
    return rows


def layer_logits(model, samples, positions):
    from rtp_llm.utils.model_weight import W

    head = model.weight.get_global_weight(W.lm_head)
    if model.parallelism_config.tp_size > 1:
        full = torch.empty(
            (head.shape[0] * 8, head.shape[1]), dtype=head.dtype, device=head.device
        )
        dist.all_gather_into_tensor(full, head.contiguous())
        head = full
    result = {}
    for index, hidden in samples.items():
        selected = hidden[positions.cpu()].cuda()
        normalized = model.norm(selected.mean(dim=-2))
        # A diagnostic readout at each layer, using the same final norm and
        # full vocabulary head. Its cost is excluded from the four-layer time.
        result[f"layer{index}.hidden"] = hidden
        result[f"layer{index}.logits"] = (
            F.linear(normalized.float(), head.float()).detach().cpu()
        )
    return result


def compare(actual, expected, relative_l2_limit=0.0, max_abs_limit=0.0):
    if actual.keys() != expected.keys():
        raise ValueError("golden tensor keys changed")
    metrics = {}
    for key in actual:
        x, ref = actual[key], expected[key]
        if x.shape != ref.shape or x.dtype != ref.dtype:
            raise ValueError(f"golden shape/dtype changed for {key}")
        finite, exact = True, True
        squared_error = squared_reference = maximum = 0.0
        # Prefill preserves billions of elements per layer. Keep comparison
        # scratch bounded instead of materializing several full FP32 copies.
        flat_x, flat_ref = x.reshape(-1), ref.reshape(-1)
        for start in range(0, flat_x.numel(), 4 * 1024 * 1024):
            a = flat_x[start : start + 4 * 1024 * 1024]
            b = flat_ref[start : start + 4 * 1024 * 1024]
            finite = finite and bool(
                torch.isfinite(a).all() and torch.isfinite(b).all()
            )
            equal = torch.equal(a, b)
            exact = exact and equal
            if not equal:
                delta = a.float() - b.float()
                squared_error += float(delta.double().square().sum())
                maximum = max(maximum, float(delta.abs().max()))
        if not exact:
            for start in range(0, flat_ref.numel(), 4 * 1024 * 1024):
                squared_reference += float(
                    flat_ref[start : start + 4 * 1024 * 1024].double().square().sum()
                )
        relative = math.sqrt(squared_error) / max(math.sqrt(squared_reference), 1e-30)
        metrics[key] = dict(
            finite=finite,
            exact=exact,
            relative_l2=relative,
            max_abs=maximum,
            passed=finite
            and relative <= relative_l2_limit
            and maximum <= max_abs_limit,
        )
    return metrics


def save_exclusive(path, tensors, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Existing golden files are never silently replaced by a regression run.
    with path.open("xb") as stream:
        torch.save({"metadata": metadata, "tensors": tensors}, stream)
    digest = file_sha256(path)
    path.with_suffix(".json").write_text(
        json.dumps(dict(metadata=metadata, sha256=digest), indent=2) + "\n"
    )
