"""Independent sparse-attention fixtures and FP64 numerical checks."""

import hashlib
import json
import os
from pathlib import Path

import torch

ATOL, RTOL = 0.0078125, 0.008


def save(name, value):
    root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if root:
        (Path(root) / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


def metadata():
    import triton

    from rtp_llm.models_py.modules.qwen4_exp.qsa import sparse_fmha, sparse_prefill_sm90

    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": prop.name,
        "uuid": str(getattr(prop, "uuid", "unknown")),
        "capability": list(torch.cuda.get_device_capability()),
        "source_sha256": {
            Path(module.__file__)
            .name: hashlib.sha256(Path(module.__file__).read_bytes())
            .hexdigest()
            for module in (sparse_fmha, sparse_prefill_sm90)
        },
        "scope": "synthetic BF16 sparse-attention component",
        "atol": ATOL,
        "rtol": RTOL,
    }


def fixture(
    length, *, batch=1, kv_length=None, strided=False, seed=20261029, amplitude=1.0
):
    total = length if kv_length is None else kv_length
    g = torch.Generator(device="cuda").manual_seed(seed + length)

    def tensor(heads, rows):
        value = (
            torch.randn(batch, rows, heads, 256, device="cuda", generator=g) * amplitude
        ).bfloat16()
        if strided:
            padded = torch.empty(
                batch, rows * 2, heads, 256, device="cuda", dtype=torch.bfloat16
            )
            padded[:, ::2].copy_(value)
            return padded[:, ::2].permute(0, 2, 1, 3)
        return value.permute(0, 2, 1, 3).contiguous()

    q, k, v = tensor(12, length), tensor(1, total), tensor(1, total)
    selected = torch.full((batch, length, 2051), -1, dtype=torch.int32)
    rng = torch.Generator().manual_seed(seed + length)
    for row in range(length):
        visible = min(total, row + 1)
        groups = torch.randperm(visible // 4, generator=rng)[:512]
        ids = (groups[:, None] * 4 + torch.arange(4)[None, :]).flatten()
        selected[:, row, : ids.numel()] = ids
        tail = torch.arange(visible // 4 * 4, visible)
        selected[:, row, 2048 : 2048 + tail.numel()] = tail
    selected = selected.cuda()
    if strided:
        padded = torch.empty(batch, length * 2, 2051, device="cuda", dtype=torch.int32)
        padded[:, ::2].copy_(selected)
        selected = padded[:, ::2]
    return q, k, v, selected


def fp64_reference(q, k, v, selected, rows=None):
    rows = list(range(q.shape[2])) if rows is None else list(rows)
    outputs = []
    for batch in range(q.shape[0]):
        pieces = []
        for start in range(0, len(rows), 4):
            rr = torch.tensor(rows[start : start + 4], device=q.device)
            ids = selected[batch, rr].long()
            valid = (ids >= 0) & (ids < k.shape[2])
            safe = ids.clamp(0, k.shape[2] - 1)
            qq = q[batch, :, rr].permute(1, 0, 2).double()
            kk, vv = k[batch, 0, safe].double(), v[batch, 0, safe].double()
            score = torch.einsum("rhd,rkd->rhk", qq, kk) / 16.0
            score.masked_fill_(~valid[:, None], -float("inf"))
            probability = torch.softmax(score, -1)
            probability = torch.where(valid.any(-1)[:, None, None], probability, 0.0)
            pieces.append(torch.einsum("rhk,rkd->rhd", probability, vv))
        outputs.append(torch.cat(pieces).permute(1, 0, 2))
    return torch.stack(outputs)


def errors(actual, reference, atol=ATOL, rtol=RTOL):
    if actual.shape != reference.shape:
        return {
            "pass": False,
            "shape": list(actual.shape),
            "reference_shape": list(reference.shape),
        }
    delta = (actual.double() - reference.double()).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    bad = delta > atol + rtol * reference.double().abs()
    result = {
        "pass": finite and not bool(bad.any()),
        "finite": finite,
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "violations": int(bad.sum()),
        "elements": actual.numel(),
        "atol": atol,
        "rtol": rtol,
    }
    if bool(bad.any()):
        index = tuple(bad.nonzero()[0].tolist())
        result.update(
            first_index=list(index),
            actual=float(actual[index]),
            reference=float(reference[index]),
        )
    return result
