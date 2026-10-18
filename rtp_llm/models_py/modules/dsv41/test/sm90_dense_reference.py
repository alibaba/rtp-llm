"""Independent reference helpers for the production SM90 dense contract."""

import copy
import hashlib

import torch

SHAPES = {
    "merged_qkv": (1792, 5120),
    "tp8_q_b": (4096, 1280),
    "cp8_q_b": (32768, 1280),
    "o_a_group": (1024, 4096),
    "tp8_o_b": (5120, 1024),
    "cp8_o_b": (5120, 8192),
    "shared_w13": (4608, 5120),
    "shared_w2": (5120, 2304),
}


def _independent_scales(packed, groups):
    shifts = torch.arange(4, dtype=torch.int64, device=packed.device) * 8
    exponents = ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 255).reshape(
        packed.shape[0], -1
    )[:, :groups]
    scales = torch.exp2(exponents.float() - 127)
    return torch.where(exponents == 255, float("nan"), scales)


def _oracle(q, packed, weight, block_scales):
    """Independent extraction and 32x32 expansion, FP32 GEMM, one BF16 cast."""
    rows, k = q.numel() // q.shape[-1], q.shape[-1]
    scales = _independent_scales(packed, k // 32)
    a = (q.reshape(rows, k // 32, 32).float() * scales.unsqueeze(-1)).reshape(rows, k)
    bs = block_scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    b = weight.float() * bs[: weight.shape[0], :k]
    return (a @ b.T).bfloat16()


def _metrics(actual, expected):
    a, b = actual.reshape_as(expected).float(), expected.float()
    diff = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    return dict(
        elements=a.numel(),
        unequal=int((a != b).sum()),
        bitwise_unequal=int(
            (
                actual.reshape_as(expected).contiguous().view(torch.int16)
                != expected.contiguous().view(torch.int16)
            ).sum()
        ),
        tolerance_violations=int(((~finite) | (diff > 0.02 + 0.015 * b.abs())).sum()),
        max_abs=float(diff.max()),
        rms_abs=float(diff.square().mean().sqrt()),
        finite=bool(finite.all()),
    )


def _hash(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def _selected_default(adapter, rows):
    from humming.kernel.humming import HummingKernel

    _, _, intervals = HummingKernel._resolve_configs(
        adapter.config.to_str(), adapter.compute_config, None, adapter.weight.device
    )
    assert isinstance(
        intervals, list
    ), "Record actual default interval dispatch, not get_config(M)"
    selected = [entry for entry in intervals if entry[0] < rows <= entry[1]]
    assert len(selected) == 1
    low, high, config = selected[0]
    return (low, high), copy.deepcopy(config)
