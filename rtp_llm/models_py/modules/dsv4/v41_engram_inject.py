# SPDX-License-Identifier: Apache-2.0
# Adapted from RTP-LLM engram.py and the checkpoint equations.
"""Engram's per-lane normalized gate, after group-32 WKV projection."""

import math

import torch


def gated_engram_residual(
    hidden: torch.Tensor,
    projected: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    eps: float,
    backend: str = "triton",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inject one shared value through a separate gate for each residual lane.

    WKV is laid out as [all lane keys, one shared value]. Dead rows pass through
    bit-for-bit, even if unused projections are nonfinite. Only an exact hidden
    alias is permitted for inference output; other shared storage is rejected.
    The eager backend is an explicit equation reference, not an automatic fallback.
    """
    if (
        hidden.ndim != 3
        or min(hidden.shape[1:]) <= 0
        or hidden.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not hidden.is_contiguous()
    ):
        raise ValueError(
            "Engram residual requires contiguous floating [tokens, lanes, dim]"
        )
    tokens, lanes, dim = hidden.shape
    if (
        projected.shape != (tokens, (lanes + 1) * dim)
        or q_weight.shape != (lanes, dim)
        or k_weight.shape != (lanes, dim)
        or any(
            t.dtype != hidden.dtype
            or t.device != hidden.device
            or not t.is_contiguous()
            for t in (projected, q_weight, k_weight)
        )
    ):
        raise ValueError(
            "Engram projection and gate weights disagree with residual shape/dtype/device"
        )
    if (
        valid_rows.shape != (tokens,)
        or valid_rows.dtype != torch.bool
        or valid_rows.device != hidden.device
        or not valid_rows.is_contiguous()
    ):
        raise ValueError(
            "Engram injection requires a contiguous device bool [tokens] mask"
        )
    if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Engram norm epsilon must be finite and positive")
    if backend not in ("triton", "eager"):
        raise ValueError("Unknown Engram injection backend")
    if backend == "triton" and (not hidden.is_cuda or (lanes, dim) != (4, 5120)):
        raise ValueError("Engram Triton injection requires CUDA [tokens, 4, 5120]")
    inputs = (hidden, projected, q_weight, k_weight)
    if (
        (backend == "triton" or out is not None)
        and torch.is_grad_enabled()
        and any(t.requires_grad for t in inputs)
    ):
        raise ValueError("Engram Triton/output-buffer injection is inference-only")
    if out is not None:
        if (
            out.shape != hidden.shape
            or out.dtype != hidden.dtype
            or out.device != hidden.device
            or not out.is_contiguous()
            or out.requires_grad
        ):
            raise ValueError(
                "Engram output must match the residual and be contiguous without gradients"
            )
        if out.numel():
            storage = out.untyped_storage().data_ptr()
            if any(
                storage == t.untyped_storage().data_ptr()
                for t in (projected, q_weight, k_weight, valid_rows)
            ):
                raise ValueError(
                    "Engram output must not share storage with projection, weights or mask"
                )
            if (
                storage == hidden.untyped_storage().data_ptr()
                and out.data_ptr() != hidden.data_ptr()
            ):
                raise ValueError("Engram output alias with hidden must be exact")
    if not tokens:
        return torch.empty_like(hidden) if out is None else out
    if backend == "eager":
        h = hidden.float()
        key = projected[:, : lanes * dim].reshape(tokens, lanes, dim).float()
        value = projected[:, lanes * dim :].float()
        weight = q_weight.float() * k_weight.float()
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
            key.square().mean(-1) + eps
        )
        dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        result = (h + gate[..., None] * value[:, None, :]).to(hidden.dtype)
        result = torch.where(valid_rows[:, None, None], result, hidden)
        return result if out is None else out.copy_(result)
    from rtp_llm.models_py.modules.dsv4._v41_engram_inject_triton import (
        engram_inject_kernel,
    )

    if out is None:
        out = torch.empty_like(hidden)
    engram_inject_kernel[(tokens, lanes)](
        hidden,
        projected,
        q_weight,
        k_weight,
        valid_rows,
        out,
        DIM=dim,
        HC=lanes,
        EPS=eps,
        BLOCK=8192,
        enable_fp_fusion=False,
    )
    return out
