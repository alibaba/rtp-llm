"""Symm-mem buffer cache + FP8 shared-expert handling for
``deep_gemm.fp8_fp4_mega_moe_fused``.

This is the fused-kernel analogue of :mod:`.mega_buf` (which serves
``fp8_fp4_mega_moe``) and :mod:`.mega_fp8_buf` (which serves
``fp8_fp8_mega_moe``).

``fp8_fp4_mega_moe_fused`` folds the GLM-5 shared expert into the routed
MegaMoE kernel. It needs its **own** symmetric-memory buffer
(``deep_gemm.get_symm_buffer_for_mega_moe_fused`` → ``SymmBufferFused``),
which is distinct from the routed ``get_symm_buffer_for_mega_moe`` buffer.

The routed experts stay FP4 (per-group, gran_k=32). The **shared expert** is
consumed as **FP8 e4m3 weights with 128×128 per-block UE8M0 scale factors**
(``mega_moe_fused`` no longer supports FP4 shared-expert weights). This module
also owns the FP8 shared-expert weight transform + scratch workspace, mirroring
DeepGEMM's reference flow in ``opt_fused/test_mega_moe_fused.py`` /
``opt_fused/fused_op.py``:

    sf_int = transform_sf_into_required_layout(sf_fp32, N, K, (128, 128))
    (l1_w, l1_sf), (l2_w, l2_sf) = \
        transform_shared_expert_weights_for_mega_moe_fused(
            (w1_fp8, l1_sf_int), (w2_fp8, l2_sf_int))

The transformed SF tensors are pre-arranged for direct SM100 UTCCP 4x32
consumption by the fused SE L1/L2 paths; the FP8 weight tensors keep the SE
kernel's gate/up row order (``w1`` = ``[gate; up]`` stacked along N).
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch

FP8_BLOCK = 128

_MEGA_FUSED_BUF_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Symmetric-memory buffer (SymmBufferFused)
# ---------------------------------------------------------------------------


def estimate_mega_moe_fused_symm_buffer_bytes(
    group_size: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
) -> int | None:
    try:
        import deep_gemm

        return int(
            deep_gemm._C.get_symm_buffer_size_for_mega_moe_fused(
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                use_fp8_dispatch,
                activation,
            )[0]
        )
    except Exception:
        return None


def get_or_create_mega_buf_fused(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
):
    """Get or create the shared symmetric memory buffer for fused Mega MoE."""
    import deep_gemm

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        bool(use_fp8_dispatch),
        activation,
    )
    buf = _MEGA_FUSED_BUF_CACHE.get(key)
    if buf is None:
        try:
            group_size = int(group.size())
        except Exception:
            group_size = 0
        estimated_bytes = (
            estimate_mega_moe_fused_symm_buffer_bytes(
                group_size=group_size,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                use_fp8_dispatch=use_fp8_dispatch,
                activation=activation,
            )
            if group_size > 0
            else None
        )

        buf = deep_gemm.get_symm_buffer_for_mega_moe_fused(
            group=group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            use_fp8_dispatch=use_fp8_dispatch,
            activation=activation,
        )
        actual_bytes = None
        try:
            actual_bytes = int(buf.buffer.numel() * buf.buffer.element_size())
        except Exception:
            pass
        if actual_bytes is not None:
            est_str = (
                f" estimated={estimated_bytes / (1024**3):.3f} GiB"
                if estimated_bytes
                else ""
            )
            logging.info(
                "[MegaMoE Fused] allocated symm buffer: group_size=%d "
                "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
                "intermediate=%d actual=%.3f GiB%s",
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                actual_bytes / (1024**3),
                est_str,
            )
        _MEGA_FUSED_BUF_CACHE[key] = buf
    return buf


def _mega_moe_fused_unavailable_reason() -> str | None:
    """Return None when fused Mega MoE can run."""
    try:
        import deep_gemm

        if not hasattr(deep_gemm, "fp8_fp4_mega_moe_fused"):
            return "deep_gemm.fp8_fp4_mega_moe_fused is missing"
        if not hasattr(deep_gemm, "get_symm_buffer_for_mega_moe_fused"):
            return "deep_gemm.get_symm_buffer_for_mega_moe_fused is missing"
        if not hasattr(deep_gemm, "transform_shared_expert_weights_for_mega_moe_fused"):
            return (
                "deep_gemm.transform_shared_expert_weights_for_mega_moe_fused "
                "is missing"
            )
    except Exception as e:
        return f"failed to import deep_gemm: {e}"
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return "torch.distributed is not initialized"
        if dist.get_world_size() <= 1:
            return f"distributed world_size={dist.get_world_size()} is not > 1"
    except Exception as e:
        return f"failed to query torch.distributed: {e}"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        return f"CUDA device capability sm{cap[0]}{cap[1]} is below SM100"
    return None


def mega_moe_fused_available() -> bool:
    """Whether DeepGEMM's fp8_fp4_mega_moe_fused is usable."""
    return _mega_moe_fused_unavailable_reason() is None


def mega_moe_fused_enabled() -> bool:
    """Default on when fused Mega MoE availability holds."""
    if os.environ.get("GLM5_USE_MEGA_MOE_FUSED", "1") == "0":
        return False
    return mega_moe_fused_available()


# ---------------------------------------------------------------------------
# FP8 shared-expert weight transform + scratch workspace
# ---------------------------------------------------------------------------


def transform_shared_expert_fp8_for_fused(
    w1_w: torch.Tensor,  # [2*inter, dim]  float8_e4m3fn  (gate||up stacked on N)
    w1_s: torch.Tensor,  # [2*inter//128, dim//128]  float32  per-block SF
    w2_w: torch.Tensor,  # [dim, inter]    float8_e4m3fn
    w2_s: torch.Tensor,  # [dim//128, inter//128]  float32  per-block SF
    dim: int,
    inter: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Transform FP8 per-block shared-expert weights for ``fp8_fp4_mega_moe_fused``.

    The checkpoint stores standard DeepSeek-style FP8 per-block weights whose
    ``weight_scale_inv`` is an arbitrary fp32 per-block scale (NOT UE8M0).
    DeepGEMM's fused SE path needs **UE8M0** scales, so — exactly like the
    routed ``fp8_fp8_mega_moe`` path (``mega_moe_fp8.py`` →
    ``requant_weight_ue8m0``) — we dequantize the FP8 blocks and requantize them
    against UE8M0 power-of-two scales. ``requant_weight_ue8m0`` returns the
    requantized FP8 weight plus the SF already in the INT32-packed MN-major
    TMA-aligned layout (equivalent to
    ``transform_sf_into_required_layout(sf, N, K, (128, 128))``), which is what
    ``transform_shared_expert_weights_for_mega_moe_fused`` consumes.

    Returns ``((l1_w, l1_sf), (l2_w, l2_sf))`` ready to pass to
    ``deep_gemm.fp8_fp4_mega_moe_fused``.
    """
    import deep_gemm

    from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0

    n1, k1 = 2 * inter, dim
    n2, k2 = dim, inter

    if w1_w.dtype != torch.float8_e4m3fn or w2_w.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "fused shared expert requires FP8 e4m3 weights, got "
            f"w13={w1_w.dtype}, w2={w2_w.dtype}"
        )
    if tuple(w1_w.shape) != (n1, k1):
        raise ValueError(
            f"shared expert w13 FP8 weight shape mismatch: expected {(n1, k1)}, "
            f"got {tuple(w1_w.shape)}"
        )
    if tuple(w2_w.shape) != (n2, k2):
        raise ValueError(
            f"shared expert w2 FP8 weight shape mismatch: expected {(n2, k2)}, "
            f"got {tuple(w2_w.shape)}"
        )
    exp_s1 = (n1 // FP8_BLOCK, k1 // FP8_BLOCK)
    exp_s2 = (n2 // FP8_BLOCK, k2 // FP8_BLOCK)
    if tuple(w1_s.shape) != exp_s1:
        raise ValueError(
            f"shared expert w13 FP8 scale shape mismatch: expected {exp_s1}, "
            f"got {tuple(w1_s.shape)}"
        )
    if tuple(w2_s.shape) != exp_s2:
        raise ValueError(
            f"shared expert w2 FP8 scale shape mismatch: expected {exp_s2}, "
            f"got {tuple(w2_s.shape)}"
        )

    # Dequantize the serialized (non-UE8M0) FP8 blocks and requantize to UE8M0.
    # The returned SF is already in the INT32-packed layout the fused SE
    # transform expects.
    w1_w_req, w1_sf_int = requant_weight_ue8m0(w1_w.contiguous(), w1_s.float())
    w2_w_req, w2_sf_int = requant_weight_ue8m0(w2_w.contiguous(), w2_s.float())

    (l1_w, l1_sf), (l2_w, l2_sf) = (
        deep_gemm.transform_shared_expert_weights_for_mega_moe_fused(
            (w1_w_req, w1_sf_int),
            (w2_w_req, w2_sf_int),
        )
    )
    return (l1_w, l1_sf), (l2_w, l2_sf)


def make_shared_mid_workspace(
    capacity: int,
    inter: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the SE L1→L2 intermediate FP8 buffer + its UE8M0 SF scratch.

    Identical layout to DeepGEMM's reference (test_mega_moe_fused.py):
      - ``mid_fp8``: ``[capacity, inter]`` float8_e4m3fn
      - ``mid_sf`` : ``[inter//128, T_pad]`` int32, transposed to a
        column-major (MN-major TMA-aligned) view; ``T_pad`` covers both the
        TMA-4 alignment and the 256-token UTCCP tile.
    """
    import deep_gemm

    mid_fp8 = torch.empty(
        (capacity, inter),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    t_pad = max(
        deep_gemm.get_tma_aligned_size(capacity, 4),
        ((capacity + 255) // 256) * 256,
    )
    mid_sf = torch.empty(
        (inter // FP8_BLOCK, t_pad),
        dtype=torch.int32,
        device=device,
    ).T
    return mid_fp8, mid_sf
