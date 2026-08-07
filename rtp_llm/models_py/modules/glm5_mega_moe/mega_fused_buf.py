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
owns the FP8 shared-expert weight transform + scratch workspace:

* legacy float32 scales → ``requant_weight_ue8m0`` then fused layout transform
* native UE8M0 / packed int32 → keep weight bits, pack SF only, then fused layout

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
    w1_s: torch.Tensor,  # float32 block SF, e8m0 block SF, or packed int32
    w2_w: torch.Tensor,  # [dim, inter]    float8_e4m3fn
    w2_s: torch.Tensor,  # float32 block SF, e8m0 block SF, or packed int32
    dim: int,
    inter: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Transform FP8 per-block shared-expert weights for ``fp8_fp4_mega_moe_fused``.

    Scale handling mirrors ``mega_moe_fp8_se``:

    * **float32** (legacy DeepSeek ``weight_scale_inv``): dequant → requant to
      UE8M0 via ``requant_weight_ue8m0`` (may rewrite FP8 weight bits).
    * **native UE8M0** (``float8_e8m0fnu`` or loader-packed ``int32``): keep
      FP8 weight bits byte-exact; only ensure the DeepGEMM TMA-packed int32
      scale layout (no requant).

    Returns ``((l1_w, l1_sf), (l2_w, l2_sf))`` ready for
    ``deep_gemm.fp8_fp4_mega_moe_fused``.
    """
    import deep_gemm

    from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0

    from .quant_layouts import prepare_fp8_weight_scale_for_deepgemm

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

    if w1_s.dtype == torch.float32 or w2_s.dtype == torch.float32:
        if w1_s.dtype != torch.float32 or w2_s.dtype != torch.float32:
            raise TypeError(
                "fused shared expert requires both scales to be raw float32 "
                f"or both native UE8M0/packed int32, got {w1_s.dtype} and "
                f"{w2_s.dtype}"
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
        w1_w_req, w1_sf_int = requant_weight_ue8m0(w1_w.contiguous(), w1_s)
        w2_w_req, w2_sf_int = requant_weight_ue8m0(w2_w.contiguous(), w2_s)
    else:
        # Already UE8M0 (raw e8m0 or loader-packed int32): no weight requant.
        w1_w_req = w1_w.contiguous()
        w2_w_req = w2_w.contiguous()
        w1_sf_int = prepare_fp8_weight_scale_for_deepgemm(w1_s, n1, k1)
        w2_sf_int = prepare_fp8_weight_scale_for_deepgemm(w2_s, n2, k2)

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
