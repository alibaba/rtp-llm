"""Symmetric memory buffer cache for GLM-5 MegaMoE.

Ported from dsv4/moe/mega_buf.py. The mega kernel needs a PyTorch
symmetric_memory buffer for cross-rank NVLink dispatch + combine.
One buffer is shared across all MoE layers (they execute sequentially).
"""

import logging
import os

import torch

_MEGA_BUF_CACHE: dict = {}
_MEGA_OUTPUT_CACHE: dict = {}


def estimate_mega_moe_symm_buffer_bytes(
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
            deep_gemm._C.get_symm_buffer_size_for_mega_moe(
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


def get_or_create_mega_buf(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
):
    """Get or create the shared symmetric memory buffer for mega MoE."""
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
    buf = _MEGA_BUF_CACHE.get(key)
    if buf is None:
        try:
            group_size = int(group.size())
        except Exception:
            group_size = 0
        estimated_bytes = (
            estimate_mega_moe_symm_buffer_bytes(
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

        buf = deep_gemm.get_symm_buffer_for_mega_moe(
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
                "[GLM5 MegaMoE] allocated symm buffer: group_size=%d "
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
        _MEGA_BUF_CACHE[key] = buf
    return buf


def get_or_create_mega_output(
    capacity: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Get or create the process-local output buffer for mega MoE."""
    key = (device, hidden, dtype)
    cached = _MEGA_OUTPUT_CACHE.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    cached = torch.empty((max(capacity, 1), hidden), dtype=dtype, device=device)
    _MEGA_OUTPUT_CACHE[key] = cached
    return cached


def _mega_moe_unavailable_reason() -> str | None:
    """Return None when Mega MoE can run, otherwise a human-readable reason."""
    try:
        import deep_gemm

        if not hasattr(deep_gemm, "fp8_fp4_mega_moe"):
            return "deep_gemm.fp8_fp4_mega_moe is missing"
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


def mega_moe_available() -> bool:
    """Whether DeepGEMM's fp8_fp4_mega_moe is usable.

    Requires: deep_gemm >= 2.5, torch >= 2.9 (symmetric_memory),
    SM100+, and an initialised process group of size > 1.
    """
    return _mega_moe_unavailable_reason() is None


def mega_moe_enabled() -> bool:
    """Default on when mega_moe_available() holds.

    GLM5_USE_MEGA_MOE=0 disables explicitly.
    """
    if os.environ.get("GLM5_USE_MEGA_MOE", "1") == "0":
        return False
    return mega_moe_available()
