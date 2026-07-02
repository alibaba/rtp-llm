"""Symmetric-memory buffer cache for DeepGEMM FP8xFP8 MegaMoE."""

import logging
import os

import torch

_MEGA_FP8_BUF_CACHE: dict = {}


def estimate_mega_moe_fp8_symm_buffer_bytes(
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
            deep_gemm._C.get_symm_buffer_size_for_mega_moe_fp8(
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


def get_or_create_mega_buf_fp8(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
):
    """Get or create the shared symmetric-memory buffer for FP8xFP8 MegaMoE."""
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
    buf = _MEGA_FP8_BUF_CACHE.get(key)
    if buf is not None:
        return buf

    try:
        group_size = int(group.size())
    except Exception:
        group_size = 0
    estimated_bytes = (
        estimate_mega_moe_fp8_symm_buffer_bytes(
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

    buf = deep_gemm.get_symm_buffer_for_mega_moe_fp8(
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
            "[MegaMoE FP8] allocated symm buffer: group_size=%d "
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
    _MEGA_FP8_BUF_CACHE[key] = buf
    return buf


def _mega_moe_fp8_unavailable_reason() -> str | None:
    """Return None when FP8xFP8 MegaMoE can run."""
    try:
        import deep_gemm

        if not hasattr(deep_gemm, "fp8_fp8_mega_moe"):
            return "deep_gemm.fp8_fp8_mega_moe is missing"
        if not hasattr(deep_gemm, "get_symm_buffer_for_mega_moe_fp8"):
            return "deep_gemm.get_symm_buffer_for_mega_moe_fp8 is missing"
        if not hasattr(deep_gemm, "transform_weights_for_mega_moe_fp8"):
            return "deep_gemm.transform_weights_for_mega_moe_fp8 is missing"
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


def mega_moe_fp8_available() -> bool:
    """Whether DeepGEMM's fp8_fp8_mega_moe is usable."""
    return _mega_moe_fp8_unavailable_reason() is None


def mega_moe_fp8_enabled() -> bool:
    """Default on when FP8xFP8 MegaMoE availability holds."""
    if os.environ.get("GLM5_USE_MEGA_MOE_FP8", "1") == "0":
        return False
    return mega_moe_fp8_available()
