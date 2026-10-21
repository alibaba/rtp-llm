"""Symm-mem buffer cache for DeepGEMM FP8xFP8 Mega MoE."""

import logging

_MEGA_FP8_BUF_CACHE: dict = {}


def estimate_mega_moe_fp8_symm_buffer_bytes(
    group_size: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
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
                num_shared_experts,
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
    num_shared_experts: int = 0,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
):
    """Get or create the shared symmetric memory buffer for FP8xFP8 Mega MoE."""
    import deep_gemm

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts,
        bool(use_fp8_dispatch),
        activation,
    )
    buf = _MEGA_FP8_BUF_CACHE.get(key)
    if buf is None:
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
                num_shared_experts=num_shared_experts,
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
            num_shared_experts=num_shared_experts,
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
                "intermediate=%d shared_experts=%d actual=%.3f GiB%s",
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                num_shared_experts,
                actual_bytes / (1024**3),
                est_str,
            )
        _MEGA_FP8_BUF_CACHE[key] = buf
    return buf
