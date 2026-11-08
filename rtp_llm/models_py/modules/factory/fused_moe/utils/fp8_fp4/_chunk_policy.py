"""Pure MoE chunk-policy decision, kept dependency-free on purpose.

This module must depend on nothing but ``torch``: tests exec it directly by
file path so they do not have to import the ``fused_moe`` package.
"""

import torch


def should_chunk_moe_input(
    layer, tokens: int, *, is_decode_forward: bool = False
) -> bool:
    """Whether the MoE forward must chunk ``tokens`` over multiple rounds.

    ``layer`` only needs the attributes read here (``max_tokens_per_rank``,
    ``_is_decode_role``, ``chunking_enabled`` and optionally ``_moe``), so a
    plain namespace works as well as the real layer.
    """
    max_tokens = layer.max_tokens_per_rank
    if max_tokens <= 0:
        raise ValueError(f"max_tokens_per_rank must be positive, got {max_tokens}")
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    is_decode = layer._is_decode_role or is_decode_forward
    if is_decode or capturing:
        if tokens > max_tokens:
            mode = "decode" if is_decode else "CUDA graph capture"
            raise ValueError(
                f"{mode} MoE input tokens={tokens} exceeds "
                f"max_tokens_per_rank={max_tokens}"
            )
        return False
    executor = getattr(
        getattr(getattr(layer, "_moe", None), "fused_moe", None),
        "fused_experts",
        None,
    )
    return (
        layer.chunking_enabled
        and tokens > max_tokens
        and getattr(executor, "single_round_dispatch", False) is not True
    )
