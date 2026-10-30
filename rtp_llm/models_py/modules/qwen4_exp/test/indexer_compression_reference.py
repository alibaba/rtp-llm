"""Flat-buffer reference for the QSA indexer compression tests."""

import torch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    exact_head_rms_norm,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import apply_partial_rope


def compress_prefill(
    hidden_states: torch.Tensor,  # [B, T, D]
    qk_proj: torch.Tensor,  # [(n_heads+1)*head_dim, D]
    k_norm_gamma: torch.Tensor,  # [head_dim]
    cos: torch.Tensor,  # [B, T, rotary_dim] (full-range)
    sin: torch.Tensor,  # [B, T, rotary_dim]
    *,
    ratio: int,
    head_dim: int,
    norm_eps: float = 1e-6,
) -> dict:
    """Full prefill compression returning the compressed pool entries + raw keys.

    This is the **flat-buffer form**: instead of writing to framework-managed paged
    pools, it returns tensors that let the caller (or a test) verify the compressed
    content by placing it in a pool buffer.

    Returns a dict::

        raw_keys      [B, T, D] fp32
        pooled_k      [B, nb, D] bf16          (one per **complete** block)
        block_starts  [B, nb] int64            first token index of each block
        num_blocks    [B] int64                how many complete blocks per batch

    The caller can copy ``pooled_k[i, b]`` directly into the bf16 view of the
    ``indexer_kv`` pool at slot ``i * num_blocks + b``.
    """
    if hidden_states.dim() != 3:
        raise ValueError(
            f"hidden_states must be [B, T, hidden], got {tuple(hidden_states.shape)}"
        )
    if qk_proj.dim() != 2 or qk_proj.shape[1] != hidden_states.shape[-1]:
        raise ValueError(
            "qk_proj must be [(n_heads + 1) * head_dim, hidden], got "
            f"{tuple(qk_proj.shape)} for hidden={hidden_states.shape[-1]}"
        )
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if head_dim <= 0 or qk_proj.shape[0] % head_dim:
        raise ValueError(
            f"qk_proj output {qk_proj.shape[0]} is not divisible by head_dim={head_dim}"
        )
    n_heads = (qk_proj.shape[0] // head_dim) - 1
    if n_heads <= 0:
        raise ValueError("qk_proj must contain at least one Q head and one K head")
    if k_norm_gamma.shape != (head_dim,):
        raise ValueError(
            f"k_norm_gamma must be [{head_dim}], got {tuple(k_norm_gamma.shape)}"
        )
    B, T, _ = hidden_states.shape
    device = hidden_states.device
    if cos.shape[:2] != (B, T) or sin.shape != cos.shape:
        raise ValueError(
            f"cos/sin must have matching [B, T, rotary_dim] shapes, got "
            f"{tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    rotary_dim = cos.shape[-1]
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even, and <= head_dim; got {rotary_dim}"
        )

    qk = hidden_states @ qk_proj.T
    _, raw_k = qk.split([n_heads * head_dim, head_dim], dim=-1)
    raw_keys = raw_k.float()  # [B, T, D] fp32

    num_blocks_full = T // ratio
    # Truncate raw_keys to the token positions that form complete blocks.
    grouped = raw_keys[:, : num_blocks_full * ratio]  # [B, nb*ratio, D]
    grouped = grouped.unflatten(1, (num_blocks_full, ratio))  # [B, nb, ratio, D]
    # Mean in fp32 (upstream orders the summation the same way).
    pooled = grouped.mean(dim=2).to(raw_k.dtype)  # [B, nb, D]

    # Apply k_layernorm (with +1) and partial RoPE at each block's first position.
    pooled = exact_head_rms_norm(pooled, k_norm_gamma[None, None, :], norm_eps)
    block_starts = torch.arange(num_blocks_full, device=device) * ratio  # [nb]
    pooled_rotated = apply_partial_rope(
        pooled,
        cos[:, block_starts],
        sin[:, block_starts],
    )
    pooled_k = pooled_rotated.to(torch.bfloat16).contiguous()

    return {
        "raw_keys": raw_keys,
        "pooled_k": pooled_k,
        "block_starts": block_starts.expand(B, -1),
        "num_blocks": torch.full(
            (B,), num_blocks_full, dtype=torch.long, device=device
        ),
    }
