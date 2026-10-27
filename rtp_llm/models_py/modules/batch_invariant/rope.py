# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unscaled base RoPE cache with vLLM's BF16 rounding boundary.

Adapted from vLLM 0.29.0 rotary_embedding/base.py. The configuration-keyed cache
is independent of RTP's process-global RoPE cache, so switching models cannot
reuse a different model's theta/dimension. FlashInfer consumes FP32 storage;
promoting the rounded BF16 values preserves the vLLM arithmetic inputs.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def base_rope_cache(rotary_dim, base, max_position_embeddings, device_index):
    if rotary_dim <= 0 or rotary_dim % 2 or max_position_embeddings <= 0:
        raise ValueError("invalid batch invariant base RoPE geometry")
    device = torch.device("cuda", device_index)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    positions = torch.arange(
        max_position_embeddings, dtype=torch.float32, device=device
    )
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache.to(torch.bfloat16).to(torch.float32)
