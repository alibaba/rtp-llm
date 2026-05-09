"""DSv4 decode SWA BF16 pool write — extracted from
``AttentionVLLM._forward_decode_body``.

Mirrors :mod:`rtp_llm.models_py.modules.dsv4.decode.write_swa` (FP8 path)
for the BF16 KV-cache. Calls :func:`write_kv_to_pool` directly — no
quantize step since the pool slot is already BF16-typed.
"""

from __future__ import annotations

from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import write_kv_to_pool


def decode_write_swa_bf16(
    kv: torch.Tensor,
    slot_mapping: Optional[torch.Tensor],
    swa_pool_view: Optional[torch.Tensor],
    bsz: int,
    q_len: int,
    head_dim: int,
) -> None:
    """Write newly-computed SWA KV into the BF16 pool slots.

    No-op when ``slot_mapping`` is empty or the pool view is unavailable
    (warmup forward before the framework allocates the pool). SWA always
    emits a valid slot per token, so ``mask_negative=False`` is the fast
    ``index_copy_`` path.
    """
    if slot_mapping is None or slot_mapping.numel() == 0 or swa_pool_view is None:
        return
    kv_flat = kv.reshape(bsz * q_len, head_dim)
    write_kv_to_pool(
        kv_flat,
        slot_mapping[: bsz * q_len],
        swa_pool_view,
        mask_negative=False,
    )
