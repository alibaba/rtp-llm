"""DeepSeek-V4 decode-path sparse attention op (BF16 KV).

Why TileLang and not FlashMLA here:
  * FlashMLA's ``flash_mla_with_kvcache`` sparse decode path asserts
    ``is_fp8_kvcache=True`` (see ``flash_mla_with_kvcache:154``) — it
    only ships an FP8-KV implementation. V4's BF16 KV cache cannot feed
    that kernel without first quantizing K/V to FP8 per-step, which is
    deferred to Phase 4 of the decode plan.
  * V4 ships its own author-authored TileLang ``sparse_attn_kernel``
    (vendored at ``dsv4/tilelang_kernels.py``), purpose-built for the
    MQA + Q-LoRA + per-head learned sink shape. Phase 1 reuses it
    verbatim — same kernel as prefill, just driven from the decode
    metadata and per-request KV slice.
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels as _tl_kernels
from rtp_llm.models_py.modules.dsv4.attention import _sparse_attn


class SparseAttnV4DecodeOp:
    """Per-step batched sparse MQA attention with per-head learned sink.

    Wraps the V4 TileLang ``sparse_attn`` kernel for the decode path,
    falling back to the PyTorch reference when tilelang is unavailable.
    """

    def __init__(self, n_heads: int, head_dim: int, softmax_scale: float):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.softmax_scale = float(softmax_scale)

    def forward(
        self,
        q: torch.Tensor,  # [B, q_len, H, D] bf16
        kv: torch.Tensor,  # [B, T_max, D] bf16  (per-request slice)
        attn_sink: torch.Tensor,  # [H] fp32
        topk_idxs: torch.Tensor,  # [B, q_len, K_total] int32; -1 == masked
    ) -> torch.Tensor:
        if q.is_cuda and _tl_kernels.tilelang_available():
            return _tl_kernels.sparse_attn(
                q,
                kv,
                attn_sink,
                topk_idxs,
                self.softmax_scale,
            )
        return _sparse_attn(q, kv, attn_sink, topk_idxs, self.softmax_scale)

    __call__ = forward
