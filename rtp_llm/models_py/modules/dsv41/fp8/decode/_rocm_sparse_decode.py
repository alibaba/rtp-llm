"""ROCm v0 decode path for V4/V4.1 packed-pool sparse attention.

FlashMLA's ``flash_mla_with_kvcache`` (which reads the packed 528B SWA
MXFP8 and 288B global FP4 pools in-kernel) is CUDA-only. This module
implements the same single- and dual-pool joint-softmax semantics for
gfx942 by gathering the selected slots through the already-verified
Triton dequant codecs (bit-exact roundtrips) and reusing the reference
sparse-MLA forward:

  primary pool slots -> ``dequantize_swa_k_cache``   -> [rows*K1, D] BF16
  extra pool slots   -> ``dequantize_k_cache_slots_fp4`` -> [rows*K2, D] BF16
  combined per-row index space -> ``rocm_sparse_mla_fwd`` (sink in denominator)

Invalid entries (index < 0, or slot >= topk_length) are masked out of both
numerator and denominator, matching FlashMLA's leftmost-valid-length scan.

CUDA/HIP-graph safety: every shape is static for a given capture bucket
(``[B, q_len, K1(+K2)]`` come from the captured metadata buffers), indices
and lengths are consumed as device tensors (no host reads, no ``.item()``),
the dequant kernels are plain Triton launches with static grids, and the
transient gathered KV (~B*(K1+K2)*D*2B per layer) is stream-ordered and
reused across layers by the caching allocator / graph pool. ``sched_meta``
is ignored (the FlashMLA tile planner does not exist on ROCm).

Cost note: this dequantizes gathered slots per step (~4 GB/step of pool
reads at B=32 with 8192-wide capture indices); a fused Triton decode FMHA
reading the packed pools directly is the follow-up performance mechanism.
"""

from __future__ import annotations

from typing import Optional

import torch


def _valid_mask(
    idxs: torch.Tensor, length: Optional[torch.Tensor], ql: int
) -> torch.Tensor:
    """[rows, K] bool: idx >= 0 and (when length given) k < length[row]."""
    valid = idxs >= 0
    if length is not None:
        per_row = length.to(torch.int64).repeat_interleave(ql).unsqueeze(1)
        k_ar = torch.arange(idxs.shape[1], device=idxs.device).unsqueeze(0)
        valid = valid & (k_ar < per_row)
    return valid


def _positions(
    rows: int, k: int, base_offset: int, device: torch.device
) -> torch.Tensor:
    r = torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1)
    c = torch.arange(k, device=device, dtype=torch.int64).unsqueeze(0)
    return base_offset + r * k + c


def rocm_sparse_decode_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    topk_idxs: torch.Tensor,
    *,
    sm_scale: float,
    topk_length: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_topk_idxs: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    return_lse: bool = False,
) -> torch.Tensor:
    from rtp_llm.models_py.modules.dsv41.fp8._rocm_sparse_fmha import (
        rocm_sparse_mla_fwd,
    )
    from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
        dequantize_k_cache_slots_fp4,
    )
    from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
        dequantize_swa_k_cache,
    )

    if return_lse:
        raise NotImplementedError(
            "ROCm sparse decode v0 does not expose LSE (no in-tree consumer)"
        )
    b, ql, heads, dim = q.shape
    rows = b * ql
    device = q.device
    minus_one = torch.full((), -1, dtype=torch.int64, device=device)

    idx1 = (
        topk_idxs.squeeze(2) if topk_idxs.dim() == 4 else topk_idxs
    ).reshape(rows, -1).to(torch.int64)
    k1 = idx1.shape[1]
    valid1 = _valid_mask(idx1, topk_length, ql)
    sel1 = dequantize_swa_k_cache(
        kv_cache, torch.where(valid1, idx1, minus_one).reshape(-1)
    )

    if extra_k_cache is not None and extra_topk_idxs is not None:
        idx2 = (
            extra_topk_idxs.squeeze(2) if extra_topk_idxs.dim() == 4 else extra_topk_idxs
        ).reshape(rows, -1).to(torch.int64)
        k2 = idx2.shape[1]
        valid2 = _valid_mask(idx2, extra_topk_length, ql)
        sel2 = dequantize_k_cache_slots_fp4(
            extra_k_cache, torch.where(valid2, idx2, minus_one).reshape(-1)
        )
        kv = torch.cat((sel1, sel2), dim=0)
        pos1 = _positions(rows, k1, 0, device)
        pos2 = _positions(rows, k2, rows * k1, device)
        new_idx = torch.cat(
            (
                torch.where(valid1, pos1, minus_one),
                torch.where(valid2, pos2, minus_one),
            ),
            dim=1,
        )
    else:
        kv = sel1
        pos1 = _positions(rows, k1, 0, device)
        new_idx = torch.where(valid1, pos1, minus_one)

    out = rocm_sparse_mla_fwd(
        q.reshape(rows, heads, dim),
        kv,
        new_idx,
        sm_scale,
        attn_sink=attn_sink,
    )
    return out.view(b, ql, heads, dim)
