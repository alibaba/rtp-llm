"""Flash-decoding attention for the Gemma4 SWA layers (head_dim 256, window).

The graph-mode SWA decode ran a page-chunked SDPA loop (~10 window pages per
layer at window 1024 / page 128, ~25-35 tiny ops per chunk across 25 SWA
layers), which dominated the decode timeline with thousands of micro-kernels
per step. This module applies the same flash-decoding pair as
``full_decode`` — the shared partial kernel (online fp32 softmax over paged
KV) plus the shared LSE-merge combine — specialized by constexpr:

- head_dim is 256, processed as two 128-wide halves (DH=128, TWOHALF);
- WINDOW=1024 adds the sliding-window lower bound
  ``key_pos > qpos - window`` to the causal mask, and the page iteration
  only covers each request's trailing ``win_pages`` pages (device-computed
  from indptr), so long contexts do not scan their whole history.

All metadata (page table, indptr, last-page-len, q positions) is read on
device; win_pages is a capture-time constant, so a fixed-grid launch is
CUDA-graph capturable. Score math is fp32 accumulate with scale 1.0 (no
1/sqrt(d)); P is rounded to bf16 before the PV dot, matching the eager
path's numeric class.
"""

from functools import lru_cache

import torch

from rtp_llm.models_py.modules.gemma4.decode_padding import pad_decode_queries
from rtp_llm.models_py.modules.gemma4.full_decode import (
    _decode_layout_supported,
    _full_decode_combine,
    _full_decode_partial,
    _next_pow2,
)

_KERNEL_VERSION = "v5-token-major-output"


@lru_cache(maxsize=None)
def _pipeline_stages(device: torch.device) -> int:
    # Resource and timing validation covers the SM103 backend specialization.
    # Other backends retain one stage until their compiled footprint is tested.
    props = torch.cuda.get_device_properties(device)
    return (
        2
        if (props.major, props.minor) == (10, 3)
        and props.shared_memory_per_block_optin >= 141312
        else 1
    )


def swa_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pages: torch.Tensor,
    indptr: torch.Tensor,
    last_page_len: torch.Tensor,
    q_pos: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    page_size: int,
    window: int,
) -> torch.Tensor:
    """q: [B, H, M, 256] bf16; caches: [pages, KVH, PS, 256].

    Q may be contiguous BHMD or the canonical BHMD view of contiguous BMHD
    storage. Its strides flow directly into query padding and the partial kernel,
    so the fused caller needs no native contiguous-query copy.

    Returns a [B, H, q_len, 256] bf16 view of contiguous [B, q_len, H, 256]
    storage, or None when unsupported (the caller falls back). The consumer's
    transpose/reshape to packed NHD shares that storage. q_pos: [B, M] int.
    All metadata stays
    on device, so a fixed-grid launch replays correctly under CUDA graphs.

    The kernel tiles each page in 64-wide sub-blocks, so only page sizes that
    are positive multiples of 64 are eligible; smaller or unaligned sizes
    would silently drop keys and must take the fallback path instead.
    """
    if window <= 0 or not _decode_layout_supported(
        q,
        k_cache,
        v_cache,
        pages,
        indptr,
        last_page_len,
        q_pos,
        head_num,
        kv_head_num,
        page_size,
        256,
    ):
        return None
    batch, _, q_len, head_dim = q.shape
    group = head_num // kv_head_num
    # tl.dot requires M >= 16; decode q_len is 1..8 typically, so pad.
    m_pow2 = max(16, _next_pow2(q_len))
    # the union of all queries' windows spans window + q_len - 1 positions
    # (plus one page of alignment slack), same bound as the chunked loop.
    win_pages = (window + q_len - 1 + page_size - 1) // page_size + 1
    # splits over the window pages only; o_acc is fp32 [B, H, splits, M, D].
    splits = min(8, win_pages)

    m_acc = torch.empty(
        (batch, head_num, splits, m_pow2), dtype=torch.float32, device=q.device
    )
    l_acc = torch.empty(
        (batch, head_num, splits, m_pow2), dtype=torch.float32, device=q.device
    )
    o_acc = torch.empty(
        (batch, head_num, splits, m_pow2, head_dim),
        dtype=torch.float32,
        device=q.device,
    )

    q_full, qpos_full = pad_decode_queries(q, q_pos, m_pow2)

    grid = (batch * head_num, splits)
    _full_decode_partial[grid](
        q_full,
        k_cache,
        v_cache,
        pages,
        indptr,
        last_page_len,
        qpos_full,
        m_acc,
        l_acc,
        o_acc,
        q_full.stride(0),
        q_full.stride(1),
        q_full.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        o_acc.stride(2),
        o_acc.stride(3),
        H=head_num,
        GROUP=group,
        DH=head_dim // 2,
        PS=page_size,
        CACHE_PAGES=k_cache.shape[0],
        M=m_pow2,
        SPLITS=splits,
        WINDOW=window,
        WINPAGES=win_pages,
        num_warps=4,
        # Keep unmeasured geometries on the conservative single-stage path.
        num_stages=(
            _pipeline_stages(q.device)
            if page_size == 128
            and window == 1024
            and m_pow2 == 16
            and torch.version.hip is None
            else 1
        ),
    )
    output = torch.empty(
        (batch, q_len, head_num, head_dim), dtype=torch.bfloat16, device=q.device
    )
    _full_decode_combine[(batch * head_num,)](
        m_acc,
        l_acc,
        o_acc,
        output,
        o_acc.stride(2),
        o_acc.stride(3),
        output.stride(0),
        output.stride(2),
        output.stride(1),
        H=head_num,
        DH=head_dim // 2,
        M=m_pow2,
        SPLITS=splits,
        TWOHALF=True,
        Q_LEN=q_len,
        num_warps=4,
    )
    return output.permute(0, 2, 1, 3)
