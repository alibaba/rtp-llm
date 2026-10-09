"""FA4 DSpark proposal on native BF16 cache, with in-place transient tail KV.

The engine reserves query slots. Only formal feature commit publishes cache
blocks; this helper must never publish its temporary proposal rows.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_query_tail(
    K, V, CACHE, TABLE, PREFIX, LIVE, KL, QL,
    WIDTH: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    PAGE: tl.constexpr, COLS: tl.constexpr, PHYSICAL: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    KB: tl.constexpr, KW: tl.constexpr, KH: tl.constexpr,
    VB: tl.constexpr, VW: tl.constexpr, VH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    row = tl.program_id(1)
    prefix = tl.load(PREFIX + b).to(tl.int64)
    live = tl.load(LIVE + b).to(tl.int32)
    tl.device_assert((live >= 0) & (live <= WIDTH), "invalid DSpark query length")
    tl.device_assert(prefix >= 0, "negative DSpark prefix")
    tl.device_assert(prefix + live <= 2147483647, "DSpark FA4 length exceeds int32 ABI")
    if row == 0:
        # Padded requests must not expose a stale historical prefix to FA4.
        tl.store(KL + b, tl.where(live > 0, prefix + live, 0))
        tl.store(QL + b, live)
    if row < live:
        pos = prefix + row
        col = pos // PAGE
        tl.device_assert(col < COLS, "DSpark tail was not reserved")
        physical = tl.load(TABLE + b * COLS + col, col < COLS, -1).to(tl.int64)
        tl.device_assert((physical >= 0) & (physical < PHYSICAL), "invalid DSpark tail page")
        x = tl.arange(0, BLOCK)
        head, dim = x // D, x % D
        valid = (x < HK * D) & (physical >= 0) & (physical < PHYSICAL)
        offset = physical * CACHE_STRIDE + (head * PAGE + pos % PAGE) * D + dim
        k = tl.load(K + b * KB + row * KW + head * KH + dim, valid, 0)
        v = tl.load(V + b * VB + row * VW + head * VH + dim, valid, 0)
        tl.store(CACHE + offset, k, valid)
        tl.store(CACHE + offset + HK * PAGE * D, v, valid)


@triton.jit
def _zero_query_padding(OUT, LIVE, WIDTH: tl.constexpr, ROW: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = tl.load(LIVE + b)
    tl.store(OUT + b * WIDTH * ROW + x, 0, (x < WIDTH * ROW) & (x // ROW >= live))


def paged_gqa_swa_fa4(
    q, k, v, cache, block_table, context_lens, query_lens,
    *, window_left: int, causal: bool, out=None,
):
    """Write only live query rows after the prefix, then read the original table.

    Cache: [pages,2,Hkv,128,128], head-major, optional padded physical stride.
    Q/K/V: [B,W,H,128], BF16. K/V row strides may be interleaved projection
    views. No history copy, shadow pages, host length readback or global scratch.
    FA4 owns capture-local partial buffers; every call owns its length metadata.
    """
    from flash_attn.cute.interface import _flash_attn_fwd

    if q.ndim != 4 or k.ndim != 4 or cache.ndim != 5:
        raise ValueError("expected rank-4 query tensors and rank-5 native cache")
    b, width, hq, d = q.shape
    hk = k.shape[2]
    if (d != 128 or hk <= 0 or hq % hk or width < 1
            or k.shape != (b, width, hk, d) or v.shape != k.shape
            or cache.shape[1:] != (2, hk, 128, d)):
        raise ValueError("incompatible DSpark FA4 paged GQA shapes")
    if not isinstance(causal, bool) or window_left < 0:
        raise ValueError("explicit causal mask and nonnegative window required")
    if (block_table.ndim != 2 or block_table.shape[0] != b
            or context_lens.shape != (b,) or query_lens.shape != (b,)):
        raise ValueError("invalid DSpark table/length shapes")
    tensors = (q, k, v, cache, block_table, context_lens, query_lens)
    if any(not t.is_cuda or t.device != q.device for t in tensors):
        raise ValueError("DSpark FA4 inputs must share one CUDA device")
    if any(t.dtype != torch.bfloat16 for t in tensors[:4]):
        raise TypeError("DSpark FA4 requires BF16 Q/K/V/cache")
    if any(t.dtype not in (torch.int32, torch.int64) for t in tensors[4:]):
        raise TypeError("DSpark FA4 metadata must be int32/int64")
    if (not q.is_contiguous() or any(not t.is_contiguous() for t in tensors[4:])
            or k.stride(-1) != 1 or v.stride(-1) != 1
            or cache.stride()[1:] != (hk * 128 * d, 128 * d, d, 1)
            or cache.stride(0) < 2 * hk * 128 * d):
        raise ValueError("invalid DSpark query/cache strides")
    if out is None:
        out = torch.empty_like(q)
    if (out.shape != q.shape or out.dtype != q.dtype or out.device != q.device
            or not out.is_contiguous()):
        raise ValueError("out must match contiguous q")
    if b == 0:
        return out
    table = block_table.to(torch.int32)
    kl = torch.empty(b, dtype=torch.int32, device=q.device)
    ql = torch.empty_like(kl)
    _write_query_tail[(b, width)](
        k, v, cache, block_table, context_lens, query_lens, kl, ql,
        width, hk, d, 128, table.shape[1], cache.shape[0], cache.stride(0),
        *k.stride()[:3], *v.stride()[:3], triton.next_power_of_2(hk * d),
        num_warps=4, debug=True,
    )
    # Zero-copy transpose: FA4 accepts RTP's head-major per-page strides.
    # Measured SM103 long-window shapes; small windows avoid split overhead.
    splits = 1
    if window_left >= 4095:
        splits = 8 if b <= 4 else 4 if b <= 8 else 2 if b <= 16 else 1
    _flash_attn_fwd(
        q, cache[:, 0].permute(0, 2, 1, 3), cache[:, 1].permute(0, 2, 1, 3),
        page_table=table, seqused_q=ql, seqused_k=kl,
        max_seqlen_q=width, max_seqlen_k=table.shape[1] * 128,
        causal=causal, window_size_left=window_left,
        window_size_right=0 if causal else window_left,
        num_splits=splits, out=out,
    )
    _zero_query_padding[(b, triton.cdiv(width * hq * d, 2048))](
        out, ql, width, hq * d, 2048,
    )
    return out
