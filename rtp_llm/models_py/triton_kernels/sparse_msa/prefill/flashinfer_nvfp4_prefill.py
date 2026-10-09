"""RTP packed working pages to FlashInfer NVFP4 sparse prefill."""

import torch
import triton
import triton.language as tl

from flashinfer.msa_ops import (
    msa_prefill_nvfp4_specialized_warmup,
    msa_sparse_attention,
)


def page_layout(heads):
    """FlashInfer planar page ABI: K, K scales, V, swizzled V scales.

    Page128/D128 stores 8192 packed data bytes and 1024 E4M3 scale
    bytes per head. Keep this layout adapter RTP-owned; computation uses
    the pinned dependency's public API, not its private implementation.
    """
    return dict(k_scale_byte_offset=heads * 8192,
                v_data_byte_offset=heads * 9216,
                v_scale_byte_offset=heads * 17408,
                page_bytes=heads * 18432)


@triton.jit
def _convert_page_head(K, V, KS, VS, DST, K_STRIDE, V_STRIDE,
                       KS_STRIDE, VS_STRIDE, PAGE_BYTES: tl.constexpr,
                       K_SCALE_OFFSET: tl.constexpr, V_DATA_OFFSET: tl.constexpr,
                       V_SCALE_OFFSET: tl.constexpr):
    # Widen before byte-stride multiplication: physical pools can exceed 2 GiB.
    page = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    data = tl.arange(0, 8192)
    scale = tl.arange(0, 1024)
    token = scale // 8
    group = scale % 8
    mma = (group // 4) * 512 + (token % 32) * 16 + (token // 32) * 4 + group % 4
    swizzled = ((token // 4) * 4 + group // 2) * 8 + (group % 2) * 4 + token % 4
    dst = DST + page * PAGE_BYTES
    k = tl.load(K + page * K_STRIDE + head * 8192 + data)
    v = tl.load(V + page * V_STRIDE + head * 8192 + data)
    ks = tl.load(KS + page * KS_STRIDE + head * 1024 + mma)
    vs = tl.load(VS + page * VS_STRIDE + head * 1024 + mma)
    tl.store(dst + head * 8192 + data, k)
    tl.store(dst + V_DATA_OFFSET + head * 8192 + data, v)
    tl.store(dst + K_SCALE_OFFSET + head * 1024 + scale, ks)
    tl.store(dst + V_SCALE_OFFSET + head * 1024 + swizzled, vs)


@triton.jit
def _prepare_topk(TOPK, POSITIONS, OUT, T0, T1, T2, P0,
                  ROWS: tl.constexpr, COLS: tl.constexpr,
                  BLOCK_ROWS: tl.constexpr):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    head = tl.program_id(1)
    slot = tl.arange(0, 16)
    ids = tl.load(TOPK + head * T0 + row[:, None] * T1 + slot[None, :] * T2,
                  mask=row[:, None] < ROWS, other=-1)
    position = tl.load(POSITIONS + row * P0, mask=row < ROWS, other=-1)
    valid = (ids >= 0) & (ids < COLS) & (ids <= position[:, None] // 128)
    ordered = tl.sort(tl.where(valid, ids, 2147483647), descending=False, dim=1)
    tl.store(OUT + head * ROWS * 16 + row[:, None] * 16 + slot[None, :],
             tl.where(ordered == 2147483647, -1, ordered),
             mask=row[:, None] < ROWS)


@triton.jit
def _clear_unwritten_page_tail(REQ_TO_TOKEN, FULL_LENGTHS, DST,
                               TABLE_STRIDE: tl.constexpr,
                               SEGMENTS_PER_REQUEST: tl.constexpr,
                               PAGE_BYTES: tl.constexpr,
                               K_SCALE_OFFSET: tl.constexpr,
                               V_DATA_OFFSET: tl.constexpr,
                               V_SCALE_OFFSET: tl.constexpr):
    request = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    length = tl.load(FULL_LENGTHS + request)
    tail = length % 128
    if tail != 0:
        page = tl.load(REQ_TO_TOKEN + request * SEGMENTS_PER_REQUEST * TABLE_STRIDE
                       + ((length - 1) // 128) * 128).to(tl.int64) // 128
        base = DST + page * PAGE_BYTES
        data = tl.arange(0, 8192)
        scale = tl.arange(0, 1024)
        token = scale // 8
        swizzled = ((token // 4) * 4 + (scale % 8) // 2) * 8 + (scale % 2) * 4 + token % 4
        tl.store(base + head * 8192 + data, 0, mask=data // 64 >= tail)
        tl.store(base + V_DATA_OFFSET + head * 8192 + data, 0,
                 mask=data // 64 >= tail)
        tl.store(base + K_SCALE_OFFSET + head * 1024 + scale, 0,
                 mask=token >= tail)
        tl.store(base + V_SCALE_OFFSET + head * 1024 + swizzled, 0,
                 mask=token >= tail)


class _PlanarPages:
    def __init__(self):
        self.pool = None

    def acquire(self, pages, heads, device):
        layout = page_layout(heads)
        shape = (pages, layout["page_bytes"])
        pool = self.pool
        if (pool is None or pool.device != device or pool.shape[0] < pages
                or pool.shape[1] != shape[1]):
            pool = torch.empty(shape, dtype=torch.uint8, device=device)
            self.pool = pool
        return pool[:pages], layout


_PLANAR_PAGES = _PlanarPages()


def flashinfer_sparse_prefill_from_topk_fp4(
    q, main_k, main_v, k_scale, v_scale, topk, req_to_token,
    cu_seqlens_q, seqused_k, positions, full_kv_lengths, sm_scale,
    segments_per_request=1,
):
    """Run FI over RTP's request-local packed working pages, preserving KV ABI."""
    if q.dtype != torch.bfloat16 or not q.is_contiguous() or q.shape[1:] != (64, 128):
        raise ValueError("FlashInfer MSA prefill requires contiguous BF16 Q [N,64,128]")
    pages, heads, page_size, packed_dim = main_k.shape
    if (main_v.shape != main_k.shape or heads != 4 or page_size != 128
            or packed_dim != 64 or k_scale.shape != v_scale.shape
            or k_scale.shape != (pages, heads * page_size * 8)):
        raise ValueError("FlashInfer MSA prefill requires RTP Hkv4/page128/D128 planes")
    if req_to_token.ndim != 2 or req_to_token.shape[0] != seqused_k.numel():
        raise ValueError("request-local working slots must match query segments")
    if (topk.shape != (heads, q.shape[0], 16) or topk.dtype != torch.int32
            or positions.shape != (q.shape[0],)):
        raise ValueError("TopK and positions must match BF16 Q rows")
    if seqused_k.dtype != torch.int32 or cu_seqlens_q.dtype != torch.int32:
        raise ValueError("FlashInfer segment lengths must be int32")
    if (full_kv_lengths.dtype != torch.int32 or full_kv_lengths.ndim != 1
            or full_kv_lengths.numel() * segments_per_request != req_to_token.shape[0]):
        raise ValueError("full KV lengths must cover each request's working pages")
    msa_prefill_nvfp4_specialized_warmup(q.device)
    pool, layout = _PLANAR_PAGES.acquire(pages, heads, q.device)
    _convert_page_head[(pages, heads)](
        main_k, main_v, k_scale.view(torch.uint8), v_scale.view(torch.uint8),
        pool, main_k.stride(0), main_v.stride(0),
        k_scale.stride(0), v_scale.stride(0),
        PAGE_BYTES=layout["page_bytes"],
        K_SCALE_OFFSET=layout["k_scale_byte_offset"],
        V_DATA_OFFSET=layout["v_data_byte_offset"],
        V_SCALE_OFFSET=layout["v_scale_byte_offset"], num_warps=8,
    )
    _clear_unwritten_page_tail[(full_kv_lengths.numel(), heads)](
        req_to_token, full_kv_lengths, pool, req_to_token.stride(0),
        segments_per_request, layout["page_bytes"],
        layout["k_scale_byte_offset"], layout["v_data_byte_offset"],
        layout["v_scale_byte_offset"], num_warps=8,
    )
    page_table = (req_to_token[:, ::128] // 128).to(torch.int32).contiguous()
    ordered_topk = torch.empty_like(topk, memory_format=torch.contiguous_format)
    _prepare_topk[(triton.cdiv(q.shape[0], 16), heads)](
        topk, positions, ordered_topk,
        topk.stride(0), topk.stride(1), topk.stride(2), positions.stride(0),
        ROWS=q.shape[0], COLS=page_table.shape[1], BLOCK_ROWS=16, num_warps=4,
    )
    data_shape = (pages, heads, 128, 64)
    scale_shape = (pages, heads, 128, 8)
    data_stride = (layout["page_bytes"], 8192, 64, 1)
    scale_stride = (layout["page_bytes"], 1024, 8, 1)
    k = torch.as_strided(pool, data_shape, data_stride)
    v = torch.as_strided(pool, data_shape, data_stride,
                         layout["v_data_byte_offset"])
    ks = torch.as_strided(pool, scale_shape, scale_stride,
                          layout["k_scale_byte_offset"])
    vs = torch.as_strided(pool, scale_shape, scale_stride,
                          layout["v_scale_byte_offset"])
    out = torch.empty_like(q)
    msa_sparse_attention(q=q, k=k, v=v, k_scale=ks, v_scale=vs,
           q2k_indices=ordered_topk, cu_seqlens_q=cu_seqlens_q,
           page_table=page_table, seqused_k=seqused_k, out=out,
           causal=True, softmax_scale=sm_scale,
           k_global_scale=1.0, v_global_scale=1.0)
    return out
