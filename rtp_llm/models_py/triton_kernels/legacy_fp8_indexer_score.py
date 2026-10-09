"""Group-128 FP8 indexer scores for the pre-MX DeepSeek cache contract.

DeepGEMM's MX-only MQA API does not accept this cache or FP32 routing weights.
FP8 values are exactly representable in BF16, so converting the dot operands
preserves their values while using FP32 accumulation. Scales and routing weights
remain FP32; there is no BF16 rounding or re-quantization of those quantities.
"""

import torch
import triton
import triton.language as tl


def is_supported(q, weights):
    return (
        q.is_cuda
        and q.dtype == torch.float8_e4m3fn
        and q.shape[-1] == 128
        and weights.dtype == torch.float32
    )


@triton.jit
def _score(
    Q,
    K,
    SCALE,
    WEIGHTS,
    START,
    END,
    TABLE,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_ROW: tl.constexpr,
    S_ROW: tl.constexpr,
    W_ROW: tl.constexpr,
    W_HEAD: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    N_PAGES: tl.constexpr,
    NEXT_N: tl.constexpr,
    PAGED: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    h = tl.arange(0, BH)
    d = tl.arange(0, 128)
    q = tl.load(
        Q + row * Q_ROW + h[:, None] * Q_HEAD + d[None, :], h[:, None] < H, 0.0
    ).to(tl.bfloat16)
    if PAGED:
        length = tl.load(END + row)
        logical_page = n // PAGE
        physical_page = tl.load(
            TABLE + (row // NEXT_N) * TABLE_WIDTH + logical_page,
            (n < N) & (n < length) & (logical_page < TABLE_WIDTH),
            -1,
        )
        valid = (
            (n < N) & (n < length) & (physical_page >= 0) & (physical_page < N_PAGES)
        )
        # DeepGEMM's fused 132-byte pool is block-planar: each page holds
        # PAGE*128 K bytes followed by PAGE*4 scale bytes. Slot (page, off)
        # therefore reads K at page*PAGE*132 + off*128 and its FP32 scale
        # at page*PAGE*132 + PAGE*128 + off*4; both are naturally aligned
        # for the page sizes DeepGEMM accepts.
        block = physical_page.to(tl.int64) * (PAGE * 132)
        off = n % PAGE
        raw = tl.load(
            K + block[None, :] + off[None, :] * 128 + d[:, None],
            valid[None, :],
            0,
        )
        k = raw.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        scale_ptr = (K + block + PAGE * 128 + off * 4).to(tl.pointer_type(tl.float32))
        scale = tl.load(scale_ptr, valid, 0)
    else:
        start = tl.load(START + row)
        end = tl.load(END + row)
        valid = (n < N) & (n >= start) & (n < end)
        k = tl.load(K + n[None, :] * K_ROW + d[:, None], valid[None, :], 0.0).to(
            tl.bfloat16
        )
        scale = tl.load(SCALE + n * S_ROW, valid, 0).to(tl.float32)
    dots = tl.dot(q, k, out_dtype=tl.float32)
    weight = tl.load(WEIGHTS + row * W_ROW + h * W_HEAD, h < H, 0).to(tl.float32)
    scores = tl.sum(tl.maximum(dots, 0.0) * weight[:, None], axis=0) * scale
    # DeepGEMM leaves out-of-range/invalid entries untouched (its
    # clean_logits contract is the caller's job); the paged fallback matches
    # that with zero fill so consumers see finite "whatever" values.
    tl.store(OUT + row * N + n, tl.where(valid, scores, 0.0), n < N)


def legacy_fp8_mqa_logits(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False, max_seqlen_k=0
):
    """Score the original FP8 K + one FP32 scale per 128-wide key format."""
    k, scales = kv
    if q.dim() != 3 or k.dim() != 2 or q.shape[-1] != k.shape[-1]:
        raise ValueError("legacy MQA expects Q[M,H,D] and K[N,D]")
    if weights.shape != q.shape[:2] or scales.numel() != k.shape[0]:
        raise ValueError("legacy MQA weight or per-key scale shape mismatch")
    if q.shape[-1] != 128:
        raise ValueError(
            "group-128 legacy MQA requires a single 128-wide quantization group"
        )
    scales = scales.reshape(-1)
    m, h, _ = q.shape
    n = k.shape[0]
    output = torch.empty((m, n), dtype=torch.float32, device=q.device)
    if m == 0 or n == 0:
        return output
    if not is_supported(q, weights):
        # Reference fallback retains full precision, including FP32 scale/weights.
        for begin in range(0, m, 16):
            end = min(m, begin + 16)
            dot = torch.einsum("mhd,nd->mhn", q[begin:end].float(), k.float())
            values = (dot.relu() * weights[begin:end].float().unsqueeze(-1)).sum(
                1
            ) * scales.float()
            columns = torch.arange(n, device=q.device)
            valid = (columns >= cu_seqlen_ks[begin:end, None]) & (
                columns < cu_seqlen_ke[begin:end, None]
            )
            output[begin:end] = values.masked_fill(~valid, -float("inf"))
        return output
    q, k = q.contiguous(), k.contiguous()
    _score[(m, triton.cdiv(n, 64))](
        q,
        k,
        scales,
        weights,
        cu_seqlen_ks.contiguous(),
        cu_seqlen_ke.contiguous(),
        cu_seqlen_ks,
        output,
        n,
        h,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        scales.stride(0),
        weights.stride(0),
        weights.stride(1),
        1,
        1,
        1,
        1,
        False,
        max(16, triton.next_power_of_2(h)),
        64,
        num_warps=4,
    )
    return output


def legacy_fp8_paged_mqa_logits(
    q, kv_cache, weights, context_lens, block_table, max_context_len
):
    """Paged score preserving 132-byte slots and full FP32 per-head weights."""
    if q.dim() != 4 or q.shape[-1] != 128:
        raise ValueError("legacy paged MQA expects Q[B,next_n,H,128]")
    batch, next_n, heads, _ = q.shape
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.dim() != 4
        or kv_cache.shape[2:] != (1, 132)
    ):
        raise ValueError("legacy paged MQA expects uint8 cache[pages,page_size,1,132]")
    if (
        weights.shape != (batch * next_n, heads)
        or context_lens.numel() != batch * next_n
    ):
        raise ValueError("legacy paged MQA weights or context lengths mismatch")
    q, kv_cache, block_table = (
        q.contiguous(),
        kv_cache.contiguous(),
        block_table.contiguous(),
    )
    context_lens = context_lens.reshape(-1).contiguous()
    m = batch * next_n
    output = torch.empty((m, max_context_len), dtype=torch.float32, device=q.device)
    if not m or not max_context_len:
        return output
    if not is_supported(q, weights):
        page_size = kv_cache.shape[1]
        pool_2d = kv_cache.view(kv_cache.shape[0], page_size * INDEXER_ENTRY_BYTES)
        for row in range(m):
            length = min(int(context_lens[row]), max_context_len)
            ids = torch.arange(length, device=q.device)
            pages = block_table[row // next_n, ids // page_size].long()
            offs = ids % page_size
            # Same block-planar decode as the kernel: [page_size*128 K |
            # page_size*4 scale] per page.
            k = (
                torch.stack(
                    [pool_2d[p, o * 128 : (o + 1) * 128] for p, o in zip(pages, offs)]
                )
                .contiguous()
                .view(torch.float8_e4m3fn)
            )
            scale = (
                torch.stack(
                    [
                        pool_2d[
                            p,
                            page_size * 128 + o * 4 : page_size * 128 + (o + 1) * 4,
                        ]
                        for p, o in zip(pages, offs)
                    ]
                )
                .contiguous()
                .view(torch.float32)
                .reshape(-1)
            )
            output[row].fill_(0.0)
            output[row, :length] = legacy_fp8_mqa_logits(
                q.view(m, heads, 128)[row : row + 1],
                (k, scale),
                weights[row : row + 1],
                torch.zeros(1, dtype=torch.int32, device=q.device),
                torch.full((1,), length, dtype=torch.int32, device=q.device),
            )[0]
        return output
    _score[(m, triton.cdiv(max_context_len, 64))](
        q,
        kv_cache,
        context_lens,
        weights,
        context_lens,
        context_lens,
        block_table,
        output,
        max_context_len,
        heads,
        q.stride(1),
        q.stride(2),
        132,
        1,
        weights.stride(0),
        weights.stride(1),
        kv_cache.shape[1],
        block_table.shape[1],
        kv_cache.shape[0],
        next_n,
        True,
        max(16, triton.next_power_of_2(heads)),
        64,
        num_warps=4,
    )
    return output
