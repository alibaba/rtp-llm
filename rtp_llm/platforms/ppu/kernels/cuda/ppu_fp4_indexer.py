"""SGLang's FP4 Indexer kernels with RTP-owned launch and cache adapters."""

from functools import lru_cache

import torch

from rtp_llm.platforms.ppu.kernels.cuda.sglang_jit import load_sglang_kernel


@lru_cache(maxsize=None)
def _module(kind, arch, parameter=0, canonical_topk=False):
    pdl = "true" if arch[0] >= 9 else "false"
    choices = {
        "q": (
            "main_norm_rope.cuh",
            f"FusedQIndexerRopeHadamardFp4QuantKernel<bf16_t,{pdl}>::forward",
        ),
        "compress": (
            "c4_v2.cuh",
            f"FlashCompress4Kernel<128,fp32_t,fp32_t,{pdl}>::run_prefill",
        ),
        "compress_decode": (
            "c4_decode_v2.cuh",
            f"FlashCompress4Kernel<128,fp32_t,fp32_t,{pdl}>::run_decode",
        ),
        "compress_decode_store": (
            "c4_decode_store.cuh",
            f"FusedC4DecodeStore<{pdl}>::forward",
        ),
        "compress_decode_queries_store": (
            "c4_decode_queries_store.cuh",
            f"FusedC4DecodeQueriesStore<false,{parameter >> 5},{parameter & 31}>::forward",
        ),
        "store": (
            "fused_norm_rope_v2.cuh",
            f"FusedNormRopeKernel<fp32_t,128,64,{parameter},{pdl}>::forward_fp4",
        ),
        "topk": ("topk_prefill_bf16.cuh", "TopKPrefillBF16Kernel::transform"),
        "topk_checked": (
            "topk_prefill_bf16_checked.cuh",
            "TopKPrefillBF16CheckedKernel::transform",
        ),
        "topk_decode": ("topk_decode_v1.cuh", f"TopKKernel<{pdl}>::transform"),
    }
    filename, symbol = choices[kind]
    source = f"csrc/deepseek_v4/{filename}"
    if kind in (
        "store",
        "topk",
        "topk_checked",
        "compress_decode",
        "compress_decode_store",
        "compress_decode_queries_store",
        "topk_decode",
    ):
        source = f"compat/{filename}"
    flags = []
    if kind in ("compress", "compress_decode"):
        flags += ["-use_fast_math"]
    if kind in ("topk", "topk_checked", "topk_decode"):
        flags += [f"-DSGL_TOPK={parameter}"]
    if canonical_topk:
        if kind != "topk_decode":
            raise ValueError("In-kernel index ordering is only available for decode")
        flags += ["-DRTP_CANONICAL_TOPK=1"]
    return load_sglang_kernel(
        f"fp4_{kind}_{parameter}", source, symbol, arch, tuple(flags)
    )


def quantize_q(q, weights, weight_scale, freqs, positions):
    """BF16 Q/projection -> packed E2M1, four UE8M0 bytes, FP32 weights."""
    m, h, d = q.shape
    if d != 128 or q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("FP4 Indexer Q requires contiguous BF16 [M,H,128]")
    packed = torch.empty((m, h, 64), dtype=torch.int8, device=q.device)
    scales = torch.empty((m, h), dtype=torch.int32, device=q.device)
    scaled_weights = torch.empty((m, h, 1), dtype=torch.float32, device=q.device)
    _module("q", torch.cuda.get_device_capability(q.device)).forward(
        q, packed, scales, weights, scaled_weights, weight_scale, freqs, positions
    )
    return packed, scales.unsqueeze(-1), scaled_weights.squeeze(-1)


def compress4(state, kv_score, ape, plan_c, plan_w):
    output = torch.empty(
        (plan_c.shape[0], 128), dtype=torch.float32, device=kv_score.device
    )
    _module("compress", torch.cuda.get_device_capability(kv_score.device)).forward(
        state.view(-1, 4, 512), kv_score, output, ape, plan_c, plan_w
    )
    return output


def compress4_decode(state, kv_score, ape, plan):
    output = torch.empty(
        (plan.shape[0], 128), dtype=torch.float32, device=kv_score.device
    )
    if plan.shape[0]:
        _module(
            "compress_decode", torch.cuda.get_device_capability(kv_score.device)
        ).forward(state.view(-1, 4, 512), kv_score, output, ape, plan)
    return output


def compress4_decode_store(
    state, kv_score, ape, plan, weight, eps, freqs, slots, cache, page_size
):
    """Return False before launch when the original two-kernel path is needed."""
    tensors = (state, kv_score, ape, plan, weight, freqs, slots, cache)
    if not (
        kv_score.is_cuda
        and kv_score.ndim == 2
        and kv_score.shape[1] == 512
        and page_size == 64
        and state.numel() % 2048 == 0
        and cache.numel() % 4352 == 0
        and all(t.is_contiguous() and t.device == kv_score.device for t in tensors)
        and all(t.dtype == torch.float32 for t in (state, kv_score, ape, weight, freqs))
        and plan.dtype == cache.dtype == torch.uint8
        and slots.dtype == torch.int32
        and plan.shape == (kv_score.shape[0], 16)
        and slots.shape == (kv_score.shape[0],)
        and ape.shape == (8, 128)
        and weight.shape == (128,)
        and freqs.ndim == 2
        and freqs.shape[1] == 64
    ):
        return False
    if kv_score.shape[0]:
        _module(
            "compress_decode_store", torch.cuda.get_device_capability(kv_score.device)
        ).forward(
            state.view(-1, 4, 512),
            kv_score,
            ape,
            plan,
            weight,
            float(eps),
            freqs,
            slots,
            cache.view(-1, 4352),
        )
    return True


def project_compress4_decode_queries_store(
    state,
    x,
    projection_weight,
    ape,
    weight,
    eps,
    freqs,
    meta,
    block_table,
    state_entries,
    state_tokens,
    cache,
    page_size,
):
    """Project MTP queries and fuse ordered C4 transitions for each request.

    Keep the original B-by-512 FP32 GEMMs: folding B and Q or using bmm
    changes the PPU reduction's rounding. Return False before any mutation
    for layouts outside the qualified decode contract.
    """
    if x.ndim != 3 or x.shape[1] not in (2, 3, 4):
        return False
    batch, width, hidden = x.shape
    metadata = (meta.positions, meta.b_idx, meta.state_slots, meta.kv_slots)
    tensors = (state, x, projection_weight, ape, weight, freqs, cache, *metadata)
    if not (
        x.is_cuda
        and 1 <= batch <= 32
        and hidden == 4096
        and page_size == 64
        and state_entries in (8, 12)
        and state_tokens > 0
        and state_tokens % 4 == 0
        and state.numel() % 2048 == 0
        and cache.numel() % 4352 == 0
        and all(t.is_contiguous() and t.device == x.device for t in tensors)
        and x.dtype == projection_weight.dtype == torch.bfloat16
        and projection_weight.shape == (512, hidden)
        and all(t.dtype == torch.float32 for t in (state, ape, weight, freqs))
        and all(
            t.shape == (batch * width,) and t.dtype in (torch.int32, torch.int64)
            for t in metadata
        )
        and block_table.ndim == 2
        and block_table.shape[1] > 0
        and block_table.stride(1) == 1
        and block_table.device == x.device
        and block_table.dtype in (torch.int32, torch.int64)
        and cache.dtype == torch.uint8
        and ape.shape == (8, 128)
        and weight.shape == (128,)
        and freqs.ndim == 2
        and freqs.shape[1] == 64
    ):
        return False
    projected = torch.empty((width, batch, 512), dtype=torch.float32, device=x.device)
    for index in range(width):
        torch.mm(
            x[:, index],
            projection_weight.t(),
            out_dtype=torch.float32,
            out=projected[index],
        )
    dtype_mask = sum(
        (t.dtype == torch.int64) << i for i, t in enumerate((*metadata, block_table))
    )
    _module(
        "compress_decode_queries_store",
        torch.cuda.get_device_capability(x.device),
        (width << 5) | dtype_mask,
    ).forward(
        state.view(-1, 4, 512),
        projected,
        ape,
        weight,
        float(eps),
        freqs,
        *metadata,
        block_table,
        state_entries,
        state_tokens,
        cache.view(-1, 4352),
    )
    return True


def norm_rope_store(
    compressed,
    plan_c,
    norm_weight,
    eps,
    freqs,
    slots,
    cache,
    page_size,
    *,
    is_decode=False,
):
    if not compressed.shape[0]:
        return
    _module("store", torch.cuda.get_device_capability(cache.device), page_size).forward(
        compressed,
        plan_c,
        norm_weight,
        eps,
        freqs,
        slots,
        cache.view(-1, page_size * 68),
        is_decode,
        4,
    )


def paged_score(q, q_scales, weights, cache, block_table, lengths, max_context):
    """Read native FP4 pages directly and retain FP32 Decode logits."""
    import deep_gemm

    if q.ndim != 4 or q.shape[1] != 1 or q.shape[-1] != 64:
        raise ValueError("FP4 paged Decode requires [B,1,H,64] packed Q")
    if cache.dtype != torch.uint8 or cache.ndim != 3 or cache.shape[-1] != 68:
        raise ValueError("FP4 paged Decode requires native 68-byte cache entries")
    if lengths.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise ValueError("FP4 paged Decode requires int32 lengths and block tables")
    page_size = cache.shape[1]
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        lengths, page_size, deep_gemm.get_num_sms()
    )
    return deep_gemm.fp8_fp4_paged_mqa_logits(
        (q.contiguous(), q_scales.contiguous()),
        cache.view(cache.shape[0], page_size, 1, 68),
        weights.contiguous(),
        lengths,
        block_table,
        schedule,
        max_context,
        clean_logits=False,
        logits_dtype=torch.float32,
    )


@lru_cache(maxsize=None)
def _identity_page(device):
    return torch.zeros((1, 1), dtype=torch.int32, device=device)


def topk_decode(scores, lengths, out, *, canonical_in_kernel=False):
    """SG FP32 Decode selection, returning logical compressed-token indices."""
    from rtp_llm.platforms.ppu.kernels.ppu_bf16_radix_topk import (
        canonicalize_topk_indices,
    )

    if scores.dtype != torch.float32 or lengths.dtype != torch.int32:
        raise ValueError("Decode TopK requires FP32 scores and int32 lengths")
    if out.shape[1] not in (512, 1024) or scores.shape[1] >= 2**31:
        raise ValueError("Decode TopK requires K=512/1024 and int32 score indices")
    if not scores.shape[0]:
        return out
    pages = _identity_page(scores.device).expand(scores.shape[0], 1)
    _module(
        "topk_decode",
        torch.cuda.get_device_capability(scores.device),
        out.shape[1],
        canonical_in_kernel,
    ).forward(scores, lengths, pages, out, 2**31, None)
    return out if canonical_in_kernel else canonicalize_topk_indices(out)


def topk_bf16(scores, starts, ends, out):
    from rtp_llm.platforms.ppu.kernels.ppu_bf16_radix_topk import (
        canonicalize_topk_indices,
    )

    if scores.dtype != torch.bfloat16 or scores.shape[1] >= 2**31:
        raise ValueError("BF16 TopK requires BF16 scores with fewer than 2**31 columns")
    # A 1M prompt has 250K compressed candidates. The vendor two-pass
    # implementation is not safe for these widths; use checked selection.
    if scores.shape[1] > 16384:
        _checked_bf16_topk(scores, starts, ends, out)
        return canonicalize_topk_indices(out)
    if not scores.shape[0]:
        return out
    pages = _identity_page(scores.device).expand(scores.shape[0], 1)
    _module(
        "topk", torch.cuda.get_device_capability(scores.device), out.shape[1]
    ).forward(scores, starts, ends, pages, out, 2**31, None)
    # Atomic scatter preserves the set but not its order. Sparse attention's
    # finite-precision reduction must see a repeatable candidate order.
    return canonicalize_topk_indices(out)


def _checked_bf16_topk(scores, starts, ends, out):
    """Native stream selection with exact per-row radix retries."""
    from rtp_llm.platforms.ppu.kernels.ppu_bf16_radix_topk import bf16_radix_topk

    if scores.shape[1] > 262144 or out.shape[1] not in (512, 1024):
        raise ValueError(
            "Wide BF16 TopK supports at most 262144 columns and K=512/1024"
        )
    if out.stride(0) != out.shape[1] or not scores.shape[0]:
        return bf16_radix_topk(scores, starts, ends, out)
    fallback = torch.empty(scores.shape[0], dtype=torch.int32, device=scores.device)
    pages = _identity_page(scores.device).expand(scores.shape[0], 1)
    _module(
        "topk_checked", torch.cuda.get_device_capability(scores.device), out.shape[1]
    ).forward(scores, starts, ends, pages, out, fallback)
    return bf16_radix_topk(scores, starts, ends, out, rows_to_compute=fallback)
