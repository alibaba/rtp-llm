"""SGLang's FP4 Indexer kernels with RTP-owned launch and cache adapters."""

from functools import lru_cache

import torch
from rtp_llm.platforms.ppu.kernels.cuda.sglang_jit import load_sglang_kernel


@lru_cache(maxsize=None)
def _module(kind, arch, parameter=0):
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
        "store": (
            "fused_norm_rope_v2.cuh",
            f"FusedNormRopeKernel<fp32_t,128,64,{parameter},{pdl}>::forward_fp4",
        ),
        "topk": ("topk_prefill_bf16.cuh", "TopKPrefillBF16Kernel::transform"),
        "topk_decode": ("topk_decode_v1.cuh", f"TopKKernel<{pdl}>::transform"),
    }
    filename, symbol = choices[kind]
    source = f"csrc/deepseek_v4/{filename}"
    if kind in ("store", "topk", "compress_decode", "topk_decode"):
        source = f"compat/{filename}"
    flags = []
    if kind in ("compress", "compress_decode"):
        flags += ["-use_fast_math"]
    if kind in ("topk", "topk_decode"):
        flags += [f"-DSGL_TOPK={parameter}"]
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


def topk_decode(scores, lengths, out):
    """SG FP32 Decode selection, returning logical compressed-token indices."""
    if scores.dtype != torch.float32 or lengths.dtype != torch.int32:
        raise ValueError("Decode TopK requires FP32 scores and int32 lengths")
    if out.shape[1] not in (512, 1024) or scores.shape[1] >= 2**31:
        raise ValueError("Decode TopK requires K=512/1024 and int32 score indices")
    if not scores.shape[0]:
        return out
    pages = _identity_page(scores.device).expand(scores.shape[0], 1)
    _module(
        "topk_decode", torch.cuda.get_device_capability(scores.device), out.shape[1]
    ).forward(scores, lengths, pages, out, 2**31, None)
    return out


def topk_bf16(scores, starts, ends, out):
    if scores.dtype != torch.bfloat16 or scores.shape[1] >= 2**31:
        raise ValueError("BF16 TopK requires BF16 scores with fewer than 2**31 columns")
    # Both frozen SG and this port fail in the PPU >16K two-pass kernel.
    if scores.shape[1] > 16384:
        raise ValueError(
            "PPU SG BF16 TopK above 16384 candidate columns is not qualified"
        )
    if not scores.shape[0]:
        return out
    pages = _identity_page(scores.device).expand(scores.shape[0], 1)
    _module(
        "topk", torch.cuda.get_device_capability(scores.device), out.shape[1]
    ).forward(scores, starts, ends, pages, out, 2**31, None)
    return out
