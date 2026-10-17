"""Native SM120 FP4 scores using RTP's DeepGEMM provider."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def provider():
    try:
        import deep_gemm
    except ImportError as error:
        raise RuntimeError(
            "FP4 indexer scoring requires DeepGEMM with SM120 FP4 support."
        ) from error
    required = (
        "fp8_fp4_mqa_logits",
        "fp8_fp4_paged_mqa_logits",
        "get_paged_mqa_logits_metadata",
        "get_num_sms",
    )
    missing = [
        name for name in required if not callable(getattr(deep_gemm, name, None))
    ]
    if missing:
        raise RuntimeError(f"DeepGEMM is missing FP4 indexer entry points: {missing}")
    return deep_gemm


def prefill_score(q, weights, packed_k, k_scales, starts, ends, *, clean_logits=False):
    payload, scales = q
    if payload.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("Native FP4 Q requires uint8 payload and UE8M0 scales")
    if payload.ndim != 3 or payload.shape[-1] != 64:
        raise ValueError("Native FP4 Q requires [rows, heads, 64] payload")
    if payload.shape[1] not in (16, 32, 64):
        raise ValueError("Native FP4 Q requires 16, 32 or 64 heads")
    if scales.shape != (*payload.shape[:-1], 4):
        raise ValueError("Native FP4 Q requires four scales per query head")
    if not payload.is_contiguous() or not scales.is_contiguous():
        raise ValueError("Native FP4 Q payload and scales must be contiguous")
    native = provider()
    result = native.fp8_fp4_mqa_logits(
        (payload.view(torch.int8), scales.view(torch.int32).squeeze(-1)),
        (packed_k.view(torch.int8), k_scales.view(torch.int32).squeeze(-1)),
        weights.contiguous(),
        starts.contiguous(),
        ends.contiguous(),
        clean_logits=clean_logits,
    )
    if not isinstance(result, torch.Tensor):
        result = torch.from_dlpack(result)
    return result[: payload.shape[0], : packed_k.shape[0]]


def paged_score(
    q, weights, pool, block_table, context_lens, *, block_size, max_ctx_len
):
    """Score consecutive decode queries against split payload/scale pages.

    Lengths must be nondecreasing within each request, as produced by RTP's
    decode metadata. DeepGEMM schedules all query atoms using the last length.
    Invalid columns are unspecified; the caller's top-k must mask by length.
    """
    payload, scales = q
    if payload.ndim != 4 or payload.shape[-1] != 64:
        raise ValueError("Native FP4 Q requires [batch, queries, heads, 64] payload")
    batch, queries, heads, _ = payload.shape
    if payload.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("Native FP4 Q requires uint8 payload and UE8M0 scales")
    if scales.shape != (*payload.shape[:-1], 4):
        raise ValueError("Native FP4 Q requires four scales per query head")
    if heads not in (16, 32, 64):
        raise ValueError("Native FP4 Q requires 16, 32 or 64 heads")
    if queries < 1:
        raise ValueError("Native FP4 Q requires nonempty queries")
    if block_size != 64 or pool.dtype != torch.uint8 or pool.shape[-1] != 68:
        raise ValueError("Native FP4 cache requires 64-entry, 68-byte pages")
    if pool.numel() % (64 * 68) or not pool.is_contiguous():
        raise ValueError("Native FP4 cache must contain contiguous whole pages")
    if weights.shape != (batch * queries, heads) or weights.dtype != torch.float32:
        raise ValueError("Native FP4 weights must be float32 [batch * queries, heads]")
    if context_lens.shape != (batch, queries) or context_lens.dtype != torch.int32:
        raise ValueError("Native FP4 lengths must be int32 [batch, queries]")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.dtype != torch.int32
        or max_ctx_len < 0
        or max_ctx_len > block_table.shape[1] * block_size
    ):
        raise ValueError("Native FP4 block table must cover each request")
    if not all(t.is_contiguous() for t in (payload, scales, context_lens)):
        raise ValueError("Native FP4 queries, scales and lengths must be contiguous")
    if not batch or not max_ctx_len:
        return torch.empty(
            (batch * queries, max_ctx_len), device=payload.device, dtype=torch.float32
        )
    native = provider()
    metadata = native.get_paged_mqa_logits_metadata(
        context_lens, block_size, native.get_num_sms()
    )
    result = native.fp8_fp4_paged_mqa_logits(
        (payload.view(torch.int8), scales.view(torch.int32).squeeze(-1)),
        pool.view(-1, block_size, 1, 68),
        weights.contiguous(),
        context_lens,
        block_table.contiguous(),
        metadata,
        max_ctx_len,
        clean_logits=False,
    )
    return result if isinstance(result, torch.Tensor) else torch.from_dlpack(result)
