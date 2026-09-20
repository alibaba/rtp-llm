"""V4.1 prefill index scoring over the FP4 index-K pool.

The pool keeps its raw V4.1-Flash FP4 bytes (64B e2m1 payload + packed-UE8M0
int32 scale per entry, planar per block). CP exchanges raw bytes, then
restores logical key order before the dense, causally bounded scorer.
Queries are quantized with the same group-32 UE8M0 FP4 form (the official
indexer-query quantization), so DeepGEMM's MX mode applies both scale sets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch

from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
    FP4_INDEXER_GROUP,
    FP4_INDEXER_HEAD_DIM,
    quantize_rows_fp4,
)


@dataclass(frozen=True)
class PrefillIndexerKeys:
    """Raw FP4 index-K bytes: ``quant`` is the packed payload, ``scale`` the packed-UE8M0 int32 scales."""

    quant: torch.Tensor  # [N, 64] int8 packed e2m1 payload
    scale: torch.Tensor  # [N] int32 packed UE8M0 scales

    def __len__(self) -> int:
        return self.quant.shape[0]


def is_supported(device: torch.device, num_heads: int, head_dim: int) -> bool:
    """Gate unsupported devices/dependencies, never catch execution failures."""
    if (
        os.environ.get("DSV41_FUSED_PREFILL_INDEXER", "1") == "0"
        or torch.device(device).type != "cuda"
        or num_heads not in (32, 64)
        or head_dim != 128
    ):
        return False
    try:
        from ._indexer_score import has_fp8_fp4_mqa_logits
    except ImportError:
        return False
    return has_fp8_fp4_mqa_logits() and torch.cuda.get_device_capability(device)[0] in (
        9,
        10,
    )


def logits_chunk_rows(num_keys: int) -> int:
    """Bound logical FP32 logits to 256 MiB, with at most 4096 queries.

    DeepGEMM additionally pads its row stride by up to 511 keys and aligns
    query rows to its small Q tile. This is independent of the head dimension;
    no [queries, heads, keys] temporary is materialized. The 4096-row query
    batch is the batched source scan: four times fewer scorer launches,
    topk calls and per-chunk glue than the 512-row cap at 16K+ contexts,
    with the same 256 MiB logits bound.
    """
    if num_keys < 0:
        raise ValueError("num_keys must be nonnegative")
    if num_keys == 0:
        return 512
    return max(1, min(4096, (256 * 1024 * 1024) // (4 * num_keys)))


def quantize_indexer_q(q: torch.Tensor):
    """Quantize queries to the official group-32 UE8M0 FP4 form.

    ``q`` is pre-RoPE BF16 [T, H, 128]; returns ``(payload [T, H, 64] int8,
    sf [T, H] int32)``. DeepGEMM's MX mode carries the query scales in ``sf``,
    so head weights are consumed unmodified by the scorer.
    """
    return quantize_rows_fp4(q.contiguous())


def _fp4_rows_torch(x: torch.Tensor):
    """Group-32 UE8M0 FP4 packing in pure torch (pool-free reference).

    Mirrors ``_v41_fp4_triton`` byte-for-byte: power-of-two scales from the
    float32 bit pattern, round-to-nearest-even e2m1 codes, group-0 scale in
    the low byte of the packed int32. Returns ``(payload [N, 64] int8,
    sf [N] int32, values [N, 128] float32)`` where ``values`` are the
    dequantized (fake-quantized) row values.
    """
    rows = int(x.numel()) // FP4_INDEXER_HEAD_DIM
    grouped = x.float().reshape(
        rows, FP4_INDEXER_HEAD_DIM // FP4_INDEXER_GROUP, FP4_INDEXER_GROUP
    )
    amax = grouped.abs().amax(-1).clamp_min(6.0 * (2.0**-126))
    scaled = amax * (1.0 / 6.0)
    bits = scaled.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    scale = (exponent << 23).view(torch.float32)
    sign = ((grouped.contiguous().view(torch.int32) >> 31) & 1).to(torch.uint8)
    normalized = (grouped / scale[:, :, None]).clamp(-6.0, 6.0)
    magnitude = normalized.abs()
    code = (
        (magnitude > 0.25).to(torch.uint8)
        + (magnitude >= 0.75).to(torch.uint8)
        + (magnitude > 1.25).to(torch.uint8)
        + (magnitude >= 1.75).to(torch.uint8)
        + (magnitude > 2.5).to(torch.uint8)
        + (magnitude >= 3.5).to(torch.uint8)
        + (magnitude > 5.0).to(torch.uint8)
    )
    code = code | (sign << 3)
    flat = code.reshape(rows, FP4_INDEXER_HEAD_DIM)
    payload = flat[:, 0::2] | (flat[:, 1::2] << 4)
    sf = (
        exponent[:, 0]
        | (exponent[:, 1] << 8)
        | (exponent[:, 2] << 16)
        | (exponent[:, 3] << 24)
    )
    mag = flat & 7
    normal = torch.exp2((mag >> 1).float() - 1.0) * (1.0 + (mag & 1).float() * 0.5)
    values = torch.where(mag < 2, mag.float() * 0.5, normal)
    values = torch.where(flat.ge(8), -values, values) * scale.repeat_interleave(
        FP4_INDEXER_GROUP, dim=-1
    ).reshape(rows, FP4_INDEXER_HEAD_DIM)
    return payload.to(torch.int8), sf, values


def quantize_indexer_k_reference(k: torch.Tensor):
    """Pool-free reference/warmup boundary: the same group-32 UE8M0 FP4 form."""
    payload, sf, _ = _fp4_rows_torch(k.contiguous())
    return payload, sf


def gather_indexer_keys(
    pool: torch.Tensor,
    slot_mapping_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    count: int,
    reqid: int,
    ratio: int,
    cp_ctx,
    owner_tokens_per_block: int,
) -> PrefillIndexerKeys:
    """Read exact FP4 cache bytes through V4.1's established slot mapping.

    Physical pool page size and CP ownership block size are independent. The
    latter is expressed in compressed entries; the callback maps original
    token positions through the typed pool tables for both ratios 1 and 2.
    """
    from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import gather_indexer_k_fp4

    device = pool.device
    if count == 0:
        return PrefillIndexerKeys(
            torch.empty(0, 64, dtype=torch.int8, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )
    sharded = cp_ctx is not None and cp_ctx.cp_size > 1 and cp_ctx.kv_cache_sharded
    if not sharded:
        ids = torch.arange(count, dtype=torch.long, device=device)
        slots = slot_mapping_fn((ids + 1) * ratio - 1, torch.full_like(ids, reqid))
        return PrefillIndexerKeys(*gather_indexer_k_fp4(pool, slots))

    from ._indexer_cp_assembler import assemble_indexer_k, build_indexer_cp_chunk_plan

    if ratio not in (1, 2) or owner_tokens_per_block % ratio:
        raise ValueError("V4.1 indexer CP ownership must align to its ratio")
    owner_entries = owner_tokens_per_block // ratio
    plan = build_indexer_cp_chunk_plan(
        cp_ctx,
        # A scalar fill stays on the device. torch.tensor([count], device=...)
        # first allocates pageable host storage and synchronizes its H2D copy
        # inside every KV producer layer.
        torch.full((1,), count, dtype=torch.long, device=device),
        block_size=pool.shape[1],
        owner_block_size=owner_entries,
        device=device,
        total_kv_len=count,
    )
    local = torch.arange(plan.total_local_T, dtype=torch.long, device=device)
    logical = (
        local // owner_entries * cp_ctx.cp_size + cp_ctx.cp_rank
    ) * owner_entries + local.remainder(owner_entries)
    # Padding must not read an unallocated virtual page. Map a valid owned row
    # first, then explicitly replace padding slots with the gather skip marker.
    safe_logical = logical.clamp_max(count - 1)
    slots = slot_mapping_fn(
        (safe_logical + 1) * ratio - 1, torch.full_like(local, reqid)
    )
    slots = torch.where(logical < count, slots, -1)
    local_payload, local_sf = gather_indexer_k_fp4(pool, slots)
    payload = torch.empty(count, 64, dtype=torch.int8, device=device)
    sf = torch.empty(count, dtype=torch.int32, device=device)
    assemble_indexer_k(
        plan=plan,
        local_k_quant=local_payload.view(torch.uint8),
        local_k_scale=local_sf.view(torch.uint8).view(-1, 4),
        out_k_quant=payload.view(torch.uint8),
        out_k_scale=sf.view(torch.uint8).view(-1, 4),
    )
    return PrefillIndexerKeys(payload, sf)


def _use_clean_logits_only(device: torch.device) -> bool:
    """Trust the verified SM100 dense-logit cleaner; keep other paths masked.

    The pinned DeepGEMM 2.8.0 (6db6ed3) SM100 implementation writes -inf both
    outside scheduled KV tiles and past each row's end inside visited tiles,
    including zero-length rows and the padded output stride. This applies to
    clean_logits=True with max_seqlen_k=0, as used below. Do not extend this
    gate to a new architecture without checking that complete-write contract.
    """
    return (
        os.environ.get("DSV41_PREFILL_CLEAN_LOGITS_ONLY", "1") != "0"
        and torch.device(device).type == "cuda"
        and torch.cuda.get_device_capability(device)[0] == 10
    )


def score_indexer_chunk(
    q_payload, q_sf, k_payload, k_sf, weights, visible, *, bounds=None
):
    """Fused dot/ReLU/head reduction; keep causal padding at negative infinity."""
    from ._indexer_score import fp8_fp4_mqa_indexer_score

    count = k_payload.shape[0]
    if q_payload.shape[0] == 0 or count == 0:
        return torch.empty(
            q_payload.shape[0], count, dtype=torch.float32, device=q_payload.device
        )
    if bounds is None:
        ends = visible.to(torch.int32).clamp(0, count).contiguous()
        starts = torch.zeros_like(ends)
    else:
        # Forward-local prefill metadata already clamps in int64 before
        # narrowing. Reuse the exact same device bounds for scoring and TopK.
        starts, ends = bounds
    # The scheduler skips future K tiles. Its SM100 cleaner writes every
    # logical output element, including rows with no completed ratio-2 pair.
    logits = fp8_fp4_mqa_indexer_score(
        q_payload,
        q_sf,
        k_payload,
        k_sf,
        weights,
        starts,
        ends,
        clean_logits=True,
        # Positive max_seqlen_k selects DG's compressed-logit layout, which
        # is incompatible with clean_logits. This caller uses ordinary [M,K].
        max_seqlen_k=0,
    )
    if not _use_clean_logits_only(q_payload.device):
        logits.masked_fill_(
            torch.arange(count, device=logits.device)[None] >= ends[:, None],
            -torch.inf,
        )
    return logits
