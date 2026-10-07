"""V4.1 decode index scoring directly from the source layer's packed FP4 cache.

The cache has grouped storage within each block: all 64-byte FP4 payload
rows, then all packed-UE8M0 int32 scales. Its nominal [blocks, entries, 68]
view is not a per-row payload/scale interleave. No full-capacity
dequantization or [T,H,K] score is made; DeepGEMM's MX-mode paged kernel
reads the fused bytes directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class DecodeIndexerKeys:
    """The owning layer's physical cache plus its logical compressed coverage."""

    pool: torch.Tensor
    block_table: torch.Tensor
    capacity: int
    logical_entries_per_block: int


@triton.jit
def _prepare_paged_metadata_kernel(
    logical_ptr,
    physical_ptr,
    table_ptr,
    safe_table_ptr,
    ROWS: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE0: tl.constexpr,
    TABLE_STRIDE1: tl.constexpr,
    LOGICAL: tl.constexpr,
    PHYSICAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(logical_ptr + row, row < ROWS, other=0)
    last = tl.maximum(length - 1, 0)
    extent = (last // LOGICAL) * PHYSICAL + last % LOGICAL + 1
    tl.store(physical_ptr + row, tl.where(length > 0, extent, 0), row < ROWS)
    block = tl.load(
        table_ptr
        + (row // TABLE_COLS) * TABLE_STRIDE0
        + (row % TABLE_COLS) * TABLE_STRIDE1,
        row < TABLE_ROWS * TABLE_COLS,
        other=0,
    )
    # RTP's block<=0 sentinel must never reach DG as a negative address.
    # Block zero is safe to read; the output kernel restores its zero-K score
    # without relying on the contents of its unallocated cache storage.
    tl.store(safe_table_ptr + row, tl.maximum(block, 0), row < TABLE_ROWS * TABLE_COLS)


@triton.jit
def _compact_logits_kernel(
    physical_ptr,
    logical_ptr,
    lengths_ptr,
    safe_table_ptr,
    CAPACITY: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    PHYSICAL_STRIDE: tl.constexpr,
    LOGICAL: tl.constexpr,
    PHYSICAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(lengths_ptr + row)
    physical_col = (col // LOGICAL) * PHYSICAL + col % LOGICAL
    block = tl.load(
        safe_table_ptr + (row // QUERY_LEN) * TABLE_COLS + col // LOGICAL,
        col < CAPACITY,
        other=0,
    )
    visible = (col < CAPACITY) & (col < length)
    value = tl.load(
        physical_ptr + row * PHYSICAL_STRIDE + physical_col,
        visible & (block > 0),
        other=0.0,
    )
    tl.store(
        logical_ptr + row * CAPACITY + col,
        tl.where(visible, value, -float("inf")),
        col < CAPACITY,
    )


def is_supported(
    device: torch.device,
    block_size: int,
    num_heads: int = 32,
    head_dim: int = 128,
) -> bool:
    """Gate known unsupported paths; execution failures must still propagate."""
    if (
        torch.device(device).type != "cuda"
        or num_heads != 32
        or head_dim != 128
        or block_size not in (64, 128)
    ):
        return False
    from ._indexer_score import has_fp8_fp4_paged_mqa_logits

    if not has_fp8_fp4_paged_mqa_logits():
        return False
    major = torch.cuda.get_device_capability(device)[0]
    return major == 10


def prepare_indexer_q(
    q: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_head_dim: int = 64,
):
    """RoPE the queries, then quantize to the group-32 UE8M0 FP4 MX form.

    ``q`` is pre-RoPE BF16 [B,S,32,128], ``weights`` is FP32 [B,S,32] (used
    raw by DeepGEMM's MX mode — the query scales live in the packed SF), and
    ``frequencies`` are complex64 [B*S, rope_head_dim/2]. Returns
    ``(payload [B,S,32,64] int8, sf [B,S,32] int32)``.
    """
    from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import quantize_rows_fp4
    from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb

    b, s, h, d = q.shape
    flat = q.reshape(b * s, h, d).clone()
    apply_rotary_emb(
        flat[..., -rope_head_dim:].unsqueeze(0), freqs_cis.reshape(b * s, -1)
    )
    payload, sf = quantize_rows_fp4(flat)
    return payload.view(b, s, h, d // 2), sf.view(b, s, h)


def prepare_indexer_q_and_weights(
    q: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q from raw head weights and a complete frequency lookup table.

    Unsupported query layouts retain the gather/cast/scale/clone/RoPE/quant
    chain; CUDA/JIT errors in the supported fused path still propagate.
    """
    from ._v41_indexer_q_triton import try_fused_indexer_q

    result = try_fused_indexer_q(q, weights, freqs_cis, positions, rope_head_dim)
    if result is not None:
        return result
    scaled_weights = weights.float() * (q.shape[-1] * q.shape[-2]) ** -0.5
    payload, sf = prepare_indexer_q(
        q, scaled_weights, freqs_cis[positions], rope_head_dim
    )
    return payload, sf, scaled_weights


def score_decode_indexer(
    q: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    pool: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    max_ctx_len: int,
    rope_head_dim: int = 64,
    logical_entries_per_block: int | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return FP32 [B*S,max_ctx_len], or None for an unsupported fast path.

    Pool and block table must come from the same KV source layer. Pool is the
    contiguous, unpadded uint8 [blocks,typed_entries_per_block,68] view over
    the per-block planar FP4 layout (payload plane then packed-UE8M0 scale
    plane); entries are physical cache rows, not raw allocator blocks. With
    uniform V4.1 cache layout, ratio-2 occupies only the first half of every
    physical block; pass that half as ``logical_entries_per_block``. Its scale
    region still follows ALL physical payload rows, so reinterpreting it as a
    half-sized cache block is incorrect. Score the physical extent and
    compact logits. ``context_lens[b,s]`` is the device-side causal visible
    key count for that individual speculative token, in [0,max_ctx_len].
    Static max_ctx_len must fit the table coverage. No device scalar is read
    during capture/replay.

    DeepGEMM leaves columns >= context_lens undefined. The output kernel sets
    them to -inf and restores score zero for visible unallocated table entries
    (block<=0), matching the old zero-K dequantization. Candidate selection,
    short-context ordering and invalid-index semantics stay in the caller.
    Returned logits are already causally masked; no repeated caller mask is
    needed. The compaction kernel never reads DeepGEMM's unwritten columns.

    With positions, weights are the unscaled projection and freqs_cis is the
    full frequency table, enabling fused Q preparation. Without positions,
    preserve the original scaled-FP32-weights/gathered-frequencies contract.
    """
    b, s, h, d = q.shape
    block_size = pool.shape[1]
    logical_entries = (
        block_size if logical_entries_per_block is None else logical_entries_per_block
    )
    if not is_supported(q.device, block_size, h, d):
        return None
    from ._indexer_score import fp8_fp4_paged_indexer_score

    if positions is None:
        q_payload, q_sf = prepare_indexer_q(q, weights, freqs_cis, rope_head_dim)
    else:
        q_payload, q_sf, weights = prepare_indexer_q_and_weights(
            q, weights, freqs_cis, positions, rope_head_dim
        )
    lengths = context_lens.to(torch.int32).contiguous()
    physical_lengths = torch.empty_like(lengths)
    safe_table = torch.empty(block_table.shape, dtype=torch.int32, device=q.device)
    _prepare_paged_metadata_kernel[
        (triton.cdiv(max(b * s, block_table.numel()), 128),)
    ](
        lengths,
        physical_lengths,
        block_table,
        safe_table,
        b * s,
        b,
        block_table.shape[1],
        block_table.stride(0),
        block_table.stride(1),
        logical_entries,
        block_size,
        128,
    )
    physical_capacity = (
        max_ctx_len
        if logical_entries == block_size
        else triton.cdiv(max_ctx_len, logical_entries) * block_size
    )
    logits = fp8_fp4_paged_indexer_score(
        q_payload,
        q_sf,
        pool,
        weights.reshape(b * s, h),
        safe_table,
        physical_lengths,
        block_size,
        physical_capacity,
    )
    compact = torch.empty(b * s, max_ctx_len, dtype=torch.float32, device=q.device)
    _compact_logits_kernel[(b * s, triton.cdiv(max_ctx_len, 1024))](
        logits,
        compact,
        lengths,
        safe_table,
        max_ctx_len,
        s,
        block_table.shape[1],
        logits.stride(0),
        logical_entries,
        block_size,
        1024,
    )
    return compact
