"""V4.1 indexer scoring for the PPU backend.

``index_score`` implements the ``V41BackendOps.index_score`` contract with
rtp-llm's reference numerics (fp8/test/test_v41_fp4_codec.py
``test_deepgemm_mx_score_matches_reference``):

1. dequantize the MX FP4 Q/K rows (products are exact in fp32),
2. per-head fp32 dot -> relu -> **BF16**,
3. multiply by the head weight (BF16) -> **BF16**,
4. accumulate the heads into 4 independent BF16 lanes (head ``h`` goes to
   lane ``h % 4``, one BF16 rounding per accumulate),
5. pairwise BF16 lane sums ``(l0+l2)`` and ``(l1+l3)``, final BF16 sum.

``head_weights`` are applied exactly as given — pre-scaling by
``softmax_scale * n_heads**-0.5`` is the caller's concern.

Causal masking follows rtp's indexer formula: a query at position ``pos``
sees ``visible = min((pos + 1) // ratio, T)`` compressed key entries, where
entry ``i`` covers raw position ``key_positions[i]`` (compressed index
``key_positions[i] // ratio``). Candidate blocks use rtp's decode layout
(``fp8/_v41_decode_topk.py`` + ``attention_v41.py:470-482``): int32 block ids
with ``-1`` for invalid entries, the newest visible block pinned to ``+inf``,
and invisible blocks scored ``-inf``.

Scoring is chunked over query rows and per-block key ranges: the working
buffer is bounded by ``chunk_rows * n_heads * block_size`` floats, never
``M * T``.
"""

# The PPU fast path tile is copied from sglang-for-sail
# python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py (512b03d),
# Apache-2.0.

from __future__ import annotations

import types

import torch

from rtp_llm.models_py.modules.dsv41.backend import V41QuantizedTensor
from rtp_llm.models_py.modules.dsv41.ppu import _quant

DEFAULT_CANDIDATE_BLOCK_SIZE = 64
DEFAULT_ROW_CHUNK = 128

# Sentinel conventions shared with rtp's decode top-k.
NEG_INF = float("-inf")
POS_INF = float("inf")


def _dequant_q(q: V41QuantizedTensor) -> torch.Tensor:
    """V41QuantizedTensor (MX FP4) -> fp32 ``[M, H, 128]``."""
    if q.format not in ("fp4_ue8m0", "mx_fp4"):
        # A plain bf16/fp32 tensor passes through: the rounding chain below
        # then operates on dequantization-free values.
        values = q.values.float()
        if values.dim() != 3 or values.shape[-1] != _quant.INDEXER_HEAD_DIM:
            raise ValueError(
                f"indexer q must be [M, H, {_quant.INDEXER_HEAD_DIM}], got {tuple(values.shape)}"
            )
        return values
    payload = q.values
    sf = q.scales
    if payload.shape[-1] != _quant.INDEXER_HEAD_DIM // 2 or sf is None:
        raise ValueError("indexer q must carry MX FP4 payload [.,.,64] and sf [.,.]")
    return _quant.dequantize_mx_fp4_rows(payload, sf)


def _dequant_k(k: V41QuantizedTensor) -> torch.Tensor:
    """V41QuantizedTensor (MX FP4) -> fp32 ``[T, 128]``."""
    values = k.values.float()
    if k.format in ("fp4_ue8m0", "mx_fp4"):
        if k.scales is None:
            raise ValueError("indexer k must carry MX FP4 scales")
        values = _quant.dequantize_mx_fp4_rows(k.values, k.scales)
    if values.dim() != 2 or values.shape[-1] != _quant.INDEXER_HEAD_DIM:
        raise ValueError(
            f"indexer k must be [T, {_quant.INDEXER_HEAD_DIM}], got {tuple(values.shape)}"
        )
    return values


def _bf16_rounding_reduce(
    dots: torch.Tensor, weights_bf16: torch.Tensor
) -> torch.Tensor:
    """``dots [R, H, S]`` fp32 -> ``[R, S]`` fp32 through the rtp oracle chain.

    relu -> bf16 -> * bf16 weight -> bf16 -> 4-lane bf16 accumulate ->
    pairwise bf16 lane sums. ``weights_bf16`` is ``[R, H]`` bf16.
    """
    values = dots.relu().to(torch.bfloat16).float()
    w = weights_bf16.float()
    lanes = [
        torch.zeros(values.shape[0], values.shape[-1], device=values.device)
        for _ in range(4)
    ]
    for head in range(dots.shape[1]):
        lane = head % 4
        lanes[lane] = (
            (values[:, head] * w[:, head, None] + lanes[lane])
            .to(torch.bfloat16)
            .float()
        )
    even = (lanes[0] + lanes[2]).to(torch.bfloat16).float()
    odd = (lanes[1] + lanes[3]).to(torch.bfloat16).float()
    return (even + odd).to(torch.bfloat16).float()


def _visible_counts(
    positions: torch.Tensor, key_positions: torch.Tensor, ratio: int, total: int
) -> torch.Tensor:
    """rtp causal formula: ``visible = min((pos + 1) // ratio, T)`` per row."""
    pos = positions.to(torch.int64).reshape(-1)
    visible = ((pos + 1) // max(int(ratio), 1)).clamp_max(total)
    return visible


def index_score(
    q: V41QuantizedTensor,
    k: V41QuantizedTensor,
    head_weights: torch.Tensor,
    *,
    layer_id: int,
    candidate_blocks: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
    context: object | None = None,
    block_size: int = DEFAULT_CANDIDATE_BLOCK_SIZE,
    ratio: int | None = None,
    chunk_rows: int = DEFAULT_ROW_CHUNK,
) -> torch.Tensor:
    """Score the indexer for the given query rows.

    Returns ``[M, T]`` logits (``-inf`` beyond the causal visible count) when
    ``candidate_blocks`` is ``None``, otherwise ``[M, selected]`` per-block
    pooled scores following rtp's candidate semantics:

    * block id ``-1`` (invalid top-k entry) -> ``-inf``,
    * the newest visible block ``(visible - 1) // block_size`` -> ``+inf``
      (pinned, exactly as ``select_candidate_blocks`` pins it),
    * otherwise the max token score over the block's *visible* entries, with
      fully invisible blocks scored ``-inf``.

    ``ratio`` defaults to 1; positions come from ``context.positions`` and
    raw key positions from ``key_positions`` (defaults to ``arange(T) * 1``).
    """
    q_hat = _dequant_q(q)  # [M, H, 128] fp32
    k_hat = _dequant_k(k)  # [T, 128] fp32
    rows, heads = q_hat.shape[0], q_hat.shape[1]
    total = k_hat.shape[0]
    weights = head_weights.to(torch.bfloat16).reshape(rows, heads)

    if ratio is None:
        ratio = 1
    positions = None
    if context is not None:
        positions = getattr(context, "positions", None)
    if positions is None:
        positions = torch.arange(rows, device=q_hat.device)
    if key_positions is None:
        # k rows are ratio-compressed entries: row i covers raw tokens
        # [i*ratio, (i+1)*ratio), so its raw position is i*ratio.
        key_positions = torch.arange(total, device=k_hat.device) * max(int(ratio), 1)
    key_compressed = key_positions.to(torch.int64).reshape(-1) // max(int(ratio), 1)
    visible = _visible_counts(positions, key_positions, ratio, total)

    if candidate_blocks is None:
        return _full_logits(
            q_hat,
            k_hat,
            weights,
            key_compressed,
            visible,
            chunk_rows,
        )
    return _candidate_scores(
        q_hat,
        k_hat,
        weights,
        candidate_blocks,
        key_compressed,
        visible,
        block_size,
        chunk_rows,
    )


def _dots_for_entries(
    q_hat: torch.Tensor, k_hat: torch.Tensor, entry_rows: torch.Tensor
) -> torch.Tensor:
    """Per-head fp32 dots of the row-chunk ``q_hat`` against gathered K rows.

    ``entry_rows``: ``[R, S]`` long tensor of K row indices (per query row its
    own key entries); out of-range rows clamp to the last row and are masked
    by the caller.
    """
    total = k_hat.shape[0]
    safe = entry_rows.clamp(0, max(total - 1, 0))
    # Explicit head dim: reshape(R, S, -1) is ambiguous when S == 0.
    gathered = k_hat[safe.reshape(-1)].reshape(
        entry_rows.shape[0], entry_rows.shape[1], k_hat.shape[1]
    )
    # [R, H, S] = sum_d q[r, h, d] * k[r, s, d]
    return torch.einsum("rhd,rsd->rhs", q_hat, gathered)


def _full_logits(
    q_hat,
    k_hat,
    weights,
    key_compressed,
    visible,
    chunk_rows,
) -> torch.Tensor:
    rows, total = q_hat.shape[0], k_hat.shape[0]
    out = torch.full((rows, total), NEG_INF, dtype=torch.float32, device=q_hat.device)
    for start in range(0, rows, max(chunk_rows, 1)):
        end = min(start + max(chunk_rows, 1), rows)
        entry_rows = torch.arange(total, device=q_hat.device).expand(end - start, total)
        dots = _dots_for_entries(q_hat[start:end], k_hat, entry_rows)
        logits = _bf16_rounding_reduce(dots, weights[start:end])
        # Causal: entry i visible iff compressed index key_positions[i]//ratio
        # is below the row's visible count (rtp's `columns < visible`).
        vis = visible[start:end, None]
        mask = key_compressed[None, :] < vis
        logits = torch.where(mask, logits, torch.full_like(logits, NEG_INF))
        out[start:end] = logits
    return out


def _candidate_scores(
    q_hat,
    k_hat,
    weights,
    candidate_blocks,
    key_compressed,
    visible,
    block_size,
    chunk_rows,
) -> torch.Tensor:
    rows, total = q_hat.shape[0], k_hat.shape[0]
    blocks = candidate_blocks.to(torch.int64).reshape(rows, -1)
    count = blocks.shape[1]
    nblocks = (total + block_size - 1) // block_size
    out = torch.full((rows, count), NEG_INF, dtype=torch.float32, device=q_hat.device)
    block_size = int(block_size)
    arange_bs = torch.arange(block_size, device=q_hat.device)
    for column in range(count):
        target = blocks[:, column]
        # Entry row range of each row's candidate block, clipped to [0, T].
        begin = (target * block_size).clamp_min(0)
        end = (begin + block_size).clamp_max(total)
        # Compute the pooled max over each row's block in row chunks.
        for start in range(0, rows, max(chunk_rows, 1)):
            stop = min(start + max(chunk_rows, 1), rows)
            begin_c = begin[start:stop, None]
            end_c = end[start:stop, None]
            entry_rows = begin_c + arange_bs[None, :]
            in_range = entry_rows < end_c
            dots = _dots_for_entries(q_hat[start:stop], k_hat, entry_rows)
            logits = _bf16_rounding_reduce(dots, weights[start:stop])
            # Causal visibility: entry visible iff compressed idx < visible.
            vis = visible[start:stop, None]
            entry_visible = (
                key_compressed[entry_rows.clamp_max(max(total - 1, 0))] < vis
            )
            pooled = torch.where(
                in_range & entry_visible, logits, torch.full_like(logits, NEG_INF)
            ).amax(dim=-1)
            valid_block = (target[start:stop] >= 0) & (target[start:stop] < nblocks)
            pooled = torch.where(valid_block, pooled, torch.full_like(pooled, NEG_INF))
            newest = (visible[start:stop] - 1).clamp_min(0) // block_size
            pinned = (
                valid_block & (visible[start:stop] > 0) & (target[start:stop] == newest)
            )
            pooled = torch.where(pinned, torch.full_like(pooled, POS_INF), pooled)
            out[start:stop, column] = pooled
    return out


# ---------------------------------------------------------------------------
# PPU fast path: the sglang-for-sail Triton scoring tile (copied).
#
# ``fp4_index_logits_decode`` scores straight from the planar index-K pool
# table with rtp's bf16 rounding points (bf16 dot -> relu -> * bf16 weight ->
# bf16 -> sum -> bf16). L3 wiring calls it directly when it owns the pool
# table; ``index_score`` above stays the portable reference.
# ---------------------------------------------------------------------------

_LOGITS_KERNELS: object | None = None


def _logits_kernels():
    global _LOGITS_KERNELS
    if _LOGITS_KERNELS is not None:
        return _LOGITS_KERNELS
    import triton
    import triton.language as tl

    ns = types.SimpleNamespace(triton=triton, tl=tl)
    INDEX_K_PAYLOAD_BYTES = 64
    INDEX_K_SCALE_BYTES = 4

    @triton.jit
    def _e2m1_decode(code):
        e = (code >> 1) & 3
        m = (code & 1).to(tl.float32)
        sub = m * 0.5
        nor = (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32))
        v = tl.where(e == 0, sub, nor)
        return tl.where((code >> 3) == 1, -v, v)

    @triton.jit
    def _fp4_index_logits_tile(
        q_ptr,
        w_ptr,
        table_ptr,
        b,
        slot,
        valid,
        page_size,
        row_stride,
        stride_qb,
        stride_qh,
        stride_wb,
        H: tl.constexpr,
        HALF_D: tl.constexpr,
        PAYLOAD_BYTES: tl.constexpr,
        SCALE_BYTES: tl.constexpr,
    ):
        offs_h = tl.arange(0, H)
        offs_i = tl.arange(0, HALF_D)
        page = slot // page_size
        off = slot % page_size
        row_base = page * row_stride
        pay = tl.load(
            table_ptr
            + row_base[:, None]
            + off[:, None] * PAYLOAD_BYTES
            + offs_i[None, :],
            mask=valid[:, None],
            other=0,
        )
        low = _e2m1_decode(pay & 0x0F)
        high = _e2m1_decode((pay >> 4) & 0x0F)
        # e8m0 block scales: element j uses block j // 32 -> byte i uses block i // 16.
        sc_idx = offs_i // 16
        exps = tl.load(
            table_ptr
            + row_base[:, None]
            + page_size * PAYLOAD_BYTES
            + off[:, None] * SCALE_BYTES
            + sc_idx[None, :],
            mask=valid[:, None],
            other=127,
        )
        scale = tl.exp2(exps.to(tl.float32) - 127.0)
        k_low = (low * scale).to(tl.bfloat16)
        k_high = (high * scale).to(tl.bfloat16)

        q_even = tl.load(
            q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + 2 * offs_i[None, :]
        )
        q_odd = tl.load(
            q_ptr
            + b * stride_qb
            + offs_h[:, None] * stride_qh
            + 2 * offs_i[None, :]
            + 1
        )

        acc = tl.dot(q_even, tl.trans(k_low))
        acc += tl.dot(q_odd, tl.trans(k_high))
        # reference rounding points: bf16 dot -> relu -> * bf16 weight -> bf16 -> sum -> bf16
        s = acc.to(tl.bfloat16).to(tl.float32)
        s = tl.maximum(s, 0.0)
        w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
        s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
        logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
        logit = tl.where(valid, logit, float("-inf"))
        return logit

    @triton.jit
    def _fp4_index_logits_kernel(
        q_ptr,
        w_ptr,
        slots_ptr,
        lens_ptr,
        table_ptr,
        out_ptr,
        L,
        page_size,
        row_stride,
        stride_qb,
        stride_qh,
        stride_wb,
        H: tl.constexpr,
        HALF_D: tl.constexpr,
        PAYLOAD_BYTES: tl.constexpr,
        SCALE_BYTES: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        tl.assume(L > 0)
        if H > 1:
            tl.assume(stride_qh == HALF_D * 2)
        b = tl.program_id(0)
        lb = tl.program_id(1)
        offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
        n_vis = tl.load(lens_ptr + b)
        if lb * BLOCK_L >= n_vis:
            tl.store(out_ptr + b * L + offs_l, float("-inf"), mask=offs_l < L)
        else:
            valid = offs_l < tl.minimum(n_vis, L)
            slot = tl.load(slots_ptr + b * L + offs_l, mask=offs_l < L, other=0).to(
                tl.int64
            )
            logit = _fp4_index_logits_tile(
                q_ptr,
                w_ptr,
                table_ptr,
                b,
                slot,
                valid,
                page_size,
                row_stride,
                stride_qb,
                stride_qh,
                stride_wb,
                H,
                HALF_D,
                PAYLOAD_BYTES,
                SCALE_BYTES,
            )
            tl.store(out_ptr + b * L + offs_l, logit, mask=offs_l < L)

    ns.index_logits_tile = _fp4_index_logits_tile
    ns.index_logits_kernel = _fp4_index_logits_kernel
    ns.INDEX_K_PAYLOAD_BYTES = INDEX_K_PAYLOAD_BYTES
    ns.INDEX_K_SCALE_BYTES = INDEX_K_SCALE_BYTES
    _LOGITS_KERNELS = ns
    return ns


def fp4_index_logits_decode_ppu(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """PPU scoring kernel copied from sglang-for-sail (fp4_indexer.py).

    ``q`` [B, H, 128] bf16, ``weights`` [B, H], ``slots`` [B, L] int64,
    ``lens`` [B], ``table`` = the layer's planar index-K pool (uint8 2D).
    Returns ``[B, L]`` fp32 logits, ``-inf`` at positions >= lens.
    """
    ns = _logits_kernels()
    assert q.dtype == torch.bfloat16 and q.shape[-1] == _quant.INDEXER_HEAD_DIM
    batch, heads, _ = q.shape
    length = slots.shape[1]
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    slots = slots.contiguous()
    out = torch.empty((batch, length), dtype=torch.float32, device=q.device)
    if length == 0:
        return out
    block_l = 64
    grid = (batch, ns.triton.cdiv(length, block_l))
    ns.index_logits_kernel[grid](
        q,
        weights,
        slots,
        lens.to(torch.int64).contiguous(),
        table,
        out,
        length,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        H=heads,
        HALF_D=_quant.INDEXER_HEAD_DIM // 2,
        PAYLOAD_BYTES=ns.INDEX_K_PAYLOAD_BYTES,
        SCALE_BYTES=ns.INDEX_K_SCALE_BYTES,
        BLOCK_L=block_l,
        num_warps=4,
    )
    return out
