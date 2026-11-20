"""Request-relative FP4 scoring over bounded groups of local Prefill requests."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_prefill_indexer as sparse
from rtp_llm.models_py.modules.dsv41.fp8._indexer_score import fp8_fp4_mqa_indexer_score
from rtp_llm.models_py.modules.dsv41.fp8._v41_grouped_prefill_score import _mask_tail_kernel
from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import PrefillIndexerKeys
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload
from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_hierarchical_topk as hierarchical
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.prefill import _quantized_indexer_query
from rtp_llm.models_py.modules.dsv41.ppu.topk import select_tokens


# Bound the live BF16 logits slab and the concatenated FP4 keys separately.
_SCORE_BYTES = 1 << 30
_MAX_BANK_ROWS = 1 << 20
_KEY_ALIGNMENT = 256


def _align(value, multiple):
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class ScoreGroup:
    first_request: int
    rows: slice
    counts: tuple
    offsets: tuple
    bank_rows: int
    width: int
    tile_rows: int

    def tiles(self):
        for start in range(self.rows.start, self.rows.stop, self.tile_rows):
            yield slice(start, min(start + self.tile_rows, self.rows.stop))


def score_groups(request_slices, counts, rows, heads, *, sparse_width=0, score_bytes=_SCORE_BYTES):
    """Partition by packed K capacity and padded score bytes, in request order."""
    if (not request_slices or len(request_slices) != len(counts)
            or heads not in (32, 64) or any(not 0 < n <= _MAX_BANK_ROWS for n in counts)):
        return None
    cursor = 0
    for span in request_slices:
        if span.step not in (None, 1) or span.start != cursor or not cursor < span.stop <= rows:
            return None
        cursor = span.stop
    if cursor != rows:
        return None

    groups = []
    first = 0
    alignment = 128 // heads
    while first < len(counts):
        last, bank_rows, offsets = first, 0, []
        while last < len(counts) and bank_rows + _align(counts[last], _KEY_ALIGNMENT) <= _MAX_BANK_ROWS:
            offsets.append(bank_rows)
            bank_rows += _align(counts[last], _KEY_ALIGNMENT)
            last += 1
        group_counts = tuple(counts[first:last])
        width = sparse_width or max(group_counts)
        tile_rows = score_bytes // (2 * _align(width, 512)) // alignment * alignment
        if tile_rows == 0:
            return None
        if sparse_width:
            tile_rows = min(tile_rows, sparse.MAX_CHUNK_ROWS)
        groups.append(ScoreGroup(
            first, slice(request_slices[first].start, request_slices[last - 1].stop),
            group_counts, tuple(offsets), offsets[-1] + group_counts[-1],
            width, tile_rows,
        ))
        first = last
    return tuple(groups)


def prepare(cache, positions, request_slices, counts, heads, *, sparse_width=0):
    """Use the caller's creation-proved local row ranges on the eager PPU path."""
    if (type(cache) is not SharedNativeCache or not cache._is_prefill or cache.sharded
            or request_slices is None or torch.is_grad_enabled()
            or not positions.is_cuda or not is_ppu()
            or torch.cuda.is_current_stream_capturing()):
        return None
    return score_groups(request_slices, counts, positions.numel(), heads, sparse_width=sparse_width)


def _pack_keys(cache, source, group):
    banks = [
        cache.read_indexer_fp4(source, rid, global_entries=count)
        for rid, count in enumerate(group.counts, group.first_request)
    ]
    if len(banks) == 1:
        return PrefillIndexerKeys(*banks[0])
    payload, scales = [], []
    for (keys, sf), start, count, stop in zip(
        banks, group.offsets, group.counts, (*group.offsets[1:], group.bank_rows),
    ):
        padding = stop - start - count
        payload.append(F.pad(keys, (0, 0, 0, padding)) if padding else keys)
        scales.append(F.pad(sf, (0, padding)) if padding else sf)
    return PrefillIndexerKeys(torch.cat(payload), torch.cat(scales))


def dense_logits(q, keys, weights, starts, visible, width):
    """The provider returns request-relative columns even for a shared K slab."""
    logits = fp8_fp4_mqa_indexer_score(
        q.values, q.scales, keys.quant, keys.scale, weights, starts, starts + visible,
        return_bf16=True, max_seqlen_k=width,
    )
    if logits.dtype != torch.bfloat16 or logits.shape != (visible.numel(), width) or logits.stride(1) != 1:
        raise RuntimeError("unexpected local Prefill FP4 logits layout")
    _mask_tail_kernel[(visible.numel(),)](logits, visible, width, logits.stride(0), 512)
    return logits


def dense_selection_tiles(logits, output, local_token_topk):
    """Keep the fast TopK domain independent of the scorer's tile size."""
    rows, width = logits.shape
    limit = (
        hierarchical.MAX_ROWS
        if local_token_topk and output.shape[1] == hierarchical.K and hierarchical.supports_width(width)
        else rows
    )
    for start in range(0, rows, limit):
        yield slice(start, min(start + limit, rows))


def run(groups, index_q, weights, context, cache, shared, out, *, source, ratio,
        candidates, block, candidate_k, dense, publish, local_token_topk,
        store_dense_topk, sparse_plan):
    """Score each tile once; retain RTP's candidate publication and remapping."""
    scores = []
    for group in groups:
        keys = _pack_keys(cache, source, group)
        owner = context.request_ids[group.rows] - group.first_request
        offsets = _upload(group.offsets, index_q.device, torch.int32).index_select(0, owner)
        counts = _upload(group.counts, index_q.device, torch.int32).index_select(0, owner)
        positions = context.positions[group.rows]
        visible = torch.minimum(((positions + 1) // ratio).clamp_min(0), counts).to(torch.int32)
        for tile in group.tiles():
            local = slice(tile.start - group.rows.start, tile.stop - group.rows.start)
            tile_visible = visible[local]
            q = _quantized_indexer_query(index_q[tile])
            if dense:
                logits = dense_logits(q, keys, weights[tile], offsets[local], tile_visible, group.width)
                output = out[tile, :min(out.shape[1], group.width)]
                publication = candidates[tile] if publish and candidates is not None else None
                for selection in dense_selection_tiles(logits, output, local_token_topk):
                    store_dense_topk(
                        logits[selection], output[selection],
                        publication[selection] if publication is not None else None,
                        tile_visible[selection], block, candidate_k, local_token_topk=local_token_topk,
                    )
            else:
                plan = sparse_plan(
                    shared, ("local_batch", tile.start, tile.stop, source, group.counts, group.offsets),
                    candidates[tile], tile_visible, group.bank_rows, block, ppu=True,
                    row_key_offsets=offsets[local], row_key_counts=counts[local],
                )
                logits = sparse.score(q.values, q.scales, keys, weights[tile], plan)
                if logits is None:
                    raise RuntimeError("local Prefill sparse scorer rejected an admitted tile")
                _mask_tail_kernel[(plan.rows,)](logits, plan.end, plan.sparse_columns, logits.stride(0), 512)
                selected = (select_tokens(logits, plan.end, contract="local", disposable=True)
                            if local_token_topk else None)
                if selected is None:
                    selected = logits.topk(out.shape[1], dim=-1).indices
                if sparse.remap(selected, plan, logits=logits, out=out[tile]) is None:
                    raise RuntimeError("local Prefill sparse remap rejected an admitted tile")
            scores.append((tile.stop - tile.start, group.width))
            del logits
    return scores
