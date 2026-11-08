"""Bounded prefill over the framework's original 528B/288B paged pools.

This is an RTP call adapter, not a new cache codec or an FP8 Q/P contract.
The ABI is ``FlashMLA-for-sail`` revision
``4305d1bdaeb8451eb5d78c67a5122ecf38f5014d`` (MIT), interface SHA256
``8861f77da7d7900ffeb60cb2f652176d4342c947d516fbcc0928eba81e952d43``.
Eligibility, physical slots and SWA residency are decided by the shared
coordinator before any cache write. Only Q is chunked here; both KV views
retain the framework allocation and block stride throughout every launch.
"""

import torch
from torch.nn.functional import pad

from rtp_llm.models_py.modules.dsv41.ppu.attention import Q_CHUNK, _pad_heads


def packed_prefill(q, plan, *, softmax_scale, attn_sink):
    """Return ``[rows, real_heads, 512]`` BF16 on the caller's current stream.

    Each row is an independent sparse query. Absolute
    causal/SWA/global visibility is already encoded in the plan's physical
    slot prefixes; the native sparse path therefore requires causal=False.
    Scheduler metadata is fresh per chunk because live lengths may differ.
    Failures propagate: a partially executed native launch is never retried
    through the reference route.
    """
    if q.ndim != 3 or q.dtype != torch.bfloat16 or q.shape[-1] != 512:
        raise ValueError("packed prefill requires BF16 Q [rows, heads, 512]")
    rows = q.shape[0]
    if q.shape[1] <= 0 or plan.valid_mask.shape != (rows,):
        raise ValueError("packed prefill has inconsistent head or live-row geometry")
    if rows == 0:
        return q.new_empty(q.shape)

    q, sink, heads = _pad_heads(q, attn_sink)
    # The H64 BF16 reference uses N64 tiles and combines two N-warps'
    # partial softmax sums. Packed Sq=1 can select N32, which rounds the
    # sum differently as soon as a key reaches the second N-warp's columns.
    # Sq=2 selects packed HS64/N64. Each query retains its own slot list,
    # so the pair only shares a scheduling bound and the original KV views.
    pair_queries = q.shape[1] == 64 and plan.global_cache is not None
    chunk = int(Q_CHUNK) if Q_CHUNK and Q_CHUNK > 0 else rows
    outputs = []
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        queries = q[start:end].contiguous().unsqueeze(1)
        swa_indices = plan.swa_indices[start:end]
        swa_lengths = plan.swa_lengths[start:end]
        global_indices = global_lengths = None
        if plan.global_cache is not None:
            global_indices = plan.global_indices[start:end]
            global_lengths = plan.global_lengths[start:end]
        if pair_queries:
            if (end - start) % 2:
                # A fully masked query fills the odd tail of the last pair.
                queries = pad(queries, (0, 0, 0, 0, 0, 0, 0, 1))
                swa_indices = pad(swa_indices, (0, 0, 0, 0, 0, 1), value=-1)
                global_indices = pad(global_indices, (0, 0, 0, 0, 0, 1), value=-1)
                swa_lengths = pad(swa_lengths, (0, 1))
                global_lengths = pad(global_lengths, (0, 1))
            queries = queries.reshape(-1, 2, q.shape[1], q.shape[2])
            swa_indices = swa_indices.reshape(-1, 2, swa_indices.shape[-1])
            global_indices = global_indices.reshape(-1, 2, global_indices.shape[-1])
            # Packed indices already carry -1 beyond each query's live prefix.
            swa_lengths = swa_lengths.reshape(-1, 2).amax(dim=1)
            global_lengths = global_lengths.reshape(-1, 2).amax(dim=1)
        metadata, _ = plan.metadata_factory()
        output, _ = plan.native(
            q=queries,
            k_cache=plan.swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=metadata,
            num_splits=None,
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=swa_indices,
            attn_sink=sink,
            extra_k_cache=plan.global_cache,
            extra_indices_in_kvcache=global_indices,
            topk_length=swa_lengths,
            extra_topk_length=global_lengths,
        )
        real_output = output.reshape(-1, q.shape[1], q.shape[2])[:end - start, :heads]
        outputs.append(
            torch.where(
                plan.valid_mask[start:end, None, None],
                real_output,
                torch.zeros_like(real_output),
            )
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
