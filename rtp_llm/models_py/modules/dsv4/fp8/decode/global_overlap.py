"""Forward-local, graph-capturable V4.1 global/QKV stream scheduling."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv4 import _record_tensor
from rtp_llm.models_py.modules.dsv4.attn_type import INDEXER_KV
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_global, _v41_decode_indexer
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)

_GLOBAL_COMPUTE_STREAMS: dict[int, torch.cuda.Stream] = {}


def is_supported(attn, x, positions, req_ids, starts) -> bool:
    if (
        os.environ.get("DSV41_OVERLAP_GLOBAL_QKV", "0") != "1"
        or os.environ.get("MOEDBG", "0") != "0"
        or not attn.is_kv_source
        or attn.compress_ratio not in (1, 2)
        or not x.is_cuda
        or x.ndim != 3
        or not 0 < x.shape[0] * x.shape[1] <= 64
        or _record_tensor.should_record_layer(attn.layer_id)
        or not _v41_decode_global.is_supported(attn, x, positions, req_ids, starts)
    ):
        return False
    # Only the native paged descriptor crosses streams. The dense fallback
    # materializes temporary keys with a different allocator lifetime.
    pool = attn._source_pool(INDEXER_KV)
    table = attn._block_tables_by_type[INDEXER_KV]
    raw_tpb = require_pool_tokens_per_block(attn._kv_cache, region=INDEXER_KV)
    entries = raw_tpb // attn.compress_ratio
    return (
        raw_tpb % attn.compress_ratio == 0
        and pool.is_contiguous()
        and entries in (pool.shape[1], pool.shape[1] // 2)
        and attn._rope_max_seq_len // attn.compress_ratio <= table.shape[1] * entries
        and _v41_decode_indexer.is_supported(
            x.device, pool.shape[1], attn.index_n_heads, attn.index_head_dim
        )
    )


@dataclass
class GlobalDecodeWork:
    current: torch.cuda.Stream
    auxiliary: torch.cuda.Stream
    inputs: tuple[torch.Tensor, ...]
    joined: bool = False

    def finish(self) -> None:
        if not self.joined:
            self.current.wait_stream(self.auxiliary)
            # Inputs belong to current: any allocator reuse is ordered after
            # the queued join. All global outputs are persistent cache pools.
            self.inputs = ()
            self.joined = True


def start_global_decode(attn, x, positions, req_ids, starts) -> GlobalDecodeWork | None:
    if not is_supported(attn, x, positions, req_ids, starts):
        return None
    from rtp_llm.models_py.modules.dsv4.fp8.attention import (
        _cuda_device_index,
        _get_process_cuda_stream,
    )

    device_index = _cuda_device_index(x.device)
    current = torch.cuda.current_stream(x.device)
    with torch.cuda.device(x.device):
        capturing = torch.cuda.is_current_stream_capturing()
    stream = _GLOBAL_COMPUTE_STREAMS.get(device_index)
    if stream is None:
        if capturing:
            return None
        stream = _get_process_cuda_stream(_GLOBAL_COMPUTE_STREAMS, x.device)
    warm_key = (device_index, tuple(x.shape), stream.cuda_stream)
    warmed = getattr(attn, "_global_decode_overlap_warmed", None)
    if capturing and (warmed is None or warm_key not in warmed):
        return None
    work = GlobalDecodeWork(current, stream, (x, positions, req_ids, starts))
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            attn._produce_global_decode(x, positions, req_ids, starts)
    except BaseException:
        work.finish()
        raise
    if not capturing:
        if warmed is None:
            warmed = set()
            attn._global_decode_overlap_warmed = warmed
        warmed.add(warm_key)
    return work
