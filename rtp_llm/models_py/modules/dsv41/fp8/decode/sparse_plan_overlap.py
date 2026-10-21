"""Prepare the forward-local reindex plan after candidate publication.

Only candidate IDs/flags, visible lengths and the source block table cross streams;
neither query projections nor KV values are read by the planner.
"""

import os
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_decode as sparse

_PLAN_STREAMS: dict[int, torch.cuda.Stream] = {}


@dataclass
class SparsePlanWork:
    current: torch.cuda.Stream
    auxiliary: torch.cuda.Stream
    inputs: tuple[torch.Tensor, ...]
    plan: sparse.SparseDecodePlan | None = None
    joined: bool = False

    def wait(self):
        if not self.joined:
            self.current.wait_stream(self.auxiliary)
            if self.plan is not None:
                # Outputs were allocated on auxiliary but outlive the join:
                # all four reindex layers consume them on current.
                for tensor in self.plan:
                    tensor.record_stream(self.current)
            self.inputs = ()
            self.joined = True
        return self.plan


def has_consumer(attn):
    """Do not fork a graph unless a local reindex layer will join it."""
    source = attn.layer_id
    return any(
        layer is not None
        and layer.layer_id > source
        and layer.is_index_source
        and layer.decode_sparse_mqa
        and layer.kv_source_layer_id == source
        and layer.compress_ratio == 1
        and (layer.index_n_heads, layer.index_head_dim, layer.index_topk)
        == (32, 128, 512)
        for layer in attn._shared_attention["layers"].values()
    )


def start_plan(attn, keys, lengths, *, query_len, producer_stream=None):
    """Default-on overlap; cold capture and unsupported layouts stay serial.

    Call only after the candidate source has published its complete result.
    The first reindex consumer must wait before using the plan. Warm-up and
    capture run the same helper: replay executes the planner, not this Python
    cache, so changes to candidates/lengths/block tables remain visible.
    """
    shared = attn._shared_attention
    candidates = shared.get("candidates")
    if (
        os.environ.get("DSV41_OVERLAP_SPARSE_PLAN", "1") != "1"
        or attn.layer_id != int(attn.v41_config.get("candidate_source_layer_id", -1))
        or attn.kv_source_layer_id != attn.layer_id
        or attn.compress_ratio != 1
        or int(attn.v41_config.get("candidate_block_size", 0)) != 8
        or candidates is None
        or not sparse.is_supported(candidates)
        or keys.pool.shape[1] != 128
        or keys.logical_entries_per_block != 128
        or not has_consumer(attn)
        or "decode_sparse_plan_ready" in shared
    ):
        return
    from rtp_llm.models_py.modules.dsv41.fp8.attention import (
        _cuda_device_index,
        _get_process_cuda_stream,
    )

    device = candidates.device
    device_index = _cuda_device_index(device)
    current = torch.cuda.current_stream(device)
    with torch.cuda.device(device):
        capturing = torch.cuda.is_current_stream_capturing()
    stream = _PLAN_STREAMS.get(device_index)
    if stream is None:
        if capturing:
            return
        stream = _get_process_cuda_stream(_PLAN_STREAMS, device)
    table = keys.block_table
    candidate_mask = shared.get("candidate_mask")
    flags = sparse._plan_candidate_flags(candidates, table, candidate_mask)
    warm_key = (
        stream.cuda_stream, tuple(candidates.shape), tuple(table.shape),
        table.stride(), lengths.dtype, query_len,
        None if flags is None else tuple(flags.shape),
    )
    warmed = shared.setdefault("decode_sparse_plan_warmed", set())
    # DeepGEMM initializes scheduler workspace once per stream, outside capture.
    if capturing and warm_key not in warmed:
        return
    inputs = (candidates, lengths, table) + (() if flags is None else (flags,))
    work = SparsePlanWork(current, stream, inputs)
    stream.wait_stream(current)
    # FULL publishes candidate tensors while their writes are still queued on
    # its indexer stream. Order the planner directly after that producer,
    # without joining it onto (and blocking) the main attention stream.
    if producer_stream is not None:
        stream.wait_stream(producer_stream)
    try:
        with torch.cuda.stream(stream):
            with record_function_range("dsv41.decode.sparse_mqa_plan_async"):
                work.plan = sparse.make_plan(
                    candidates, lengths, table, query_len=query_len,
                    candidate_mask=candidate_mask,
                )
    except BaseException:
        work.wait()
        raise
    if not capturing:
        warmed.add(warm_key)
    shared["decode_sparse_plan_ready"] = work


def wait_plan(shared):
    work = shared.pop("decode_sparse_plan_ready", None)
    if work is not None:
        shared["decode_sparse_plan"] = work.wait()
