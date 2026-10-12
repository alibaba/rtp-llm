"""Snapshot maintained stage-zero state before prefill cleanup.

The caller owns dispatch spans, cache allocation leases and transport tickets.
This exporter does not admit worker PP execution or retain native allocations.
"""

from collections.abc import Sequence
from typing import Any

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41L20Span
from rtp_llm.models_py.modules.dsv4.v41_pp import V41PPFrame, _tensor, _validate_frame


def snapshot_native_l20_frame(
    block: Any,
    hidden: torch.Tensor,
    spans: Sequence[V41L20Span],
    expected_rows: torch.Tensor,
) -> V41PPFrame:
    """Clone the current short eager L20 frame, with explicit causal candidates."""
    attention = getattr(block, "attn", None)
    unit = getattr(block, "ffn_hc", None)
    if (
        getattr(block, "layer_id", None) != 20
        or not isinstance(attention, AttentionV41FP8)
        or not isinstance(unit, DelayedHCUnit)
        or attention.layer_id != 20
        or attention.cache_layer_id != 20
        or attention.tp_size != 1
        or attention.compress_ratio != 1
        or attention.kv_source_layer_id != 20
        or attention.index_source_layer_id != 20
        or not attention.is_kv_source
        or not attention.is_index_source
        or attention._cache_layouts != dsv41_page_layouts(0)
    ):
        raise NotImplementedError(
            "Native L20 export requires its maintained stage-zero producer"
        )
    shared = attention._shared_attention
    config = attention.v41_config
    if (
        set(shared.get("layers", ())) != set(range(21))
        or shared["layers"].get(20) is not attention
        or config.get("candidate_source_layer_id") != 20
        or config.get("candidate_topk_blocks") != 2048
        or config.get("candidate_block_size") != 8
        or shared.get("global", {}).get(20) is None
        or "candidates" not in shared
        or shared["candidates"] is not None
    ):
        raise NotImplementedError(
            "Native L20 export requires current all-causal stage state"
        )
    if not isinstance(hidden, torch.Tensor) or not hidden.is_cuda:
        raise NotImplementedError("Native L20 export requires SM120 CUDA residuals")
    device = hidden.device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing() or torch.cuda.get_device_capability(
            device
        ) != (12, 0):
            raise NotImplementedError(
                "Native L20 export requires eager SM120 execution"
            )
    spans = tuple(spans)
    if (
        not 1 <= len(spans) <= 2
        or any(not isinstance(span, V41L20Span) for span in spans)
        or len({span.identity.request_id for span in spans}) != len(spans)
        or any(span.start + span.count > 256 for span in spans)
    ):
        raise NotImplementedError(
            "Native L20 export requires distinct short request spans"
        )
    common = attention._prefill_meta_shared
    if common is None or not common.use_varlen or common.batch_size != len(spans):
        raise ValueError("Native L20 export requires current prefill metadata")
    count = common.seqlen
    _tensor(hidden, "export residual", (count, 4, 5120), torch.bfloat16, device)
    _tensor(
        expected_rows,
        "trusted export rows",
        (count, 3),
        torch.int64,
        torch.device("cpu"),
    )
    _tensor(common.position_ids, "export positions", (count,), torch.int64, device)
    _tensor(common.req_id_per_token, "export requests", (count,), torch.int32, device)
    cp = getattr(attention, "_cp_ctx", None)
    if common.cp_on:
        from rtp_llm.models_py.distributed.collective_torch import Group, _get_group
        from rtp_llm.models_py.modules.dsv4.cp import CPContext

        if not isinstance(cp, CPContext) or cp.cp_size != 4 or cp.kv_cache_sharded:
            raise NotImplementedError("Native L20 export requires unsharded CP4")
        group = _get_group(Group.TP)
        if (
            torch.distributed.get_backend(group) != "nccl"
            or torch.distributed.get_world_size(group) != 4
            or torch.distributed.get_rank(group) != cp.cp_rank
            or cp.chunk_length != count
            or cp.padded_seq_len != 4 * count
        ):
            raise ValueError("Native L20 export disagrees with its registered CP lane")
        lengths, prefixes = cp.input_lengths_global, cp.prefix_lengths
        _tensor(cp.local_is_real, "export real mask", (count,), torch.bool, device)
        _tensor(
            cp.global_positions, "export CP positions", (count,), torch.int64, device
        )
        _tensor(
            cp.req_id_per_token, "export CP requests", (count,), torch.int32, device
        )
        if not torch.equal(common.position_ids, cp.global_positions) or not torch.equal(
            common.req_id_per_token, cp.req_id_per_token
        ):
            raise ValueError("Native L20 export metadata differs from current CP rows")
        real = cp.local_is_real
    else:
        if cp is not None and cp.cp_size != 1:
            raise ValueError("Native L20 export has stale CP metadata")
        lengths, prefixes = common.input_lengths, common.prefix_lengths
        real = torch.ones(count, dtype=torch.bool, device=device)
    if (
        lengths is None
        or prefixes is None
        or tuple(lengths.tolist()) != tuple(span.count for span in spans)
        or tuple(prefixes.tolist()) != tuple(span.start for span in spans)
        or common.seqlen_full != sum(span.count for span in spans)
    ):
        raise ValueError("Native L20 export spans disagree with current requests")
    rows = torch.stack(
        (common.req_id_per_token.long(), common.position_ids, real.long()), -1
    ).cpu()
    rows[~rows[:, 2].bool()] = torch.tensor([-1, -1, 0])
    if not torch.equal(rows, expected_rows):
        raise ValueError("Native L20 export rows disagree with current dispatch")
    premix = _tensor(
        unit.pre_mix_out, "export pre-mix", (count, 4), torch.float32, device
    )
    topk = _tensor(
        shared.get("topk", {}).get(20),
        "export top-k",
        (count, 512),
        torch.int32,
        device,
    )
    blocks = torch.arange(2048, dtype=torch.int32, device=device)
    counts = torch.where(real, common.position_ids // 8 + 1, 0).to(torch.int32)
    candidates = torch.where(
        blocks[None] < counts[:, None], blocks[None], -1
    ).contiguous()
    selections = topk.clone()
    selections[~real] = -1
    frame = V41PPFrame(
        hidden.clone(), premix.clone(), rows.clone(), selections, candidates, counts
    )
    _validate_frame(frame, spans, expected_rows)
    return frame


__all__ = ["snapshot_native_l20_frame"]
