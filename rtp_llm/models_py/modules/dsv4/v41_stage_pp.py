"""Synchronous stage execution over the validated L20 PP frame/delta protocol.

The scheduler supplies the same trusted dispatch on both partners. Complete CP
publication and send/receive tickets remain caller-owned. Plans, allocations,
dispatch metadata and wire buffers must remain stable for each call; exported
buffers stay live until the send completes. No transport or admission is inferred.
"""

import hashlib
import json
from dataclasses import dataclass, replace

import torch

from rtp_llm.config.dsv41_config import V41LayerSources
from rtp_llm.models_py.modules.dsv4.v41_block import V41ResidualState
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41L20Span,
    _cp_rank,
    _integer,
    _spans,
    _tables,
    export_l20_delta,
)
from rtp_llm.models_py.modules.dsv4.v41_indexer import V41IndexResult
from rtp_llm.models_py.modules.dsv4.v41_pp import (
    V41PPFrame,
    pack_v41_pp_frame,
    validate_v41_pp_frame,
)
from rtp_llm.models_py.modules.dsv4.v41_stage import (
    V41Stage,
    V41StageResult,
    _same_cache,
)
from rtp_llm.models_py.modules.dsv4.v41_stage_attention import (
    V41AttentionPlan,
    V41ForwardRows,
    V41Selections,
)

_DISPATCH = "v41_stage_dispatch"
_L20 = V41LayerSources(20, 1, 20, 20, 20)


@dataclass(frozen=True)
class V41PPDispatch:
    invocation_id: str
    cp_rank: int
    round_id: int
    spans: tuple[V41L20Span, ...]
    row_map: torch.Tensor

    def snapshot(self, contract):
        _cp_rank(self.cp_rank)
        _integer(self.round_id, "PP dispatch round")
        spans = _spans(self.spans, contract)
        if (
            type(self.invocation_id) is not str
            or not self.invocation_id
            or not 1 <= contract.max_sequence_length <= 32768
            or any(span.identity.request_id < 0 for span in spans)
        ):
            raise ValueError(
                "V4.1 stage PP dispatch requires a bounded forward identity"
            )
        rows = self.row_map
        if (
            not isinstance(rows, torch.Tensor)
            or rows.device.type != "cpu"
            or rows.dtype != torch.int64
            or rows.ndim != 2
            or rows.shape[1] != 3
            or rows.shape[0] > 128
            or not rows.is_contiguous()
        ):
            raise ValueError("V4.1 stage PP row map requires CPU int64 [T<=128,3]")
        rows = rows.clone()
        for request, position, valid in rows.tolist():
            if valid == 0 and (request, position) == (-1, -1):
                continue
            if valid != 1 or not 0 <= request < len(spans):
                raise ValueError(
                    "V4.1 stage PP row has invalid span or padding identity"
                )
            span = spans[request]
            if not span.start <= position < span.start + span.count:
                raise ValueError("V4.1 stage PP row is outside the dispatched chunk")
        active = rows[rows[:, 2] == 1, :2]
        if active.unique(dim=0).shape[0] != active.shape[0]:
            raise ValueError("V4.1 stage PP dispatch repeats a query")
        return replace(self, spans=spans, row_map=rows)

    def _fingerprint(self, contract):
        manifest = {
            "version": 1,
            "contract": contract.fingerprint.hex(),
            "invocation": self.invocation_id,
            "cp_rank": self.cp_rank,
            "round": self.round_id,
            "spans": [span.wire_values() for span in self.spans],
            "rows": self.row_map.tolist(),
        }
        digest = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
        return torch.tensor(list(digest), dtype=torch.uint8)

    def forward_rows(self, device):
        """Translate validated span indices to stage request IDs, preserving order."""
        requests = [
            self.spans[index].identity.request_id if valid else -1
            for index, _, valid in self.row_map.tolist()
        ]
        return V41ForwardRows(
            self.invocation_id,
            torch.tensor(requests, dtype=torch.int64, device=device),
            self.row_map[:, 1].to(device=device).contiguous(),
            self.row_map[:, 2].to(device=device, dtype=torch.bool),
        ).snapshot()


def _bind_plan(plan, dispatch, caches, block_tables, max_positions):
    if not isinstance(plan, V41AttentionPlan) or plan.kv_source_layer_id != 20:
        raise ValueError("V4.1 stage PP requires an L20 source cache plan")
    for tag, cache, pages in (
        ("hca_kv", plan.global_cache, plan.global_pages),
        ("indexer_kv", plan.index_cache, plan.index_pages),
    ):
        if cache is None or not _same_cache(cache, caches[tag]):
            raise ValueError("V4.1 stage PP plan is not bound to its admitted cache")
        width = (max_positions + 255) // 256
        if not isinstance(pages, torch.Tensor) or pages.shape != (
            len(dispatch.row_map),
            width,
        ):
            raise ValueError("V4.1 stage PP page map does not match its context")
        for row, (index, _, valid) in enumerate(dispatch.row_map.tolist()):
            if not valid:
                continue
            admitted = block_tables[dispatch.spans[index].identity][tag]
            if len(admitted) > width:
                raise ValueError("V4.1 stage PP allocation exceeds its context")
            expected = list(admitted) + [-1] * (width - len(admitted))
            if pages[row].tolist() != expected:
                raise ValueError(
                    "V4.1 stage PP page map differs from admitted allocation"
                )


def export_v41_stage_result(
    result, dispatch, contract, block_tables, *, backend="triton"
):
    """Export the completed L20 result and CP-complete producer cache deltas."""
    dispatch = dispatch.snapshot(contract)
    if (
        not isinstance(result, V41StageResult)
        or result.state.next_layer_id != 21
        or result.hidden is not None
        or not isinstance(result.l20_plan, V41AttentionPlan)
    ):
        raise ValueError("V4.1 PP export requires a completed stage-zero result")
    device = result.state.residual.device
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("V4.1 stage PP export cannot run during capture")
    rows = dispatch.forward_rows(device)
    rows.require_same(result.rows)
    result.selections.require(_L20, rows)
    result.selections.validate()
    plan = result.l20_plan
    plan.validate(_L20, rows, contract.max_sequence_length)
    caches = {"hca_kv": plan.global_cache, "indexer_kv": plan.index_cache}
    tables = {}
    for span in dispatch.spans:
        if span.identity not in block_tables:
            raise ValueError("V4.1 stage PP is missing producer admission")
        tables[span.identity] = _tables(block_tables[span.identity], caches)
    _bind_plan(plan, dispatch, caches, tables, contract.max_sequence_length)
    selected = result.selections.result
    frame = V41PPFrame(
        result.state.residual,
        result.state.previous_pre_mix,
        dispatch.row_map,
        selected.topk,
        selected.candidate_blocks,
        selected.candidate_counts,
    )
    delta = export_l20_delta(
        contract,
        dispatch.cp_rank,
        dispatch.round_id,
        dispatch.spans,
        caches,
        tables,
        backend=backend,
    )
    wire = pack_v41_pp_frame(delta, frame, dispatch.spans, dispatch.row_map)
    wire[_DISPATCH] = dispatch._fingerprint(contract)
    return wire


def run_v41_pp_stage(stage, tensors, receiver, dispatch, plans):
    """Preflight stage one, import both replicas, then execute L21 through L39.

    Calls are serialized with receiver admission/release and Engram shutdown.
    Validation failures leave replicas untouched. Execution failures after import
    require aborting the forward; this adapter does not roll back device writes.
    """
    if (
        not isinstance(stage, V41Stage)
        or stage.role != "prefill"
        or stage.pp_rank != 1
        or stage.dim != 5120
        or stage.max_positions != receiver.contract.max_sequence_length
    ):
        raise ValueError("V4.1 PP execution requires a matching full-width stage one")
    stage._check_execution()
    dispatch = dispatch.snapshot(receiver.contract)
    if dispatch.cp_rank != receiver.cp_rank:
        raise ValueError("V4.1 stage PP dispatch has the wrong CP partner")
    fingerprint = tensors.get(_DISPATCH)
    if (
        not isinstance(fingerprint, torch.Tensor)
        or fingerprint.device.type != "cpu"
        or fingerprint.dtype != torch.uint8
        or fingerprint.shape != (32,)
        or not fingerprint.is_contiguous()
        or not torch.equal(fingerprint, dispatch._fingerprint(receiver.contract))
    ):
        raise ValueError("V4.1 stage PP frame belongs to a different dispatch")
    frame = validate_v41_pp_frame(tensors, receiver, dispatch.spans, dispatch.row_map)
    rows = dispatch.forward_rows(stage.device)
    state = V41ResidualState(frame.hidden_states, frame.prev_pre_mix, 21)
    selections = V41Selections(
        rows,
        20,
        20,
        1,
        V41IndexResult(frame.topk, frame.candidate_blocks, frame.candidate_counts),
    )
    plans = dict(plans)
    state, rows, hashes = stage._prepare_forward(state, rows, plans, None, selections)
    tables = {
        span.identity: receiver.block_tables(span.identity) for span in dispatch.spans
    }
    # Stage preflight requires every L20 consumer to share this cache and page map.
    _bind_plan(plans[21], dispatch, receiver.caches, tables, stage.max_positions)
    receiver.apply(tensors, dispatch.spans)
    for span in dispatch.spans:
        receiver.require_ready(span.identity, span.start + span.count, span.chunk_id)
    return stage._run_prepared(state, rows, plans, hashes, selections)
