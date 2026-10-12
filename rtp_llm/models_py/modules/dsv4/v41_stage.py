"""Eager rank-local V4.1 stage loop with explicit caches and forward state.

This assembles numerical execution, not worker/scheduler admission. Caller-owned
plans must already admit request generations, CP visibility, pair predecessors,
SWA retention and (on stage one) completed L20 replica imports. No live CP/PP/PD
transport is inferred here. A failed forward can leave cache writes and must be
aborted by the caller. Graph capture is rejected pending lifecycle integration.
"""

from dataclasses import dataclass, replace

import torch

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models_py.modules.dsv4.v41_block import (
    V41Block,
    V41ResidualState,
    _clean_state,
    readout_v41_residual,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import _overlaps
from rtp_llm.models_py.modules.dsv4.v41_collective import V41CollectiveMoE
from rtp_llm.models_py.modules.dsv4.v41_engram_stage import V41EngramStage
from rtp_llm.models_py.modules.dsv4.v41_projection import _tensor
from rtp_llm.models_py.modules.dsv4.v41_stage_attention import (
    V41AttentionPlan,
    V41ForwardRows,
    V41Selections,
    V41StageAttention,
)


@dataclass(frozen=True)
class V41StageResult:
    state: V41ResidualState
    rows: V41ForwardRows
    selections: V41Selections
    hidden: torch.Tensor | None
    l20_plan: V41AttentionPlan | None = None


def _same_cache(left, right):
    return (
        left.tag == right.tag
        and left.raw.device == right.raw.device
        and left.raw.data_ptr() == right.raw.data_ptr()
        and left.raw.shape == right.raw.shape
        and left.raw.stride() == right.raw.stride()
    )


class V41Stage:
    """Exact 21/19 prefill or 40-layer decode ordering, with before-block Engram.

    Blocks and the externally initialized Engram owner stay on their final
    device. Every call supplies a fresh invocation identity and cache plans.
    The object retains no predecessor, hashes, selections or request history.
    """

    def __init__(
        self,
        blocks,
        config,
        *,
        role,
        pp_rank,
        engram,
        final_norm=None,
        max_positions=32768
    ):
        if not isinstance(config, V41Config):
            raise ValueError("V4.1 stage requires validated checkpoint configuration")
        config.validate()
        if (
            config.text.get("scoring_func") != "sqrtsoftplus"
            or config.text.get("n_shared_experts") != 1
            or config.text.get("rms_norm_eps") != 1e-20
        ):
            raise ValueError(
                "V4.1 stage requires the checkpoint router/shared-expert/norm recipe"
            )
        if (
            config.text["num_hidden_layers"] != 40
            or role not in ("prefill", "decode")
            or type(pp_rank) is not int
            or pp_rank not in ((0, 1) if role == "prefill" else (0,))
        ):
            raise ValueError("V4.1 stage requires 21/19 prefill or 40-layer decode")
        begin, end = (
            ((0, 21) if pp_rank == 0 else (21, 40)) if role == "prefill" else (0, 40)
        )
        blocks = tuple(blocks)
        if len(blocks) != end - begin or any(
            not isinstance(op, V41Block) or op.layer_id != layer
            for op, layer in zip(blocks, range(begin, end))
        ):
            raise ValueError(
                "V4.1 stage blocks must cover their exact ordered global range"
            )
        if type(max_positions) is not int or not 1 <= max_positions <= 32768:
            raise ValueError("V4.1 stage requires a bounded 32K context")
        sources = config.layer_sources()
        dim, device = config.text["hidden_size"], blocks[0].device
        backend = blocks[0].kv.backend
        for op in blocks:
            if (
                op.dim != dim
                or op.device != device
                or op.attention.sources != sources[op.layer_id]
                or op.kv.backend != backend
                or len(op.attention.frequencies) < max_positions
            ):
                raise ValueError(
                    "V4.1 stage block geometry/source/backend/RoPE mismatch"
                )
            router = op.feed_forward.router
            if (
                router.experts != config.text["n_routed_experts"]
                or router.topk != config.text["num_experts_per_tok"]
                or router.gate_temp != config.text.get("gate_temp", 1.0)
                or router.route_scale != config.text["routed_scaling_factor"]
                or router.norm_topk_prob != config.text["norm_topk_prob"]
                or op.feed_forward.shared.inter_dim
                != config.text["moe_intermediate_size"]
                or op.feed_forward.shared.swiglu_limit != config.text["swiglu_limit"]
            ):
                raise ValueError(
                    "V4.1 stage FFN does not match checkpoint routing and expert configuration"
                )
            routed = op.feed_forward.routed
            if isinstance(routed, V41CollectiveMoE) and (
                routed.plan.role != role or routed.plan.pp_rank != pp_rank
            ):
                raise ValueError("V4.1 stage and collective topology disagree")
        owned = tuple(
            layer for layer in config.text["engram_layer_ids"] if begin <= layer < end
        )
        if (
            not isinstance(engram, V41EngramStage)
            or engram.layer_ids != owned
            or (
                owned
                and (
                    engram.device != device
                    or (engram.backend == "eager") != (backend == "eager")
                )
            )
        ):
            raise ValueError("V4.1 stage Engram ownership/device/backend mismatch")
        if end == 40:
            _tensor(final_norm, (dim,), torch.bfloat16, device, "stage final norm")
        elif final_norm is not None:
            raise ValueError("V4.1 stage zero cannot own final readout")
        self.blocks, self.engram, self.final_norm = blocks, engram, final_norm
        self.begin, self.end, self.role, self.pp_rank = begin, end, role, pp_rank
        self.dim, self.device, self.max_positions = dim, device, max_positions
        self.max_rows = min(op.max_rows for op in blocks)
        self.attentions = tuple(
            V41StageAttention(
                op.attention.sources, max_positions=max_positions, backend=backend
            )
            for op in blocks
        )

    def _preflight(self, state, rows, plans, selections, cache_rows=None):
        cache_rows = rows if cache_rows is None else cache_rows
        if rows.valid_rows.device != self.device:
            raise ValueError("V4.1 forward rows and stage must share a device")
        residual, previous = _clean_state(
            state,
            rows.valid_rows,
            dim=self.dim,
            device=self.device,
            max_rows=self.max_rows,
            next_layer_id=self.begin,
        )
        if set(plans) != set(range(self.begin, self.end)):
            raise ValueError("V4.1 forward requires exactly one plan per local layer")
        if self.begin == 21:
            if not isinstance(selections, V41Selections):
                raise ValueError("V4.1 stage one requires imported L20 selections")
            selections.require(self.blocks[0].attention.sources, rows)
            selections.validate()
        elif selections is not None:
            raise ValueError("V4.1 stage zero/decode must start with fresh selections")
        swas, sources = [], {}
        for op in self.blocks:
            routed = op.feed_forward.routed
            if isinstance(routed, V41CollectiveMoE):
                routed.require_admitted()
            plan = plans[op.layer_id]
            if not isinstance(plan, V41AttentionPlan):
                raise ValueError("V4.1 layer requires an explicit attention plan")
            plan.validate(op.attention.sources, cache_rows, self.max_positions)
            if any(_overlaps(plan.swa.raw, prior) for prior in swas):
                raise ValueError("V4.1 layers require disjoint SWA cache allocations")
            swas.append(plan.swa.raw)
            source = plan.kv_source_layer_id
            if source is not None:
                prior = sources.setdefault(source, plan)
                if (
                    not _same_cache(plan.global_cache, prior.global_cache)
                    or not _same_cache(plan.index_cache, prior.index_cache)
                    or not torch.equal(plan.global_pages, prior.global_pages)
                    or not torch.equal(plan.index_pages, prior.index_pages)
                ):
                    raise ValueError(
                        "V4.1 source consumers must share source caches and logical page maps"
                    )
        pools = swas + [
            cache.raw
            for plan in sources.values()
            for cache in (plan.global_cache, plan.index_cache)
        ]
        pools += [
            plan.pair_state.raw
            for plan in plans.values()
            if plan.pair_state is not None
        ]
        protected = [
            state.residual,
            state.previous_pre_mix,
            rows.positions,
            rows.request_ids,
            rows.valid_rows,
            cache_rows.positions,
            cache_rows.request_ids,
            cache_rows.valid_rows,
        ]
        for plan in plans.values():
            protected.extend(
                value
                for value in (
                    plan.swa_writes,
                    plan.swa_reads,
                    plan.global_pages,
                    plan.index_pages,
                    plan.global_writes,
                    plan.index_writes,
                )
                if value is not None
            )
            if plan.pair_metadata is not None:
                protected.extend(vars(plan.pair_metadata).values())
        if selections is not None:
            protected.extend(
                value for value in vars(selections.result).values() if value is not None
            )
        for i, value in enumerate(pools):
            if any(_overlaps(value, other) for other in pools[:i]):
                raise ValueError(
                    "V4.1 cache pools must not alias across layers or tags"
                )
            if any(_overlaps(value, tensor) for tensor in protected):
                raise ValueError(
                    "V4.1 cache pools cannot alias forward inputs or metadata"
                )
        return V41ResidualState(residual, previous, self.begin)

    def _check_execution(self):
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "V4.1 stage capture requires integrated cache and Engram lifetimes"
                    )
            if torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError("V4.1 stage FP32 operations require TF32 disabled")

    def _prepare_forward(
        self, state, rows, plans, windows, selections, cache_rows=None
    ):
        self._check_execution()
        if not isinstance(rows, V41ForwardRows):
            raise ValueError("V4.1 stage requires explicit forward rows")
        rows = rows.snapshot()
        state = self._preflight(state, rows, plans, selections, cache_rows)
        if self.engram.layer_ids:
            if windows is None or windows.shape != (rows.valid_rows.numel(), 4):
                raise ValueError(
                    "V4.1 Engram stage requires one explicit token window per row"
                )
        elif windows is not None:
            raise ValueError("V4.1 stage one must not receive Engram windows")
        hashes = self.engram.prepare(windows)
        self.engram.validate_forward(hashes, rows.valid_rows, dim=self.dim)
        return state, rows, hashes

    def _run_prepared(self, state, rows, plans, hashes, selections, cp_forward=None):
        cache_rows = rows if cp_forward is None else cp_forward.rows
        for op, attention in zip(self.blocks, self.attentions):
            if op.layer_id in self.engram.layer_ids:
                state = replace(
                    state,
                    residual=self.engram.apply(op.layer_id, state.residual, hashes),
                )
            plan = plans[op.layer_id]
            meta = plan.pair_metadata
            if meta is not None:
                meta = replace(
                    meta,
                    positions=cache_rows.positions,
                    valid_rows=cache_rows.valid_rows,
                    request_ids=cache_rows.request_ids,
                )
            frame = op.prepare(
                state,
                rows.positions,
                rows.valid_rows,
                pair_state=plan.pair_state,
                pair_metadata=meta,
                kv_exchange=None if cp_forward is None else cp_forward.exchange,
            )
            attended, selections = attention(
                frame,
                op.attention.sinks,
                plan,
                rows,
                selections,
                read_rows=None if cp_forward is None else cp_forward.local_slice,
            )
            state = op.finish(frame, attended)
        hidden = (
            readout_v41_residual(state, self.final_norm, rows.valid_rows)
            if self.end == 40
            else None
        )
        l20_plan = plans[20] if self.end == 21 else None
        if l20_plan is not None and cp_forward is not None:
            l20_plan = cp_forward.local_plan(l20_plan)
        return V41StageResult(state, rows, selections, hidden, l20_plan)

    def run(self, state, rows, plans, *, windows=None, selections=None):
        state, rows, hashes = self._prepare_forward(
            state, rows, plans, windows, selections
        )
        return self._run_prepared(state, rows, plans, hashes, selections)
