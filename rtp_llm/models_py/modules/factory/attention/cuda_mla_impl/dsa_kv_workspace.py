"""Model-owned DSA scratch; target/draft and mutable Graph metadata stay isolated."""

import math
from dataclasses import dataclass
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _element_size(dtype):
    return torch.empty((), dtype=dtype, device="cpu").element_size()


def _bytes(shape, dtype):
    return math.prod(shape) * _element_size(dtype)


def indexer_candidate_workspace_shapes(
    rows, columns, topk, *, prefix, native, needs_block_output=True
):
    """Reserve only the chosen selector's buffers, before execution/capture."""
    from rtp_llm.models_py.modules.dsv4.indexer_topk import (
        native_candidate_selection_supported,
    )

    use_native = native and native_candidate_selection_supported(
        columns, topk, prefix=prefix
    )
    shapes = {}
    if use_native:
        output_prefix = prefix + ".native"
    else:
        shapes[prefix + ".keys"] = ((rows, columns), torch.int64)
        for name in ("selected_keys", "selected_slots"):
            shapes[prefix + "." + name] = ((rows, min(topk, columns)), torch.int64)
        output_prefix = prefix
    # merge_candidates supplies running as its output. Only block selection
    # owns a separate pair; allocating merge outputs would double-charge it.
    if prefix == "block" and needs_block_output:
        shapes[output_prefix + ".scores"] = ((rows, topk), torch.float32)
        shapes[output_prefix + ".positions"] = ((rows, topk), torch.int32)
    return shapes


def staged_workspace_layout(stages):
    """Reuse disjoint phase buffers in dtype-specific, capture-stable arenas."""
    if not stages:
        return {}, {}, {}
    shared = set(stages[0]).intersection(*(set(s) for s in stages[1:]))
    common = {n: stages[0][n] for n in shared}
    pools, views = {}, {}
    definitions = {}
    for stage in stages:
        offsets = {}
        for name, (shape, dtype) in stage.items():
            definition = (shape, dtype)
            if name in definitions and definitions[name] != definition:
                raise ValueError("DSA stage buffer definitions must agree")
            definitions[name] = definition
            if name in common:
                continue
            count = math.prod(shape)
            offset = offsets.get(dtype, 0)
            views[name] = (dtype, offset, count)
            offsets[dtype] = offset + count
        for dtype, count in offsets.items():
            pools[dtype] = max(pools.get(dtype, 0), count)
    return common, pools, views


def staged_workspace_bytes(stages):
    common, pools, _ = staged_workspace_layout(stages)
    return sum(_bytes(*v) for v in common.values()) + sum(
        _bytes((count,), dtype) for dtype, count in pools.items()
    )


class DsaPhaseWorkspace:
    """Grow only during prepare; every execution lease is allocation-free."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.buffers = {}
        self.arenas = {}
        self.staged_views = {}
        self.graph_capacity = None
        self.graph_geometry = {}
        self.metadata_budget_bytes = 0

    @property
    def resident_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (*self.buffers.values(), *self.arenas.values())
        )

    def required_staged_bytes(self, stages):
        common, pools, _ = staged_workspace_layout(stages)
        total = self.required_bytes(common)
        for dtype, count in pools.items():
            old = self.arenas.get(dtype)
            total += max(
                0,
                _bytes((count,), dtype)
                - (old.numel() * old.element_size() if old is not None else 0),
            )
        return total

    def reserve_stages(self, stages, *, forbid_realloc=False):
        common, pools, views = staged_workspace_layout(stages)
        self.reserve(common, forbid_realloc=forbid_realloc)
        for dtype, count in pools.items():
            old = self.arenas.get(dtype)
            if old is None or old.numel() < count:
                if forbid_realloc:
                    raise ValueError("DSA Graph stage arena cannot grow")
                self.arenas[dtype] = torch.empty(count, dtype=dtype, device=self.device)
        self.staged_views = {
            name: self.arenas[dtype][offset : offset + count]
            for name, (dtype, offset, count) in views.items()
        }

    def required_bytes(self, shapes):
        total = self.resident_bytes
        for name, (shape, dtype) in shapes.items():
            old = self.buffers.get(name)
            if old is not None and old.dtype != dtype:
                raise ValueError("a DSA workspace cannot change a buffer dtype")
            total += max(
                0,
                _bytes(shape, dtype)
                - (old.numel() * old.element_size() if old is not None else 0),
            )
        return total

    def reserve(self, shapes, *, forbid_realloc=False):
        for name, (shape, dtype) in shapes.items():
            count = math.prod(shape)
            old = self.buffers.get(name)
            if old is None or old.numel() < count:
                if forbid_realloc:
                    raise ValueError(f"DSA Graph workspace cannot grow: {name}")
                self.buffers[name] = torch.empty(count, dtype=dtype, device=self.device)
            elif old.dtype != dtype:
                raise ValueError("a DSA workspace cannot change a buffer dtype")

    def __call__(self, name, shape, dtype):
        old = self.staged_views.get(name, self.buffers.get(name))
        count = math.prod(shape)
        if old is None or old.numel() < count or old.dtype != dtype:
            raise ValueError(f"DSA buffer was not reserved during prepare: {name}")
        return old[:count].view(shape)

    def release(self):
        self.buffers.clear()
        self.staged_views.clear()
        self.arenas.clear()


@dataclass(frozen=True)
class ExternalActivationBudget:
    """Model-supplied live bytes for global and rank-local token rows."""

    bytes_per_token: int
    indexer_bytes_per_token: int = 0
    local_bytes_per_token: int = 0
    local_token_divisor: int = 1

    def bytes_for(self, tokens: int, *, indexer: bool = False) -> int:
        per_token = self.bytes_per_token
        if indexer:
            per_token += self.indexer_bytes_per_token
        return (
            tokens * per_token
            + (tokens // self.local_token_divisor) * self.local_bytes_per_token
        )


class DsaWorkspaceOwner:
    """One owner per model, shared across its serial DSA layers and prepares."""

    def __init__(
        self,
        *,
        external_activation_budget: ExternalActivationBudget,
        index_heads=32,
        index_dim=128,
        indexer_use_fp8=False,
        sfa_use_fp8=False,
        topk=2048,
        heads=96,
        latent_dim=512,
        suffix_dim=64,
        value_dim=128,
        tp_size=1,
        device="cuda",
        indexer_budget_bytes=None,
        sfa_budget_bytes=None,
        prefill_kv_workspace_budget_bytes=8 * 1024**3,
    ):
        self.index_heads, self.index_dim, self.topk = index_heads, index_dim, topk
        self.indexer_use_fp8 = indexer_use_fp8
        self.sfa_use_fp8 = sfa_use_fp8
        if sfa_use_fp8 and (latent_dim, suffix_dim) != (512, 64):
            raise ValueError(
                "fp8_ds_mla requires 512 latent and 64 physical suffix dimensions"
            )
        self.mega_topk = indexer_use_fp8 and tp_size > 1
        self.mega_a2a = indexer_use_fp8 and tp_size > 1
        if (
            type(prefill_kv_workspace_budget_bytes) is not int
            or prefill_kv_workspace_budget_bytes <= 0
        ):
            raise ValueError(
                "Prefill KV workspace budget must be a positive integer byte count"
            )
        self.prefill_kv_workspace_budget_bytes = prefill_kv_workspace_budget_bytes
        # Only the metadata-free FP8 decode operator consumes a per-row topk_length,
        # and it only pays off when this owner holds a fraction of the row's top-k:
        # with one rank per domain every row still walks all `topk` slots, where the
        # kernel's asymmetric 256/320 query split measures as a small net loss.
        self.sfa_metadata_free = sfa_use_fp8 and tp_size > 1
        self.indexer_score = None
        self.heads, self.latent_dim = heads, latent_dim
        self.suffix_dim, self.value_dim = suffix_dim, value_dim
        self.tp_size = tp_size
        self.external_activation_budget = external_activation_budget
        self.device = torch.device(device)
        from rtp_llm.models_py.modules.dsv4.indexer_topk import (
            native_exact_topk_enabled,
        )

        self.native_exact_topk = native_exact_topk_enabled()
        # Decode retains the original default phase capacities.
        self.budgets = {
            "indexer": (
                8 * 1024**3 if indexer_budget_bytes is None else indexer_budget_bytes
            ),
            "sfa": 12 * 1024**3 if sfa_budget_bytes is None else sfa_budget_bytes,
        }
        if any(n <= 0 for n in self.budgets.values()):
            raise ValueError("DSA phase memory budgets must be positive")
        self.phases = {}
        self.value_weights = {}
        self.prefill_query_cp = False
        self.query_layout = None
        self.producer_layout = None
        self.routing_plan = None
        self.row_exchange = None
        self.route_bytes = 0
        self.route_capacity = 0
        self.route_chunk_rows = 0
        self.route_allocation_granularity = 0
        self.layout_bytes = 0
        self.graph_max_requests = 0
        self.graph_max_queries = 1
        self.graph_cache_groups = 1

    def phase(self, name, *, graph_key=None):
        # Proposal, verify and update Graphs execute serially on the model
        # stream. Reserve scratch for maximum B/Q before the first capture;
        # each bucket retains its own mutable page tables and schedules.
        if graph_key is not None:
            graph_key = "graph"
        key = (name, graph_key)
        if key not in self.phases:
            self.phases[key] = DsaPhaseWorkspace(self.device)
        return self.phases[key]

    def other_resident_bytes(self, phase):
        return sum(
            p.resident_bytes + p.metadata_budget_bytes
            for p in self.phases.values()
            if p is not phase
        )

    def release_eager_buffers(self):
        # A DSA workspace must not inflate a following KDA/MoE activation
        # peak. Keep capacities/plans at model scope, but relinquish eager
        # tensor leases between attention regions. PyTorch's allocator can
        # reuse that storage; captured bucket pointers are never released.
        for (name, graph_key), phase in self.phases.items():
            if graph_key is None:
                phase.release()

    def external_projection_bytes(self, tokens, *, indexer=False):
        return (
            self.external_activation_budget.bytes_for(tokens, indexer=indexer)
            + self.layout_bytes
            + self.route_bytes
        )

    def bind_value_weights(self, layers):
        from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
        from rtp_llm.utils.model_weight import W

        for index, weights in layers:
            local = weights[W.mla_vc]
            if local.dtype != torch.bfloat16 or local.shape != (
                self.heads if self.prefill_query_cp else self.heads // self.tp_size,
                self.latent_dim,
                self.value_dim,
            ):
                raise ValueError(
                    "DSA per-head V-up weights have an incompatible layout"
                )
            self.value_weights[index] = (
                all_gather(local.contiguous(), Group.TP)
                if self.tp_size > 1 and not self.prefill_query_cp
                else local.contiguous()
            )
