"""Model-owned DSA scratch; target/draft and mutable Graph metadata stay isolated."""

import math
from functools import lru_cache

import torch

from rtp_llm.ops import KvCacheDataType


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


class DsaWorkspaceOwner:
    """One owner per model, shared across its serial DSA layers and prepares."""

    def __init__(
        self,
        *,
        index_heads=32,
        index_dim=128,
        indexer_use_fp8=False,
        sfa_use_fp8=False,
        topk=2048,
        heads=96,
        latent_dim=512,
        suffix_dim=64,
        value_dim=128,
        hidden_dim=0,
        q_rank=0,
        nope_dim=128,
        tp_size=1,
        device="cuda",
        attn_res_blocks=0,
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
        self.hidden_dim, self.q_rank, self.tp_size = hidden_dim, q_rank, tp_size
        self.query_heads = heads // tp_size
        self.nope_dim = nope_dim
        if not isinstance(attn_res_blocks, int) or attn_res_blocks < 0:
            raise ValueError("DSA AttnRes block capacity must be a nonnegative integer")
        self.attn_res_blocks = attn_res_blocks
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

    @classmethod
    def from_model(cls, config, parallelism):
        dsa, a = config.k3_runtime_config.dsa, config.attn_config
        block_size = config.k3_runtime_config.attn_res_block_size
        blocks = (config.num_layers + block_size - 1) // block_size if block_size else 0
        owner = cls(
            prefill_kv_workspace_budget_bytes=config.prefill_kv_workspace_budget_bytes,
            index_heads=dsa.index_num_heads,
            index_dim=dsa.index_head_dim,
            indexer_use_fp8=dsa.indexer_use_fp8,
            sfa_use_fp8=a.kv_cache_dtype == KvCacheDataType.FP8,
            topk=dsa.index_topk,
            heads=a.head_num,
            latent_dim=a.kv_lora_rank,
            suffix_dim=a.rope_head_dim,
            value_dim=a.v_head_dim,
            hidden_dim=config.hidden_size,
            q_rank=a.q_lora_rank,
            nope_dim=a.nope_head_dim,
            attn_res_blocks=blocks,
            tp_size=parallelism.tp_size,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        from rtp_llm.ops import RoleType

        cp = parallelism.prefill_cp_config
        owner.prefill_query_cp = (
            parallelism.role_type == RoleType.PREFILL
            and cp.is_enabled()
            and getattr(cp, "preserve_tp_layout", False)
        )
        if owner.prefill_query_cp or parallelism.role_type == RoleType.DECODE:
            owner.query_heads = owner.heads
        return owner

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
        if self.routing_plan is not None:
            # Charge complete local projections before planning history KV
            # chunks. Residual, normalized H, O and reverse destination are
            # separate from the persistent published source.
            rows = tokens
            full = rows * (
                4 * self.hidden_dim * 2
                + 2 * (self.latent_dim + self.suffix_dim) * 2
                + self.heads * (self.nope_dim + self.suffix_dim) * 2
                + 2 * self.heads * self.value_dim * 2
                + self.index_dim * 2
                + self.index_heads * (self.index_dim + 4)
            )
            # Full receive activation and its FP8 encoding if needed, the
            # latent columns retained by Gate, and raw BF16 Indexer Q.
            temporary = rows * (
                3 * self.hidden_dim
                + 4 * ((self.hidden_dim + 127) // 128)
                + 2 * self.q_rank * 2
                + self.index_heads * (2 * self.index_dim + 8)
            )
            anchors = rows * self.attn_res_blocks * self.hidden_dim * 2
            # TopK running arrays belong to Indexer phase specifications;
            # SFA's incoming TopK and absorbed/padded Q belong to its execution
            # planner. Shared projection weights are model storage, not scratch.
            return full + temporary + anchors + self.layout_bytes + self.route_bytes
        # Gathered hidden (or subsequent O-projection), fused QKV/gate, Q
        # latent/norm and unabsorbed Q survive the phase boundary. Include
        # them before choosing KV capacity; they are never Q-chunked here.
        local_heads = (
            self.heads if self.prefill_query_cp else self.heads // self.tp_size
        )
        features = (
            self.hidden_dim
            + 2 * self.q_rank
            + self.latent_dim
            + self.suffix_dim
            + local_heads * self.value_dim
            + self.query_heads * (self.nope_dim + self.suffix_dim)
        )
        if indexer:
            features += (
                self.index_heads * self.index_dim
                + self.index_dim
                + 4 * self.index_heads
            )
        # The decoder allocates every AttnRes anchor before its layer loop.
        # These model-owned BF16 SP rows stay live across both DSA phases;
        # 93-layer capacity must not reuse the four-layer activation estimate.
        local_rows = tokens if self.prefill_query_cp else tokens // self.tp_size
        anchors = local_rows * self.attn_res_blocks * self.hidden_dim * 2
        return tokens * features * 2 + anchors + self.layout_bytes + self.route_bytes

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
