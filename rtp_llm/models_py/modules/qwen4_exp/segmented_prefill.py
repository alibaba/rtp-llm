"""Opt-in exact-shape static segments around the live attention/cache boundary.

Segments of one exact shape share a pool and stage-specific input slots in
decoder execution order. Outputs must be consumed before the next layer reuses
that stage's slots. Execution uses one serialized stream. No padding, eviction,
attention metadata, cache mutation, or fallback after capture failure is allowed.
"""

import logging
import os
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class SegmentBudget:
    """Model-wide shape budget, deliberately independent of allocator state."""

    def __init__(self, max_entries=4, max_tokens=16896):
        if max_entries < 1 or max_tokens < 1:
            raise ValueError("Segment cache budgets must be positive")
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.keys = set()
        self.tokens = 0

    def admit(self, key):
        if key in self.keys:
            return True
        if not self.can_admit(key):
            return False
        self.keys.add(key)
        self.tokens += key[0]
        return True

    def can_admit(self, key):
        if key in self.keys:
            return True
        rows = key[0]
        if rows <= 0 or len(self.keys) >= self.max_entries:
            return False
        if self.tokens + rows > self.max_tokens:
            return False
        return True


def require_rank_agreement(value):
    """Cold-path agreement only, before any capture containing collectives.

    This path is restricted to TP8/EP8/DP1, so the default world is the model group.
    Device indices differ by rank and must never be part of the wire signature.
    """
    peers = [None] * dist.get_world_size()
    dist.all_gather_object(peers, value)
    if any(peer != value for peer in peers):
        raise RuntimeError("Qwen segmented prefill rank shape/cache disagreement")


class SegmentAdmission:
    def __init__(self):
        # One model-owned admission and residency budget for both graph paths.
        # Only measured hot shapes may capture; transient scheduler splits stay
        # eager even after repeated visits. No online LRU or frequency promotion.
        self.moe_rows = frozenset(
            int(row)
            for row in os.environ.get(
                "QWEN_PREFILL_GRAPH_MOE_ROWS", "96,768,3072,5376"
            ).split(",")
            if row.strip()
        )
        self.budget = SegmentBudget(
            int(os.environ.get("QWEN_PREFILL_GRAPH_MAX_ENTRIES", "8")),
            int(os.environ.get("QWEN_PREFILL_GRAPH_MAX_TOKENS", "26208")),
        )
        self.segment_budget = SegmentBudget(
            int(os.environ.get("QWEN_PREFILL_SEGMENT_MAX_ENTRIES", "4")),
            int(os.environ.get("QWEN_PREFILL_SEGMENT_MAX_TOKENS", "16896")),
        )
        self.rows = frozenset(
            int(row)
            for row in os.environ.get(
                "QWEN_PREFILL_SEGMENT_ROWS", "512,2048,6144,8192"
            ).split(",")
        )
        if not self.rows or any(row <= 32 or row > 8192 for row in self.rows):
            raise ValueError("Segment rows must be exact prefill sizes in [33, 8192]")
        if self.rows & self.moe_rows or any(
            row <= 32 or row > 8192 for row in self.moe_rows
        ):
            raise ValueError(
                "MoE hot rows must be in [33, 8192] and disjoint from segments"
            )
        self.max_memory = (
            int(os.environ.get("QWEN_PREFILL_GRAPH_MAX_MEMORY_MIB", "28672")) * 2**20
        )
        self.capture_reserve = (
            int(os.environ.get("QWEN_PREFILL_GRAPH_CAPTURE_RESERVE_MIB", "1024"))
            * 2**20
        )
        if not 0 < self.capture_reserve <= self.max_memory:
            raise ValueError(
                "Graph capture reserve must be positive and fit the budget"
            )
        self.memory_used = 0
        self.capture_closed = False
        self.denied_moe_rows = set()
        self.decisions = {}
        self.pools = {}
        self.capture_streams = {}
        self.input_slots = {}

    def static_inputs(self, name, inputs):
        # The decoder consumes hc_attn outputs before hc_moe, and consumes the
        # latter before the next layer's hc_attn. Separate stage slots preserve
        # that boundary, including layer 0's residual alias of its input slot.
        # Shared inputs are outside graph pools and stay alive for the model.
        result = []
        for index, tensor in enumerate(inputs):
            key = (name, index, tuple(tensor.shape), tensor.dtype, tensor.device)
            if key not in self.input_slots:
                self.input_slots[key] = torch.empty_like(tensor)
            result.append(self.input_slots[key])
        return tuple(result)

    def capture_resources(self, x):
        # Pool sharing reuses dead capture temporaries between successive
        # layer segments. Retained outputs remain live allocations. Different
        # shapes have separate pools and can be interleaved between forwards.
        key = (x.shape[0], x.dtype, x.device)
        if x.device not in self.capture_streams:
            self.capture_streams[x.device] = torch.cuda.Stream(device=x.device)
        if key not in self.pools:
            self.pools[key] = torch.cuda.graph_pool_handle()
        return self.pools[key], self.capture_streams[x.device]

    def supports(self, x, moe=False):
        if moe and x.shape[0] not in self.moe_rows:
            if x.shape[0] not in self.denied_moe_rows:
                self.denied_moe_rows.add(x.shape[0])
                logger.info(
                    "[QwenPrefillBudget] eager rows=%d reason=not_hot_shape", x.shape[0]
                )
            return False
        if not (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape[0] in (self.moe_rows if moe else self.rows)
            and x.is_contiguous()
            and not x.requires_grad
            and not torch.cuda.is_current_stream_capturing()
        ):
            return False
        key = (x.shape[0], str(x.dtype))
        if key not in self.decisions:
            require_rank_agreement(
                (
                    key,
                    sorted(self.budget.keys),
                    self.budget.tokens,
                    self.budget.max_entries,
                    self.budget.max_tokens,
                    sorted(self.rows),
                    sorted(self.moe_rows),
                    self.segment_budget.max_entries,
                    self.segment_budget.max_tokens,
                    sorted(self.segment_budget.keys),
                    self.segment_budget.tokens,
                    self.max_memory,
                    self.capture_reserve,
                )
            )
            accepted = self.budget.can_admit(key) and (
                moe or self.segment_budget.can_admit(key)
            )
            if accepted:
                self.budget.admit(key)
                if not moe:
                    self.segment_budget.admit(key)
            self.decisions[key] = accepted
        return self.decisions[key]

    def allow_capture(self, device, min_free):
        if self.capture_closed:
            return False
        # memory_used is the previous cold-path all-rank maximum. Replays never
        # snapshot the allocator, synchronize the CPU, or add a collective.
        free = torch.tensor(
            torch.cuda.mem_get_info(device)[0], device=device, dtype=torch.int64
        )
        dist.all_reduce(free, op=dist.ReduceOp.MIN)
        if (
            self.memory_used + self.capture_reserve > self.max_memory
            or free.item() < min_free + self.capture_reserve
        ):
            self.capture_closed = True
            logger.info(
                "[QwenPrefillBudget] closed used_mib=%.1f budget_mib=%.1f",
                self.memory_used / 2**20,
                self.max_memory / 2**20,
            )
            return False
        return True

    def record_capture(self, device):
        pools = {tuple(pool) for key, pool in self.pools.items() if key[2] == device}
        # Charge the full reserved private pools (including inactive blocks),
        # plus retained input storage outside the pools. Other allocator/KV
        # memory is deliberately not mislabelled as graph memory.
        snapshot = torch.cuda.memory_snapshot(include_traces=False)
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in self.input_slots.values()
            if tensor.device == device
        }
        used = graph_memory_bytes(snapshot, pools, sum(storages.values()))
        maximum = torch.tensor(used, device=device, dtype=torch.int64)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        self.memory_used = int(maximum.item())
        logger.info(
            "[QwenPrefillBudget] charged local_mib=%.1f max_rank_mib=%.1f budget_mib=%.1f",
            used / 2**20,
            self.memory_used / 2**20,
            self.max_memory / 2**20,
        )
        # Admission headroom is not an allocator hard limit. A capture larger
        # than the remaining budget fails on every rank, never silently admits
        # an over-budget graph or tries unsafe rank-local eviction/recovery.
        if self.memory_used > self.max_memory:
            raise RuntimeError("Unified prefill graph memory budget exceeded")


def graph_memory_bytes(snapshot, pools, input_bytes):
    total = input_bytes
    for segment in snapshot:
        if "segment_pool_id" not in segment:
            raise RuntimeError("CUDA allocator snapshot lacks graph pool ownership")
        if tuple(segment["segment_pool_id"]) in pools:
            total += segment["total_size"]
    return total


def reuse_moe_input_slot(slot, result):
    """Publish the result after the graph has finished consuming its input.

    The slot is a private copy, shared across decoder layers of this shape.
    Its previous result must be consumed before the next MoE replay. Copying
    here keeps each layer's temporary output reclaimable in the shared pool.
    """
    if (
        not isinstance(result, torch.Tensor)
        or result.shape != slot.shape
        or result.dtype != slot.dtype
        or result.device != slot.device
    ):
        raise RuntimeError("MoE graph output must match its private input slot")
    return slot.copy_(result)


class StaticPrefillSegment:
    def __init__(self, layer_idx, name, admission, mega=None):
        self.layer_idx = layer_idx
        self.name = name
        self.admission = admission
        self.mega = mega
        self.graphs = {}
        self.rejected = set()
        self.stream_id = None
        self.min_free = (
            int(os.environ.get("QWEN_PREFILL_SEGMENT_MIN_FREE_MIB", "2048")) * 2**20
        )
        if self.min_free < 0:
            raise ValueError("Segment memory reserve must be nonnegative")

    def run(self, eager, *inputs):
        x = inputs[0]
        supported = (
            self.admission.supports(x, moe=True)
            if self.name == "moe"
            else self.admission.supports(x)
        )
        if not supported:
            return eager(*inputs)
        # Include optional injection inputs, strides, and all output-affecting
        # tensor metadata. No live batch or mutable Python object is captured.
        key = tuple(
            (tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in inputs
        )
        if any(
            t.requires_grad
            or t.device != x.device
            or not t.is_contiguous()
            or t.dtype != x.dtype
            or t.shape[0] != x.shape[0]
            for t in inputs
        ):
            return eager(*inputs)
        main = torch.cuda.current_stream(x.device)
        if self.stream_id is None:
            self.stream_id = main.cuda_stream
        if self.stream_id != main.cuda_stream:
            raise RuntimeError("Segment replay requires one serialized main stream")
        if key in self.rejected:
            return eager(*inputs)
        if key not in self.graphs:
            if any(old[0][0][0] == x.shape[0] for old in self.graphs):
                raise RuntimeError(
                    "Segment input signature changed within a shared shape pool"
                )
            require_rank_agreement(
                (
                    self.layer_idx,
                    self.name,
                    tuple((s, stride, str(dtype)) for s, stride, dtype, _ in key),
                    len(self.graphs),
                    len(self.rejected),
                    self.min_free,
                )
            )
            if not self.admission.allow_capture(x.device, self.min_free):
                self.rejected.add(key)
                logger.info(
                    "[QwenStaticPrefill] rejected layer=%s segment=%s rows=%s memory",
                    self.layer_idx,
                    self.name,
                    x.shape[0],
                )
                return eager(*inputs)
            begin = time.perf_counter()
            # Warm both actual execution streams. Retain no warm output: it
            # may alias input and is not the result of the triggering forward.
            eager(*inputs)
            static = self.admission.static_inputs(self.name, inputs)
            for dst, src in zip(static, inputs):
                dst.copy_(src)
            pool, stream = self.admission.capture_resources(x)
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                eager(*static)
            torch.cuda.synchronize(x.device)
            graph = torch.cuda.CUDAGraph()
            if self.mega is not None:
                self.mega._prefill_graph_recording = True
            try:
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    # Keep the 3072-row latency-sensitive path copy-free.
                    # Its per-layer outputs use the same shared byte budget;
                    # other MoE shapes reclaim outputs through the input slot.
                    if self.name == "moe" and x.shape[0] != 3072:
                        output = reuse_moe_input_slot(static[0], eager(*static))
                    else:
                        output = eager(*static)
            finally:
                if self.mega is not None:
                    self.mega._prefill_graph_recording = False
            main.wait_stream(stream)
            self.graphs[key] = (graph, static, output, stream)
            self.admission.record_capture(x.device)
            logger.info(
                "[QwenStaticPrefill] captured layer=%s rank=%s segment=%s rows=%s capture_ms=%.3f allocated_mib=%.1f reserved_mib=%.1f",
                self.layer_idx,
                dist.get_rank(),
                self.name,
                x.shape[0],
                (time.perf_counter() - begin) * 1000,
                torch.cuda.memory_allocated(x.device) / 2**20,
                torch.cuda.memory_reserved(x.device) / 2**20,
            )
        graph, static, output, _ = self.graphs[key]
        for dst, src in zip(static, inputs):
            dst.copy_(src)
        graph.replay()
        return output
