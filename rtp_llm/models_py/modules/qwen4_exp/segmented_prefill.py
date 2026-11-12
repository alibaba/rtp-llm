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
        rows = key[0]
        if rows <= 0 or len(self.keys) >= self.max_entries:
            return False
        if self.tokens + rows > self.max_tokens:
            return False
        self.keys.add(key)
        self.tokens += rows
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
        self.budget = SegmentBudget(
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

    def supports(self, x):
        if not (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape[0] in self.rows
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
                )
            )
            self.decisions[key] = self.budget.admit(key)
        return self.decisions[key]


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
        if not self.admission.supports(x):
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
            free = torch.tensor(
                torch.cuda.mem_get_info(x.device)[0], device=x.device, dtype=torch.int64
            )
            dist.all_reduce(free, op=dist.ReduceOp.MIN)
            if free.item() < self.min_free:
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
                    output = eager(*static)
            finally:
                if self.mega is not None:
                    self.mega._prefill_graph_recording = False
            main.wait_stream(stream)
            self.graphs[key] = (graph, static, output, stream)
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
