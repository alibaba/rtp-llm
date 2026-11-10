"""Bounded, opt-in prefill graph replay of the unchanged SM90 MoE arithmetic.

Each graph owns its input and output. The caller consumes outputs on the same
main stream before another invocation; all layers serialize use of the shared
MegaMoE symmetric workspace. Input copy and first capture remain explicit costs.
Capture errors fail the process; no rank-local collective fallback is attempted.
"""

import logging
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class QwenMegaGraphAdmission:
    """Deterministic model-wide budget shared by all 48 layers on each rank.

    Admission depends only on replicated shapes and fixed budgets, never local
    free memory. All ranks must construct and call the same model in TP order.
    Retained entries are not evicted while downstream consumers may use them.
    """

    def __init__(self):
        self.keys = set()
        self.tokens = 0
        self.max_entries = 16
        self.max_total_tokens = 32768

    def admit(self, key):
        if key in self.keys:
            return True
        tokens = key[-1][0]
        if (
            len(self.keys) >= self.max_entries
            or self.tokens + tokens > self.max_total_tokens
        ):
            return False
        self.keys.add(key)
        self.tokens += tokens
        return True


class QwenMegaPrefillGraph:
    def __init__(self, layer_idx, admission):
        self.layer_idx = layer_idx
        # Exact shapes only: no padding, changed quantization domain or routing.
        self.admission = admission
        self.graphs = {}
        self.stream_id = None

    def supports(self, x):
        return (
            not torch.cuda.is_current_stream_capturing()
            and x.shape[0] <= 4096
            and self.admission.admit((x.device.index, x.dtype, tuple(x.shape)))
        )

    def forward(self, layer, x, eager):
        main = torch.cuda.current_stream(x.device)
        if self.stream_id is None:
            self.stream_id = main.cuda_stream
        if self.stream_id != main.cuda_stream:
            raise RuntimeError(
                "MegaMoE prefill graph requires one serialized main stream"
            )
        key = (x.device.index, x.dtype, tuple(x.shape))
        if key not in self.graphs:
            capture_started = time.perf_counter()
            # Warm all JIT, cuBLAS and TP collective paths before capture.
            eager(x)
            static_input = torch.empty_like(x)
            static_input.copy_(x)
            capture_stream = torch.cuda.Stream(device=x.device)
            capture_stream.wait_stream(main)
            # The actual capture stream also needs its first cuBLAS/collective
            # use outside capture. Join before any graph accesses the workspace.
            with torch.cuda.stream(capture_stream):
                eager(static_input)
            torch.cuda.synchronize(x.device)
            graph = torch.cuda.CUDAGraph()
            layer._sm90_mega_moe._prefill_graph_recording = True
            try:
                with torch.cuda.graph(graph, stream=capture_stream):
                    output = eager(static_input)
            finally:
                layer._sm90_mega_moe._prefill_graph_recording = False
            main.wait_stream(capture_stream)
            self.graphs[key] = (graph, static_input, output, capture_stream)
            logger.info(
                "[QwenMegaPrefillGraph] captured layer=%s rank=%d tokens=%d "
                "graphs=%d capture_ms=%.3f allocated_mib=%.1f reserved_mib=%.1f "
                "peak_allocated_mib=%.1f peak_reserved_mib=%.1f",
                self.layer_idx,
                dist.get_rank(),
                x.shape[0],
                len(self.graphs),
                (time.perf_counter() - capture_started) * 1000,
                torch.cuda.memory_allocated(x.device) / 2**20,
                torch.cuda.memory_reserved(x.device) / 2**20,
                torch.cuda.max_memory_allocated(x.device) / 2**20,
                torch.cuda.max_memory_reserved(x.device) / 2**20,
            )
        graph, static_input, output, _ = self.graphs[key]
        static_input.copy_(x)
        graph.replay()
        return output
