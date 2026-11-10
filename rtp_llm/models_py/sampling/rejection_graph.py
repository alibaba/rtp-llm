"""Bounded, stream-local replay for point-mass draft verification."""

import os
import threading
from collections import OrderedDict
from collections.abc import Callable

import torch

_LIMIT_BYTES = 256 << 20
_entries: OrderedDict[tuple, "_Entry"] = OrderedDict()
_lock = threading.Lock()
# Installed by the C++ sampler bridge; standalone Python keeps the same fallback.
_native_replay: Callable | None = None
_register_native_plan: Callable | None = None
_try_native_guard: Callable | None = None
_release_native_guard: Callable | None = None


class _Entry:
    def __init__(self, inputs, forced, compute, stream, key):
        from rtp_llm.models_py.sampling.adapter import _rejection_sample_logits_eager

        self.stream = stream
        self.key = key[:7]
        self.native_eligible = compute is _rejection_sample_logits_eager
        self.native_plan = None
        capture_stream = torch.cuda.Stream(device=stream.device)
        capture_stream.wait_stream(stream)
        with torch.cuda.stream(capture_stream):
            # Keep the initial tensor for shape/stride metadata and a bounded
            # lifetime, but read each replay's logits through a device address.
            self.inputs = [inputs[0], *[tensor.clone() for tensor in inputs[1:]]]
            logits, ids, seeds, positions, sample = self.inputs[:5]
            cap = self.inputs[5] if len(self.inputs) == 6 else None
            batch, rows, _ = logits.shape
            self.tokens = torch.empty(
                (batch, rows), device=logits.device, dtype=torch.int32
            )
            self.lengths = torch.empty(batch, device=logits.device, dtype=torch.int32)
            self.input_addresses = torch.tensor(
                [t.data_ptr() for t in inputs]
                + ([0] if len(inputs) == 5 else [])
                + [self.tokens.data_ptr(), self.lengths.data_ptr()]
                + [0] * (2 * batch),
                dtype=torch.int64,
                device=logits.device,
            )[:8]

            def run():
                compute(
                    logits,
                    None,
                    ids,
                    seeds,
                    positions,
                    sample,
                    [forced] * batch,
                    cap,
                    output_ids=self.tokens,
                    output_lengths=self.lengths,
                    clean_output=True,
                    graph_input_addresses=self.input_addresses,
                )

            # JIT and allocator warmup must finish before capture. Only this
            # cold entry waits; steady-state replay has no host synchronization.
            run()
            capture_stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            self.graph.capture_begin(capture_error_mode="thread_local")
            try:
                run()
            finally:
                self.graph.capture_end()
        self.ready = torch.cuda.Event()
        self.ready.record(capture_stream)
        # This graph contains only Triton kernels with explicit seed inputs,
        # so replay does not require PyTorch generator-state prologues.
        self.graph_exec = (
            self.graph.raw_cuda_graph_exec()
            if hasattr(self.graph, "raw_cuda_graph_exec")
            else None
        )
        self.ready_event = self.ready.cuda_event
        self.bytes = (
            self.inputs[0].untyped_storage().nbytes()
            + sum(t.numel() * t.element_size() for t in self.inputs[1:])
            + (8 << 20)
        )

    def replay(self, inputs, tokens, lengths):
        if (
            self.native_eligible
            and _native_replay is not None
            and self.graph_exec is not None
        ):
            if self.native_plan is None and _register_native_plan is not None:
                self.native_plan = _register_native_plan(
                    self,
                    self.graph_exec,
                    self.ready_event,
                    self.input_addresses,
                    self.key,
                )
            _native_replay(
                self.graph_exec,
                self.ready_event,
                self.input_addresses,
                inputs,
                tokens,
                lengths,
            )
            return
        # The address upload and replay are ordered on the caller's stream.
        # Kernels write caller-owned outputs directly, with no staging copies.
        # A fresh pinned allocation is retained by PyTorch's copy allocator
        # until DMA completes; never mutate a shared host pointer while in flight.
        addresses = torch.tensor(
            [t.data_ptr() for t in inputs]
            + ([0] if len(inputs) == 5 else [])
            + [tokens.data_ptr(), lengths.data_ptr()],
            dtype=torch.int64,
            pin_memory=True,
        )
        self.input_addresses.copy_(addresses, non_blocking=True)
        for tensor in (*inputs, tokens, lengths):
            tensor.record_stream(self.stream)
        self.graph.replay()
        self.ready.record(self.stream)


def try_replay(
    compute: Callable,
    logits: torch.Tensor,
    ids: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    sample: torch.Tensor,
    forced: list[bool] | None,
    cap: torch.Tensor | None,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
) -> bool:
    if os.environ.get("RTP_LLM_GUMBEL_REJECTION_GRAPH", "1") != "1":
        return False
    if not logits.is_cuda or logits.ndim != 3 or logits.dtype != torch.float32:
        return False
    batch, rows, vocab = logits.shape
    device = logits.device
    if not 8 <= batch <= 128 or not 2 <= rows <= 9:
        return False
    if (
        ids.shape != (batch, rows - 1)
        or ids.dtype != torch.int32
        or seeds.shape != (batch,)
        or seeds.dtype != torch.int64
        or positions.shape != (batch,)
        or positions.dtype != torch.int64
        or sample.shape != (batch,)
        or sample.dtype != torch.bool
        or tokens.shape != (batch, rows)
        or tokens.dtype != torch.int32
        or lengths.shape != (batch,)
        or lengths.dtype != torch.int32
        or tokens.device != device
        or lengths.device != device
        or not tokens.is_contiguous()
        or not lengths.is_contiguous()
    ):
        return False
    if cap is not None and (cap.shape != (batch,) or cap.dtype != torch.int32):
        return False
    forced_value = False
    if forced:
        if len(forced) != batch or forced.count(forced[0]) != batch:
            return False
        forced_value = forced[0]
    inputs = [logits, ids, seeds, positions, sample]
    if cap is not None:
        inputs.append(cap)
    if any(t.device != device or not t.is_contiguous() for t in inputs):
        return False
    if torch.cuda.current_device() != device.index:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    stream = torch.cuda.current_stream(device)
    # Shapes, dtypes and contiguity were checked above; this fully identifies
    # the captured layout without rebuilding a Tensor metadata tuple per input.
    key = (
        device.index,
        stream.cuda_stream,
        forced_value,
        batch,
        rows,
        vocab,
        cap is not None,
        compute,
    )
    if not _lock.acquire(blocking=False):
        return False
    native_guard = False
    try:
        if _try_native_guard is not None:
            native_guard = _try_native_guard()
            if not native_guard:
                return False
        if key not in _entries:
            required = (
                logits.untyped_storage().nbytes()
                + sum(t.numel() * t.element_size() for t in inputs[1:])
                + (8 << 20)
            )
            if required > _LIMIT_BYTES:
                return False
            same_device = [k for k in _entries if k[0] == key[0]]
            used = sum(_entries[k].bytes for k in same_device)
            while same_device and (
                len(same_device) >= 2 or used + required > _LIMIT_BYTES
            ):
                old = same_device.pop(0)
                if not _entries[old].ready.query():
                    return False
                used -= _entries.pop(old).bytes
            _entries[key] = _Entry(inputs, forced_value, compute, stream, key)
        entry = _entries[key]
        _entries.move_to_end(key)
        entry.replay(inputs, tokens, lengths)
        return True
    finally:
        if native_guard:
            _release_native_guard()
        _lock.release()
