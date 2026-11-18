"""Explicit model-owned graph regions, serialized with their live eager work."""

import sys
import threading

import torch

# A failed CUDA drain cannot prove that its allocations are safe to release.
_QUARANTINED = []


class GraphSegment:
    """A captured function writes a caller-owned bridge; no returned pool view."""

    def __init__(self, function, arguments, output, pool, stream):
        self.arguments = arguments
        self.output = output
        self.graph = torch.cuda.CUDAGraph()

        def run():
            result = function(*arguments)
            # Only an identical view already contains the requested result.
            # A different offset/layout in the same storage still needs a copy.
            if not (
                result.data_ptr() == output.data_ptr()
                and result.shape == output.shape
                and result.stride() == output.stride()
                and result.dtype == output.dtype
                and result.device == output.device
            ):
                output.copy_(result)

        current = torch.cuda.current_stream(output.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        stream.synchronize()
        with torch.cuda.graph(self.graph, pool=pool, stream=stream):
            run()
        current.wait_stream(stream)

    def replay(self):
        self.graph.replay()
        return self.output


class SegmentedPrefillGraphRunner:
    """The adapter owns the plan and mathematics; this owner holds its lease.

    The native wrapper still owns H2D staging and the final output transfer.
    prepare() here starts only in forward(), after native H2D has been flushed.
    No fixed bank is modified by the wrapper's speculative prepare call.
    """

    def __init__(self, adapter):
        self.adapter = adapter
        self.lock = threading.Lock()
        self.completion = torch.cuda.Event()
        self.pending = False
        self.failed = False
        self.closed = False
        self.unsafe = False
        self.inflight = None
        try:
            adapter.capture()
            self.completion.record()
            self.pending = True
        except BaseException:
            self.failed = True
            self.unsafe = True
            _QUARANTINED.append(self)
            raise

    def can_run(self, inputs, preparing=False):
        if self.failed or self.closed:
            raise RuntimeError("segmented prefill runner is not reusable")
        return self.adapter.select(inputs, preparing)

    def _wait_for_previous(self):
        if self.failed or self.closed:
            raise RuntimeError("segmented prefill runner is not reusable")
        if self.pending:
            try:
                self.completion.synchronize()
            except BaseException:
                self.failed = True
                self.unsafe = True
                _QUARANTINED.append(self)
                raise
            self.pending = False
            self.inflight = None

    def wait_for_previous(self):
        # Called before native host staging is reused, including on a subsequent
        # decode/eager invocation. A device wait alone cannot protect CPU writes.
        with self.lock:
            self._wait_for_previous()

    def forward(self, inputs):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("segmented prefill invocations must be serialized")
        try:
            self._wait_for_previous()
            self.inflight = inputs
            self.adapter.prepare(inputs)
            # The model's outer forward/finally runs for every invocation. Its
            # eager PLE/QSA/MoE and publication callbacks see current live inputs.
            return self.adapter.forward(inputs)
        except BaseException:
            # State writers may have run. Never restart this call as eager.
            self.failed = True
            raise
        finally:
            original_error = sys.exc_info()[1]
            try:
                self.completion.record()
                self.pending = True
            except BaseException:
                self.failed = True
                self.unsafe = True
                _QUARANTINED.append(self)
                if original_error is None:
                    raise
            finally:
                self.lock.release()

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.unsafe:
                raise RuntimeError(
                    "segmented prefill completion could not be established"
                )
            try:
                if self.pending:
                    self.completion.synchronize()
                self.adapter.close()
                self.inflight = None
                self.closed = True
            except BaseException:
                self.failed = True
                _QUARANTINED.append(self)
                raise
