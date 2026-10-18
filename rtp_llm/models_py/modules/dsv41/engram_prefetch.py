"""Shared NVIDIA UVA Engram decode prefetch scheduling."""

import torch


class UvaEngramPrefetch:
    """Bounded all-layer lookup on a graph-capturable auxiliary stream.

    Construct before capture. ``begin`` forks from the calling stream after
    hashes are ready; each consumer waits its own ready event. ``finish`` must
    run in a finally block on that same calling stream, including failures.
    Eager record_stream plus the final join protect allocation lifetime; graph
    capture retains fixed tensor addresses through its private allocation pool.
    This object is request-local in use, not safe for concurrent forwards.
    """

    def __init__(self, device, layer_ids):
        self.device = torch.device(device)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Create UVA prefetch streams before graph capture")
        self.stream = torch.cuda.Stream(device=self.device)
        self.fork = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.ready = {layer_id: torch.cuda.Event() for layer_id in layer_ids}
        self.rows = {}
        self._hashes = ()
        self._origin = None

    def begin(self, lookups):
        if self._origin is not None:
            raise RuntimeError("Previous UVA Engram prefetch has not been joined")
        lookups = tuple(lookups)
        if not lookups:
            return
        # Validate and allocate on the consumer stream before the fork. No
        # large embedding table is copied; only this bounded batch's rows.
        for layer_id, embedding, indices in lookups:
            if layer_id not in self.ready or embedding._uva is None:
                raise RuntimeError(
                    "UVA Engram prefetch requires registered layer tables"
                )
            if not indices.is_cuda or indices.device != self.device:
                raise ValueError(
                    "UVA Engram prefetch hashes must reside on its CUDA device"
                )
        self.rows = {
            layer_id: torch.empty(
                (*indices.shape, embedding.weight.shape[1]),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for layer_id, embedding, indices in lookups
        }
        self._hashes = tuple(indices for _, _, indices in lookups)
        origin = torch.cuda.current_stream(self.device)
        capturing = torch.cuda.is_current_stream_capturing()
        self._origin = origin
        try:
            self.fork.record(origin)
            self.stream.wait_event(self.fork)
            with torch.cuda.stream(self.stream):
                try:
                    for layer_id, embedding, indices in lookups:
                        if not capturing:
                            indices.record_stream(self.stream)
                        embedding(indices, self.device, out=self.rows[layer_id])
                        self.ready[layer_id].record(self.stream)
                finally:
                    # Join all enqueued side work even if a later lookup fails.
                    self.done.record(self.stream)
        except BaseException:
            self.finish()
            raise

    def wait(self, layer_id):
        if self._origin is None or layer_id not in self.rows:
            raise RuntimeError("No active UVA Engram prefetch for this layer")
        current = torch.cuda.current_stream(self.device)
        current.wait_event(self.ready[layer_id])
        rows = self.rows[layer_id]
        if not torch.cuda.is_current_stream_capturing():
            rows.record_stream(current)
        return rows

    def finish(self):
        if self._origin is None:
            return
        try:
            current = torch.cuda.current_stream(self.device)
            if current != self._origin:
                raise RuntimeError("Join UVA Engram prefetch on its originating stream")
            current.wait_event(self.done)
        finally:
            self.rows = {}
            self._hashes = ()
            self._origin = None
