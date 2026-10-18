"""Hopper Engram registration and bounded UVA prefetch.

Imported only by the SM90 execution path.
"""

import fcntl
import logging
import mmap
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _register_shared_host_memory(runtime, pointer, nbytes):
    """Bound concurrent large host registrations across local model ranks.

    On H20/CUDA13, concurrent registration of two full Engram tables by
    eight ranks has failed with cudaErrorInvalidValue. Serialization bounds
    registration concurrency but does not guarantee success: later full-table
    probes still failed on some ranks while registering the second table.
    The creation lock is per table and cannot protect overlapping calls
    for different layers. This lock covers registration only, never lookup.
    """
    path = Path("/tmp") / f"rtp_llm_engram_{os.getuid()}_cuda_register.lock"
    with open(path, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return runtime.cudaHostRegister(pointer, nbytes, 3)  # Portable | Mapped


def _register_shared_host_ranges(runtime, pointer, nbytes, chunk_bytes):
    """Register bounded ranges while retaining a contiguous device view.

    Only the last range may end within a host page. No table bytes move.
    A failed call rolls back all successful ranges; never fall back to CPU.
    The caller must keep the host storage alive until every range is released.
    """
    if nbytes <= 0 or pointer % mmap.PAGESIZE:
        raise ValueError("Engram registration needs nonempty page-aligned storage")
    if chunk_bytes < 0 or chunk_bytes % mmap.PAGESIZE:
        raise ValueError(
            "Engram registration chunk must be a nonnegative page multiple"
        )
    registered = []
    device_base = None
    try:
        for offset in range(0, nbytes, chunk_bytes or nbytes):
            address = pointer + offset
            size = min(chunk_bytes or nbytes, nbytes - offset)
            (result,) = _register_shared_host_memory(runtime, address, size)
            if int(result) != 0:
                _, name = runtime.cudaGetErrorName(result)
                raise RuntimeError(
                    f"Engram cudaHostRegister failed: {result} ({name}); "
                    f"offset={offset}, bytes={size}, total_bytes={nbytes}"
                )
            registered.append(address)
            result, device_pointer = runtime.cudaHostGetDevicePointer(address, 0)
            if int(result) != 0:
                raise RuntimeError(f"Engram chunk device pointer failed: {result}")
            if device_base is None:
                device_base = device_pointer
            if device_pointer != device_base + offset:
                raise RuntimeError(
                    "Engram registered ranges have noncontiguous device addresses"
                )
        return tuple(registered)
    except BaseException:
        for address in reversed(registered):
            (result,) = runtime.cudaHostUnregister(address)
            if int(result) != 0:
                logger.error(
                    "Engram registration rollback failed: ptr=%#x error=%s",
                    address,
                    result,
                )
        raise


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
