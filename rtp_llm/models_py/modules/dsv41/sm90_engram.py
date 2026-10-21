"""Hopper Engram host-table registration.

Imported only by the SM90 execution path.
"""

import fcntl
import logging
import mmap
import os
from pathlib import Path

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
