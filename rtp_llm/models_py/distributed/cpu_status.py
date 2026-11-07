"""Local TP host consensus for speculative transaction status.

Only startup uses Torch distributed collectives. Each inference rendezvous
publishes the host boolean through acquire/release atomics in shared memory,
following vLLM's host publication design. Every original transaction phase is
retained. A timeout poisons the mapping and propagates without GPU fallback.
"""

import ctypes
import mmap
import os
import stat
import tempfile
import threading

import torch.distributed as dist


class CpuTpStatus:
    def __init__(self, ops, mapping, rank, timeout_ms=60000, world_size=2):
        if (
            not 0 <= rank < world_size
            or not ops.cpu_status_size(world_size)
            or len(mapping) < ops.cpu_status_size(world_size)
            or timeout_ms <= 0
        ):
            raise ValueError("Invalid TP status mapping or rank")
        self.ops = ops
        self.mapping = mapping
        self.pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        if self.pointer % 64:
            raise ValueError("Status mapping must be cache-line aligned")
        self.rank = rank
        self.timeout_ms = timeout_ms
        self.generation = 0
        self.closed = False
        self.lock = threading.Lock()

    def reduce(self, local_ok):
        with self.lock:
            if self.closed:
                raise RuntimeError("Status mapping is closed")
            self.generation += 1
            rc = self.ops.cpu_status_reduce(
                self.pointer,
                self.rank,
                self.generation,
                int(bool(local_ok)),
                self.timeout_ms,
            )
            if rc < 0:
                raise RuntimeError(
                    f"TP CPU status failed: errno={-rc}, generation={self.generation}"
                )
            return bool(rc)

    def close(self, cancel=False):
        with self.lock:
            if self.closed:
                return
            if cancel:
                self.ops.cpu_status_break(self.pointer)
            # A peer may still be reading the final successful generation.
            # Its own mapping remains valid after this process unmaps.
            self.closed = True
            self.mapping.close()


def _identity(info):
    return info.st_dev, info.st_ino, info.st_uid, info.st_size


def _open_mapping(path, identity):
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (
            _identity(info) != tuple(identity)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise RuntimeError("TP2 status file identity or mode changed")
        return mmap.mmap(fd, info.st_size)
    finally:
        os.close(fd)


def _unlink_owned(path, identity):
    info = os.stat(path, follow_symlinks=False)
    if _identity(info) != tuple(identity) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError("TP2 status file identity changed before unlink")
    os.unlink(path)


def create_cpu_tp_status(group, ops, timeout_ms=60000):
    """Collective startup; all ranks either map the same file or raise."""
    world_size = dist.get_world_size(group)
    required_bytes = ops.cpu_status_size(world_size)
    if not required_bytes:
        raise ValueError("Unsupported CPU target status group size")
    rank = dist.get_rank(group)
    descriptor = [None]
    if rank == 0:
        path = None
        identity = None
        try:
            fd, path = tempfile.mkstemp(prefix="rtp-tp-status-", dir="/dev/shm")
            try:
                size = max(mmap.PAGESIZE, required_bytes)
                os.ftruncate(fd, size)
                identity = _identity(os.fstat(fd))
                with mmap.mmap(fd, size) as mapped:
                    pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapped))
                    if ops.cpu_status_initialize(pointer, size, world_size) != 0:
                        raise RuntimeError("TP status initialization failed")
            finally:
                os.close(fd)
            descriptor[0] = dict(path=path, identity=identity)
        except Exception as exc:
            if path is not None and identity is not None:
                _unlink_owned(path, identity)
            descriptor[0] = dict(error=str(exc))
    dist.broadcast_object_list(
        descriptor, src=dist.get_global_rank(group, 0), group=group
    )
    data = descriptor[0]
    if "error" in data:
        raise RuntimeError("CPU status startup failed: " + data["error"])
    owner = None
    mapped = None
    error = None
    try:
        mapped = _open_mapping(data["path"], data["identity"])
        owner = CpuTpStatus(ops, mapped, rank, timeout_ms, world_size)
    except Exception as exc:
        error = str(exc)
    errors = [None] * world_size
    dist.all_gather_object(errors, error, group=group)
    # All ranks have opened (or reported failure) before the creator unlinks.
    # Transfer any unlink error too, so rank admission remains collective.
    cleanup = [None]
    if rank == 0:
        try:
            _unlink_owned(data["path"], data["identity"])
        except Exception as exc:
            cleanup[0] = str(exc)
    dist.broadcast_object_list(cleanup, src=dist.get_global_rank(group, 0), group=group)
    if any(errors) or cleanup[0]:
        if mapped is not None:
            mapped.close()
        raise RuntimeError(f"CPU status startup failed: {errors}, cleanup={cleanup[0]}")
    return owner
