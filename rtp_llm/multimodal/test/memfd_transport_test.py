"""CPU IPC correctness tests and benchmark-only memfd transport.

The baseline is Pool's normal PyTorch tensor reducer. The candidate copies the
same existing CPU storage to memfd and uses multiprocessing's FD broker. Neither
is installed into the serving path. Cancellation/failure cleanup needs a separate
design before production use; these tests exercise successful transfers.
"""

import ctypes
import gc
import mmap
import multiprocessing as mp
import os
import sys
import time
import unittest
from multiprocessing.reduction import DupFd

import torch


def _memfd_create():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    libc.memfd_create.restype = ctypes.c_int
    fd = libc.memfd_create(b"rtp-mm-benchmark", 1)  # MFD_CLOEXEC
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _rebuild_memfd(handle, nbytes, shape, stride, offset, dtype):
    if not nbytes:
        return torch.empty_strided(shape, stride, dtype=dtype)
    fd = handle.detach()
    try:
        buffer = mmap.mmap(
            fd, nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
        )
    finally:
        os.close(fd)
    # frombuffer keeps the mmap alive through the tensor storage. Views must
    # remain usable after the worker exits and after the local variable dies.
    return torch.frombuffer(buffer, dtype=dtype).as_strided(shape, stride, offset)


class _MemfdTensor:
    def __init__(self, tensor):
        self.tensor = tensor

    def __reduce__(self):
        tensor = self.tensor
        nbytes = tensor.untyped_storage().nbytes()
        metadata = (
            nbytes,
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.dtype,
        )
        if not nbytes:
            return _rebuild_memfd, (None, *metadata)
        fd = _memfd_create()
        try:
            os.ftruncate(fd, nbytes)
            with mmap.mmap(
                fd, nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
            ) as buffer:
                destination = torch.frombuffer(buffer, dtype=torch.uint8)
                source = torch.empty(0, dtype=torch.uint8).set_(
                    tensor.untyped_storage(), 0, (nbytes,), (1,)
                )
                try:
                    destination.copy_(source)
                finally:
                    del destination, source
            # Same FD broker as the default torch reducer. This duplicate owns
            # the backing object until the consumer detaches its descriptor.
            handle = DupFd(fd)
        finally:
            os.close(fd)
        return _rebuild_memfd, (handle, *metadata)


class _ReceiveClock:
    def __reduce__(self):
        # Unpickle this before the tensor so receiver mapping is timed too.
        return time.monotonic_ns, ()


class _SendClock:
    def __reduce__(self):
        # Pickle this after the tensor so torch's deferred storage copy counts.
        return int, (time.monotonic_ns(),)


def _receive(receiver_start_ns, tensor, export_end_ns, metadata):
    first = tensor[(0,) * tensor.ndim].item() if tensor.numel() else None
    ready_ns = time.monotonic_ns()
    metadata.update(
        receiver_start_ns=receiver_start_ns,
        ready_ns=ready_ns,
        export_end_ns=export_end_ns,
        first=first,
    )
    return tensor, metadata


class Envelope:
    def __init__(self, tensor, mode, metadata=None):
        self.tensor = tensor
        self.mode = mode
        self.metadata = dict(metadata or {}, producer_ready_ns=time.monotonic_ns())

    def __reduce__(self):
        self.metadata["export_start_ns"] = time.monotonic_ns()
        payload = self.tensor if self.mode == "torch" else _MemfdTensor(self.tensor)
        return _receive, (_ReceiveClock(), payload, _SendClock(), self.metadata)


def initialize_worker(threads=1, worker_cpus=None, ready_barrier=None):
    if worker_cpus:
        index = (mp.current_process()._identity[0] - 1) % len(worker_cpus)
        os.sched_setaffinity(0, [worker_cpus[index]])
    torch.set_num_threads(threads)
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    if ready_barrier is not None:
        ready_barrier.wait(timeout=60)


def video_payload(mode, nhwc_shape, request_id):
    started_ns = time.monotonic_ns()
    value = request_id % 200 + 1
    # Fresh decoder-like allocation for EVERY request; no previously-shared
    # storage or cached handle is reused in either backend.
    native = torch.full(nhwc_shape, value, dtype=torch.uint8)
    native.reshape(-1)[0] = 251
    native.reshape(-1)[-1] = 253
    tensor = native.permute(0, 3, 1, 2)
    return Envelope(
        tensor,
        mode,
        dict(
            request_id=request_id,
            mode=mode,
            shape=tuple(tensor.shape),
            stride=tensor.stride(),
            logical_bytes=tensor.numel(),
            expected_sum=value * (tensor.numel() - 2) + 504,
            worker_pid=os.getpid(),
            worker_cpu_affinity=sorted(os.sched_getaffinity(0)),
            producer_prepare_ms=(time.monotonic_ns() - started_ns) / 1e6,
        ),
    )


def _pattern_payload(mode, dtype, layout):
    base = torch.arange(120, dtype=torch.float32).to(dtype).reshape(4, 5, 6)
    if layout == "permuted":
        tensor = base.permute(2, 0, 1)
    elif layout == "offset":
        tensor = base[1:, ::2, 1::2]
    elif layout == "empty":
        tensor = torch.empty((0, 3), dtype=dtype)
    else:
        tensor = base
    return Envelope(tensor, mode)


def _fd_count():
    return len(os.listdir("/proc/self/fd"))


def _maps_count():
    with open("/proc/self/maps") as maps:
        return sum("memfd:rtp-mm-benchmark" in line for line in maps)


@unittest.skipUnless(sys.platform == "linux", "Linux memfd")
class MemfdTransportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        context = mp.get_context("spawn")
        ready = context.Barrier(3)
        cls.pool = context.Pool(
            2, initializer=initialize_worker, initargs=(1, None, ready)
        )
        ready.wait(timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()
        cls.pool.join()

    def test_values_dtype_shape_stride_and_offset(self):
        for dtype in (torch.uint8, torch.float32, torch.bfloat16):
            for layout in ("contiguous", "permuted", "offset", "empty"):
                with self.subTest(dtype=dtype, layout=layout):
                    expected = _pattern_payload("torch", dtype, layout).tensor
                    for mode in ("torch", "memfd"):
                        actual, _ = self.pool.apply_async(
                            _pattern_payload, (mode, dtype, layout)
                        ).get(30)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.stride(), expected.stride())
                        self.assertEqual(
                            actual.storage_offset(), expected.storage_offset()
                        )

    def test_concurrent_requests_are_distinct(self):
        tasks = [
            self.pool.apply_async(video_payload, (mode, (2, 64, 64, 3), i))
            for i in range(16)
            for mode in ("torch", "memfd")
        ]
        for task in tasks:
            tensor, metadata = task.get(30)
            self.assertEqual(tensor.sum().item(), metadata["expected_sum"])
            self.assertEqual(tuple(tensor.shape), metadata["shape"])
            self.assertEqual(tensor.stride(), metadata["stride"])

    def test_receiver_outlives_producer(self):
        pool = mp.get_context("spawn").Pool(1, initializer=initialize_worker)
        try:
            tensor, _ = pool.apply_async(
                _pattern_payload, ("memfd", torch.uint8, "offset")
            ).get(30)
        finally:
            pool.close()
            pool.join()
        expected = _pattern_payload("torch", torch.uint8, "offset").tensor
        self.assertTrue(torch.equal(tensor, expected))
        tensor.fill_(17)
        self.assertTrue(torch.all(tensor == 17).item())

    def test_repeated_transfers_release_mappings_and_fds(self):
        counts = []
        for _ in range(3):
            for _ in range(8):
                task = self.pool.apply_async(
                    _pattern_payload, ("memfd", torch.uint8, "offset")
                )
                tensor, _ = task.get(30)
                del tensor, task
            gc.collect()
            # AsyncResult becomes ready just before Pool's result handler drops
            # its last reference; let the handler and FD-broker cleanup finish.
            time.sleep(0.05)
            counts.append(_fd_count())
            self.assertEqual(_maps_count(), 0)
        self.assertEqual(len(set(counts)), 1, counts)


if __name__ == "__main__":
    unittest.main()
