"""Manual diagnostic: register two existing full-size shared tables, no copies."""

import contextlib
import ctypes
import mmap
import os
import unittest

import torch
from cuda.bindings import runtime


class EngramRegistrationTest(unittest.TestCase):
    def test_loader_pinned_memory_probe(self):
        """Bound loader failures to registration/allocation, before model load."""
        ranks = int(os.environ.get("ENGRAM_REGISTRATION_RANKS", "1"))
        if ranks > 1:
            ctx = torch.multiprocessing.get_context("spawn")
            barrier = ctx.Barrier(ranks, timeout=180)
            torch.multiprocessing.spawn(
                _probe_loader_rank, args=(barrier,), nprocs=ranks, join=True
            )
        else:
            self._probe_loader_pinned_memory(0)

    def _probe_loader_pinned_memory(self, rank, barrier=None):
        torch.cuda.set_device(rank)
        torch.cuda.synchronize()
        failures = []
        for size in (1024 * 1024, 2 * 1024**3):
            if barrier is not None:
                barrier.wait()
            error, pointer = runtime.cudaHostAlloc(size, 0)
            print(f"RANK={rank} HOST_ALLOC bytes={size} code={int(error)}", flush=True)
            if barrier is not None:
                barrier.wait()
            if int(error):
                failures.append(("alloc", size, int(error)))
                runtime.cudaGetLastError()
            else:
                self.assertEqual(int(runtime.cudaFreeHost(pointer)[0]), 0)
            with mmap.mmap(-1, size) as mapping:
                buffer = (ctypes.c_uint8 * 1).from_buffer(mapping)
                pointer = ctypes.addressof(buffer)
                del buffer
                error = runtime.cudaHostRegister(pointer, size, 2)[0]
                print(
                    f"RANK={rank} HOST_REGISTER bytes={size} code={int(error)}",
                    flush=True,
                )
                if barrier is not None:
                    barrier.wait()
                if int(error):
                    failures.append(("register", size, int(error)))
                    runtime.cudaGetLastError()
                else:
                    self.assertEqual(int(runtime.cudaHostUnregister(pointer)[0]), 0)
        self.assertEqual(failures, [])

    def test_sequential_full_tables(self):
        ranks = int(os.environ.get("ENGRAM_REGISTRATION_RANKS", "1"))
        if ranks > 1:
            ctx = torch.multiprocessing.get_context("spawn")
            lock = (
                ctx.Lock() if os.environ.get("ENGRAM_SERIAL_REGISTER") == "1" else None
            )
            barrier = ctx.Barrier(ranks, timeout=180)
            torch.multiprocessing.spawn(
                _register_rank, args=(lock, barrier), nprocs=ranks, join=True
            )
        else:
            self._register_tables(0)

    def _register_tables(self, rank, lock=None, barrier=None):
        paths = os.environ["ENGRAM_REGISTRATION_PATHS"].split(":")
        self.assertEqual(len(paths), 2)
        flags = int(os.environ.get("ENGRAM_REGISTRATION_FLAGS", "3"))
        chunk_bytes = int(os.environ.get("ENGRAM_REGISTRATION_CHUNK_MB", "0")) * 1024**2
        self.assertGreaterEqual(chunk_bytes, 0)
        if os.environ.get("ENGRAM_REGISTRATION_REVERSE", "0") == "1":
            paths.reverse()
        mappings, tensors, registered = [], [], []
        failures = []
        torch.cuda.set_device(rank)
        torch.cuda.synchronize()
        print(
            f"RANK={rank} CUDA runtime={runtime.cudaRuntimeGetVersion()} driver={runtime.cudaDriverGetVersion()}",
            flush=True,
        )
        for attr in (
            runtime.cudaDeviceAttr.cudaDevAttrHostRegisterSupported,
            runtime.cudaDeviceAttr.cudaDevAttrPageableMemoryAccessUsesHostPageTables,
        ):
            print(
                f"RANK={rank} attr={attr} value={runtime.cudaDeviceGetAttribute(attr, rank)}",
                flush=True,
            )
        try:
            for path in paths:
                with open(path, "r+b") as file:
                    size = os.fstat(file.fileno()).st_size
                    mapping = mmap.mmap(file.fileno(), size)
                mappings.append(mapping)
                tensor = torch.frombuffer(mapping, dtype=torch.uint8)
                tensors.append(tensor)
                ptr = tensor.data_ptr()
                print(
                    f"RANK={rank} REGISTER path={path} ptr={ptr:#x} end={ptr+size:#x} bytes={size} flags={flags} prior={runtime.cudaGetLastError()}",
                    flush=True,
                )
                with open("/proc/self/maps") as maps:
                    for line in maps:
                        if path in line:
                            print(line.strip(), flush=True)
                device_base = None
                # Keep every successful chunk of BOTH tables registered until
                # the final all-rank barrier. This changes per-call size only,
                # not simultaneous coverage or the amount of pinned table data.
                for offset in range(0, size, chunk_bytes or size):
                    length = min(chunk_bytes or size, size - offset)
                    with lock if lock is not None else contextlib.nullcontext():
                        (result,) = runtime.cudaHostRegister(ptr + offset, length, flags)
                    print(
                        f"RANK={rank} RESULT offset={offset} bytes={length} code={int(result)} "
                        f"name={runtime.cudaGetErrorName(result)} message={runtime.cudaGetErrorString(result)}",
                        flush=True,
                    )
                    if int(result) != 0:
                        failures.append((path, offset, int(result)))
                        break
                    registered.append(ptr + offset)
                    self.assertTrue(tensor[offset:].is_pinned())
                    result, device_ptr = runtime.cudaHostGetDevicePointer(ptr + offset, 0)
                    self.assertEqual(int(result), 0)
                    if device_base is None:
                        device_base = device_ptr
                    self.assertEqual(device_ptr, device_base + offset)
                    # Verify both ends of each chunk through the CUDA mapping,
                    # including the table tail and every registration boundary.
                    for local_offset in (0, max(0, length - 16)):
                        host_sample = (ctypes.c_uint8 * 16)()
                        (result,) = runtime.cudaMemcpy(
                            ctypes.addressof(host_sample), device_ptr + local_offset,
                            16, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        )
                        self.assertEqual(int(result), 0)
                        start = offset + local_offset
                        self.assertEqual(bytes(host_sample), mapping[start:start + 16])
            if barrier is not None:
                barrier.wait()
            self.assertEqual(failures, [])
        finally:
            for ptr in reversed(registered):
                (result,) = runtime.cudaHostUnregister(ptr)
                print(
                    f"RANK={rank} UNREGISTER ptr={ptr:#x} code={int(result)}",
                    flush=True,
                )
            tensors.clear()
            tensor = None
            for mapping in mappings:
                mapping.close()


def _register_rank(rank, lock, barrier):
    EngramRegistrationTest()._register_tables(rank, lock, barrier)


def _probe_loader_rank(rank, barrier):
    EngramRegistrationTest()._probe_loader_pinned_memory(rank, barrier)


if __name__ == "__main__":
    unittest.main()
