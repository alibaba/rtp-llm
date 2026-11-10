"""ROCm/HIP UVA registration for host-resident V4.1 Engram tables.

The CUDA path pins a tmpfs shared copy through cuda.bindings
(cudaHostRegister + cudaHostGetDevicePointer). ROCm has no cuda.bindings;
this module provides the same three-entry surface through ctypes on
libamdhip64.so and a table wrapper matching ``_PinnedSharedTable``'s
consumer interface (``.weight``, ``.scales``, ``.device``,
``.cuda_view()``, ``._lookup_events``).

Design differences from the CUDA tmpfs table, both deliberate:

1. Registration is IN PLACE on the checkpoint tensors. Those tensors are
   safetensors mmap views, so every TP rank on the host shares ONE physical
   page-cache copy of each ~190 GiB table. A tmpfs or anonymous copy would
   duplicate the table per registering process.
2. Page ranges are merged before registration: several checkpoint tensors
   live in the same file mapping, and hipHostRegister on overlapping ranges
   would fail with already-registered errors and make unregistration
   ownership ambiguous.

Offload policy is preserved: the tables stay in host RAM (device=cpu,
persistent_gpu_bytes=0 in the AWI census); device kernels read selected rows
over UVA, which is transient staging for computation, not a persistent GPU
mirror.
"""

import ctypes
import logging
import threading
import weakref

import torch

logger = logging.getLogger(__name__)

# hipHostRegisterDefault=0, Portable=0x1, Mapped=0x2, IoMemory=0x4
_HIP_HOST_REGISTER_PORTABLE = 0x1
_HIP_HOST_REGISTER_MAPPED = 0x2
_HIP_HOST_REGISTER_FLAGS = _HIP_HOST_REGISTER_PORTABLE | _HIP_HOST_REGISTER_MAPPED


class HipUvaError(RuntimeError):
    pass


class _HipRuntime:
    """ctypes wrapper for the hipHost* registration API slice we need."""

    def __init__(self):
        last_error = None
        for name in ("libamdhip64.so", "libamdhip64.so.6"):
            try:
                self._lib = ctypes.CDLL(name)
                break
            except OSError as error:  # noqa: PERF203
                last_error = error
        else:
            raise HipUvaError(f"libamdhip64 not loadable: {last_error}")
        lib = self._lib
        # hipError_t hipHostRegister(void* ptr, size_t sizeBytes, unsigned flags)
        lib.hipHostRegister.restype = ctypes.c_int
        lib.hipHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        # hipError_t hipHostGetDevicePointer(void** devPtr, void* hostPtr,
        #                                    unsigned flags)
        lib.hipHostGetDevicePointer.restype = ctypes.c_int
        lib.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        # hipError_t hipHostUnregister(void* ptr)
        lib.hipHostUnregister.restype = ctypes.c_int
        lib.hipHostUnregister.argtypes = [ctypes.c_void_p]

    def host_register(self, ptr: int, nbytes: int) -> None:
        status = self._lib.hipHostRegister(
            ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), _HIP_HOST_REGISTER_FLAGS
        )
        if status != 0:
            raise HipUvaError(f"hipHostRegister({ptr:#x}, {nbytes}) failed: {status}")

    def host_unregister(self, ptr: int) -> None:
        status = self._lib.hipHostUnregister(ctypes.c_void_p(ptr))
        if status != 0:
            logger.warning("hipHostUnregister(%#x) failed: %s", ptr, status)

    def device_pointer(self, ptr: int) -> int:
        device_ptr = ctypes.c_void_p()
        status = self._lib.hipHostGetDevicePointer(
            ctypes.byref(device_ptr), ctypes.c_void_p(ptr), 0
        )
        if status != 0:
            raise HipUvaError(f"hipHostGetDevicePointer({ptr:#x}) failed: {status}")
        if device_ptr.value is None:
            raise HipUvaError(f"hipHostGetDevicePointer({ptr:#x}) returned NULL")
        return int(device_ptr.value)


_runtime_lock = threading.Lock()
_runtime = None
# Registered page ranges are refcounted process-wide: tensors from the same
# safetensors file can share boundary pages, so two tables' merged ranges may
# overlap. Only the first registration calls hipHostRegister and only the
# last release calls hipHostUnregister.
_registered_ranges: dict = {}


def hip_runtime() -> _HipRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = _HipRuntime()
        return _runtime


def _acquire_ranges(runtime, ranges, device) -> list:
    """Register merged page ranges (refcounted); return the owned list."""
    owned = []
    try:
        with torch.cuda.device(device):
            torch.zeros(1, device=device)  # ensure context init
            for span in ranges:
                with _runtime_lock:
                    count = _registered_ranges.get(span, 0)
                    _registered_ranges[span] = count + 1
                if count == 0:
                    runtime.host_register(span[0], span[1] - span[0])
                owned.append(span)
    except BaseException:
        _release_ranges(runtime, owned)
        raise
    return owned


def _release_ranges(runtime, owned) -> None:
    for span in owned:
        with _runtime_lock:
            count = _registered_ranges.get(span, 0)
            if count <= 1:
                _registered_ranges.pop(span, None)
            else:
                _registered_ranges[span] = count - 1
                count = count  # still referenced; do not unpin
        if count <= 1:
            runtime.host_unregister(span[0])


def _merged_page_ranges(spans):
    """Merge (start, end) byte spans into page-aligned, non-overlapping ranges."""
    page = 4096
    aligned = sorted(
        (start & ~(page - 1), -(-(end) // page) * page) for start, end in spans
    )
    merged = []
    for start, end in aligned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class _HipArrayView:
    """Expose a registered host pointer through __cuda_array_interface__.

    PyTorch's ROCm build consumes the same interface as CUDA, so
    ``torch.as_tensor(view, device=...)`` yields a device tensor reading the
    pinned host pages over UVA. Kept byte-typed ("|u1") exactly like the CUDA
    ``_CudaArrayView``; kernels bitcast the raw FP8/E8M0 bytes.
    """

    def __init__(self, owner, tensor, pointer):
        self.owner = owner
        self.__cuda_array_interface__ = {
            "shape": tuple(tensor.shape),
            "strides": tuple(tensor.stride()),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }


class HipRegisteredTable:
    """_PinnedSharedTable-compatible wrapper registering host tables in place.

    ``weight`` and ``scales`` are the resident host tensors (safetensors mmap
    views). Their storages' page ranges are merged and hipHostRegister'ed once
    for the pair; the registration is released by a weakref finalizer after
    every recorded asynchronous lookup event completes, mirroring the CUDA
    table's teardown ordering.
    """

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor, source_identity, device):
        if weight.device.type != "cpu" or scales.device.type != "cpu":
            raise ValueError("HipRegisteredTable requires host-resident tensors")
        if torch.version.hip is None:
            raise HipUvaError("HipRegisteredTable is ROCm-only")
        self.device = torch.device(device)
        self.weight = weight
        self.scales = scales
        self.source_identity = source_identity
        self._lookup_events = []
        runtime = hip_runtime()
        spans = []
        for tensor in (weight, scales):
            storage = tensor.untyped_storage()
            spans.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
        ranges = _merged_page_ranges(spans)
        # Pin under the target device's context so the mapping is device
        # visible; the portable flag keeps it valid on the process's devices.
        owned = _acquire_ranges(runtime, ranges, self.device)
        self._ranges = ranges
        total_gib = sum(end - start for start, end in ranges) / 1024**3
        self._finalizer = weakref.finalize(
            self, self._unregister, runtime, owned, self._lookup_events
        )
        logger.info(
            "Engram HIP-registered %.2f GiB host pages in place on %s (%d ranges)",
            total_gib,
            self.device,
            len(ranges),
        )

    @staticmethod
    def _unregister(runtime, owned, lookup_events):
        # Mapped host storage has no record_stream support; wait for every
        # asynchronous reader before unpinning (same contract as CUDA table).
        for event in lookup_events:
            event.synchronize()
        _release_ranges(runtime, owned)

    def cuda_view(self, tensor: torch.Tensor) -> torch.Tensor:
        runtime = hip_runtime()
        pointer = runtime.device_pointer(tensor.data_ptr())
        with torch.cuda.device(self.device):
            return torch.as_tensor(
                _HipArrayView(self, tensor, pointer), device=self.device
            )

    def release(self):
        if getattr(self, "_finalizer", None) is not None:
            self._finalizer()
