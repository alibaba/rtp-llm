"""ROCm/HIP host-registration runtime for the shared pinned Engram table.

cuda.bindings is unavailable on ROCm, so ``_PinnedSharedTable``'s three
runtime calls (cudaHostRegister / cudaHostGetDevicePointer /
cudaHostUnregister) are provided here through ctypes on libamdhip64.so with
the same call signatures and tuple return shapes, letting the exact
tmpfs-shared-table design run unmodified on HIP.

Scale lesson (load21, kept for reviewers): registering the safetensors
MAP_PRIVATE mmap IN PLACE looked free in a 4 GiB probe (d_anon=0) but at
production scale long-term pinning COWed the prefaulted file pages into
~91.5 GiB of PRIVATE anonymous memory PER RANK (host hit 973 GiB used,
OOM kill). The tmpfs MAP_SHARED copy is what makes registration share one
physical, COW-free copy across all ranks - the same reason the CUDA path
was built this way.
"""

import ctypes
import logging
import threading

logger = logging.getLogger(__name__)

# hipHostRegisterDefault=0, Portable=0x1, Mapped=0x2, IoMemory=0x4.
# The CUDA call site passes 3 (Portable|Mapped); identical values on HIP.
_HIP_HOST_REGISTER_PORTABLE = 0x1
_HIP_HOST_REGISTER_MAPPED = 0x2


class HipUvaError(RuntimeError):
    pass


class _HipHostRuntime:
    """cuda.bindings-runtime-compatible shim over libamdhip64.

    Method names and return tuples deliberately mirror the cuda.bindings
    runtime calls used by _PinnedSharedTable:
      cudaHostRegister(ptr, nbytes, flags) -> (err,)
      cudaHostGetDevicePointer(ptr, flags) -> (err, device_ptr)
      cudaHostUnregister(ptr)              -> (err,)
    """

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
        lib.hipHostRegister.restype = ctypes.c_int
        lib.hipHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        lib.hipHostGetDevicePointer.restype = ctypes.c_int
        lib.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        lib.hipHostUnregister.restype = ctypes.c_int
        lib.hipHostUnregister.argtypes = [ctypes.c_void_p]

    def cudaHostRegister(self, ptr, nbytes, flags):  # noqa: N802
        status = self._lib.hipHostRegister(
            ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(flags)
        )
        return (status,)

    def cudaHostGetDevicePointer(self, ptr, flags):  # noqa: N802
        device_ptr = ctypes.c_void_p()
        status = self._lib.hipHostGetDevicePointer(
            ctypes.byref(device_ptr), ctypes.c_void_p(ptr), ctypes.c_uint(flags)
        )
        return (status, device_ptr.value if device_ptr.value is not None else 0)

    def cudaHostUnregister(self, ptr):  # noqa: N802
        status = self._lib.hipHostUnregister(ctypes.c_void_p(ptr))
        return (status,)


_runtime_lock = threading.Lock()
_runtime = None


def host_pin_runtime():
    """Return the process host-registration runtime for the active backend.

    ROCm: the ctypes HIP shim. CUDA: the real cuda.bindings runtime, keeping
    the frozen reference path byte-identical.
    """
    global _runtime
    import torch

    if torch.version.hip is None:
        from cuda.bindings import runtime

        return runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = _HipHostRuntime()
        return _runtime
