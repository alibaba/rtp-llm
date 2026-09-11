"""CUDA registration and lifetime for per-host shared Engram tables.

Uses CUDA Runtime's supported registered-host pointer API. No full-table
Torch tensor, pin_memory copy, or query collective is created.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
import resource
import threading
from contextlib import contextmanager
from pathlib import Path

import torch

_LIVE_LOOKUPS = set()
_REGISTRY_LOCK = threading.RLock()


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_void_p),
        ("shape", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
        ("suboffsets", ctypes.c_void_p),
        ("internal", ctypes.c_void_p),
    ]


class _CudaRuntime:
    # CUDA Runtime registration flags and corresponding Driver attribute enums.
    HOST_REGISTER_SUPPORTED = 99
    USES_HOST_PAGE_TABLES = 100
    READ_ONLY_SUPPORTED = 113
    PORTABLE, MAPPED, READ_ONLY = 1, 2, 8

    def __init__(self):
        torch.cuda.init()
        # Torch initializes devices lazily; a real allocation activates its
        # primary context before Driver/Runtime interoperation.
        torch.empty(1, dtype=torch.uint8, device="cuda")
        self.native = torch.cuda.cudart()
        self.compiled_version = torch._C._cuda_getCompiledVersion()
        if not 13000 <= self.compiled_version < 14000:
            raise RuntimeError(
                f"Engram host registration requires CUDA13 Torch, got {self.compiled_version}"
            )
        self.library = ctypes.CDLL("libcuda.so.1")
        signatures = {
            "cuDriverGetVersion": [ctypes.POINTER(ctypes.c_int)],
            "cuCtxGetDevice": [ctypes.POINTER(ctypes.c_int)],
            "cuDeviceGetAttribute": [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
            ],
            "cuMemHostGetDevicePointer_v2": [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_void_p,
                ctypes.c_uint,
            ],
        }
        for name, args in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = args, ctypes.c_int
        self.library.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.library.cuGetErrorString.restype = ctypes.c_int
        version = ctypes.c_int()
        self._driver_call("cuDriverGetVersion", ctypes.byref(version))
        self.driver_version = version.value

    def _driver_call(self, name, *arguments):
        code = getattr(self.library, name)(*arguments)
        if code:
            detail = ctypes.c_char_p()
            self.library.cuGetErrorString(code, ctypes.byref(detail))
            message = (
                detail.value.decode("utf-8", errors="replace")
                if detail.value
                else "unknown driver error"
            )
            raise RuntimeError(f"{name} failed ({code}): {message}")

    def call(self, name, *arguments):
        if name == "cudaHostGetDevicePointer":
            pointer, host, flags = arguments
            self._driver_call(
                "cuMemHostGetDevicePointer_v2",
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint64)),
                host,
                flags,
            )
            return
        if name not in ("cudaHostRegister", "cudaHostUnregister"):
            raise ValueError(f"unsupported Engram runtime operation {name}")
        # PyTorch's native binding invokes the exact runtime linked by Torch.
        # Driver interop above uses that runtime's already-current context.
        code = getattr(self.native, name)(*arguments)
        if code != self.native.cudaError.success:
            raise RuntimeError(
                f"{name} failed ({code}): {self.native.cudaGetErrorString(code)}"
            )

    def attribute(self, attribute, device):
        if torch.cuda.current_device() != device:
            raise RuntimeError(
                "Engram capabilities require the selected Torch CUDA context"
            )
        actual_device = ctypes.c_int()
        self._driver_call("cuCtxGetDevice", ctypes.byref(actual_device))
        result = ctypes.c_int()
        self._driver_call(
            "cuDeviceGetAttribute", ctypes.byref(result), attribute, actual_device.value
        )
        return result.value

    def capabilities(self, device):
        loaded_runtimes = sorted(
            {
                fields[5]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if len(fields := line.split(maxsplit=5)) == 6
                and Path(fields[5]).name.startswith("libcudart")
            }
        )
        return {
            "host_register_supported": self.attribute(
                self.HOST_REGISTER_SUPPORTED, device
            ),
            "uses_host_page_tables": self.attribute(self.USES_HOST_PAGE_TABLES, device),
            "read_only_supported": self.attribute(self.READ_ONLY_SUPPORTED, device),
            "torch_cuda_compiled_version": self.compiled_version,
            "driver_version": self.driver_version,
            "registration_binding": "torch_native_cudart",
            "device_pointer_binding": "cuda_driver_current_context",
            "loaded_cuda_runtimes": loaded_runtimes,
        }


class _RegisteredBuffer:
    dtype = torch.uint8

    def __init__(self, view, runtime, device):
        self.device = torch.device("cuda", device)
        self.view = view
        self.runtime = runtime
        self.registered = False
        self.export = _PyBuffer()
        self.device_address = None
        get_buffer = ctypes.pythonapi.PyObject_GetBuffer
        get_buffer.argtypes = [
            ctypes.py_object,
            ctypes.POINTER(_PyBuffer),
            ctypes.c_int,
        ]
        get_buffer.restype = ctypes.c_int
        get_buffer(view, ctypes.byref(self.export), 0)
        if not self.export.readonly or self.export.itemsize != 1 or not self.export.buf:
            self._release_buffer()
            raise ValueError(
                "Engram registration requires a contiguous read-only byte mapping"
            )

    def register(self, flags):
        self.runtime.call("cudaHostRegister", self.export.buf, self.export.len, flags)
        self.registered = True
        pointer = ctypes.c_void_p()
        self.runtime.call(
            "cudaHostGetDevicePointer", ctypes.byref(pointer), self.export.buf, 0
        )
        self.device_address = pointer.value
        if self.device_address is None:
            raise RuntimeError("CUDA returned a null registered-host device pointer")

    def data_ptr(self):
        if not self.registered:
            raise RuntimeError("registered Engram buffer is closed")
        return self.device_address

    def _release_buffer(self):
        if self.export.obj:
            release = ctypes.pythonapi.PyBuffer_Release
            release.argtypes = [ctypes.POINTER(_PyBuffer)]
            release.restype = None
            release(ctypes.byref(self.export))
        self.view.release()

    def close(self):
        if self.registered:
            self.runtime.call("cudaHostUnregister", self.export.buf)
            self.registered = False
        self._release_buffer()


class SharedEngramGraph:
    def __init__(self, owner):
        self.owner = owner
        self.graph = torch.cuda.CUDAGraph()
        self.inputs = []
        self.closed = False
        self.captured = False

    def _hold(self, tensor):
        if tensor is not None:
            self.inputs.append(
                (tensor, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
            )

    @contextmanager
    def capture(self, stream=None):
        owner = self.owner
        with owner._lock, torch.cuda.device(owner.device):
            owner._check_open()
            if self.closed or self.captured or owner._capture is not None:
                raise RuntimeError(
                    "Engram graph is closed or another capture is active"
                )
            owner._capture = self
            try:
                with torch.cuda.graph(self.graph, stream=stream):
                    yield self
                self.captured = True
            finally:
                owner._capture = None

    def replay(self):
        with self.owner._lock, torch.cuda.device(self.owner.device):
            self.owner._check_open()
            if self.closed or not self.captured:
                raise RuntimeError("Engram graph has been reset")
            for tensor, pointer, shape, stride in self.inputs:
                if (
                    tensor.data_ptr() != pointer
                    or tuple(tensor.shape) != shape
                    or tuple(tensor.stride()) != stride
                ):
                    raise RuntimeError(
                        "Engram graph input/output storage changed; recapture is required"
                    )
            self.graph.replay()

    def close(self):
        with self.owner._lock, torch.cuda.device(self.owner.device):
            if self.closed:
                return
            if self.owner._capture is not None:
                raise RuntimeError(
                    "finish Engram graph capture before closing its binding"
                )
            torch.cuda.synchronize(self.owner.device)
            self.graph.reset()
            self.closed = True
            self.inputs.clear()
            self.owner._graphs.discard(self)


class SharedEngramLookup:
    def __init__(self, shared, *, device=None):
        self.device = _device_index(device)
        self.shared = shared
        self.pid = os.getpid()
        self._lock = threading.RLock()
        self._buffers = {}
        self._tables = {}
        self._graphs = set()
        self._capture = None
        self._closed = False
        self._closing = False
        with torch.cuda.device(self.device):
            if torch.cuda.get_device_capability(self.device)[0] != 10:
                raise RuntimeError(
                    "Engram GPU lookup requires a Blackwell SM100-family device"
                )
            self.runtime = _CudaRuntime()
            self.capabilities = self.runtime.capabilities(self.device)
            if not self.capabilities["host_register_supported"]:
                raise RuntimeError("CUDA device does not support host registration")
            self.writable_mapping = not (
                self.capabilities["read_only_supported"]
                or self.capabilities["uses_host_page_tables"]
            )
            self.flags = self.runtime.PORTABLE | self.runtime.MAPPED
            if self.capabilities["read_only_supported"]:
                self.flags |= self.runtime.READ_ONLY
            self.mode = (
                "registered_read_only_ats"
                if self.capabilities["uses_host_page_tables"]
                else (
                    "registered_shared_rw_api_read_only"
                    if self.writable_mapping
                    else "registered_read_only_pinned"
                )
            )
            self.total_bytes = sum(
                item["nbytes"] for item in shared.manifest["tensors"].values()
            )
            soft_limit, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            if (
                not self.capabilities["uses_host_page_tables"]
                and soft_limit != resource.RLIM_INFINITY
                and soft_limit < self.total_bytes
            ):
                raise RuntimeError(
                    f"memlock limit {soft_limit} is below Engram registration size {self.total_bytes}"
                )
            self._validate_tables()
            with _REGISTRY_LOCK:
                _LIVE_LOOKUPS.add(self)
            try:
                for name in sorted(shared.manifest["tensors"]):
                    buffer = _RegisteredBuffer(
                        shared._cuda_view(name, writable_mapping=self.writable_mapping),
                        self.runtime,
                        self.device,
                    )
                    self._buffers[name] = buffer
                    try:
                        buffer.register(self.flags)
                    except RuntimeError as error:
                        raise RuntimeError(
                            f"Engram registration failed for {name} "
                            f"({buffer.export.len} bytes, mode={self.mode}, "
                            f"backing={shared.directory}). The backing filesystem "
                            "must support CUDA host registration; use a shared "
                            "tmpfs mount for pinned mappings."
                        ) from error
            except BaseException:
                self.close()
                raise

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        store_root,
        revision,
        *,
        device=None,
        chunk_bytes=16 * 1024 * 1024,
    ):
        from rtp_llm.config.dsv41_config import V41Config
        from rtp_llm.model_loader.host_shared_weights import (
            HostSharedWeightStore,
            engram_checkpoint_slices,
        )

        config = V41Config.from_path(checkpoint)
        supported, reason, capabilities = is_supported(device)
        if not supported:
            raise RuntimeError(reason)
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if (
            not capabilities["uses_host_page_tables"]
            and soft_limit != resource.RLIM_INFINITY
            and soft_limit < config.engram_host_bytes
        ):
            raise RuntimeError(
                f"memlock limit {soft_limit} is below full Engram size {config.engram_host_bytes}"
            )
        slices = engram_checkpoint_slices(checkpoint, config)
        shared = HostSharedWeightStore(store_root).open_or_publish(
            revision, slices, chunk_bytes=chunk_bytes
        )
        try:
            return cls(shared, device=device)
        except BaseException:
            # A failed unregister retains its exported buffers until process
            # shutdown; BufferError here must not force-close a live mapping.
            try:
                shared.close()
            except BufferError:
                pass
            raise

    def _validate_tables(self):
        entries = self.shared.manifest["tensors"]
        expected = set()
        for layer in (1, 14):
            prefix = f"layers.{layer}.engram.embed."
            weight, scale = entries[prefix + "weight"], entries[prefix + "scale"]
            if weight["dtype"] != "F8_E4M3" or scale["dtype"] != "F8_E8M0":
                raise ValueError("Engram lookup requires FP8 E4M3 and UE8M0 scales")
            rows, dim = weight["shape"]
            if dim != 256 or scale["shape"] != [rows, dim // 32]:
                raise ValueError(
                    "Engram lookup requires 256-dimensional rows and group32 scales"
                )
            if weight["nbytes"] != rows * dim or scale["nbytes"] != rows * (dim // 32):
                raise ValueError("Engram backing extent does not cover its full table")
            self._tables[layer] = (prefix, rows, dim)
            expected.update((prefix + "weight", prefix + "scale"))
        if set(entries) != expected:
            raise ValueError(
                "shared Engram backing must contain exactly both tables and their scales"
            )

    def _check_open(self):
        if os.getpid() != self.pid:
            raise RuntimeError("CUDA Engram mappings cannot be inherited across fork")
        if self._closed or self._closing:
            raise RuntimeError("shared Engram lookup is closing or closed")

    def graph(self):
        with self._lock, torch.cuda.device(self.device):
            self._check_open()
            binding = SharedEngramGraph(self)
            self._graphs.add(binding)
            return binding

    def lookup(self, layer, indices, *, valid_mask=None, out=None):
        from rtp_llm.models_py.modules.dsv41._engram_lookup_triton import (
            engram_gather_kernel,
        )

        with self._lock, torch.cuda.device(self.device):
            self._check_open()
            prefix, rows, dim = self._tables[layer]
            device = torch.device("cuda", self.device)
            if (
                indices.device != device
                or indices.dtype != torch.int64
                or not indices.is_contiguous()
            ):
                raise ValueError(
                    "Engram indices must be contiguous int64 on the registered GPU"
                )
            if valid_mask is not None and (
                valid_mask.device != device
                or valid_mask.dtype != torch.bool
                or valid_mask.shape != indices.shape
                or not valid_mask.is_contiguous()
            ):
                raise ValueError(
                    "Engram validity mask must be a contiguous bool tensor matching indices"
                )
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing and (self._capture is None or out is None):
                raise RuntimeError(
                    "capture requires an Engram graph resource binding and a fixed output buffer"
                )
            shape = tuple(indices.shape) + (dim,)
            if out is None:
                out = torch.empty(shape, device=device, dtype=torch.bfloat16)
            if (
                out.device != device
                or out.dtype != torch.bfloat16
                or tuple(out.shape) != shape
                or not out.is_contiguous()
            ):
                raise ValueError(
                    "Engram output must be contiguous BF16 with shape indices+[256]"
                )
            in_bounds = (indices >= 0) & (indices < rows)
            if valid_mask is not None:
                in_bounds = in_bounds | ~valid_mask
            torch._assert_async(
                in_bounds.all(), "Engram query ID is outside the host table"
            )
            if indices.numel():
                engram_gather_kernel[(indices.numel(),)](
                    self._buffers[prefix + "weight"],
                    self._buffers[prefix + "scale"],
                    indices,
                    indices if valid_mask is None else valid_mask,
                    out,
                    ROWS=rows,
                    DIM=dim,
                    HAS_VALID=valid_mask is not None,
                    BLOCK=256,
                )
            if capturing:
                for tensor in (indices, valid_mask, out):
                    self._capture._hold(tensor)
            return out

    def warmup(self, stream=None):
        with self._lock, torch.cuda.device(self.device):
            self._check_open()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Engram warmup must finish before CUDA Graph capture"
                )
            stream = stream or torch.cuda.current_stream(self.device)
            with torch.cuda.stream(stream):
                indices = torch.zeros((6, 24), dtype=torch.int64, device=self.device)
                valid = torch.ones_like(indices, dtype=torch.bool)
                for layer in self._tables:
                    self.lookup(layer, indices)
                    self.lookup(layer, indices, valid_mask=valid)
            stream.synchronize()

    def accounting(self):
        from rtp_llm.model_loader.host_shared_metrics import memory_accounting

        return memory_accounting(self)

    def close(self):
        with self._lock, torch.cuda.device(self.device):
            if self._closed:
                return
            if os.getpid() != self.pid:
                raise RuntimeError("only the CUDA registration owner may close it")
            if self._capture is not None or torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "finish Engram graph capture before closing its resources"
                )
            self._closing = True
            # No buffer or lease is released if completion cannot be confirmed.
            torch.cuda.synchronize(self.device)
            for binding in list(self._graphs):
                binding.graph.reset()
                binding.closed = True
                binding.inputs.clear()
            self._graphs.clear()
            for name in list(self._buffers):
                self._buffers[name].close()
                del self._buffers[name]
            self.shared.close()
            self._closed = True
            with _REGISTRY_LOCK:
                _LIVE_LOOKUPS.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def shutdown_shared_engram():
    with _REGISTRY_LOCK:
        lookups = [lookup for lookup in _LIVE_LOOKUPS if lookup.pid == os.getpid()]
    failures = []
    for lookup in lookups:
        try:
            lookup.close()
        except Exception as error:
            failures.append(str(error))
    if failures:
        raise RuntimeError(
            "Engram shutdown retained mappings after CUDA cleanup failures: "
            + "; ".join(failures)
        )


def _device_index(device):
    if device is None:
        return torch.cuda.current_device()
    if type(device) is int:
        if device < 0:
            raise ValueError("CUDA device index must be nonnegative")
        return device
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError("shared Engram lookup requires a CUDA device")
    return torch.cuda.current_device() if parsed.index is None else parsed.index


def is_supported(device=None):
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable", {}
    device = _device_index(device)
    if torch.cuda.get_device_capability(device)[0] != 10:
        return False, "a Blackwell SM100-family device is required", {}
    with torch.cuda.device(device):
        runtime = _CudaRuntime()
        capabilities = runtime.capabilities(device)
    if not capabilities["host_register_supported"]:
        return False, "CUDA host registration is unsupported", capabilities
    return True, "supported", capabilities


def _atexit_cleanup():
    try:
        shutdown_shared_engram()
    except Exception:
        logging.exception("Shared Engram cleanup failed during interpreter shutdown")


atexit.register(_atexit_cleanup)
