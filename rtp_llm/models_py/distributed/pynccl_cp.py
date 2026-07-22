"""Low-overhead NCCL AllGather for GLM5 context-parallel prefill.

The regular ``torch.distributed`` path creates a ProcessGroupNCCL ``Work``
object and manages internal CUDA events for every collective. GLM5 prefill
issues several small CP AllGathers per sparse-MLA layer, so that host-side
control cost is measurable. ``GLM5_CP_OPT=1`` enables the complete optimization
stack: direct ``ctypes`` submission of ``ncclAllGather``, registered symmetric
output windows, and packed Indexer K gather. With the switch unset or set to
zero, callers use the original torch/c10d and separate-gather paths.

The direct path owns a second NCCL communicator containing exactly the ranks
from the supplied torch process group.  Callers in the GLM5 hot path consume
the result on the same stream, so CUDA stream ordering replaces the C10D Work
object.  The feature is opt-in; with the flag unset the existing collective
wrapper is used unchanged.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist

_ncclComm_t = ctypes.c_void_p
_cudaStream_t = ctypes.c_void_p
_buffer_t = ctypes.c_void_p
_ncclDataType_t = ctypes.c_int
_ncclResult_t = ctypes.c_int
_ncclWindow_t = ctypes.c_void_p
_NCCL_WIN_COLL_SYMMETRIC = 1


class _ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


# Values are the public ncclDataType_t enum. AllGather is a byte-preserving
# move, so one-byte FP8 values can be transported as ncclUint8.
_NCCL_DT = {
    torch.int8: 0,
    torch.uint8: 1,
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
}
for _fp8_name in (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
):
    _fp8 = getattr(torch, _fp8_name, None)
    if _fp8 is not None:
        _NCCL_DT[_fp8] = 1


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


# Resolve once at import. Production sets the flag before the worker starts;
# avoiding an os.environ lookup in every layer keeps the dispatcher negligible.
_OPT_ENABLED = _env_enabled("GLM5_CP_OPT")
_PYNCCL_VALIDATE = _env_enabled("GLM5_CP_PYNCCL_VALIDATE")


class _NCCLLib:
    def __init__(self) -> None:
        candidates = glob.glob(
            os.path.join(os.path.dirname(torch.__file__), "lib", "libnccl.so*")
        )
        self._lib_path = candidates[0] if candidates else "libnccl.so.2"
        self._lib = ctypes.CDLL(self._lib_path)

        self._lib.ncclGetErrorString.restype = ctypes.c_char_p
        self._lib.ncclGetErrorString.argtypes = [_ncclResult_t]
        self._lib.ncclGetUniqueId.restype = _ncclResult_t
        self._lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_ncclUniqueId)]
        self._lib.ncclCommInitRank.restype = _ncclResult_t
        self._lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(_ncclComm_t),
            ctypes.c_int,
            _ncclUniqueId,
            ctypes.c_int,
        ]
        self._lib.ncclCommDestroy.restype = _ncclResult_t
        self._lib.ncclCommDestroy.argtypes = [_ncclComm_t]
        self._lib.ncclAllGather.restype = _ncclResult_t
        self._lib.ncclAllGather.argtypes = [
            _buffer_t,
            _buffer_t,
            ctypes.c_size_t,
            _ncclDataType_t,
            _ncclComm_t,
            _cudaStream_t,
        ]
        # Symmetric window registration was added in NCCL 2.27. Keep it
        # optional so GLM5_CP_OPT can fall back to ordinary pynccl.
        self._has_symm = hasattr(self._lib, "ncclCommWindowRegister")
        if self._has_symm:
            self._lib.ncclCommWindowRegister.restype = _ncclResult_t
            self._lib.ncclCommWindowRegister.argtypes = [
                _ncclComm_t,
                _buffer_t,
                ctypes.c_size_t,
                ctypes.POINTER(_ncclWindow_t),
                ctypes.c_int,
            ]
        self._has_window_deregister = hasattr(
            self._lib, "ncclCommWindowDeregister"
        )
        if self._has_window_deregister:
            self._lib.ncclCommWindowDeregister.restype = _ncclResult_t
            self._lib.ncclCommWindowDeregister.argtypes = [
                _ncclComm_t,
                _ncclWindow_t,
            ]

    def _check(self, result: int) -> None:
        if result == 0:
            return
        error = self._lib.ncclGetErrorString(result)
        message = error.decode() if error is not None else f"error code {result}"
        raise RuntimeError(f"pynccl: {message}")

    def make_comm(self, process_group, device: torch.device) -> _ncclComm_t:
        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        root_global_rank = dist.get_global_rank(process_group, 0)

        unique_id = _ncclUniqueId()
        unique_id_cpu = torch.empty(128, dtype=torch.uint8)
        if rank == 0:
            self._check(self._lib.ncclGetUniqueId(ctypes.byref(unique_id)))
            ctypes.memmove(
                unique_id_cpu.data_ptr(), ctypes.addressof(unique_id), 128
            )

        # A NCCL process group cannot broadcast a CPU tensor. Use the existing
        # torch communicator once to distribute the new communicator's ID.
        unique_id_device = unique_id_cpu.to(device)
        dist.broadcast(
            unique_id_device, src=root_global_rank, group=process_group
        )
        unique_id_cpu = unique_id_device.cpu().contiguous()
        ctypes.memmove(
            ctypes.addressof(unique_id), unique_id_cpu.data_ptr(), 128
        )

        comm = _ncclComm_t()
        self._check(
            self._lib.ncclCommInitRank(
                ctypes.byref(comm), world_size, unique_id, rank
            )
        )
        return comm

    def destroy_comm(self, comm: _ncclComm_t) -> None:
        self._check(self._lib.ncclCommDestroy(comm))

    def window_register(
        self, comm: _ncclComm_t, ptr: int, nbytes: int
    ) -> _ncclWindow_t:
        if not self._has_symm:
            raise RuntimeError("pynccl: NCCL symmetric windows are unavailable")
        window = _ncclWindow_t()
        self._check(
            self._lib.ncclCommWindowRegister(
                comm,
                _buffer_t(ptr),
                nbytes,
                ctypes.byref(window),
                _NCCL_WIN_COLL_SYMMETRIC,
            )
        )
        return window

    def window_deregister(self, comm: _ncclComm_t, window: _ncclWindow_t) -> None:
        if self._has_window_deregister:
            self._check(self._lib.ncclCommWindowDeregister(comm, window))

    def all_gather(
        self,
        send_ptr: int,
        recv_ptr: int,
        count: int,
        dtype: torch.dtype,
        comm: _ncclComm_t,
        stream_ptr: int,
    ) -> None:
        self._check(
            self._lib.ncclAllGather(
                _buffer_t(send_ptr),
                _buffer_t(recv_ptr),
                count,
                _NCCL_DT[dtype],
                comm,
                _cudaStream_t(stream_ptr),
            )
        )


_LIB: "_NCCLLib | None" = None
# A communicator is CUDA-device-specific. One-GPU-per-process is the normal
# deployment, but include the device index in the cache key for correctness.
_COMMS: Dict[Tuple[Any, int], Tuple["_NCCLLib", Any]] = {}
_WORLD_SIZES: Dict[Any, int] = {}

# Symmetric-memory state is process-persistent. Each role gets a distinct
# buffer because MLA cKV and kPE must coexist until their common restore has
# consumed both outputs. Packed Indexer FP8 K+scale intentionally stays on
# ordinary pynccl and therefore does not reserve a symmetric window.
_SYMM_INIT_BUDGET_NBYTES = 1024 * 1024 * 1024
_SYMM_POOLS: Dict[Tuple[Any, int], Any] = {}
_SYMM_POOL_FAILED: set[Tuple[Any, int]] = set()
_SYMM_INIT_DONE: set[Tuple[Any, int]] = set()
_SYMM_BASES: Dict[Tuple[Any, int, str, torch.dtype], torch.Tensor] = {}
_SYMM_VIEWS: Dict[
    Tuple[Any, int, str, torch.dtype, Tuple[int, ...]], torch.Tensor
] = {}
_SYMM_WINDOWS: Dict[Tuple[Any, int, int], Any] = {}
_SYMM_CAPACITY_NBYTES: Dict[Tuple[Any, int, int], int] = {}
_SYMM_ROLE_STREAMS: Dict[Tuple[Any, int, str], int] = {}

_SYMM_ROLE_DTYPES = {
    "indexer_k_bf16": frozenset((torch.bfloat16,)),
    "mla_ckv": frozenset((torch.bfloat16,)),
    "mla_kpe": frozenset((torch.bfloat16,)),
    "mla_history": frozenset((torch.bfloat16,)),
}


def enabled() -> bool:
    return _OPT_ENABLED


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _comm_key(process_group, device: torch.device) -> Tuple[Any, int]:
    return process_group, _device_index(device)


def _resolve_process_group(
    group: Any,
) -> dist.ProcessGroup:
    # Keep this low-level module importable without loading RTP-LLM's pybind
    # config libraries. The collective wrapper is needed only when callers pass
    # the public Group enum instead of an already-resolved ProcessGroup.
    from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

    if group is None:
        group = Group.TP
    if isinstance(group, Group):
        return _get_group(group)
    return group


def _torch_all_gather(input: torch.Tensor, group: Any) -> torch.Tensor:
    from rtp_llm.models_py.distributed.collective_torch import all_gather

    return all_gather(input, group=group)


def _get_comm(process_group, device: torch.device):
    """Collectively create and cache the direct NCCL communicator at startup."""
    global _LIB

    key = _comm_key(process_group, device)
    cached = _COMMS.get(key)
    if cached is not None:
        return cached

    resolved_device = torch.device("cuda", key[1])
    with torch.cuda.device(resolved_device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GLM5 pynccl communicator must be initialized before CUDA graph capture"
            )
        if _LIB is None:
            _LIB = _NCCLLib()
        comm = _LIB.make_comm(process_group, resolved_device)
    _WORLD_SIZES[process_group] = dist.get_world_size(process_group)
    cached = (_LIB, comm)
    _COMMS[key] = cached
    return cached


def _require_comm(process_group, device: torch.device):
    cached = _COMMS.get(_comm_key(process_group, device))
    if cached is None:
        raise RuntimeError(
            "GLM5 pynccl communicator is not initialized; initialize the "
            "distributed environment before entering CP prefill"
        )
    return cached


def destroy() -> None:
    """Release symmetric windows and direct communicators before PG teardown."""
    global _LIB

    for window_key, window in list(_SYMM_WINDOWS.items()):
        comm_entry = _COMMS.get(window_key[:2])
        if comm_entry is None:
            continue
        lib, comm = comm_entry
        try:
            lib.window_deregister(comm, window)
        except RuntimeError as error:
            logging.warning(
                "Failed to deregister GLM5 pynccl symmetric window: %s", error
            )

    for lib, comm in list(_COMMS.values()):
        try:
            lib.destroy_comm(comm)
        except RuntimeError as error:
            logging.warning("Failed to destroy GLM5 pynccl communicator: %s", error)

    # Keep registered allocations alive until their communicator has been
    # destroyed, including on NCCL builds without an explicit deregister API.
    _SYMM_VIEWS.clear()
    _SYMM_BASES.clear()
    _SYMM_WINDOWS.clear()
    _SYMM_CAPACITY_NBYTES.clear()
    _SYMM_ROLE_STREAMS.clear()
    _SYMM_INIT_DONE.clear()
    _SYMM_POOLS.clear()
    _SYMM_POOL_FAILED.clear()
    _COMMS.clear()
    _WORLD_SIZES.clear()
    _LIB = None


def _world_size(process_group) -> int:
    world_size = _WORLD_SIZES.get(process_group)
    if world_size is None:
        world_size = dist.get_world_size(process_group)
        _WORLD_SIZES[process_group] = world_size
    return world_size


def _symm_max_nbytes(process_group, device: torch.device) -> int | None:
    """Use the same topology and size gate as RTP-LLM's Torch SYMM backend."""
    try:
        from rtp_llm.models_py.distributed.symm_mem import (
            TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES,
        )

        capability_major = torch.cuda.get_device_capability(device)[0]
        return TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES.get(
            capability_major, {}
        ).get(_world_size(process_group))
    except Exception:
        return None


def _symm_pool(process_group, device: torch.device, *, create: bool) -> Any | None:
    key = _comm_key(process_group, device)
    pool = _SYMM_POOLS.get(key)
    if pool is not None:
        return pool
    if key in _SYMM_POOL_FAILED or not create:
        return None
    try:
        backend = process_group._get_backend(device)
        pool = torch.cuda.MemPool(backend.mem_allocator)
    except Exception as error:
        _SYMM_POOL_FAILED.add(key)
        logging.warning("GLM5 CP symmetric MemPool unavailable: %s", error)
        return None
    _SYMM_POOLS[key] = pool
    return pool


def _init_symm_windows(process_group, device: torch.device) -> None:
    """Collectively allocate and register all GLM5 role windows at startup."""
    if not enabled():
        return
    group_key = _comm_key(process_group, device)
    if group_key in _SYMM_INIT_DONE:
        return

    lib, comm = _require_comm(process_group, device)
    if not lib._has_symm:
        logging.warning(
            "GLM5_CP_OPT requested, but NCCL lacks ncclCommWindowRegister; "
            "falling back to ordinary pynccl"
        )
        _SYMM_INIT_DONE.add(group_key)
        return

    pool = _symm_pool(process_group, device, create=True)
    max_nbytes = _symm_max_nbytes(process_group, device)
    if pool is None or max_nbytes is None:
        logging.warning(
            "GLM5_CP_OPT symmetric mode is unsupported for this GPU/world-size topology; "
            "falling back to ordinary pynccl"
        )
        _SYMM_INIT_DONE.add(group_key)
        return

    used_nbytes = sum(
        capacity
        for key, capacity in _SYMM_CAPACITY_NBYTES.items()
        if key[:2] == group_key
    )
    for role, dtypes in _SYMM_ROLE_DTYPES.items():
        for dtype in dtypes:
            base_key = (*group_key, role, dtype)
            if base_key in _SYMM_BASES:
                continue
            if used_nbytes + int(max_nbytes) > _SYMM_INIT_BUDGET_NBYTES:
                logging.warning(
                    "GLM5 CP symmetric registration reached the 1-GiB/rank budget"
                )
                _SYMM_INIT_DONE.add(group_key)
                return

            element_size = torch.empty((), dtype=dtype).element_size()
            capacity_numel = int(max_nbytes) // int(element_size)
            with torch.cuda.use_mem_pool(pool):
                base = torch.empty(capacity_numel, dtype=dtype, device=device)

            ptr_key = (*group_key, int(base.data_ptr()))
            # Window registration is collective and must remain in this
            # deterministic, rank-uniform startup loop.
            window = lib.window_register(comm, base.data_ptr(), int(max_nbytes))
            _SYMM_BASES[base_key] = base
            _SYMM_WINDOWS[ptr_key] = window
            _SYMM_CAPACITY_NBYTES[ptr_key] = int(max_nbytes)
            used_nbytes += int(max_nbytes)

    _SYMM_INIT_DONE.add(group_key)
    logging.info(
        "GLM5 CP symmetric windows initialized: %.1f MiB/rank",
        used_nbytes / (1024 * 1024),
    )


def _symm_output(
    role: str | None,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    process_group,
) -> torch.Tensor | None:
    """Return a registered persistent prefix view, or None for fallback."""
    if not enabled() or role is None:
        return None
    if dtype not in _SYMM_ROLE_DTYPES.get(role, ()):
        return None
    group_key = _comm_key(process_group, device)
    base = _SYMM_BASES.get((*group_key, role, dtype))
    if base is None:
        return None

    numel = 1
    for dim in shape:
        numel *= int(dim)
    ptr_key = (*group_key, int(base.data_ptr()))
    capacity_nbytes = _SYMM_CAPACITY_NBYTES.get(ptr_key, 0)
    if numel * base.element_size() > capacity_nbytes:
        return None

    view_key = (*group_key, role, dtype, shape)
    view = _SYMM_VIEWS.get(view_key)
    if view is None:
        view = base[:numel].view(shape)
        _SYMM_VIEWS[view_key] = view
    return view


def _pynccl_symm_all_gather(
    output: torch.Tensor,
    input: torch.Tensor,
    stream: torch.cuda.Stream,
    process_group,
) -> None:
    lib, comm = _require_comm(process_group, input.device)
    ptr_key = (*_comm_key(process_group, output.device), int(output.data_ptr()))
    if ptr_key not in _SYMM_WINDOWS:
        raise RuntimeError("GLM5 symmetric AllGather output window is not registered")
    lib.all_gather(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        input.dtype,
        comm,
        stream.cuda_stream,
    )


def _symm_stream_allowed(
    role: str,
    stream: torch.cuda.Stream,
    process_group,
    device: torch.device,
) -> bool:
    """Persistent role buffers are safe only on one stream's ordered timeline."""
    key = (*_comm_key(process_group, device), role)
    stream_ptr = int(stream.cuda_stream)
    previous = _SYMM_ROLE_STREAMS.get(key)
    if previous is None:
        _SYMM_ROLE_STREAMS[key] = stream_ptr
        return True
    return previous == stream_ptr


def warmup(
    group: Any = None,
    device: Any = None,
) -> None:
    """Initialize the second communicator before a captured/hot forward."""
    if not enabled():
        return
    if not dist.is_initialized():
        raise RuntimeError("GLM5 pynccl warmup requires torch.distributed")
    if device is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        resolved_device = torch.device("cuda", device)
    else:
        resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise RuntimeError(f"GLM5 pynccl requires a CUDA device, got {resolved_device}")
    process_group = _resolve_process_group(group)
    key = _comm_key(process_group, resolved_device)
    symm_ready = not enabled() or key in _SYMM_INIT_DONE
    if key in _COMMS and symm_ready:
        return

    with torch.cuda.device(resolved_device):
        if key not in _COMMS:
            backend = str(dist.get_backend(process_group)).lower()
            if backend != "nccl":
                raise RuntimeError(
                    f"GLM5 pynccl requires an NCCL process group, got {backend}"
                )
            _get_comm(process_group, resolved_device)
        _init_symm_windows(process_group, resolved_device)


def init(process_group, device: Any) -> None:
    """Distributed-startup entry point for all optional GLM5 CP resources."""
    warmup(process_group, device)


def _validate_all_gather(
    output: torch.Tensor, input: torch.Tensor, process_group
) -> None:
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("GLM5 pynccl AllGather requires CUDA tensors")
    if input.device != output.device:
        raise ValueError(
            f"AllGather input/output devices differ: {input.device} vs {output.device}"
        )
    if input.dtype != output.dtype:
        raise ValueError(
            f"AllGather input/output dtypes differ: {input.dtype} vs {output.dtype}"
        )
    if input.dtype not in _NCCL_DT:
        raise TypeError(f"unsupported pynccl AllGather dtype: {input.dtype}")
    if not input.is_contiguous() or not output.is_contiguous():
        raise ValueError("GLM5 pynccl AllGather requires contiguous tensors")

    world_size = _world_size(process_group)
    expected = world_size * input.numel()
    if output.numel() != expected:
        raise ValueError(
            f"AllGather output has {output.numel()} elements, expected {expected}"
        )


def all_gather_into_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    group: Any = None,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Gather into ``output`` using pynccl or the original C10D fallback.

    The pynccl path is stream-asynchronous. Its GLM5 callers immediately consume
    ``output`` on the same stream, which supplies the required dependency.
    """
    process_group = _resolve_process_group(group)
    if not enabled():
        dist.all_gather_into_tensor(output, input, group=process_group)
        return

    # Full Python-side validation is useful while bringing up a new shape, but
    # costs a meaningful fraction of an 11-us pynccl launch. Production relies
    # on the fixed GLM5 call-site contracts; opt in when diagnosing a new path.
    if _PYNCCL_VALIDATE:
        _validate_all_gather(output, input, process_group)
    gather_stream = stream or torch.cuda.current_stream(input.device)
    lib, comm = _require_comm(process_group, input.device)
    lib.all_gather(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        input.dtype,
        comm,
        gather_stream.cuda_stream,
    )


def all_gather(
    input: torch.Tensor,
    group: Any = None,
    *,
    role: str | None = None,
) -> torch.Tensor:
    """GLM5 dispatcher: symmetric window, ordinary pynccl, then C10D."""
    if not enabled():
        return _torch_all_gather(input, group=group)
    if input.dim() == 0:
        raise ValueError("GLM5 CP AllGather expects a tensor with at least one dimension")

    process_group = _resolve_process_group(group)
    world_size = _world_size(process_group)
    output_shape = (world_size * input.shape[0], *input.shape[1:])
    output = _symm_output(
        role,
        output_shape,
        input.dtype,
        input.device,
        process_group,
    )
    gather_stream = None
    if output is not None:
        gather_stream = torch.cuda.current_stream(input.device)
        if not _symm_stream_allowed(
            role, gather_stream, process_group, input.device
        ):
            # A separate stream could still be reading the persistent role
            # buffer. Use an ordinary output instead of adding hot-path events.
            output = None
    if output is None:
        output = torch.empty(
            output_shape,
            dtype=input.dtype,
            device=input.device,
        )
        all_gather_into_tensor(output, input, group=process_group)
        return output

    if _PYNCCL_VALIDATE:
        _validate_all_gather(output, input, process_group)
    assert gather_stream is not None
    _pynccl_symm_all_gather(output, input, gather_stream, process_group)
    return output
