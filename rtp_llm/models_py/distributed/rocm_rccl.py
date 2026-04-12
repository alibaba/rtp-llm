"""ROCm-only RCCL graph-capture helpers.

This module exists solely for the ROCm graph-capture path. CUDA runtime does
not use these helpers and continues to use the default
torch.distributed/NCCL flow. All public entry points in this module are
guarded by ROCm runtime checks so importing it on CUDA remains harmless.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.distributed

from rtp_llm.ops import ParallelismConfig, rtp_llm_ops

_NCCL_SUCCESS = 0
_NCCL_SUM = 0
# ncclDataType_t enum values from NCCL/RCCL 2.x headers (nccl.h).
# See: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html
#   0 = ncclInt8,   1 = ncclUint8,  2 = ncclInt32,  3 = ncclUint32,
#   4 = ncclInt64,  5 = ncclUint64, 6 = ncclFloat16, 7 = ncclFloat32,
#   8 = ncclFloat64, 9 = ncclBfloat16, 10 = ncclFp8E4M3, 11 = ncclFp8E5M2
_NCCL_DTYPE_MAP = {
    torch.int8: 0,  # ncclInt8
    torch.uint8: 1,  # ncclUint8
    torch.int32: 2,  # ncclInt32
    torch.int64: 4,  # ncclInt64
    torch.float16: 6,  # ncclFloat16
    torch.float32: 7,  # ncclFloat32
    torch.float64: 8,  # ncclFloat64
    torch.bfloat16: 9,  # ncclBfloat16
}
if hasattr(torch, "uint32"):
    _NCCL_DTYPE_MAP[torch.uint32] = 3
if hasattr(torch, "uint64"):
    _NCCL_DTYPE_MAP[torch.uint64] = 5
# RCCL only exposes two FP8 enums today: E4M3(10) and E5M2(11). PyTorch's
# fn/fnuz variants map to the same RCCL enum values.
if hasattr(torch, "float8_e4m3fn"):
    _NCCL_DTYPE_MAP[torch.float8_e4m3fn] = 10
if hasattr(torch, "float8_e4m3fnuz"):
    _NCCL_DTYPE_MAP[torch.float8_e4m3fnuz] = 10
if hasattr(torch, "float8_e5m2"):
    _NCCL_DTYPE_MAP[torch.float8_e5m2] = 11
if hasattr(torch, "float8_e5m2fnuz"):
    _NCCL_DTYPE_MAP[torch.float8_e5m2fnuz] = 11

_rccl_lib: Optional[ctypes.CDLL] = None
_rccl_comm: Optional[ctypes.c_void_p] = None
_rccl_world_size: int = 1
_rccl_comm_owned_by_python: bool = False
_is_rocm_runtime: bool = getattr(torch.version, "hip", None) is not None
# Thread safety: protected by GIL in CPython. If nogil builds are adopted,
# this global must be guarded by an explicit lock or replaced with thread-local storage.
_HipgraphAllGatherCacheKey = Tuple[Tuple[int, ...], torch.dtype, str, int]
_hipgraph_allgather_outputs: Dict[_HipgraphAllGatherCacheKey, torch.Tensor] = {}


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def is_rocm_runtime() -> bool:
    """Whether the current torch runtime is ROCm/HIP."""
    return _is_rocm_runtime


def _get_nccl_dtype(tensor: torch.Tensor) -> int:
    nccl_dtype = _NCCL_DTYPE_MAP.get(tensor.dtype)
    if nccl_dtype is not None:
        return nccl_dtype
    supported = ", ".join(sorted(str(dtype) for dtype in _NCCL_DTYPE_MAP))
    raise TypeError(
        f"Unsupported dtype {tensor.dtype} for HIPGraph RCCL collectives. Supported dtypes: {supported}"
    )


def _get_or_create_allgather_output(tensor: torch.Tensor) -> torch.Tensor:
    expected_shape = (_rccl_world_size * tensor.shape[0], *tensor.shape[1:])
    device_index = tensor.device.index if tensor.device.index is not None else -1
    cache_key: _HipgraphAllGatherCacheKey = (
        tuple(expected_shape),
        tensor.dtype,
        tensor.device.type,
        device_index,
    )
    output = _hipgraph_allgather_outputs.get(cache_key)
    if output is not None:
        return output

    if not _is_hipgraph_capture_active():
        raise RuntimeError(
            "HIPGraph all_gather output cache miss while capture is inactive. "
            f"Refusing to allocate replay-time buffer (shape={expected_shape}, "
            f"dtype={tensor.dtype}, device={tensor.device})."
        )

    output = torch.zeros(expected_shape, device=tensor.device, dtype=tensor.dtype)
    _hipgraph_allgather_outputs[cache_key] = output
    return output


def _load_rccl() -> Optional[ctypes.CDLL]:
    global _rccl_lib
    if _rccl_lib is not None:
        return _rccl_lib
    for name in ("librccl.so.1", "librccl.so"):
        try:
            _rccl_lib = ctypes.CDLL(name)
            logging.info(f"Loaded RCCL library: {name}")
            break
        except OSError:
            continue
    if _rccl_lib is None:
        logging.warning(
            "Failed to load RCCL library (tried librccl.so.1, librccl.so). "
            "HIPGraph capture collectives will not be available."
        )
    return _rccl_lib


def _setup_rccl_api(lib: ctypes.CDLL) -> None:
    lib.ncclGetUniqueId.restype = ctypes.c_int
    lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
    lib.ncclCommInitRank.restype = ctypes.c_int
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        _NcclUniqueId,
        ctypes.c_int,
    ]
    lib.ncclCommDestroy.restype = ctypes.c_int
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ncclAllReduce.restype = ctypes.c_int
    lib.ncclAllReduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.ncclAllGather.restype = ctypes.c_int
    lib.ncclAllGather.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]


def _is_hipgraph_capture_active() -> bool:
    checker = getattr(rtp_llm_ops, "is_hipgraph_capture_enabled", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception as e:
        logging.warning(f"Failed to query HIPGraph capture state: {e}")
        return False


def _get_rccl_runtime() -> Tuple[ctypes.CDLL, ctypes.c_void_p]:
    lib = _rccl_lib if _rccl_lib is not None else _load_rccl()
    if lib is None:
        raise RuntimeError(
            "RCCL library is not available for HIPGraph capture collectives"
        )
    if _rccl_comm is None or _rccl_comm.value is None:
        raise RuntimeError(
            "RCCL communicator is not initialized for HIPGraph capture collectives"
        )
    return lib, _rccl_comm


def _clear_hipgraph_capture_nccl_comm() -> None:
    global _rccl_comm_owned_by_python
    global _rccl_comm, _rccl_world_size
    if (
        _rccl_lib is not None
        and _rccl_comm_owned_by_python
        and _rccl_comm is not None
        and _rccl_comm.value is not None
    ):
        try:
            _rccl_lib.ncclCommDestroy(_rccl_comm)
        except Exception as e:
            logging.warning(f"Failed to destroy python-owned RCCL comm: {e}")
    _rccl_comm = None
    _rccl_world_size = 1
    _rccl_comm_owned_by_python = False
    _hipgraph_allgather_outputs.clear()


def set_hipgraph_capture_nccl_comm(
    nccl_comm_handle: int, world_size: int, rank: int
) -> None:
    comm_rank = rank
    global _rccl_comm_owned_by_python
    global _rccl_comm, _rccl_world_size
    if not _is_rocm_runtime:
        return
    if nccl_comm_handle == 0 or world_size <= 1:
        _clear_hipgraph_capture_nccl_comm()
        return
    lib = _load_rccl()
    if lib is None:
        logging.warning("set_hipgraph_capture_nccl_comm: RCCL library not available")
        _clear_hipgraph_capture_nccl_comm()
        return
    _setup_rccl_api(lib)
    _rccl_comm = ctypes.c_void_p(nccl_comm_handle)
    _rccl_world_size = world_size
    _rccl_comm_owned_by_python = False
    logging.info(
        "Registered HIPGraph RCCL comm handle from C++ "
        f"(rank={comm_rank}, world_size={world_size}, handle={nccl_comm_handle})"
    )
    # Communicator/world-size changes invalidate cached all-gather buffers.
    # enter/exit capture should not clear this cache because replay relies on
    # stable addresses recorded during capture.
    _hipgraph_allgather_outputs.clear()


def bootstrap_hipgraph_capture_rccl_comm_from_tp_group(
    tp_group: torch.distributed.ProcessGroup,
) -> None:
    global _rccl_comm_owned_by_python
    global _rccl_comm, _rccl_world_size
    if not _is_rocm_runtime:
        return
    if _rccl_comm is not None and _rccl_comm.value is not None:
        return
    if not torch.distributed.is_initialized():
        return

    try:
        group_world_size = torch.distributed.get_world_size(tp_group)
        if group_world_size <= 1:
            return
        group_rank = torch.distributed.get_rank(tp_group)
        try:
            tp_ranks = torch.distributed.get_process_group_ranks(tp_group)
            src_rank = int(tp_ranks[0])
        except Exception:
            src_rank = 0

        lib = _load_rccl()
        if lib is None:
            return
        _setup_rccl_api(lib)

        uid_buffer = _NcclUniqueId()
        if group_rank == 0:
            result = lib.ncclGetUniqueId(ctypes.byref(uid_buffer))
            if result != _NCCL_SUCCESS:
                raise RuntimeError(f"ncclGetUniqueId failed with error code {result}")

        uid_bytes = ctypes.string_at(
            ctypes.byref(uid_buffer), ctypes.sizeof(uid_buffer)
        )
        uid_tensor = torch.tensor(
            list(uid_bytes),
            dtype=torch.uint8,
            device=torch.cuda.current_device(),
        )
        torch.distributed.broadcast(uid_tensor, src=src_rank, group=tp_group)

        uid_values = bytes(int(v) for v in uid_tensor.cpu().tolist())
        uid_buffer = _NcclUniqueId.from_buffer_copy(uid_values)

        comm_ptr = ctypes.c_void_p()
        result = lib.ncclCommInitRank(
            ctypes.byref(comm_ptr), group_world_size, uid_buffer, group_rank
        )
        if result != _NCCL_SUCCESS:
            raise RuntimeError(f"ncclCommInitRank failed with error code {result}")

        _rccl_comm = comm_ptr
        _rccl_world_size = group_world_size
        _rccl_comm_owned_by_python = True
        _hipgraph_allgather_outputs.clear()
        logging.info(
            "Bootstrapped HIPGraph RCCL comm from TP group "
            f"(group_rank={group_rank}, world_size={group_world_size})"
        )
    except Exception as e:
        logging.warning(
            "Failed to bootstrap HIPGraph RCCL comm from TP group: "
            f"{e}. Capture will fallback to torch.distributed path."
        )


def prepare_hipgraph_capture_rccl_comm_if_needed(
    parallelism_config: ParallelismConfig,
    tp_group: torch.distributed.ProcessGroup,
) -> None:
    if not _is_rocm_runtime:
        return
    if parallelism_config.tp_size <= 1:
        return
    # IMPORTANT: bootstrap must happen before graph capture begins.
    bootstrap_hipgraph_capture_rccl_comm_from_tp_group(tp_group)


def enter_hipgraph_capture_mode(
    nccl_comm_handle: int = 0, world_size: int = 0, rank: int = 0
) -> None:
    """Called by C++ enter_graph_capture() after setHipGraphCaptureEnabled(true).

    State ownership contract:
    - C++ owns `in_hip_graph_capture` (the atomic bool); Python only reads it
      via `is_hipgraph_capture_enabled()`.
    - Python owns `_rccl_comm` / `_rccl_world_size`; C++ only passes the handle
      as a uintptr_t, never reads it back.

    If this function raises, C++ enter_graph_capture() will:
      1. call setHipGraphCaptureEnabled(false)       — resets C++ state
      2. call set_graph_capture_nccl_comm(0, 0, rank) — triggers
         _clear_hipgraph_capture_nccl_comm(), resetting Python state
    So both sides are left in a clean state on failure.
    """
    if nccl_comm_handle != 0 and world_size > 1:
        set_hipgraph_capture_nccl_comm(nccl_comm_handle, world_size, rank)
    # Keep previously registered comm when no valid handle is provided.
    # C++ registration path is responsible for explicit clear via
    # set_hipgraph_capture_nccl_comm(0, 0, rank) when needed.


def exit_hipgraph_capture_mode() -> None:
    """Called by C++ exit_graph_capture() before setHipGraphCaptureEnabled(false).

    Capture-active state is owned exclusively by C++ (the atomic bool
    `in_hip_graph_capture`). Python reads it via `is_hipgraph_capture_enabled()`.
    Python does NOT reset it here — that is done by C++ unconditionally after
    this function returns (or throws).

    `_rccl_comm` is intentionally NOT cleared here: the communicator is reused
    across multiple capture sessions (one per graph key). It is only cleared when
    the C++ side calls set_graph_capture_nccl_comm(0, 0, rank) explicitly.
    Adding comm cleanup here would break replay on subsequent capture rounds.
    """
    return


def should_use_hipgraph_capture_rccl(is_tp_group: bool) -> bool:
    return (
        _is_rocm_runtime
        and is_tp_group
        and _is_hipgraph_capture_active()
        and _rccl_comm is not None
        and _rccl_comm.value is not None
    )


def ensure_tp_rccl_comm_for_capture(is_tp_group: bool) -> None:
    if (
        _is_rocm_runtime
        and is_tp_group
        and _is_hipgraph_capture_active()
        and (_rccl_comm is None or _rccl_comm.value is None)
    ):
        raise RuntimeError(
            "HIPGraph capture on ROCm requires an initialized TP RCCL communicator, "
            "but none is available. Please ensure TP RCCL bootstrap/registration "
            "succeeds before graph capture."
        )


def hipgraph_capture_all_reduce(tensor: torch.Tensor) -> None:
    lib, rccl_comm = _get_rccl_runtime()
    result = lib.ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        _get_nccl_dtype(tensor),
        _NCCL_SUM,
        rccl_comm,
        torch.cuda.current_stream().cuda_stream,
    )
    if result != _NCCL_SUCCESS:
        raise RuntimeError(f"ncclAllReduce failed with error code {result}")


def hipgraph_capture_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    lib, rccl_comm = _get_rccl_runtime()
    output_tensor = _get_or_create_allgather_output(tensor)
    result = lib.ncclAllGather(
        tensor.data_ptr(),
        output_tensor.data_ptr(),
        tensor.numel(),
        _get_nccl_dtype(tensor),
        rccl_comm,
        torch.cuda.current_stream().cuda_stream,
    )
    if result != _NCCL_SUCCESS:
        raise RuntimeError(f"ncclAllGather failed with error code {result}")
    return output_tensor


def configure_rocm_pg_for_hipgraph(parallelism_config: ParallelismConfig) -> None:
    if not _is_rocm_runtime:
        return
    if parallelism_config.tp_size <= 1:
        return
    # ProcessGroupNCCL watchdog/event-query path is not graph-capture-safe on ROCm.
    # Force blocking/no-async mode before any ProcessGroup is created.
    env_updates = {
        "TORCH_NCCL_ENABLE_MONITORING": "0",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
        "NCCL_ASYNC_ERROR_HANDLING": "0",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "NCCL_BLOCKING_WAIT": "1",
        "TORCH_NCCL_ENABLE_TIMING": "0",
        "NCCL_ENABLE_TIMING": "0",
        "TORCH_NCCL_RETHROW_CUDA_ERRORS": "0",
    }
    for key, value in env_updates.items():
        os.environ[key] = value


# C++ shim compatibility entrypoints.
# NOTE: They are ROCm-only in effect because underlying handlers are runtime-gated.
set_graph_capture_nccl_comm = set_hipgraph_capture_nccl_comm
enter_graph_capture_mode = enter_hipgraph_capture_mode
exit_graph_capture_mode = exit_hipgraph_capture_mode


# Backend-neutral wrappers used by collective_torch.py.
def is_available_runtime() -> bool:
    return is_rocm_runtime()


def set_capture_comm(nccl_comm_handle: int, world_size: int, rank: int) -> None:
    set_hipgraph_capture_nccl_comm(nccl_comm_handle, world_size, rank)


def prepare_comm_if_needed(
    parallelism_config: ParallelismConfig,
    tp_group: torch.distributed.ProcessGroup,
) -> None:
    prepare_hipgraph_capture_rccl_comm_if_needed(parallelism_config, tp_group)


def enter_capture_mode(
    nccl_comm_handle: int = 0, world_size: int = 0, rank: int = 0
) -> None:
    enter_hipgraph_capture_mode(nccl_comm_handle, world_size, rank)


def exit_capture_mode() -> None:
    exit_hipgraph_capture_mode()


def should_use_capture_collectives(is_tp_group: bool) -> bool:
    return should_use_hipgraph_capture_rccl(is_tp_group)


def ensure_capture_comm_ready(is_tp_group: bool) -> None:
    ensure_tp_rccl_comm_for_capture(is_tp_group)


def capture_all_reduce(tensor: torch.Tensor) -> None:
    hipgraph_capture_all_reduce(tensor)


def capture_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    return hipgraph_capture_all_gather(tensor)


def destroy_capture_comm() -> None:
    """Clean up RCCL capture comm state during distributed environment teardown.

    Must be called from destroy_distributed_environment() so that a subsequent
    re-init + bootstrap_hipgraph_capture_rccl_comm_from_tp_group() creates a
    fresh communicator instead of reusing the stale one from the destroyed
    process group.
    """
    _clear_hipgraph_capture_nccl_comm()


def configure_process_groups(parallelism_config: ParallelismConfig) -> None:
    configure_rocm_pg_for_hipgraph(parallelism_config)
