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
        return torch.empty(expected_shape, device=tensor.device, dtype=tensor.dtype)

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


def _ensure_rccl_comm_from_process_group(
    process_group: Optional[torch.distributed.ProcessGroup],
) -> bool:
    global _rccl_comm, _rccl_world_size
    global _hipgraph_allgather_outputs
    if process_group is None:
        return False
    if _rccl_comm is not None and _rccl_comm.value is not None:
        return True
    try:
        comm_ptr = int(process_group._comm_ptr())
    except Exception as e:
        logging.warning("Failed to fetch NCCL comm from process group: %s", e)
        return False
    if comm_ptr == 0:
        return False
    lib = _load_rccl()
    if lib is None:
        return False
    _setup_rccl_api(lib)
    _rccl_comm = ctypes.c_void_p(comm_ptr)
    try:
        _rccl_world_size = torch.distributed.get_world_size(process_group)
    except Exception:
        try:
            _rccl_world_size = torch.distributed.get_world_size()
        except Exception:
            _rccl_world_size = 1
    _hipgraph_allgather_outputs.clear()
    return True


def _get_rccl_runtime(
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tuple[ctypes.CDLL, ctypes.c_void_p]:
    lib = _rccl_lib if _rccl_lib is not None else _load_rccl()
    if lib is None:
        raise RuntimeError(
            "RCCL library is not available for HIPGraph capture collectives"
        )
    if (
        _rccl_comm is None or _rccl_comm.value is None
    ) and not _is_hipgraph_capture_active():
        _ensure_rccl_comm_from_process_group(process_group)
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
    if (
        _rccl_comm is not None
        and _rccl_comm.value == nccl_comm_handle
        and _rccl_world_size == world_size
    ):
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
    # NOTE: Do NOT pre-init allreduce strategies here.
    # The C++ comm-handle registration path doesn't know which torch
    # ProcessGroup corresponds to the TP comm; pre-initializing with
    # torch.distributed.group.WORLD is incorrect when TP < WORLD (multi-node)
    # and would force a redundant re-init (with IPC-buffer churn) the moment
    # `prepare_hipgraph_capture_rccl_comm_if_needed` runs with the right
    # tp_group. Pre-init is the sole responsibility of the latter path.


# ---------------------------------------------------------------------------
# ROCm AllReduce strategy flags (single source of truth — read once at import).
#
# ROCM_ALLREDUCE_STRATEGY is an *enable-set*, not a priority list. The env
# value is parsed as a comma-separated set of tokens; token order is ignored.
# Dispatch always uses a fixed precision/speed priority:
#
#     quick → trtllm → aiter → symm_mem → NCCL
#
# Valid tokens: quick, trtllm, aiter, none (default: none)
#   quick  — aiter Quick AllReduce (quantised, fastest, lossy)
#   trtllm — trtllm allreduce kernel (lossless, hidden-size restricted)
#   aiter  — aiter P2P custom allreduce (lossless, most general)
#   none   — no accelerated kernel, fall through to symm_mem / NCCL
#
# To opt out of a tier, omit it from the env value.
# ---------------------------------------------------------------------------
_VALID_STRATEGIES = {"quick", "trtllm", "aiter", "none"}


def _parse_enabled_strategies() -> set:
    """Parse ROCM_ALLREDUCE_STRATEGY into a set of enabled tier tokens.

    Order of tokens in the env string is intentionally ignored — dispatch
    priority is fixed by the call sites (quick → trtllm → aiter).
    """
    raw = os.environ.get("ROCM_ALLREDUCE_STRATEGY", "none").lower()
    tokens = {s.strip() for s in raw.split(",") if s.strip()}
    invalid = tokens - _VALID_STRATEGIES
    if invalid:
        logging.warning(
            "ROCM_ALLREDUCE_STRATEGY: ignoring unknown strategies: %s", invalid
        )
        tokens -= invalid
    return tokens if tokens else {"none"}


_rocm_allreduce_strategies: set = (
    _parse_enabled_strategies() if _is_rocm_runtime else {"none"}
)
_enable_quick_allreduce: bool = "quick" in _rocm_allreduce_strategies
_enable_trtllm_allreduce: bool = "trtllm" in _rocm_allreduce_strategies
_enable_aiter_custom_ar: bool = "aiter" in _rocm_allreduce_strategies

if _is_rocm_runtime and _rocm_allreduce_strategies != {"none"}:
    logging.info(
        "ROCm AllReduce enabled tiers (dispatch order is fixed "
        "quick→trtllm→aiter→symm_mem→NCCL): %s",
        sorted(_rocm_allreduce_strategies - {"none"}),
    )


def _pre_init_allreduce_strategies(
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Pre-initialize allreduce backends before graph capture.

    Must be called before entering graph capture mode so that operations
    like hipMalloc, all_gather_object, and init_custom_qr run outside of
    stream capture where they are forbidden.

    Pre-initializes every strategy enabled via ROCM_ALLREDUCE_STRATEGY,
    including aiter custom AR (even though it is not used during capture)
    so that the first eager-mode prefill doesn't pay the IPC-handshake
    cost on the critical path.

    Each manager owns its own intra-init dist.barrier(); calling them in
    sequence yields N barriers when N strategies are enabled. Acceptable
    one-time cost — kept in the manager so lazy-init paths stay correct.
    """
    if not _is_rocm_runtime:
        return
    if _rccl_world_size <= 1:
        return
    if not torch.distributed.is_initialized():
        return
    if tp_group is None:
        tp_group = torch.distributed.group.WORLD
    device_id = torch.cuda.current_device()

    if _enable_trtllm_allreduce:
        try:
            from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
                ensure_trtllm_comm_initialized,
            )

            ensure_trtllm_comm_initialized(group=tp_group, device_id=device_id)
            logging.info(
                "Pre-init trtllm_allreduce succeeded (device_id=%s)", device_id
            )
        except Exception as exc:
            logging.warning("Pre-init trtllm_allreduce failed (non-fatal): %s", exc)

    if _enable_quick_allreduce:
        try:
            from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
                quick_ar_manager,
            )

            quick_ar_manager.ensure_initialized(group=tp_group, device_id=device_id)
            logging.info("Pre-init quick_allreduce succeeded (device_id=%s)", device_id)
        except Exception as exc:
            logging.warning("Pre-init quick_allreduce failed (non-fatal): %s", exc)

    if _enable_aiter_custom_ar:
        # aiter custom AR is not used during graph capture, but pre-initializing
        # here keeps the first-iteration latency predictable (avoids running
        # all_gather_into_tensor + cudaSynchronize on the critical path of the
        # first eager-mode prefill).
        try:
            from rtp_llm.models_py.modules.base.rocm.aiter_custom_allreduce import (
                aiter_ar_manager,
            )

            aiter_ar_manager.ensure_initialized(group=tp_group, device_id=device_id)
            logging.info(
                "Pre-init aiter_custom_allreduce succeeded (device_id=%s)",
                device_id,
            )
        except Exception as exc:
            logging.warning(
                "Pre-init aiter_custom_allreduce failed (non-fatal): %s",
                exc,
            )


def _warmup_rccl_collectives(
    lib: ctypes.CDLL, comm: ctypes.c_void_p, world_size: int
) -> None:
    """Warmup RCCL collectives to force internal buffer allocation before graph capture."""
    try:
        device = torch.cuda.current_device()
        dummy_in = torch.zeros(1, dtype=torch.float32, device=device)
        ag_out = torch.zeros(world_size, dtype=torch.float32, device=device)
        ar_out = torch.zeros(1, dtype=torch.float32, device=device)
        stream = torch.cuda.current_stream().cuda_stream
        nccl_float = _get_nccl_dtype(dummy_in)

        res = lib.ncclAllGather(
            dummy_in.data_ptr(),
            ag_out.data_ptr(),
            dummy_in.numel(),
            nccl_float,
            comm,
            stream,
        )
        if res != _NCCL_SUCCESS:
            logging.warning("RCCL AllGather warmup returned error %d", res)

        res = lib.ncclAllReduce(
            ar_out.data_ptr(),
            ar_out.data_ptr(),
            ar_out.numel(),
            nccl_float,
            _NCCL_SUM,
            comm,
            stream,
        )
        if res != _NCCL_SUCCESS:
            logging.warning("RCCL AllReduce warmup returned error %d", res)

        torch.cuda.synchronize(device)
        logging.info("RCCL collective warmup succeeded (world_size=%d)", world_size)
    except Exception as e:
        logging.warning("RCCL collective warmup failed (non-fatal): %s", e)


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
        _warmup_rccl_collectives(lib, comm_ptr, group_world_size)
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
    # Pre-initialize enabled allreduce strategies with the correct TP group
    # so that hipgraph_capture_all_reduce can use them during graph capture.
    _pre_init_allreduce_strategies(tp_group)


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

    consume_capture is deferred to finish_hipgraph_capture_session() to
    avoid ProcessGroupNCCL watchdog hipErrorCapturedEvent races.
    """
    return


def finish_hipgraph_capture_session() -> None:
    """Finalize pending trt_allreduce IPC handles after a full capture loop.

    Must be called outside of any graph capture, after captureDecode/capturePrefill
    completes.  This is separated from exit_hipgraph_capture_mode to avoid
    ProcessGroupNCCL watchdog races between consecutive graph captures.
    """
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            consume_capture,
            has_pending_capture,
        )
    except ImportError:
        return
    if has_pending_capture():
        consume_capture()


def should_use_hipgraph_capture_rccl(is_tp_group: bool) -> bool:
    return _is_rocm_runtime and is_tp_group and _is_hipgraph_capture_active()


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


def _is_hidden_size_supported_for_trtllm(hidden_size: int) -> bool:
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            ALLREDUCE_SUPPORTED_HIDDEN_SIZES,
        )

        return hidden_size in ALLREDUCE_SUPPORTED_HIDDEN_SIZES
    except Exception:
        return False


def _is_trtllm_allreduce_ready() -> bool:
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            is_trt_allreduce_ready,
        )

        return is_trt_allreduce_ready()
    except ImportError:
        return False


_trtllm_fallback_warned: bool = False
_quick_fallback_warned: bool = False


def hipgraph_capture_all_reduce(
    tensor: torch.Tensor,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Allreduce during HIPGraph capture.

    Priority: quick_AR -> trtllm_AR -> ncclAllReduce.

    Both quick_AR and trtllm_AR must have been pre-initialized via
    _pre_init_allreduce_strategies() before capture begins, since their
    init paths call hipMalloc/all_gather_object which are forbidden under
    stream capture. ``should_use`` is therefore safe to call here because
    it only triggers lazy init when ``initialized`` is False — which
    cannot happen if pre-init ran for the enabled strategies.

    Note: aiter P2P custom AR is intentionally skipped in capture mode
    (its handshake/buffer-registration path is not graph-capture safe).
    """
    global _trtllm_fallback_warned, _quick_fallback_warned

    # Tier 1: Quick AllReduce (opt-in, fastest)
    if _enable_quick_allreduce and process_group is not None:
        _quick_eligible = False
        try:
            from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
                quick_ar_manager,
            )

            device_id = torch.cuda.current_device()
            _quick_eligible = quick_ar_manager.should_use(tensor, process_group, device_id)
        except Exception as exc:
            if not _quick_fallback_warned:
                logging.warning(
                    "quick_allreduce eligibility check failed in graph capture: %s",
                    exc,
                )
                _quick_fallback_warned = True
        if _quick_eligible:
            return quick_ar_manager.allreduce(tensor, inplace=True)

    # Tier 2: trtllm allreduce
    _trtllm_eligible = False
    if (
        _enable_trtllm_allreduce
        and process_group is not None
        and _is_hidden_size_supported_for_trtllm(tensor.shape[-1])
        and _is_trtllm_allreduce_ready()
    ):
        try:
            from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
                _trtllm_comm_manager,
            )
            from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
                allreduce as trtllm_allreduce,
            )

            if (
                tensor.numel() * tensor.element_size()
                <= _trtllm_comm_manager.dist_env.max_size_in_bytes
            ):
                _trtllm_eligible = True
        except ImportError as exc:
            if not _trtllm_fallback_warned:
                logging.warning(
                    "trtllm_allreduce import failed in graph capture: %s", exc,
                )
                _trtllm_fallback_warned = True
    if _trtllm_eligible:
        device_id = torch.cuda.current_device()
        return trtllm_allreduce(
            allreduce_in=tensor,
            group=process_group,
            device_id=device_id,
        )

    # Fallback to lib.ncclAllReduce (in-place, returns original tensor)
    lib, rccl_comm = _get_rccl_runtime(process_group)
    nccl_result = lib.ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        _get_nccl_dtype(tensor),
        _NCCL_SUM,
        rccl_comm,
        torch.cuda.current_stream().cuda_stream,
    )
    if nccl_result != _NCCL_SUCCESS:
        raise RuntimeError(f"ncclAllReduce failed with error code {nccl_result}")
    return tensor


def hipgraph_capture_all_gather(
    tensor: torch.Tensor,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    lib, rccl_comm = _get_rccl_runtime(process_group)
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


def capture_all_reduce(
    tensor: torch.Tensor,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    return hipgraph_capture_all_reduce(tensor, process_group)


def capture_all_gather(
    tensor: torch.Tensor,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    return hipgraph_capture_all_gather(tensor, process_group)


def _close_allreduce_strategies() -> None:
    """Release IPC buffers held by enabled allreduce strategies.

    Idempotent — safe to call when nothing was initialized. Wraps each
    strategy in its own try/except so one failure cannot prevent the
    others from cleaning up.
    """
    if _enable_quick_allreduce:
        try:
            from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
                quick_ar_manager,
            )

            quick_ar_manager.close()
        except Exception as exc:
            logging.warning("quick_ar_manager.close() failed: %s", exc)
    if _enable_aiter_custom_ar:
        try:
            from rtp_llm.models_py.modules.base.rocm.aiter_custom_allreduce import (
                aiter_ar_manager,
            )

            aiter_ar_manager.close()
        except Exception as exc:
            logging.warning("aiter_ar_manager.close() failed: %s", exc)
    if _enable_trtllm_allreduce:
        try:
            from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
                _trtllm_comm_manager,
            )

            _trtllm_comm_manager.cleanup()
        except Exception as exc:
            logging.warning("trtllm_comm_manager.cleanup() failed: %s", exc)


def destroy_capture_comm() -> None:
    """Clean up RCCL capture comm state during distributed environment teardown.

    Must be called from destroy_distributed_environment() so that a subsequent
    re-init + bootstrap_hipgraph_capture_rccl_comm_from_tp_group() creates a
    fresh communicator instead of reusing the stale one from the destroyed
    process group.
    """
    _close_allreduce_strategies()
    _clear_hipgraph_capture_nccl_comm()


def configure_process_groups(parallelism_config: ParallelismConfig) -> None:
    configure_rocm_pg_for_hipgraph(parallelism_config)
