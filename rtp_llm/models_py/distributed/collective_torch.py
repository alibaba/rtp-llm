from __future__ import annotations

import ctypes
import gc
import logging
import os
from datetime import timedelta
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import torch
import torch.distributed

from rtp_llm.models_py.distributed.symm_mem import (
    get_symm_mem_communicator,
    init_symm_mem_communicator,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig, rtp_llm_ops


class Group(Enum):
    """Process group types for collective operations"""

    DP = "DP"
    TP = "TP"
    DP_AND_TP = "DP_AND_TP"


# Global process group storage
# Key can be Group enum or string (for multiple DP/TP groups)
_group_map: Dict[Union[Group, str], torch.distributed.ProcessGroup] = {}
_parallelism_config: Optional[ParallelismConfig] = None
_initialized: bool = False  # Track if we've initialized (to prevent double init)

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
# ncclDataType_t additions available on newer NCCL/RCCL.
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
_is_rocm_runtime: bool = getattr(torch.version, "hip", None) is not None
_enable_aiter_custom_ar: bool = os.environ.get("ENABLE_AITER_CUSTOM_AR", "0") == "1"
_enable_quick_allreduce: bool = (
    os.environ.get("AITER_QUICK_REDUCE_QUANTIZATION", "NONE") != "NONE"
)
# Thread safety: protected by GIL in CPython. If nogil builds are adopted,
# this global must be guarded by an explicit lock or replaced with thread-local storage.
_HipgraphAllGatherCacheKey = Tuple[Tuple[int, ...], torch.dtype, str, int]
_hipgraph_allgather_outputs: Dict[_HipgraphAllGatherCacheKey, torch.Tensor] = {}


def _is_hidden_size_supported_for_trtllm(hidden_size: int) -> bool:
    """Check if hidden_size is supported by trtllm allreduce kernels.

    In TP (Tensor Parallelism), allreduce happens after row-parallel linear
    layers, so the input tensor's last dimension is the full hidden_size
    (not TP-split). This matches the kernel's hidden_dim parameter which
    is taken from allreduce_in.size(-1) in C++.
    """
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            ALLREDUCE_SUPPORTED_HIDDEN_SIZES,
        )

        return hidden_size in ALLREDUCE_SUPPORTED_HIDDEN_SIZES
    except Exception:
        return False


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
    global _rccl_comm, _rccl_world_size
    global _hipgraph_allgather_outputs
    _rccl_comm = None
    _rccl_world_size = 1
    _hipgraph_allgather_outputs.clear()


def set_hipgraph_capture_nccl_comm(
    nccl_comm_handle: int, world_size: int, rank: int
) -> None:
    del rank
    global _rccl_comm, _rccl_world_size
    global _hipgraph_allgather_outputs
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
    # Communicator/world-size changes invalidate cached all-gather buffers.
    # enter/exit capture should not clear this cache because replay relies on
    # stable addresses recorded during capture.
    _hipgraph_allgather_outputs.clear()
    _pre_init_trtllm_allreduce()


def _pre_init_trtllm_allreduce() -> None:
    """Pre-initialize trt_allreduce before graph capture.

    Must be called before entering graph capture mode so that
    TrtllmDistEnv.__init__ (which does hipMalloc and dist.all_gather_object)
    runs outside of stream capture where those operations are forbidden.
    """
    if not _is_rocm_runtime:
        return
    if _parallelism_config is None or _parallelism_config.tp_size <= 1:
        return
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            ensure_trtllm_comm_initialized,
        )

        tp_group = _get_group(Group.TP)
        device_id = _parallelism_config.tp_rank
        ensure_trtllm_comm_initialized(
            dtype=torch.bfloat16,
            group=tp_group,
            device_id=device_id,
        )
    except Exception as exc:
        logging.warning("Pre-init trtllm_allreduce failed (non-fatal): %s", exc)
    # Pre-initialize quick allreduce if enabled
    if _enable_quick_allreduce:
        _pre_init_quick_allreduce()


def _pre_init_quick_allreduce() -> None:
    """Pre-initialize quick allreduce before graph capture.

    Must be called outside of stream capture because init_custom_qr
    and all_gather_object are forbidden during capture.
    """
    if not _is_rocm_runtime:
        return
    if _parallelism_config is None or _parallelism_config.tp_size <= 1:
        return
    try:
        from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
            ensure_quick_ar_initialized,
        )

        tp_group = _get_group(Group.TP)
        device_id = _parallelism_config.tp_rank
        ensure_quick_ar_initialized(group=tp_group, device_id=device_id)
    except Exception as exc:
        logging.warning("Pre-init quick_allreduce failed (non-fatal): %s", exc)


def enter_hipgraph_capture_mode(
    nccl_comm_handle: int = 0, world_size: int = 0, rank: int = 0
) -> None:
    if nccl_comm_handle != 0 and world_size > 1:
        set_hipgraph_capture_nccl_comm(nccl_comm_handle, world_size, rank)
    # Keep previously registered comm when no valid handle is provided.
    # C++ registration path is responsible for explicit clear via
    # set_hipgraph_capture_nccl_comm(0, 0, rank) when needed.


def exit_hipgraph_capture_mode() -> None:
    # Capture state is owned by C++ side and queried via is_hipgraph_capture_enabled().
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import consume_capture

        consume_capture()
    except Exception:
        pass
    return


def _should_use_hipgraph_capture_rccl(group: Group) -> bool:
    return (
        _is_rocm_runtime
        and group == Group.TP
        and _is_hipgraph_capture_active()
        and _rccl_comm is not None
        and _rccl_comm.value is not None
    )


def _is_trtllm_allreduce_ready() -> bool:
    """Check if trt_allreduce is already initialized and usable.

    Must not trigger initialization during graph capture, because
    TrtllmDistEnv.__init__ does hipMalloc and dist.all_gather_object
    which are forbidden during stream capturing.
    """
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            _trtllm_comm_manager,
        )

        return (
            _trtllm_comm_manager is not None
            and _trtllm_comm_manager.initialized
            and _trtllm_comm_manager.dist_env is not None
            and not _trtllm_comm_manager.dist_env.disabled
        )
    except Exception:
        return False


def _hipgraph_capture_all_reduce(
    tensor: torch.Tensor,
    process_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Tiered allreduce routing for HIPGraph capture (decode) mode.

    Priority order (based on benchmark results):
      1. Quick AllReduce (if enabled via AITER_QUICK_REDUCE_QUANTIZATION)
         - Fastest at all sizes, but has precision loss (~3e-2 for FP, ~3.75e-1 for FP8)
         - Controlled by env var, disabled by default
      2. trtllm allreduce
         - Best lossless option for small tensors (≤256MB)
         - Requires hidden_size in supported set
      3. NCCL (fallback)
         - Always available, best for large tensors (>256MB)
    """
    # Tier 1: Quick AllReduce (opt-in, fastest but lossy)
    if _enable_quick_allreduce:
        try:
            from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
                ensure_quick_ar_initialized,
                quick_allreduce,
                should_quick_allreduce,
            )

            if ensure_quick_ar_initialized(
                process_group, _parallelism_config.tp_rank
            ) and should_quick_allreduce(tensor):
                return quick_allreduce(tensor)
        except Exception as exc:
            logging.warning(
                "Quick AllReduce failed in decode mode, fallback to next tier: %s", exc
            )

    # Tier 2: trtllm allreduce (best lossless for small tensors)
    if (
        _is_hidden_size_supported_for_trtllm(tensor.shape[-1])
        and _is_trtllm_allreduce_ready()
    ):
        try:
            from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
                allreduce as trtllm_allreduce,
            )

            return trtllm_allreduce(
                allreduce_in=tensor,
                group=process_group,
                device_id=_parallelism_config.tp_rank,
            )
        except Exception as e:
            logging.warning(
                "trtllm_allreduce failed in graph capture mode, "
                "fallback to ncclAllReduce: %s",
                e,
            )

    # Fallback to lib.ncclAllReduce (in-place, returns original tensor)
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
    return tensor


def _hipgraph_capture_all_gather(tensor: torch.Tensor) -> torch.Tensor:
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


def init_distributed_environment(
    parallelism_config: ParallelismConfig,
    nccl_comm_config: NcclCommConfig,
    nccl_init_port: int,
    backend: str = "nccl",
    timeout: Optional[int] = None,
):
    """Initialize distributed environment and create process groups.

    This function creates DP, TP, and DP_AND_TP process groups using torch.distributed.
    It can only be called once unless destroy_distributed_environment() has been called.

    Args:
        parallelism_config: Configuration for parallelism setup (sizes, ranks, etc.)
        nccl_comm_config: NCCL config with nccl_ip (and other ports for C++ init).
        nccl_init_port: Port for torch.distributed init_process_group (tcp://ip:port).
        backend: Distributed backend (default: "nccl")
        timeout: Timeout in seconds for process group initialization

    Raises:
        RuntimeError: If already initialized and not destroyed
    """
    global _group_map, _parallelism_config, _initialized

    # Check if already initialized (and not destroyed)
    if _initialized and torch.distributed.is_initialized():
        logging.warning(
            "Distributed environment already initialized, skipping initialization"
        )
        # Still need to create groups if they don't exist
        if not _group_map:
            _create_process_groups(
                parallelism_config, backend, timedelta(seconds=timeout)
            )
        return

    assert backend in ["nccl"], "backend current only supports nccl"
    ip = nccl_comm_config.nccl_ip
    port = nccl_init_port
    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    local_rank = parallelism_config.local_rank

    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"

    # If torch.distributed is already initialized (e.g., by external code),
    # we still need to create our process groups
    if torch.distributed.is_initialized():
        logging.info("torch.distributed already initialized, creating process groups")
        _create_process_groups(parallelism_config, backend, timedelta(seconds=timeout))
        _parallelism_config = parallelism_config
        _initialized = True
        return

    logging.info(
        f"[rank: {world_rank}] initialize process_group: {ip}:{port}, rank: {world_rank}, world_size: {world_size}, "
        f"local_rank: {local_rank}, backend: {backend}, timeout: {timeout}",
    )

    if timeout is not None:
        assert isinstance(timeout, (int)), "timeout must be a number"
        assert timeout > 0, "timeout must be positive"
        timeout = timedelta(seconds=timeout)  # pyright: ignore[reportAssignmentType]

    # DP_AND_TP (global group) - initialized via init_process_group
    torch.distributed.init_process_group(
        backend=backend,
        init_method=f"tcp://{ip}:{port}",
        world_size=world_size,
        rank=world_rank,
        # device_id=torch.device(f"cuda:{local_rank}"), # https://github.com/pytorch/pytorch/pull/149144
        timeout=timeout,  # pyright: ignore[reportArgumentType]
    )
    torch.distributed.barrier(group=torch.distributed.group.WORLD)
    _group_map[Group.DP_AND_TP] = torch.distributed.group.WORLD
    logging.info(
        f"[rank: {world_rank}] Created DP_AND_TP group {torch.distributed.group.WORLD} with ranks: {list(range(world_size))}"
    )

    # Create DP and TP groups
    _create_process_groups(parallelism_config, backend, timeout)
    _parallelism_config = parallelism_config
    _initialized = True
    init_user_buffers_environment(parallelism_config)


def _create_process_groups(
    parallelism_config: ParallelismConfig,
    backend: str,
    timeout: Optional[timedelta],
):
    """Create DP and TP process groups.

    Args:
        parallelism_config: Configuration for parallelism setup
        backend: Distributed backend
        timeout: Timeout for process group creation
    """
    global _group_map

    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    tp_size = parallelism_config.tp_size
    dp_size = parallelism_config.dp_size

    if dp_size > 1 and world_size != dp_size:
        # Create all DP groups - all ranks must participate in creating all DP groups
        # DP group: ranks with the same tp_rank (i.e., world_rank % tp_size)
        # There are tp_size DP groups (one for each tp_rank value)
        for tp_rank_val in range(tp_size):
            dp_ranks = [r for r in range(world_size) if r % tp_size == tp_rank_val]
            if len(dp_ranks) > 0:
                logging.info(
                    f"[rank: {world_rank}] Creating DP group for tp_rank {tp_rank_val} with ranks: {dp_ranks}"
                )
                dp_group = torch.distributed.new_group(
                    ranks=dp_ranks,
                    backend=backend,
                    timeout=timeout,  # pyright: ignore[reportArgumentType]
                )
                # Only store the group if this rank is part of it
                if world_rank in dp_ranks:
                    group_key = Group.DP.name + str(tp_rank_val)
                    _group_map[group_key] = dp_group
                    logging.info(
                        f"[rank: {world_rank}] Stored DP group with key: {group_key} {dp_group} with ranks: {dp_ranks}"
                    )
                # All ranks must wait for group creation to complete
                torch.distributed.barrier()

    if tp_size > 1 and world_size != tp_size:
        # Create all TP groups - all ranks must participate in creating all TP groups
        # TP group: ranks with the same dp_rank (i.e., world_rank // tp_size)
        # There are dp_size TP groups (one for each dp_rank value)
        for dp_rank_val in range(dp_size):
            tp_ranks = [r for r in range(world_size) if r // tp_size == dp_rank_val]
            if len(tp_ranks) > 0:
                logging.info(
                    f"[rank: {world_rank}] Creating TP group for dp_rank {dp_rank_val} with ranks: {tp_ranks}"
                )
                tp_group = torch.distributed.new_group(
                    ranks=tp_ranks,
                    backend=backend,
                    timeout=timeout,  # pyright: ignore[reportArgumentType]
                )
                # Only store the group if this rank is part of it
                if world_rank in tp_ranks:
                    group_key = Group.TP.name + str(dp_rank_val)
                    _group_map[group_key] = tp_group
                    logging.info(
                        f"[rank: {world_rank}] Stored TP group with key: {group_key} {tp_group} with ranks: {tp_ranks}"
                    )

                init_symm_mem_communicator(tp_group)

                # All ranks must wait for group creation to complete
                torch.distributed.barrier()
    elif tp_size > 1 and world_size == tp_size:
        # Single TP group: WORLD is the TP group, init symm_mem for it
        init_symm_mem_communicator(torch.distributed.group.WORLD)


def distributed_environment_initialized() -> bool:
    """Check if distributed environment is initialized.

    Returns:
        True if distributed environment is initialized, False otherwise
    """
    return torch.distributed.is_initialized()


def init_user_buffers_environment(parallelism_config: ParallelismConfig):
    """Initialize user buffers communicator for context parallelism.

    This function initializes the user buffers communicator for CP (Context Parallelism).
    It should be called after init_distributed_environment() if cp_size > 1.

    Args:
        parallelism_config: Configuration for parallelism setup
        buffer_size: Size of the communication buffer in bytes (default: 512MB)
                    Recommended calculation: max_seq_len * sizeof(act_type) * num_kv_head * head_dim
                    where:
                    - max_seq_len: maximum sequence length
                    - sizeof(act_type): size of activation data type (e.g., 2 for fp16, 4 for fp32)
                    - num_kv_head: number of key-value heads
                    - head_dim: dimension of each attention head
    Raises:
        RuntimeError: If distributed environment is not initialized
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed environment is not initialized. "
            "Call init_distributed_environment(parallelism_config) first."
        )

    if parallelism_config.prefill_cp_config.is_enabled():
        from rtp_llm.models_py.utils.arch import is_cuda

        if is_cuda():
            from rtp_llm.models_py.distributed.user_buffers import (
                init_user_buffers_communicator,
            )

            local_rank = parallelism_config.local_rank
            world_size = parallelism_config.world_size

            buffer_size = parallelism_config.prefill_cp_config.comm_buffer_size

            logging.info(
                f"[rank: {parallelism_config.world_rank}] Initializing user buffers communicator "
                f"with buffer_size: {buffer_size}, local_rank: {local_rank}, world_size: {world_size}"
            )
            init_user_buffers_communicator(
                _get_group(Group.TP), local_rank, world_size, buffer_size
            )


def destroy_distributed_environment():
    """Destroy distributed environment and clean up process groups.

    After calling this function, init_distributed_environment() can be called again
    to reinitialize the distributed environment.
    """
    global _group_map, _parallelism_config, _initialized

    rank = torch.distributed.get_rank()
    logging.info(f"[rank: {rank}] Destroying distributed environment")

    from rtp_llm.models_py.utils.arch import is_cuda

    if is_cuda():
        from rtp_llm.models_py.distributed.user_buffers import (
            destroy_user_buffers_communicator,
        )

        destroy_user_buffers_communicator()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    _group_map.clear()
    logging.info(f"[rank: {rank}] Distributed environment destroyed")
    _parallelism_config = None
    _initialized = False
    gc.collect()


def _get_group(group: Group) -> torch.distributed.ProcessGroup:
    """Get process group for the specified group type.

    This function checks if the distributed environment is initialized.
    If not initialized and _parallelism_config is available, it will attempt to initialize.

    Args:
        group: Group type (DP, TP, or DP_AND_TP)

    Returns:
        Process group for the specified group type

    Raises:
        RuntimeError: If distributed environment is not initialized and cannot be auto-initialized
        ValueError: If group type is invalid
    """
    global _parallelism_config, _initialized

    # Check if we need to initialize
    if not torch.distributed.is_initialized() or not _initialized:
        if _parallelism_config is not None:
            # Auto-initialize if we have the config
            logging.info(
                "Auto-initializing distributed environment from stored parallelism_config"
            )
            init_distributed_environment(_parallelism_config)
        else:
            raise RuntimeError(
                "Distributed environment is not initialized. "
                "Call init_distributed_environment(parallelism_config) first, "
                "or ensure parallelism_config is available for auto-initialization."
            )

    # Determine the actual key to use in _group_map
    group_key = group
    tp_size = _parallelism_config.tp_size
    dp_size = _parallelism_config.dp_size
    world_size = _parallelism_config.world_size
    if group == Group.DP and dp_size > 1 and world_size != dp_size:
        tp_rank = torch.distributed.get_rank() % tp_size
        group_key = Group.DP.name + str(tp_rank)
    elif group == Group.TP and tp_size > 1 and world_size != tp_size:
        dp_rank = torch.distributed.get_rank() // tp_size
        group_key = Group.TP.name + str(dp_rank)
    else:
        # DP_AND_TP always uses Group.DP_AND_TP as key
        group_key = Group.DP_AND_TP

    if group_key not in _group_map:
        raise ValueError(
            f"Process group {group_key} not found. Make sure init_distributed_environment() was called."
        )

    return _group_map[group_key]


# 需要注意：调用 send/recv 时如果某些 rank 没有操作，就没有对应的 ncclgroupstart/ncclgroupend
# 这样直接使用 torch 的 send/recv 是错误的。
def send(tensor: torch.Tensor, dst: int, group: Group) -> None:
    """Send a tensor to a destination rank.

    Args:
        tensor: Tensor to send
        dst: Destination global rank
        group: Process group to use
    """
    process_group = _get_group(group)
    torch.distributed.send(tensor, dst, group=process_group)


def recv(tensor: torch.Tensor, src: int, group: Group) -> torch.Tensor:
    """Receive a tensor from a source rank.

    Args:
        tensor: Tensor to receive into
        src: Source global rank
        group: Process group to use

    Returns:
        Received tensor (same as input tensor)
    """
    process_group = _get_group(group)
    torch.distributed.recv(tensor, src, group=process_group)
    return tensor


def broadcast(tensor: torch.Tensor, src: int, group: Group) -> None:
    """Broadcast a tensor from source rank to all ranks in the group.

    Args:
        tensor: Tensor to broadcast (will be modified on non-source ranks)
        src: Source global rank
        group: Process group to use
    """
    process_group = _get_group(group)
    torch.distributed.broadcast(tensor, src, group=process_group)


def _should_use_trtllm_allreduce(tensor: torch.Tensor, group: Group) -> bool:
    """Check if trtllm allreduce should be used outside of graph capture.

    When HIP Graph mode is enabled, trtllm allreduce comm is pre-initialized.
    We can reuse it for prefill (non-capture) passes as well, as long as the
    tensor fits within the IPC shared-memory buffer.
    """
    if not _is_rocm_runtime or group != Group.TP:
        return False
    if not _is_hidden_size_supported_for_trtllm(tensor.shape[-1]):
        return False
    if not _is_trtllm_allreduce_ready():
        return False
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            _trtllm_comm_manager,
        )

        max_size = _trtllm_comm_manager.dist_env.max_size_in_bytes
        tensor_bytes = tensor.numel() * tensor.element_size()
        return tensor_bytes <= max_size
    except Exception:
        return False


def _should_use_aiter_custom_ar(tensor: torch.Tensor, group: Group) -> bool:
    """Check if aiter CustomAllreduce should be used for prefill AllReduce."""
    if not _enable_aiter_custom_ar:
        return False
    if not _is_rocm_runtime or group != Group.TP:
        return False
    if _is_hipgraph_capture_active():
        return False
    try:
        from rtp_llm.models_py.modules.base.rocm.aiter_custom_allreduce import (
            ensure_aiter_ar_initialized,
            should_aiter_custom_ar,
        )

        if not ensure_aiter_ar_initialized(
            _get_group(group), _parallelism_config.tp_rank
        ):
            return False
        return should_aiter_custom_ar(tensor)
    except Exception:
        return False


def _should_use_quick_allreduce(tensor: torch.Tensor, group: Group) -> bool:
    """Check if aiter Quick AllReduce should be used."""
    if not _enable_quick_allreduce:
        return False
    if not _is_rocm_runtime or group != Group.TP:
        return False
    try:
        from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
            ensure_quick_ar_initialized,
            should_quick_allreduce,
        )

        if not ensure_quick_ar_initialized(
            _get_group(group), _parallelism_config.tp_rank
        ):
            return False
        return should_quick_allreduce(tensor)
    except Exception:
        return False


def all_reduce(tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """All-reduce a tensor across all ranks in the group.

    priority: quick_AR → trtllm → aiter custom AR → symm_mem → NCCL

    Args:
        tensor: Tensor to all-reduce (will be modified in-place)
        group: Process group to use

    Returns:
        All-reduced tensor (same as input tensor)
    """
    if _should_use_hipgraph_capture_rccl(group):
        process_group = _get_group(group)
        return _hipgraph_capture_all_reduce(tensor, process_group)

    # Tier 1: Quick AllReduce (opt-in, fastest)
    if _should_use_quick_allreduce(tensor, group):
        try:
            from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
                quick_allreduce,
            )

            return quick_allreduce(tensor)
        except Exception as e:
            logging.warning(
                "Quick AllReduce failed in prefill mode, fallback to next tier: %s",
                e,
            )

    # Tier 2: trtllm allreduce
    if _should_use_trtllm_allreduce(tensor, group):
        try:
            from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
                allreduce as trtllm_allreduce,
            )

            process_group = _get_group(group)
            return trtllm_allreduce(
                allreduce_in=tensor,
                group=process_group,
                device_id=_parallelism_config.tp_rank,
            )
        except Exception as e:
            logging.warning(
                "trtllm_allreduce failed in prefill mode, " "fallback to next tier: %s",
                e,
            )

    # Tier 3: aiter CustomAllreduce
    if _should_use_aiter_custom_ar(tensor, group):
        try:
            from rtp_llm.models_py.modules.base.rocm.aiter_custom_allreduce import (
                aiter_custom_allreduce,
            )

            return aiter_custom_allreduce(tensor)
        except Exception as e:
            logging.warning(
                "aiter CustomAllreduce failed in prefill mode, "
                "fallback to next tier: %s",
                e,
            )

    # Tier 4: symm_mem (CUDA only)
    if group == Group.TP:
        symm_mem_comm = get_symm_mem_communicator()
        if symm_mem_comm is not None and symm_mem_comm.should_torch_symm_mem_allreduce(
            tensor
        ):
            return symm_mem_comm.all_reduce(tensor)

    # Tier 5: NCCL fallback
    process_group = _get_group(group)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.SUM, group=process_group
    )
    return tensor


def all_gather(tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Gather tensors from all ranks in the group.

    Args:
        tensor: Tensor to gather from this rank
        group: Process group to use

    Returns:
        Concatenated tensor containing all gathered tensors
        (shape: [world_size * tensor.shape[0]] + list(tensor.shape)[1:])
    """
    if _should_use_hipgraph_capture_rccl(group):
        return _hipgraph_capture_all_gather(tensor)

    if group == Group.TP:
        symm_mem_comm = get_symm_mem_communicator()
        if symm_mem_comm is not None and symm_mem_comm.should_torch_symm_mem_allgather(
            tensor
        ):
            gathered = symm_mem_comm.all_gather(tensor)
            if gathered is not None:
                world_size = gathered.shape[0]
                return gathered.view(
                    [world_size * tensor.shape[0]] + list(tensor.shape)[1:]
                )

    process_group = _get_group(group)
    world_size = torch.distributed.get_world_size(process_group)

    tensor_list = torch.zeros(
        [world_size * tensor.shape[0]] + list(tensor.shape)[1:],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    torch.distributed.all_gather_into_tensor(tensor_list, tensor, group=process_group)
    return tensor_list

    # reference old implementation
    # tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    # torch.distributed.all_gather(tensor_list, tensor, group=process_group)
    # return torch.cat(tensor_list, dim=0)


def barrier(group: Group) -> None:
    """Barrier all ranks in the group.

    Args:
        group: Process group to use
    """
    process_group = _get_group(group)
    torch.distributed.barrier(group=process_group)


__all__ = [
    "Group",
    "init_distributed_environment",
    "init_user_buffers_environment",
    "distributed_environment_initialized",
    "destroy_distributed_environment",
    "set_hipgraph_capture_nccl_comm",
    "enter_hipgraph_capture_mode",
    "exit_hipgraph_capture_mode",
    "send",
    "recv",
    "broadcast",
    "all_reduce",
    "all_gather",
    "barrier",
]
