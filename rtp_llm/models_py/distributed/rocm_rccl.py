"""ROCm HIPGraph collectives over a communicator owned by ProcessGroupNCCL.

The native communicator stored here is borrowed.  This module never creates,
aborts, or destroys an RCCL communicator.  Preparation is an eager, collective
operation; capture and replay are read-only users of the published descriptor.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
)

import torch
import torch.distributed

from rtp_llm.ops import ParallelismConfig, rtp_llm_ops

if TYPE_CHECKING:
    from rtp_llm.models_py.distributed.collective_torch import GroupRecord

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

_is_rocm_runtime = getattr(torch.version, "hip", None) is not None
_GRAPH_CONTROL_GROUP_KEY = "GRAPH_CONTROL"
_HIPGRAPH_PROCESS_GROUP_ENV = {
    "TORCH_NCCL_ENABLE_MONITORING": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
    "NCCL_ASYNC_ERROR_HANDLING": "0",
    "TORCH_NCCL_BLOCKING_WAIT": "1",
    "NCCL_BLOCKING_WAIT": "1",
    "TORCH_NCCL_ENABLE_TIMING": "0",
    "NCCL_ENABLE_TIMING": "0",
    "TORCH_NCCL_RETHROW_CUDA_ERRORS": "0",
}


def is_rocm_runtime() -> bool:
    return _is_rocm_runtime


def _is_hipgraph_capture_active() -> bool:
    try:
        checker = rtp_llm_ops.is_hipgraph_capture_enabled
    except AttributeError as exc:
        raise RuntimeError(
            "ROCm graph communication requires the "
            "is_hipgraph_capture_enabled runtime binding"
        ) from exc
    try:
        return bool(checker())
    except Exception as exc:
        raise RuntimeError("Failed to query HIPGraph capture state") from exc


def _get_nccl_dtype(tensor: torch.Tensor) -> int:
    value = _NCCL_DTYPE_MAP.get(tensor.dtype)
    if value is not None:
        return value
    supported = ", ".join(sorted(str(dtype) for dtype in _NCCL_DTYPE_MAP))
    raise TypeError(
        f"Unsupported dtype {tensor.dtype} for HIPGraph RCCL collectives. "
        f"Supported dtypes: {supported}"
    )


_rccl_lib: Optional[ctypes.CDLL] = None


def _load_rccl() -> Optional[ctypes.CDLL]:
    global _rccl_lib
    if _rccl_lib is not None:
        return _rccl_lib
    for name in ("librccl.so.1", "librccl.so"):
        try:
            _rccl_lib = ctypes.CDLL(name)
            logging.info("Loaded RCCL library: %s", name)
            break
        except OSError as exc:
            logging.warning("Failed to load RCCL library %s: %s", name, exc)
    return _rccl_lib


def _setup_rccl_api(lib: ctypes.CDLL) -> None:
    """Register only the two borrowed-communicator operations we call."""
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


@dataclass(frozen=True)
class ProcessGroupPreflight:
    one: torch.Tensor
    gathered: torch.Tensor
    accessor_kind: str
    accessor: Callable[[], int] = field(repr=False, compare=False)


class ProcessGroupCommAccessor:
    """Compatibility boundary for private ProcessGroupNCCL pointer access."""

    def _resolve(self, tp: "GroupRecord") -> Tuple[str, Callable[[], int]]:
        pg = tp.process_group
        errors = []
        device = torch.device("cuda", tp.device_index)
        try:
            accessor = pg._get_backend(device)._comm_ptr
            if callable(accessor):
                return "device_backend._comm_ptr", accessor
            errors.append("device backend _comm_ptr is not callable")
        except (AttributeError, RuntimeError) as exc:
            errors.append(f"device backend _comm_ptr failed: {exc}")
        try:
            accessor = pg._comm_ptr
            if callable(accessor):
                return "process_group._comm_ptr", accessor
            errors.append("ProcessGroup _comm_ptr is not callable")
        except (AttributeError, RuntimeError) as exc:
            errors.append(f"ProcessGroup _comm_ptr failed: {exc}")
        raise RuntimeError("; ".join(errors))

    def preflight(self, tp: "GroupRecord") -> ProcessGroupPreflight:
        if tp.device_index is None:
            raise RuntimeError("TP ProcessGroup has no bound device")
        accessor_kind, accessor = self._resolve(tp)
        device = torch.device("cuda", tp.device_index)
        one = torch.ones(1, dtype=torch.float32, device=device)
        gathered = torch.empty(len(tp.ranks), dtype=one.dtype, device=device)
        return ProcessGroupPreflight(one, gathered, accessor_kind, accessor)

    def materialize(
        self,
        tp: "GroupRecord",
        buffers: Optional[ProcessGroupPreflight] = None,
    ) -> None:
        resources = buffers if buffers is not None else self.preflight(tp)
        one, gathered = resources.one, resources.gathered
        torch.distributed.all_reduce(one, group=tp.process_group)
        torch.distributed.all_gather_into_tensor(gathered, one, group=tp.process_group)
        torch.cuda.synchronize(tp.device_index)

    def extract(
        self,
        tp: "GroupRecord",
        buffers: Optional[ProcessGroupPreflight] = None,
    ) -> int:
        accessor_kind, accessor = (
            (buffers.accessor_kind, buffers.accessor)
            if buffers is not None
            else self._resolve(tp)
        )
        pointer = int(accessor())
        if not pointer:
            raise RuntimeError(f"{accessor_kind} returned zero")
        return pointer


@dataclass(frozen=True)
class BorrowedRcclDescriptor:
    handle: int = field(repr=False)
    source_group: "GroupRecord"
    control_group: "GroupRecord"
    world_size: int
    device_index: int
    generation: int


@dataclass(frozen=True)
class GraphOwnerToken:
    token_id: int
    generation: int
    owner_id: int


CaptureSignature = Tuple[torch.Size, torch.dtype, torch.device]
CaptureBufferKey = Tuple[int, CaptureSignature]


@dataclass
class CaptureArena:
    generation: int
    graph_owner_id: int
    buffers: Dict[CaptureBufferKey, torch.Tensor] = field(default_factory=dict)
    required_signatures: List[CaptureSignature] = field(default_factory=list)
    _capture_occurrences: Dict[CaptureSignature, int] = field(default_factory=dict)

    def validate(self, token: GraphOwnerToken) -> None:
        if self.generation != token.generation or self.graph_owner_id != token.owner_id:
            raise RuntimeError(
                "HIPGraph capture arena owner/generation mismatch: "
                f"arena=({self.graph_owner_id}, {self.generation}), "
                f"token=({token.owner_id}, {token.generation})"
            )

    def record(self, signature: CaptureSignature) -> None:
        self.required_signatures.append(signature)

    def begin_planning(self) -> None:
        self.required_signatures.clear()

    def cancel_planning(self) -> None:
        self.required_signatures.clear()

    def begin_capture(self) -> None:
        self._capture_occurrences.clear()

    def prepare(
        self,
        signatures: Optional[Iterable[CaptureSignature]] = None,
    ) -> None:
        if _is_hipgraph_capture_active():
            raise RuntimeError("CaptureArena.prepare must run before graph capture")
        occurrence_counts: Dict[CaptureSignature, int] = {}
        for signature in signatures or tuple(self.required_signatures):
            occurrence = occurrence_counts.get(signature, 0)
            occurrence_counts[signature] = occurrence + 1
            key = (occurrence, signature)
            if key not in self.buffers:
                shape, dtype, device = signature
                self.buffers[key] = torch.empty(shape, dtype=dtype, device=device)
        self.required_signatures.clear()

    def require(self, signature: CaptureSignature) -> torch.Tensor:
        occurrence = self._capture_occurrences.get(signature, 0)
        self._capture_occurrences[signature] = occurrence + 1
        key = (occurrence, signature)
        try:
            return self.buffers[key]
        except KeyError as exc:
            raise RuntimeError(
                "HIPGraph all-gather output occurrence "
                f"{occurrence} for {signature} was not planned before capture"
            ) from exc


class ManagerState(Enum):
    EMPTY = "EMPTY"
    MATERIALIZING = "MATERIALIZING"
    CONSENSUS = "CONSENSUS"
    READY = "READY"
    FAILED = "FAILED"
    CLOSING = "CLOSING"


class RcclGraphCommManager:
    def __init__(self, accessor: Optional[ProcessGroupCommAccessor] = None):
        self._accessor = accessor or ProcessGroupCommAccessor()
        self._state = ManagerState.EMPTY
        self._descriptor: Optional[BorrowedRcclDescriptor] = None
        self._tokens: Dict[int, GraphOwnerToken] = {}
        self._arenas: Dict[int, CaptureArena] = {}
        self._next_token = 1
        self._planning_token: Optional[int] = None
        self._capture_token: Optional[int] = None
        self._lock = threading.RLock()

    @property
    def state(self) -> ManagerState:
        with self._lock:
            return self._state

    @property
    def descriptor(self) -> Optional[BorrowedRcclDescriptor]:
        with self._lock:
            return self._descriptor

    @property
    def owner_count(self) -> int:
        with self._lock:
            return len(self._tokens)

    def assert_can_shutdown(self) -> None:
        with self._lock:
            if self._tokens:
                owners = sorted(token.owner_id for token in self._tokens.values())
                raise RuntimeError(
                    "Cannot destroy distributed environment with live graph owners: "
                    f"{owners}"
                )

    def _matches(self, tp: "GroupRecord", control: "GroupRecord") -> bool:
        desc = self._descriptor
        return bool(
            desc
            and desc.source_group == tp
            and desc.control_group == control
            and desc.generation == tp.generation
            and desc.device_index == tp.device_index
        )

    def prepare(
        self, tp: "GroupRecord", control: "GroupRecord"
    ) -> BorrowedRcclDescriptor:
        try:
            return self._prepare(tp, control)
        except Exception:
            with self._lock:
                if self._state in (
                    ManagerState.MATERIALIZING,
                    ManagerState.CONSENSUS,
                ):
                    self._state = ManagerState.FAILED
                    self._descriptor = None
            raise

    def _prepare(
        self, tp: "GroupRecord", control: "GroupRecord"
    ) -> BorrowedRcclDescriptor:
        with self._lock:
            if self._state == ManagerState.READY:
                if self._matches(tp, control):
                    return self._descriptor  # type: ignore[return-value]
                raise RuntimeError(
                    "RCCL graph communicator is ready for a different group"
                )
            if self._state == ManagerState.FAILED:
                raise RuntimeError(
                    "RCCL graph communicator preparation previously failed in this generation"
                )
            if self._state != ManagerState.EMPTY:
                raise RuntimeError(
                    f"Invalid RCCL graph manager state {self._state.value}"
                )
            self._state = ManagerState.MATERIALIZING

        local_error = None
        handle = 0
        buffers = None
        try:
            buffers = self._accessor.preflight(tp)
            lib = _load_rccl()
            if lib is None:
                raise RuntimeError("RCCL library or required symbols are unavailable")
            _setup_rccl_api(lib)
        except Exception as exc:
            local_error = str(exc)

        try:
            preflight = {
                "success": local_error is None,
                "rank": torch.distributed.get_rank(),
                "tp_ranks": tuple(tp.ranks),
                "device": tp.device_index,
                "backend": tp.backend,
                "generation": tp.generation,
                "accessor": (
                    buffers.accessor_kind if buffers is not None else "unavailable"
                ),
                "torch_version": torch.__version__,
                "error": local_error,
            }
            preflight_results = [None] * len(tp.ranks)
            torch.distributed.all_gather_object(
                preflight_results, preflight, group=control.process_group
            )
        except Exception:
            with self._lock:
                self._state = ManagerState.FAILED
                self._descriptor = None
            raise
        preflight_keys = (
            "tp_ranks",
            "backend",
            "generation",
            "accessor",
            "torch_version",
        )
        preflight_reference = tuple(preflight[key] for key in preflight_keys)
        if not all(
            item is not None
            and item["success"]
            and tuple(item[key] for key in preflight_keys) == preflight_reference
            for item in preflight_results
        ):
            with self._lock:
                self._state = ManagerState.FAILED
            raise RuntimeError(
                "Uniform RCCL capability preflight failed before materialization: "
                f"{preflight_results}"
            )

        local_error = None
        try:
            self._accessor.materialize(tp, buffers)
            handle = self._accessor.extract(tp, buffers)
        except Exception as exc:
            local_error = str(exc)

        metadata = {
            "success": local_error is None and handle != 0,
            "rank": torch.distributed.get_rank(),
            "tp_ranks": tuple(tp.ranks),
            "world_size": len(tp.ranks),
            "device": tp.device_index,
            "backend": tp.backend,
            "generation": tp.generation,
            "purpose": tp.purpose,
            "accessor": preflight["accessor"],
            "torch_version": torch.__version__,
            "error": local_error,
        }
        with self._lock:
            self._state = ManagerState.CONSENSUS
        gathered = [None] * len(tp.ranks)
        try:
            torch.distributed.all_gather_object(
                gathered, metadata, group=control.process_group
            )
        except Exception:
            with self._lock:
                self._state = ManagerState.FAILED
            raise

        comparable_keys = (
            "tp_ranks",
            "world_size",
            "backend",
            "generation",
            "purpose",
            "accessor",
            "torch_version",
        )
        reference = tuple(metadata[key] for key in comparable_keys)
        consensus = all(
            item is not None
            and item["success"]
            and tuple(item[key] for key in comparable_keys) == reference
            for item in gathered
        )
        devices = [item.get("device") for item in gathered if item is not None]
        if any(not isinstance(device, int) or device < 0 for device in devices):
            consensus = False
        if not consensus:
            with self._lock:
                self._state = ManagerState.FAILED
                self._descriptor = None
            details = [dict(item) for item in gathered if item is not None]
            raise RuntimeError(
                "Uniform RCCL ProcessGroup communicator preparation failed: "
                f"{details}"
            )

        descriptor = BorrowedRcclDescriptor(
            handle=handle,
            source_group=tp,
            control_group=control,
            world_size=len(tp.ranks),
            device_index=int(tp.device_index),
            generation=tp.generation,
        )
        with self._lock:
            self._descriptor = descriptor
            self._state = ManagerState.READY
        return descriptor

    def require_ready(
        self, tp: Optional["GroupRecord"] = None, device_index: Optional[int] = None
    ) -> BorrowedRcclDescriptor:
        with self._lock:
            desc = self._descriptor
            if self._state != ManagerState.READY or desc is None:
                raise RuntimeError("ROCm HIPGraph communication was not prepared")
            if tp is not None and desc.source_group != tp:
                raise RuntimeError("ROCm HIPGraph descriptor source group mismatch")
            if device_index is not None and desc.device_index != device_index:
                raise RuntimeError("ROCm HIPGraph descriptor device mismatch")
            return desc

    def acquire_graph_owner(self, owner_id: Optional[int] = None) -> GraphOwnerToken:
        with self._lock:
            desc = self.require_ready()
            token_id = self._next_token
            self._next_token += 1
            owner = token_id if owner_id is None else int(owner_id)
            token = GraphOwnerToken(token_id, desc.generation, owner)
            self._tokens[token_id] = token
            self._arenas[token_id] = CaptureArena(desc.generation, owner)
            return token

    def begin_planning(self, token_id: int, generation: int) -> None:
        with self._lock:
            token = self._require_token(token_id, generation)
            if self._planning_token not in (None, token.token_id):
                raise RuntimeError("Concurrent HIPGraph planning is not supported")
            self._arenas[token.token_id].begin_planning()
            self._planning_token = token.token_id

    def cancel_planning(self, token_id: int, generation: int) -> None:
        with self._lock:
            token = self._require_token(token_id, generation)
            if self._planning_token == token.token_id:
                self._arenas[token.token_id].cancel_planning()
                self._planning_token = None

    def _require_token(self, token_id: int, generation: int) -> GraphOwnerToken:
        token = self._tokens.get(int(token_id))
        if token is None or token.generation != int(generation):
            raise RuntimeError(
                f"Unknown or stale HIPGraph owner token {token_id} for generation {generation}"
            )
        self.require_ready()
        return token

    def prepare_arena(self, token_id: int, generation: int) -> None:
        with self._lock:
            token = self._require_token(token_id, generation)
            arena = self._arenas[token.token_id]
            arena.validate(token)
            try:
                arena.prepare()
            finally:
                if self._planning_token == token.token_id:
                    self._planning_token = None

    def enter_capture(self, token_id: int, generation: int) -> None:
        with self._lock:
            token = self._require_token(token_id, generation)
            descriptor = self.require_ready()
            current_handle = self._accessor.extract(descriptor.source_group)
            if current_handle != descriptor.handle:
                self._state = ManagerState.FAILED
                self._descriptor = None
                raise RuntimeError(
                    "TP ProcessGroup RCCL communicator changed during the active generation"
                )
            if self._capture_token is not None:
                raise RuntimeError("Concurrent HIPGraph capture is not supported")
            arena = self._arenas[token.token_id]
            arena.validate(token)
            arena.begin_capture()
            self._capture_token = token.token_id

    def exit_capture(self, token_id: int, generation: int) -> None:
        with self._lock:
            captured_token = self._capture_token
            # Capture exit must be recoverable even when validation reports an
            # owner/generation mismatch. Never strand the manager in capture.
            self._capture_token = None
            self._require_token(token_id, generation)
            if captured_token != int(token_id):
                raise RuntimeError("HIPGraph capture owner mismatch")

    def current_arena(self, capture_only: bool = False) -> Optional[CaptureArena]:
        with self._lock:
            token_id = self._capture_token
            if token_id is None and not capture_only:
                token_id = self._planning_token
            return self._arenas.get(token_id) if token_id is not None else None

    def release_graph_owner(self, token: GraphOwnerToken) -> None:
        self.release_owner(token.token_id, token.generation)

    def validate_owner(self, token_id: int, generation: int) -> None:
        with self._lock:
            self._require_token(token_id, generation)

    def release_owner_after_acquire_failure(self, owner_id: int) -> None:
        """Rollback a token when the C++ caller could not decode its result."""
        with self._lock:
            matches = [
                token
                for token in self._tokens.values()
                if token.owner_id == int(owner_id)
            ]
            if not matches:
                return
            if len(matches) != 1:
                raise RuntimeError(
                    f"Multiple HIPGraph owner tokens match owner_id {owner_id}"
                )
            token = matches[0]
        self.release_owner(token.token_id, token.generation)

    def release_owner(self, token_id: int, generation: int) -> None:
        with self._lock:
            current = self._tokens.get(int(token_id))
            if current is None or current.generation != int(generation):
                raise RuntimeError(
                    f"Unknown or stale HIPGraph owner token {token_id} "
                    f"for generation {generation}"
                )
            if self._capture_token == current.token_id:
                raise RuntimeError("Cannot release a graph owner during capture")
            self._arenas.pop(current.token_id, None)
            self._tokens.pop(current.token_id, None)
            if self._planning_token == current.token_id:
                self._planning_token = None

    def shutdown(self) -> None:
        with self._lock:
            self.assert_can_shutdown()
            self._state = ManagerState.CLOSING
            self._descriptor = None
            self._arenas.clear()
            self._planning_token = None
            self._capture_token = None
            self._state = ManagerState.EMPTY


_graph_comm_manager = RcclGraphCommManager()
_graph_communication_required: Optional[bool] = None


def _is_degenerate_graph_topology() -> bool:
    if _graph_communication_required is not None:
        return not _graph_communication_required
    return (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() <= 1
    )


def _signature(tensor: torch.Tensor, world_size: int):
    shape = torch.Size((world_size * tensor.shape[0], *tensor.shape[1:]))
    return shape, tensor.dtype, tensor.device


def record_allgather_signature(tensor: torch.Tensor, world_size: int) -> None:
    arena = _graph_comm_manager.current_arena()
    if arena is not None and not _is_hipgraph_capture_active():
        arena.record(_signature(tensor, world_size))


def _get_capture_allgather_output(tensor: torch.Tensor) -> torch.Tensor:
    desc = _graph_comm_manager.require_ready(device_index=tensor.device.index)
    arena = _graph_comm_manager.current_arena(capture_only=True)
    if arena is None:
        raise RuntimeError("HIPGraph all-gather has no active runner-owned arena")
    return arena.require(_signature(tensor, desc.world_size))


def _get_rccl_runtime() -> Tuple[ctypes.CDLL, ctypes.c_void_p]:
    desc = _graph_comm_manager.require_ready(device_index=torch.cuda.current_device())
    if _rccl_lib is None:
        raise RuntimeError("RCCL library was not loaded during graph preparation")
    return _rccl_lib, ctypes.c_void_p(desc.handle)


def _is_hidden_size_supported_for_trtllm(hidden_size: int) -> bool:
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            ALLREDUCE_SUPPORTED_HIDDEN_SIZES,
        )

        return hidden_size in ALLREDUCE_SUPPORTED_HIDDEN_SIZES
    except ImportError:
        return False


def _is_trtllm_allreduce_ready() -> bool:
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            is_trt_allreduce_ready,
        )

        return is_trt_allreduce_ready()
    except ImportError:
        return False


def hipgraph_capture_all_reduce(
    tensor: torch.Tensor,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    _validate_capture_tensor(tensor, "all-reduce input")
    _graph_comm_manager.require_ready(device_index=tensor.device.index)
    if (
        process_group is not None
        and _is_hidden_size_supported_for_trtllm(tensor.shape[-1])
        and _is_trtllm_allreduce_ready()
    ):
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
            # Once TRT has been selected, failure aborts capture on every rank;
            # there is deliberately no rank-local RCCL fallback.
            return trtllm_allreduce(
                allreduce_in=tensor,
                group=process_group,
                device_id=torch.cuda.current_device(),
            )

    lib, comm = _get_rccl_runtime()
    result = lib.ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        _get_nccl_dtype(tensor),
        _NCCL_SUM,
        comm,
        torch.cuda.current_stream().cuda_stream,
    )
    if result != _NCCL_SUCCESS:
        raise RuntimeError(f"ncclAllReduce failed with error code {result}")
    return tensor


def hipgraph_capture_all_gather(
    tensor: torch.Tensor,
) -> torch.Tensor:
    _validate_capture_tensor(tensor, "all-gather input")
    lib, comm = _get_rccl_runtime()
    output = _get_capture_allgather_output(tensor)
    _validate_capture_tensor(output, "all-gather output")
    expected_numel = tensor.numel() * _graph_comm_manager.require_ready().world_size
    if output.numel() != expected_numel:
        raise RuntimeError(
            f"HIPGraph all-gather output has {output.numel()} elements; "
            f"expected {expected_numel}"
        )
    result = lib.ncclAllGather(
        tensor.data_ptr(),
        output.data_ptr(),
        tensor.numel(),
        _get_nccl_dtype(tensor),
        comm,
        torch.cuda.current_stream().cuda_stream,
    )
    if result != _NCCL_SUCCESS:
        raise RuntimeError(f"ncclAllGather failed with error code {result}")
    return output


def process_group_device_id(
    backend: str, device_index: Optional[int], graph_required: bool
) -> Optional[torch.device]:
    """Bind ROCm NCCL groups only when graph communication needs the handle."""
    if (
        graph_required
        and _is_rocm_runtime
        and backend == "nccl"
        and device_index is not None
    ):
        return torch.device("cuda", device_index)
    return None


def _validate_capture_tensor(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(f"HIPGraph {name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise RuntimeError(f"HIPGraph {name} must be contiguous")
    descriptor = _graph_comm_manager.require_ready()
    if tensor.device.index != descriptor.device_index:
        raise RuntimeError(
            f"HIPGraph {name} device {tensor.device.index} does not match "
            f"prepared device {descriptor.device_index}"
        )


def prepare_distributed_environment(
    parallelism_config: ParallelismConfig, graph_required: bool
) -> None:
    """Apply ROCm-only settings and device binding before ProcessGroup creation."""
    global _graph_communication_required
    if not _is_rocm_runtime:
        return
    _graph_communication_required = bool(graph_required)
    if graph_required:
        _is_hipgraph_capture_active()
    # Preserve the baseline ROCm TP ProcessGroup tuning that predates graph
    # communication; it is not a graph-communication fallback mechanism.
    configure_rocm_pg_for_hipgraph(parallelism_config)
    local_rank = parallelism_config.local_rank
    device_count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(
            f"Invalid ROCm local_rank {local_rank}; visible device count is {device_count}"
        )
    torch.cuda.set_device(local_rank)


def configure_rocm_pg_for_hipgraph(parallelism_config: ParallelismConfig) -> None:
    if not _is_rocm_runtime or parallelism_config.tp_size <= 1:
        return
    for key, value in _HIPGRAPH_PROCESS_GROUP_ENV.items():
        if key in os.environ and os.environ[key] != value:
            logging.warning(
                "Overriding user-provided %s=%s with %s for ROCm HIPGraph",
                key,
                os.environ[key],
                value,
            )
        os.environ[key] = value
    logging.info(
        "Configured ROCm HIPGraph ProcessGroup environment: %s",
        _HIPGRAPH_PROCESS_GROUP_ENV,
    )


class GroupRegistry(Protocol):
    def get(self, key): ...
    def create(self, ranks, backend, timeout, device_index, graph_required): ...
    def record(self, key, record) -> None: ...


def prepare_rocm_graph_communication(
    parallelism_config: ParallelismConfig,
    tp: "GroupRecord",
    registry: GroupRegistry,
    group_timeout: timedelta,
) -> Optional[BorrowedRcclDescriptor]:
    """Create ROCm graph control state and publish the borrowed TP handle."""
    if not _is_rocm_runtime or parallelism_config.tp_size <= 1:
        return None

    control = registry.get(_GRAPH_CONTROL_GROUP_KEY)
    if control is None:
        local_control = None
        world_rank = parallelism_config.world_rank
        for dp_rank in range(parallelism_config.dp_size):
            ranks = tuple(
                range(
                    dp_rank * parallelism_config.tp_size,
                    (dp_rank + 1) * parallelism_config.tp_size,
                )
            )
            group = registry.create(list(ranks), "gloo", group_timeout, None, False)
            if world_rank in ranks:
                local_control = replace(
                    tp,
                    process_group=group,
                    ranks=tuple(tp.ranks),
                    backend="gloo",
                    device_index=None,
                    owned_by_rtp=True,
                    purpose="graph_control",
                )
        if local_control is None:
            raise RuntimeError("Failed to create the local TP graph control group")
        registry.record(_GRAPH_CONTROL_GROUP_KEY, local_control)
        control = local_control

    if _graph_comm_manager.state == ManagerState.READY:
        # Re-entry is validation only. Do not repeat TRT readiness consensus or
        # rebuild/clean a workspace that captured graphs may already reference.
        return _graph_comm_manager.prepare(tp, control)

    descriptor = _graph_comm_manager.prepare(tp, control)
    logging.info(
        "Prepared ROCm graph communication: control_ranks=%s device=%d generation=%d",
        control.ranks,
        descriptor.device_index,
        descriptor.generation,
    )
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            cleanup as cleanup_trtllm,
        )
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            ensure_trtllm_comm_initialized,
        )
    except ImportError:

        def cleanup_trtllm() -> None:
            return None

        local_trt_ready = False
    else:
        local_trt_ready = ensure_trtllm_comm_initialized(
            group=tp.process_group,
            device_id=descriptor.device_index,
            generation=descriptor.generation,
            control_group=control.process_group,
        )
    trt_readiness = [None] * len(control.ranks)
    torch.distributed.all_gather_object(
        trt_readiness,
        bool(local_trt_ready),
        group=control.process_group,
    )
    if not all(trt_readiness):
        cleanup_trtllm()
        logging.warning(
            "TRT-LLM IPC workspace is unavailable on ranks %s; all TP ranks "
            "cleaned up the workspace and will use the borrowed ProcessGroup "
            "RCCL communicator",
            [rank for rank, ready in zip(control.ranks, trt_readiness) if not ready],
        )
    return descriptor


def acquire_graph_owner(owner_id: int = 0) -> Tuple[int, int]:
    if _graph_comm_manager.state == ManagerState.EMPTY:
        if _graph_communication_required is True:
            raise RuntimeError(
                "ROCm HIPGraph communication was required but was not prepared "
                "before graph-runner construction"
            )
        if not _is_degenerate_graph_topology():
            raise RuntimeError(
                "ROCm HIPGraph topology was not declared before graph-runner "
                "construction in a multi-rank process"
            )
        return 0, 0
    token = _graph_comm_manager.acquire_graph_owner(owner_id or None)
    return token.token_id, token.generation


def prepare_capture_arena(token_id: int, generation: int) -> None:
    if token_id == 0:
        return
    _graph_comm_manager.prepare_arena(token_id, generation)


def begin_capture_planning(token_id: int, generation: int) -> None:
    if token_id == 0:
        return
    _graph_comm_manager.begin_planning(token_id, generation)


def cancel_capture_planning(token_id: int, generation: int) -> None:
    if token_id == 0:
        return
    _graph_comm_manager.cancel_planning(token_id, generation)


def enter_graph_capture_mode(token_id: int = 0, generation: int = 0) -> None:
    if token_id == 0:
        return
    _graph_comm_manager.enter_capture(token_id, generation)


def exit_graph_capture_mode(token_id: int = 0, generation: int = 0) -> None:
    if token_id == 0:
        return
    _graph_comm_manager.exit_capture(token_id, generation)


def release_graph_owner(token_id: int, generation: int) -> None:
    if token_id == 0:
        return
    _graph_comm_manager.release_owner(token_id, generation)


def release_graph_owner_after_acquire_failure(owner_id: int) -> None:
    _graph_comm_manager.release_owner_after_acquire_failure(owner_id)


def finish_hipgraph_capture_session(token_id: int, generation: int) -> None:
    if token_id == 0:
        return
    _graph_comm_manager.validate_owner(token_id, generation)
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
            finish_capture_session,
        )
    except ImportError:
        return
    finish_capture_session()


def should_use_hipgraph_capture_rccl(is_tp_group: bool) -> bool:
    return (
        _is_rocm_runtime
        and is_tp_group
        and _graph_comm_manager.state == ManagerState.READY
        and _is_hipgraph_capture_active()
    )


def ensure_tp_rccl_comm_for_capture(is_tp_group: bool) -> None:
    if not _is_rocm_runtime or not is_tp_group:
        return
    if _graph_comm_manager.state != ManagerState.READY:
        if (
            _graph_comm_manager.state == ManagerState.EMPTY
            and _is_degenerate_graph_topology()
        ):
            return
        try:
            capture_active = _is_hipgraph_capture_active()
        except RuntimeError:
            # A non-graph environment does not require the ROCm capture-state
            # binding. graph_required initialization probes it fail-fast.
            return
        if capture_active:
            _graph_comm_manager.require_ready(device_index=torch.cuda.current_device())
        return
    if _is_hipgraph_capture_active():
        _graph_comm_manager.require_ready(device_index=torch.cuda.current_device())


def try_capture_all_reduce(
    tensor: torch.Tensor,
    is_tp_group: bool,
    get_process_group: Callable[[], torch.distributed.ProcessGroup],
) -> Optional[torch.Tensor]:
    """Handle a ROCm graph all-reduce, or return ``None`` for eager routing."""
    ensure_tp_rccl_comm_for_capture(is_tp_group)
    if not should_use_hipgraph_capture_rccl(is_tp_group):
        return None
    return hipgraph_capture_all_reduce(tensor, get_process_group())


def try_capture_all_gather(
    tensor: torch.Tensor,
    is_tp_group: bool,
) -> Optional[torch.Tensor]:
    """Handle a ROCm graph all-gather, or return ``None`` for eager routing."""
    ensure_tp_rccl_comm_for_capture(is_tp_group)
    if not should_use_hipgraph_capture_rccl(is_tp_group):
        return None
    return hipgraph_capture_all_gather(tensor)


def record_eager_allgather_signature(
    tensor: torch.Tensor, is_tp_group: bool, world_size: int
) -> None:
    if _is_rocm_runtime and is_tp_group:
        record_allgather_signature(tensor, world_size)


def shutdown_graph_comm() -> None:
    global _graph_communication_required
    if not _is_rocm_runtime:
        return
    try:
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import cleanup

        cleanup()
    except ImportError:
        pass
    _graph_comm_manager.shutdown()
    _graph_communication_required = None


def assert_graph_comm_can_shutdown() -> None:
    if _is_rocm_runtime:
        _graph_comm_manager.assert_can_shutdown()
