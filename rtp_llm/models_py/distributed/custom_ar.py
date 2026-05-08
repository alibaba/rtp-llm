"""
Custom AllReduce via CUDA IPC + NVLink P2P.

Compiled into librtp_compute_ops via Bazel (cpp/cuda/custom_allreduce/).
Used for small tensors during decode (low-latency), falls back to NCCL for prefill.

IPC initialization is deferred to the first all_reduce call so that
cudaMalloc / cudaIpcOpenMemHandle run on the actual CUDA worker thread,
matching the CUDA context that will later execute the kernels.
(RTP-LLM initialises distributed comms on MainThread but runs model
forward on a C++ worker thread named Dummy-N.)
"""

import ctypes
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_CUSTOM_AR_SIZE_THRESHOLD = 8 * 1024 * 1024  # 8MB

# ── ctypes CUDA wrappers (same approach as sglang / vllm) ────────────

cudaError_t = ctypes.c_int


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


class _CudaRT:
    """Thin ctypes wrapper around libcudart (loaded once)."""

    _lib = None

    @classmethod
    def _get_lib(cls):
        if cls._lib is None:
            cls._lib = ctypes.CDLL("libcudart.so")
        return cls._lib

    @classmethod
    def malloc(cls, size: int) -> ctypes.c_void_p:
        lib = cls._get_lib()
        ptr = ctypes.c_void_p()
        err = lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
        if err != 0:
            raise RuntimeError(f"cudaMalloc({size}) failed: err={err}")
        return ptr

    @classmethod
    def ipc_get_handle(cls, ptr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        lib = cls._get_lib()
        handle = cudaIpcMemHandle_t()
        err = lib.cudaIpcGetMemHandle(ctypes.byref(handle), ptr)
        if err != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed: err={err}")
        return handle

    @classmethod
    def ipc_open_handle(cls, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        lib = cls._get_lib()
        ptr = ctypes.c_void_p()
        cudaIpcMemLazyEnablePeerAccess = 1
        err = lib.cudaIpcOpenMemHandle(
            ctypes.byref(ptr), handle, ctypes.c_uint(cudaIpcMemLazyEnablePeerAccess)
        )
        if err != 0:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed: err={err}")
        return ptr


# ── Helper ────────────────────────────────────────────────────────────


def _is_custom_ar_disabled() -> bool:
    val = os.environ.get("FT_DISABLE_CUSTOM_AR", "1")
    return val.lower() in ("1", "true", "yes", "on")


def _detect_full_nvlink(world_size: int) -> bool:
    """Detect whether all GPUs are connected via NVLink.

    Checks nvidia-smi topo output. Falls back to True for <=2 GPUs.
    Can be overridden via CUSTOM_AR_FULL_NVLINK=0|1 env var.
    """
    env_override = os.environ.get("CUSTOM_AR_FULL_NVLINK")
    if env_override is not None:
        return env_override.lower() in ("1", "true", "yes")
    if world_size <= 2:
        return True
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi topo failed, assuming full_nvlink=False")
            return False
        gpu_lines_checked = 0
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped.startswith("GPU"):
                continue
            parts = stripped.split()
            if len(parts) < 1 + world_size:
                continue
            gpu_lines_checked += 1
            for tok in parts[1 : 1 + world_size]:
                if tok == "X":
                    continue
                if not tok.startswith("NV"):
                    logger.info(
                        f"_detect_full_nvlink: non-NVLink connection found: {tok} in line: {stripped[:80]}"
                    )
                    return False
        if gpu_lines_checked >= world_size:
            return True
        logger.warning(
            f"_detect_full_nvlink: only found {gpu_lines_checked} GPU lines, expected {world_size}"
        )
        return False
    except Exception as e:
        logger.warning(f"_detect_full_nvlink failed: {e}, assuming False")
        return False


def _get_ops():
    """Get custom allreduce ops from librtp_compute_ops."""
    try:
        from librtp_compute_ops import rtp_llm_ops

        if hasattr(rtp_llm_ops, "custom_ar_meta_size"):
            return rtp_llm_ops
    except ImportError:
        pass
    return None


# ── Communicator ──────────────────────────────────────────────────────


class CustomAllReduceCommunicator:
    """IPC-based custom all-reduce for TP decode.

    IPC buffers are lazily initialized on the first all_reduce() call,
    ensuring cudaMalloc / cudaIpcOpenMemHandle run in the same CUDA
    context (worker thread) that executes the kernels.
    """

    def __init__(self, tp_group: ProcessGroup, device: torch.device):
        self.disabled = True
        self._ptr = None
        self.max_size = _CUSTOM_AR_SIZE_THRESHOLD
        self._device = device

        self._ops = _get_ops()
        if self._ops is None:
            return

        self.rank = dist.get_rank(group=tp_group)
        self.world_size = dist.get_world_size(group=tp_group)
        if self.world_size not in (2, 4, 6, 8):
            return

        self._tp_group = tp_group
        self._gloo = dist.new_group(
            ranks=dist.get_process_group_ranks(tp_group), backend="gloo"
        )
        self._ipc_ready = False
        self._ipc_lock = threading.Lock()
        self.disabled = False
        logger.info(
            f"CustomAllReduce: rank={self.rank}, ws={self.world_size}, "
            f"device={device} (IPC deferred)"
        )

    # ── IPC buffer allocation (runs on worker thread) ─────────────

    @staticmethod
    def _create_shared_buffer(
        size: int, rank: int, world_size: int, gloo_group
    ) -> List[int]:
        """Allocate IPC-capable buffer via ctypes cudaMalloc + IPC handle exchange."""
        pointer = _CudaRT.malloc(size)
        handle = _CudaRT.ipc_get_handle(pointer)

        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=gloo_group)

        ptrs: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                ptrs.append(pointer.value)
            else:
                ptrs.append(_CudaRT.ipc_open_handle(h).value)
        logger.info(
            f"_create_shared_buffer(size={size}): rank={rank}, "
            f"ptrs=[{', '.join(hex(p) for p in ptrs)}]"
        )
        return ptrs

    def _init_ipc(self):
        """Full IPC setup — called once from worker thread."""
        device = self._device
        torch.cuda.set_device(device)
        # Also set via ctypes to make sure cudaMalloc targets the right GPU
        _CudaRT._get_lib().cudaSetDevice(ctypes.c_int(device.index))
        logger.info(
            f"CustomAllReduce._init_ipc (lazy): device={device}, "
            f"thread={threading.current_thread().name}, "
            f"cuda_device={torch.cuda.current_device()}"
        )

        meta_sz = self._ops.custom_ar_meta_size()
        self._meta_ptrs = self._create_shared_buffer(
            meta_sz + self.max_size, self.rank, self.world_size, self._gloo
        )
        self._buf_ptrs = self._create_shared_buffer(
            self.max_size, self.rank, self.world_size, self._gloo
        )

        self._rank_data = torch.empty(self.max_size, dtype=torch.uint8, device=device)
        full_nvlink = _detect_full_nvlink(self.world_size)
        logger.info(
            f"CustomAllReduce: full_nvlink={full_nvlink} (world_size={self.world_size})"
        )
        self._ptr = self._ops.custom_ar_init(
            self._meta_ptrs, self._rank_data, self.rank, full_nvlink
        )
        self._ops.custom_ar_register_buffer(self._ptr, self._buf_ptrs)
        torch.cuda.synchronize(device)
        logger.info(f"CustomAllReduce._init_ipc complete on {device}")

    def _ensure_ipc(self) -> bool:
        """Thread-safe lazy IPC initialization."""
        if self._ipc_ready:
            return True
        with self._ipc_lock:
            if self._ipc_ready:
                return True
            try:
                self._init_ipc()
                self._ipc_ready = True
                return True
            except Exception as e:
                logger.error(
                    f"CustomAllReduce lazy IPC init failed: {e}", exc_info=True
                )
                self.disabled = True
                return False

    # ── Public API ────────────────────────────────────────────────

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        sz = inp.numel() * inp.element_size()
        return inp.is_contiguous() and sz % 16 == 0 and sz <= self.max_size

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        if not self._ensure_ipc():
            raise RuntimeError("CustomAllReduce IPC not initialised")
        out = torch.empty_like(inp)
        self._ops.custom_ar_all_reduce(
            self._ptr, inp, out, self._buf_ptrs[self.rank], self.max_size
        )
        return out

    def close(self):
        if self._ptr is not None:
            try:
                self._ops.custom_ar_dispose(self._ptr)
            except Exception:
                pass
            self._ptr = None


# ── Module-level singleton ────────────────────────────────────────────

_comm: Optional[CustomAllReduceCommunicator] = None


def init_custom_ar_communicator(
    tp_group: ProcessGroup,
    device: torch.device,
) -> Optional[CustomAllReduceCommunicator]:
    global _comm
    if _is_custom_ar_disabled():
        logger.info("CustomAllReduce disabled by FT_DISABLE_CUSTOM_AR")
        return None
    if _get_ops() is None:
        return None
    try:
        c = CustomAllReduceCommunicator(tp_group, device)
        if c.disabled:
            return None
        _comm = c
        return c
    except Exception as e:
        logger.warning(f"CustomAllReduce init failed: {e}")
        return None


def get_custom_ar_communicator() -> Optional[CustomAllReduceCommunicator]:
    return _comm
