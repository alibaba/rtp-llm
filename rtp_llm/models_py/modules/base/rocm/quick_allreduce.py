"""Aiter Quick AllReduce wrapper for ROCm.

Quick AllReduce leverages quantization (FP, FP8, INT6, INT4) for further
acceleration on ROCm MI300 series. It is designed as a complement to
custom allreduce, providing better throughput at the cost of some precision.

Uses aiter low-level ops (``init_custom_qr``, ``qr_all_reduce``, etc.)
directly, exchanging IPC handles via the NCCL group.

Environment variables:
    ROCM_ALLREDUCE_STRATEGY: set to "quick" to enable Quick AllReduce
    ROCM_QUICK_AR_QUANTIZATION: FP / FP8 / INT6 / INT4 (default: FP)
    ROCM_QUICK_AR_CAST_BF16_TO_FP16: 1 / 0 (default: 1, cast bf16 to fp16 for faster kernels)
    ROCM_QUICK_AR_MAX_SIZE_MB: max buffer size in MB (default: 0 = aiter default ~2GB)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_MB = 1024 * 1024

_QUANT_MAP = {"FP": 0, "FP8": 1, "INT6": 2, "INT4": 3, "NONE": 4}

_SUPPORTED_WORLD_SIZES = [2, 4, 8]
_SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]

_BOOL_TRUE = {"1", "true", "yes", "on", "y", "t"}
_BOOL_FALSE = {"0", "false", "no", "off", "n", "f"}


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    logger.warning(
        "Quick AllReduce: invalid %s=%r, using default=%s",
        name,
        raw,
        default,
    )
    return default


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Quick AllReduce: invalid %s=%r (expected int), using default=%d",
            name,
            raw,
            default,
        )
        return default


# Min size thresholds from aiter kernel tests (order: [FP, FP8, INT6, INT4], unit: bytes).
# Source: aiter/dist/device_communicators/quick_all_reduce.py & quick_ar_comm.h
#
# A very large threshold (e.g. 2048MB) is the aiter authors' way of saying
# "this quant level is not recommended for that (dtype, world_size)" — it
# effectively disables that level via should_use(). It is NOT a typo.
_QR_MIN_SIZE = {
    (torch.float16, 2): [1 * _MB, 2 * _MB, 2 * _MB, 1 * _MB],
    (torch.float16, 4): [1 * _MB, 16 * _MB, 4 * _MB, 2 * _MB],
    (torch.float16, 8): [16 * _MB, 4 * _MB, 4 * _MB, 8 * _MB],
    (torch.bfloat16, 2): [2 * _MB, 8 * _MB, 8 * _MB, 8 * _MB],
    (torch.bfloat16, 4): [8 * _MB, 64 * _MB, 64 * _MB, 16 * _MB],
    (torch.bfloat16, 8): [16 * _MB, 2048 * _MB, 2048 * _MB, 2048 * _MB],
}


def _all_ranks_ok(ok: bool, group: ProcessGroup, world_size: int) -> bool:
    """Return True only if every rank passed ok=True."""
    try:
        flags = [None] * world_size
        dist.all_gather_object(flags, ok, group=group)
        return all(flags)
    except Exception:
        return False


class _QuickARManager:
    """Singleton that manages aiter Quick AllReduce."""

    def __init__(self) -> None:
        self.group: Optional[ProcessGroup] = None
        self.device_id: Optional[int] = None
        self.rank: int = 0
        self.world_size: int = 1
        self.ptr: int = 0
        self.max_size: int = 0
        self.quant_level: int = 4  # NONE
        self.use_fp16_kernels: int = 1
        self.initialized: bool = False
        self.disabled: bool = False

    def initialize(self, group: ProcessGroup, device_id: int) -> None:
        if self.initialized and group == self.group and device_id == self.device_id:
            return
        # If this is a re-init (different group/device), free the previous
        # IPC buffer first to avoid leaking hipDeviceMallocUncached memory.
        if self.ptr != 0:
            try:
                import aiter as ops

                ops.qr_destroy(self.ptr)
            except Exception as exc:
                logger.warning("Quick AllReduce qr_destroy on re-init failed: %s", exc)
            self.ptr = 0
        self.initialized = False
        self.disabled = False
        self.group = group
        self.device_id = device_id

        # Read quantization level from env (default: FP = lossless)
        quant_str = os.environ.get("ROCM_QUICK_AR_QUANTIZATION", "FP").upper()
        if quant_str not in _QUANT_MAP or quant_str == "NONE":
            logger.warning(
                "Quick AllReduce: invalid ROCM_QUICK_AR_QUANTIZATION=%s, "
                "falling back to FP",
                quant_str,
            )
            quant_str = "FP"

        self.quant_level = _QUANT_MAP[quant_str]
        self.use_fp16_kernels = (
            1 if _parse_bool_env("ROCM_QUICK_AR_CAST_BF16_TO_FP16", True) else 0
        )

        # Two-phase init with all-rank consensus to avoid deadlocks when
        # init succeeds on some ranks and fails on others (e.g. one rank
        # OOMs in init_custom_qr but the others don't).
        local_ok = False
        local_err: Optional[str] = None
        try:
            import aiter as ops

            self.rank = dist.get_rank(group=group)
            self.world_size = dist.get_world_size(group=group)

            if self.world_size == 1 or self.world_size not in _SUPPORTED_WORLD_SIZES:
                local_err = f"unsupported world_size={self.world_size}"
            else:
                props = torch.cuda.get_device_properties(device_id)
                gcn_arch = getattr(props, "gcnArchName", "")
                supported_archs = ["gfx94", "gfx50"]
                if not any(gfx in gcn_arch for gfx in supported_archs):
                    local_err = f"unsupported arch {gcn_arch}"
                else:
                    torch.cuda.set_device(device_id)
                    max_size_mb = _parse_int_env("ROCM_QUICK_AR_MAX_SIZE_MB", 0)
                    max_size_bytes = max_size_mb * _MB if max_size_mb > 0 else 0

                    self.ptr = ops.init_custom_qr(
                        self.rank,
                        self.world_size,
                        max_size_bytes,
                    )
                    self.max_size = (
                        max_size_bytes if max_size_bytes > 0 else ops.qr_max_size()
                    )
                    local_ok = True
        except ImportError:
            local_err = "aiter not available"
        except Exception as exc:
            local_err = f"init_custom_qr failed: {exc}"

        # Phase 1 consensus: every rank must succeed before IPC exchange.
        if not _all_ranks_ok(local_ok, group, self.world_size):
            if local_ok and self.ptr != 0:
                try:
                    import aiter as ops

                    ops.qr_destroy(self.ptr)
                except Exception:
                    pass
                self.ptr = 0
            logger.info(
                "Quick AllReduce disabled: %s",
                local_err or "a peer rank failed init",
            )
            self.disabled = True
            self.initialized = True
            return

        # Phase 2: exchange IPC handles with consensus to prevent deadlock.
        phase2_ok = False
        try:
            import aiter as ops

            handle = ops.qr_get_handle(self.ptr)
            handles = [None] * self.world_size
            dist.all_gather_object(handles, handle, group=group)
            ops.qr_open_handles(self.ptr, handles)
            phase2_ok = True
        except Exception as exc:
            logger.warning("Quick AllReduce phase-2 IPC exchange failed: %s", exc)

        if not _all_ranks_ok(phase2_ok, group, self.world_size):
            try:
                import aiter as ops

                ops.qr_destroy(self.ptr)
            except Exception:
                pass
            self.ptr = 0
            self.disabled = True
            self.initialized = True
            return

        dist.barrier(group=group)
        self.initialized = True
        logger.info(
            "Quick AllReduce ready (device=%d, rank=%d, world_size=%d, "
            "quant=%s, use_fp16=%d, max_size=%dMB)",
            device_id,
            self.rank,
            self.world_size,
            quant_str,
            self.use_fp16_kernels,
            self.max_size // _MB,
        )

    def close(self) -> None:
        if self.ptr != 0:
            try:
                import aiter as ops

                ops.qr_destroy(self.ptr)
            except Exception as exc:
                logger.warning("Quick AllReduce qr_destroy on close failed: %s", exc)
            self.ptr = 0
        self.disabled = True
        self.initialized = False

    def ensure_initialized(self, group: ProcessGroup, device_id: int) -> bool:
        """Lazily initialize and return True if quick allreduce is usable."""
        if not self.initialized:
            self.initialize(group, device_id)
        return self.initialized and not self.disabled

    def should_use(self, tensor: Tensor, group: ProcessGroup, device_id: int) -> bool:
        """Check whether *tensor* is eligible for quick allreduce."""
        if not self.initialized or self.disabled or self.ptr == 0:
            return False
        if self.group is not group or self.device_id != device_id:
            return False
        if tensor.dtype not in _SUPPORTED_DTYPES:
            return False

        inp_size = tensor.numel() * tensor.element_size()

        # Must be multiples of 16
        if inp_size % 16 != 0:
            return False

        # Check max size
        if inp_size > self.max_size:
            return False

        # Check min size threshold.
        # When casting bf16→fp16 (use_fp16_kernels=1), the fp16 kernel thresholds
        # are only safe for FP lossless mode (quant_level==0). For lossy quant
        # levels the original dtype thresholds must be respected because aiter's
        # high thresholds intentionally disable unreliable (dtype, world_size,
        # quant) combinations.
        if self.use_fp16_kernels and self.quant_level == 0:
            dtype_for_check = torch.float16
        else:
            dtype_for_check = tensor.dtype
        min_sizes = _QR_MIN_SIZE.get((dtype_for_check, self.world_size))
        if min_sizes is not None and inp_size < min_sizes[self.quant_level]:
            return False

        return True

    def allreduce(self, tensor: Tensor, inplace: bool = False) -> Tensor:
        """AllReduce *tensor* via aiter quick reduce."""
        import aiter as ops

        out = tensor if inplace else torch.empty_like(tensor)
        ops.qr_all_reduce(
            self.ptr,
            tensor,
            out,
            self.quant_level,
            self.use_fp16_kernels,
        )
        return out


quick_ar_manager = _QuickARManager()
