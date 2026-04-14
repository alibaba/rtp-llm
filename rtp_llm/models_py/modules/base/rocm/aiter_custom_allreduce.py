"""Aiter CustomAllreduce wrapper for ROCm prefill AllReduce.

Uses aiter low-level ops (``init_custom_ar``, ``all_reduce``, etc.)
directly, exchanging IPC handles via the NCCL group (same approach as
the C++ ``CustomAllReduceComm``).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 128 * 1024 * 1024  # 128 MB


class _AiterARManager:
    """Singleton that manages aiter custom AllReduce via low-level ops."""

    def __init__(self) -> None:
        self.group: Optional[ProcessGroup] = None
        self.device_id: Optional[int] = None
        self.rank: int = 0
        self.world_size: int = 1
        self.fa: int = 0
        self.buffer: Optional[Tensor] = None
        self.max_size: int = _DEFAULT_MAX_SIZE
        self.initialized = False
        self.disabled = False

    def _exchange_ipc_handles(self, local_buffer: Tensor):
        """Exchange IPC handles via NCCL all_gather.

        Same approach as C++ CustomAllReduceComm::prepareP2PBuffer_:
        copy the local IPC handle to GPU, NCCL all_gather, then copy back.
        """
        import aiter as ops

        handle_tensor = ops.get_meta_buffer_ipc_handle(local_buffer)
        handle_size = handle_tensor.numel()

        device = f"cuda:{self.device_id}"
        local_gpu = torch.empty(handle_size, dtype=torch.uint8, device=device)
        gathered_gpu = torch.empty(
            handle_size * self.world_size, dtype=torch.uint8, device=device
        )

        local_gpu.copy_(handle_tensor)
        dist.all_gather_into_tensor(gathered_gpu, local_gpu, group=self.group)
        torch.cuda.synchronize()

        gathered_cpu = gathered_gpu.cpu()
        handles = []
        offsets = []
        for i in range(self.world_size):
            start = i * handle_size
            end = start + handle_size
            handles.append(gathered_cpu[start:end].clone())
            offsets.append(0)
        return handles, offsets

    def initialize(self, group: ProcessGroup, device_id: int) -> None:
        if self.initialized and group == self.group and device_id == self.device_id:
            return
        self.fa = 0
        self.initialized = False
        self.disabled = False
        self.group = group
        self.device_id = device_id

        try:
            import aiter as ops

            self.rank = dist.get_rank(group=group)
            self.world_size = dist.get_world_size(group=group)

            if self.world_size == 1 or self.world_size not in {2, 4, 6, 8}:
                self.disabled = True
                self.initialized = True
                return

            torch.cuda.set_device(device_id)

            # 2-stage write mode needs 2x temp buffer inside meta
            meta = ops.allocate_meta_buffer(ops.meta_size() + self.max_size * 2)
            rank_data = torch.empty(
                8 * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{device_id}"
            )

            meta_handles, meta_offsets = self._exchange_ipc_handles(meta)
            self.fa = ops.init_custom_ar(
                meta,
                rank_data,
                meta_handles,
                meta_offsets,
                self.rank,
                True,
            )

            # Must use allocate_meta_buffer (hipDeviceMallocUncached) instead of
            # torch.empty.  On some ROCm platforms hipMalloc memory does not
            # support IPC (hipIpcOpenMemHandle returns error 17).
            self.buffer = ops.allocate_meta_buffer(self.max_size)
            buf_handles, buf_offsets = self._exchange_ipc_handles(self.buffer)
            ops.register_buffer(self.fa, self.buffer, buf_handles, buf_offsets)

            self._meta = meta
            self._rank_data = rank_data

            dist.barrier(group=group)

            self.initialized = True
        except ImportError:
            self.disabled = True
            self.initialized = True
        except Exception as exc:
            logger.warning("Aiter CustomAR init failed: %s", exc)
            self.disabled = True
            self.initialized = True


_mgr = _AiterARManager()


def ensure_aiter_ar_initialized(group: ProcessGroup, device_id: int) -> bool:
    """Lazily initialize and return True if comm is usable."""
    if not _mgr.initialized:
        _mgr.initialize(group, device_id)
    return _mgr.initialized and not _mgr.disabled


def should_aiter_custom_ar(tensor: Tensor) -> bool:
    """Check whether *tensor* is eligible for aiter CustomAllreduce."""
    if _mgr.disabled or _mgr.fa == 0:
        return False
    inp_size = tensor.numel() * tensor.element_size()
    if inp_size % 16 != 0:
        return False
    # 2-stage allreduce write mode uses 2x temp buffer,
    # so effective limit is max_size / 2
    return inp_size <= _mgr.max_size // 2


def aiter_custom_allreduce(tensor: Tensor) -> Tensor:
    """AllReduce *tensor* via aiter P2P custom allreduce.

    Supports BF16 and FP8 dtypes.
    """
    import aiter as ops

    out = torch.empty_like(tensor)
    is_fp8 = tensor.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

    ops.all_reduce(
        _mgr.fa,
        tensor,
        out,
        False,
        is_fp8,
        _mgr.buffer,
    )
    return out
