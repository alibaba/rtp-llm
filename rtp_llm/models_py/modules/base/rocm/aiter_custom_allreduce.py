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

# Aiter custom AR kernel only handles BF16 and FP8 (e4m3fn / e4m3fnuz).
# FP16/FP32 must fall through to the next tier; otherwise the BF16-typed
# kernel reads them as garbage and produces wrong values.
_SUPPORTED_DTYPES = (
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
)


def _all_ranks_ok(ok: bool, group: ProcessGroup, world_size: int) -> bool:
    """Return True only if every rank passed ok=True."""
    try:
        flags = [None] * world_size
        dist.all_gather_object(flags, ok, group=group)
        return all(flags)
    except Exception:
        return False


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
        # Sync only the current stream — full device sync would block
        # unrelated streams (e.g. another process group's collectives).
        torch.cuda.current_stream().synchronize()

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
        # If this is a re-init (different group/device), free the previous
        # custom-AR handle and IPC buffer first to avoid leaking
        # hipDeviceMallocUncached memory across re-inits.
        if self.fa != 0:
            try:
                import aiter as ops

                ops.dispose(self.fa)
            except Exception as exc:
                logger.warning("Aiter CustomAR dispose on re-init failed: %s", exc)
            self.fa = 0
        self.buffer = None
        self._meta = None
        self._rank_data = None
        self.initialized = False
        self.disabled = False
        self.group = group
        self.device_id = device_id

        # Phase 0: every rank locally allocates the meta + buffer.
        # We must reach a consensus that ALL ranks succeeded BEFORE issuing
        # any collective (otherwise a peer's allocate_meta_buffer OOM would
        # deadlock the survivors at all_gather_into_tensor).
        local_ok = False
        local_err: Optional[str] = None
        meta = None
        rank_data = None
        buffer = None
        try:
            import aiter as ops

            self.rank = dist.get_rank(group=group)
            self.world_size = dist.get_world_size(group=group)

            if self.world_size == 1 or self.world_size not in {2, 4, 6, 8}:
                local_err = f"unsupported world_size={self.world_size}"
            else:
                torch.cuda.set_device(device_id)
                meta = ops.allocate_meta_buffer(ops.meta_size() + self.max_size * 2)
                rank_data = torch.empty(
                    8 * 1024 * 1024,
                    dtype=torch.uint8,
                    device=f"cuda:{device_id}",
                )
                # Must use allocate_meta_buffer (hipDeviceMallocUncached)
                # instead of torch.empty. On some ROCm platforms hipMalloc
                # memory does not support IPC (hipIpcOpenMemHandle returns
                # error 17).
                buffer = ops.allocate_meta_buffer(self.max_size)
                local_ok = True
        except ImportError:
            local_err = "aiter not available"
        except Exception as exc:
            local_err = f"local allocation failed: {exc}"

        # Phase 1 consensus: every rank must succeed before IPC exchange.
        if not _all_ranks_ok(local_ok, group, self.world_size):
            logger.info(
                "Aiter CustomAllreduce disabled: %s",
                local_err or "a peer rank failed init",
            )
            self.disabled = True
            self.initialized = True
            return

        # Phase 2: exchange IPC handles.
        phase2_ready = meta is not None and buffer is not None and rank_data is not None
        if not _all_ranks_ok(phase2_ready, group, self.world_size):
            logger.info("Aiter CustomAllreduce disabled: a rank not ready for phase-2")
            self.disabled = True
            self.initialized = True
            return

        phase2_ok = False
        try:
            import aiter as ops

            meta_handles, meta_offsets = self._exchange_ipc_handles(meta)
            self.fa = ops.init_custom_ar(
                meta,
                rank_data,
                meta_handles,
                meta_offsets,
                self.rank,
                True,
            )
            buf_handles, buf_offsets = self._exchange_ipc_handles(buffer)
            ops.register_buffer(self.fa, buffer, buf_handles, buf_offsets)
            self.buffer = buffer
            self._meta = meta
            self._rank_data = rank_data
            phase2_ok = True
        except Exception as exc:
            logger.warning("Aiter CustomAR phase-2 IPC exchange failed: %s", exc)

        if not _all_ranks_ok(phase2_ok, group, self.world_size):
            if self.fa != 0:
                try:
                    import aiter as ops

                    ops.dispose(self.fa)
                except Exception:
                    pass
                self.fa = 0
            self.buffer = None
            self._meta = None
            self._rank_data = None
            self.disabled = True
            self.initialized = True
            return

        dist.barrier(group=group)
        self.initialized = True

    def close(self) -> None:
        """Release the custom-AR handle and IPC buffer.

        Safe to call multiple times. Call from teardown paths
        (e.g. destroy_distributed_environment) so IPC buffers don't leak
        across process-group re-creation cycles.
        """
        if self.fa != 0:
            try:
                import aiter as ops

                ops.dispose(self.fa)
            except Exception as exc:
                logger.warning("Aiter CustomAR dispose on close failed: %s", exc)
            self.fa = 0
        self.buffer = None
        self._meta = None
        self._rank_data = None
        self.disabled = True
        self.initialized = False

    def ensure_initialized(self, group: ProcessGroup, device_id: int) -> bool:
        """Lazily initialize and return True if comm is usable."""
        if not self.initialized:
            self.initialize(group, device_id)
        return self.initialized and not self.disabled

    def should_use(self, tensor: Tensor, group: ProcessGroup, device_id: int) -> bool:
        """Check whether *tensor* is eligible for aiter CustomAllreduce.

        State-only — never triggers (re-)initialization. Call sites must
        run ``ensure_initialized`` outside of stream capture beforehand.
        """
        if not self.initialized or self.disabled or self.fa == 0:
            return False
        if self.group is not group or self.device_id != device_id:
            return False
        if tensor.dtype not in _SUPPORTED_DTYPES:
            return False
        inp_size = tensor.numel() * tensor.element_size()
        if inp_size % 16 != 0:
            return False
        # 2-stage allreduce write mode uses 2x temp buffer,
        # so effective limit is max_size / 2
        return inp_size <= self.max_size // 2

    def allreduce(self, tensor: Tensor) -> Tensor:
        """AllReduce *tensor* via aiter P2P custom allreduce.

        Supports BF16 and FP8 dtypes.
        """
        import aiter as ops

        out = torch.empty_like(tensor)
        is_fp8 = tensor.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

        ops.all_reduce(
            self.fa,
            tensor,
            out,
            False,
            is_fp8,
            self.buffer,
        )
        return out


aiter_ar_manager = _AiterARManager()
