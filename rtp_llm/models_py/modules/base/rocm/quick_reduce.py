# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's QuickReduce communicator for RTP-LLM ROCm.

from __future__ import annotations

import logging
import os
import socket
from enum import Enum
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

KB = 1024
MB = 1024 * KB


def _get_ops():
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops


class QuickReduceRegime(Enum):
    FP = 0
    INT8 = 1
    INT6 = 2
    INT4 = 3


def is_weak_contiguous(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous() or (
        tensor.untyped_storage().nbytes()
        - tensor.storage_offset() * tensor.element_size()
        == tensor.numel() * tensor.element_size()
    )


def _group_backend_name(group: ProcessGroup) -> str:
    return str(dist.get_backend(group)).lower()


def _in_the_same_node(group: ProcessGroup) -> bool:
    world_size = dist.get_world_size(group=group)
    local_hostname = socket.gethostname()
    hostnames: List[Optional[str]] = [None] * world_size
    dist.all_gather_object(hostnames, local_hostname, group=group)
    return all(hostname == hostnames[0] for hostname in hostnames)


class RocmQuickReduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
    _QR_MIN_SIZE = {
        (torch.float16, 2): [1 * MB, 2 * MB, 2 * MB, 1 * MB],
        (torch.float16, 4): [1 * MB, 16 * MB, 4 * MB, 2 * MB],
        (torch.float16, 8): [16 * MB, 4 * MB, 4 * MB, 2 * MB],
        (torch.bfloat16, 2): [2 * MB, 8 * MB, 8 * MB, 8 * MB],
        (torch.bfloat16, 4): [8 * MB, 64 * MB, 64 * MB, 16 * MB],
        (torch.bfloat16, 8): [16 * MB, 2048 * MB, 2048 * MB, 2048 * MB],
    }

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        quantization: str = "FP",
        cast_bf16_to_fp16: bool = True,
        max_size_mb: Optional[int] = None,
        min_size_mb: Optional[int] = None,
        quantization_min_size_kb: Optional[int] = None,
    ) -> None:
        self.disabled = True
        self.group = group
        self._handle = None
        self._ptr = 0

        assert "nccl" not in _group_backend_name(
            group
        ), "RocmQuickReduce should be attached to a non-NCCL group."

        if not _in_the_same_node(group):
            logger.warning(
                "ROCm QuickReduce is disabled because the TP metadata group "
                "spans multiple nodes"
            )
            return

        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        self.rank = rank
        self.world_size = world_size
        if world_size == 1:
            return
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "ROCm QuickReduce is disabled due to unsupported world size %d. "
                "Supported world sizes: %s",
                world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            device_ids = [
                int(device_id) for device_id in cuda_visible_devices.split(",")
            ]
        else:
            device_ids = list(range(torch.cuda.device_count()))
        physical_device_id = device_ids[device_index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=group)
        physical_device_ids = [int(t.item()) for t in gather_list]

        fully_connected = True
        for lhs in physical_device_ids:
            for rhs in physical_device_ids:
                if lhs != rhs and not torch.cuda.can_device_access_peer(lhs, rhs):
                    fully_connected = False
                    break
            if not fully_connected:
                break
        if world_size > 2 and not fully_connected:
            logger.warning(
                "ROCm QuickReduce is disabled because more than two GPUs are "
                "not fully peer-connected"
            )
            return

        quantization = quantization.upper()
        if quantization not in QuickReduceRegime.__members__:
            raise ValueError(f"Invalid QuickReduce quantization: {quantization}")
        self.qr_quant_level = QuickReduceRegime[quantization]
        self.use_fp16_kernels = cast_bf16_to_fp16
        self.qr_quantization_min_size = (
            quantization_min_size_kb * KB
            if quantization_min_size_kb is not None
            else None
        )

        ops = _get_ops()
        qr_max_size = max_size_mb * MB if max_size_mb is not None else None
        self._handle = ops.RocmQuickReduceHandle(rank, world_size, qr_max_size)
        self._ptr = self._handle.ptr()
        self.qr_max_size = (
            qr_max_size if qr_max_size is not None else ops.rocm_quick_reduce_max_size()
        )
        self.qr_min_size = min_size_mb * MB if min_size_mb is not None else None
        self.create_shared_buffer()
        self.disabled = False

    def create_shared_buffer(self) -> None:
        ops = _get_ops()
        handle = ops.rocm_quick_reduce_get_handle(self._ptr)
        handles = [None] * self.world_size
        dist.all_gather_object(handles, handle, group=self.group)
        ops.rocm_quick_reduce_open_handles(self._ptr, handles)

    def should_quick_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        dtype = torch.float16 if self.use_fp16_kernels else inp.dtype
        min_size = self.qr_min_size
        if min_size is None:
            min_size = self._QR_MIN_SIZE[(dtype, self.world_size)][
                self.qr_quant_level.value
            ]
        return inp_size <= self.qr_max_size and inp_size >= min_size

    def _get_qr_quant_level(self, inp: torch.Tensor) -> int:
        if (
            self.qr_quantization_min_size is not None
            and inp.numel() * inp.element_size() < self.qr_quantization_min_size
        ):
            return QuickReduceRegime.FP.value
        return self.qr_quant_level.value

    def quick_all_reduce(
        self, inp: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(inp)
        _get_ops().rocm_quick_reduce_all_reduce(
            self._ptr,
            inp,
            out,
            self._get_qr_quant_level(inp),
            self.use_fp16_kernels,
        )
        return out

    def close(self) -> None:
        self._handle = None
        self._ptr = 0
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
