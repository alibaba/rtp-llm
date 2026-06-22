# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's custom all-reduce Python communicator for RTP-LLM ROCm.

from __future__ import annotations

import logging
import os
import socket
from contextlib import contextmanager
from typing import List, Optional, cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


def _get_ops():
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops


def _custom_ar_available() -> bool:
    try:
        _get_ops().vllm_custom_ar_meta_size()
        return True
    except Exception:
        return False


custom_ar = _custom_ar_available()


def is_weak_contiguous(tensor: torch.Tensor) -> bool:
    return bool(_get_ops().vllm_custom_ar_is_weak_contiguous(tensor))


def _group_backend_name(group: ProcessGroup) -> str:
    return str(dist.get_backend(group)).lower()


def _in_the_same_node(group: ProcessGroup) -> bool:
    world_size = dist.get_world_size(group=group)
    local_hostname = socket.gethostname()
    hostnames: List[Optional[str]] = [None] * world_size
    dist.all_gather_object(hostnames, local_hostname, group=group)
    return all(hostname == hostnames[0] for hostname in hostnames)


class RocmVllmCustomAllReduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = 8192 * 1024,
    ) -> None:
        self._is_capturing = False
        self.disabled = True
        self.group = group
        self.handle = None
        self.meta_ptrs: List[int] = []
        self.buffer_ptrs: List[int] = []

        if not custom_ar:
            logger.info(
                "ROCm vLLM CustomAllreduce is disabled because native ops are unavailable"
            )
            return

        assert "nccl" not in _group_backend_name(
            group
        ), "RocmVllmCustomAllReduce should be attached to a non-NCCL group."

        if not _in_the_same_node(group):
            logger.warning(
                "ROCm vLLM CustomAllreduce is disabled because the TP metadata "
                "group spans multiple nodes"
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
                "ROCm vLLM CustomAllreduce is disabled due to unsupported world "
                "size %d. Supported world sizes: %s",
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
                "ROCm vLLM CustomAllreduce is disabled because more than two GPUs "
                "are not fully peer-connected"
            )
            return

        self.disabled = False
        self.max_size = max_size
        self.fully_connected = fully_connected
        ops = _get_ops()
        self.meta_ptrs = self.create_shared_buffer(
            ops.vllm_custom_ar_meta_size() + max_size, group=group, uncached=True
        )
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.handle = ops.VllmCustomAllReduceHandle(
            self.meta_ptrs,
            self.rank_data,
            rank,
            fully_connected,
        )
        ops.vllm_custom_ar_register_buffer(self.handle.ptr(), self.buffer_ptrs)

    @contextmanager
    def capture(self):
        try:
            self._is_capturing = True
            yield
        finally:
            self._is_capturing = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        if self.disabled or self.handle is None:
            return
        ops = _get_ops()
        handle, offset = ops.vllm_custom_ar_get_graph_buffer_ipc_meta(self.handle.ptr())
        logger.info("Registering %d HIP graph CustomAllreduce addresses", len(offset))
        all_data: list[list[list[int] | None]]
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )
        handles = cast(list[list[int]], [data[0] for data in all_data])
        offsets = cast(list[list[int]], [data[1] for data in all_data])
        ops.vllm_custom_ar_register_graph_buffers(self.handle.ptr(), handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.max_size
        return False

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        registered: bool = False,
    ) -> torch.Tensor:
        if self.handle is None:
            raise RuntimeError("ROCm vLLM CustomAllreduce handle is not initialized")
        if out is None:
            out = torch.empty_like(inp)
        ops = _get_ops()
        if registered:
            ops.vllm_custom_ar_all_reduce(self.handle.ptr(), inp, out, 0, 0)
        else:
            ops.vllm_custom_ar_all_reduce(
                self.handle.ptr(), inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_all_reduce(self, input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if self.disabled or not self.should_custom_ar(input_tensor):
            return None
        if self._is_capturing:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input_tensor, registered=True)
            return torch.empty_like(input_tensor)
        return self.all_reduce(input_tensor, registered=False)

    def close(self) -> None:
        ops = _get_ops() if self.meta_ptrs or self.buffer_ptrs else None
        if self.handle is not None:
            self.handle = None
        if ops is not None:
            if self.meta_ptrs:
                self.free_shared_buffer(
                    self.meta_ptrs, rank=getattr(self, "rank", None)
                )
                self.meta_ptrs = []
            if self.buffer_ptrs:
                self.free_shared_buffer(
                    self.buffer_ptrs, rank=getattr(self, "rank", None)
                )
                self.buffer_ptrs = []
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: Optional[ProcessGroup] = None,
        uncached: Optional[bool] = False,
    ) -> List[int]:
        del uncached
        ops = _get_ops()
        pointer, handle = ops.vllm_custom_ar_allocate_shared_buffer_and_handle(
            size_in_bytes
        )

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for idx, remote_handle in enumerate(handles):
            if idx == rank:
                pointers.append(pointer)
            else:
                pointers.append(ops.vllm_custom_ar_open_mem_handle(remote_handle))
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int],
        group: Optional[ProcessGroup] = None,
        rank: Optional[int] = None,
    ) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        _get_ops().vllm_custom_ar_free_shared_buffer(pointers[rank])
