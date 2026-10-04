# SPDX-License-Identifier: Apache-2.0
"""Opt-in small BF16 sums for eight SM120 GPUs with direct peer access.

All ranks stage their input, read peers in a fixed order with FP32 accumulation,
and synchronize before reusing the shared buffer. Large or unsupported tensors
keep the existing collective backend. Graphs and streams must execute these
collectives in the same serial order on every rank, as required by NCCL.
"""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.symm_mem import _agree_across_group

_MAX_ELEMENTS = 20480  # 40 KiB; larger PCIe messages use the existing backend.


class CanonicalPeerAllReduce:
    def __init__(self, group, device, buffer, handle, peers, kernel):
        self.group = group
        self.device = device
        self.buffer = buffer
        self.handle = handle
        self.peers = peers
        self.kernel = kernel

    def should_use(self, tensor: torch.Tensor) -> bool:
        return (
            tensor.is_cuda
            and tensor.device == self.device
            and tensor.dtype == torch.bfloat16
            and tensor.is_contiguous()
            and 0 < tensor.numel() <= _MAX_ELEMENTS
        )

    def all_reduce(self, tensor: torch.Tensor, *, inplace: bool = True):
        if not self.should_use(tensor):
            raise ValueError("unsupported canonical peer all-reduce tensor")
        out = tensor if inplace else torch.empty_like(tensor)
        count = tensor.numel()
        self.buffer[:count].copy_(tensor.view(-1))
        self.handle.barrier(channel=0, timeout_ms=10000)
        self.kernel[((count + 255) // 256,)](
            *self.peers,
            out,
            count,
            BLOCK=256,
            ALIGNED=count % 256 == 0,
            num_warps=4,
            enable_fp_fusion=False,
        )
        self.handle.barrier(channel=1, timeout_ms=10000)
        return out


def init_canonical_p2p(
    group, device: torch.device, *, disable_custom_all_reduce: Optional[bool] = None
) -> Optional[CanonicalPeerAllReduce]:
    if (
        not torch.cuda.is_available()
        or torch.version.hip is not None
        or dist.get_world_size(group) != 8
    ):
        return None
    requested = (
        os.getenv("RTP_LLM_CANONICAL_P2P_ALL_REDUCE", "0") == "1"
        and disable_custom_all_reduce is not True
    )
    # Called by every member of the eligible single-node TP8 group, even when
    # disabled locally. A missing flag on one rank must not split the backend.
    if not _agree_across_group(group, requested, "canonical_p2p_enabled"):
        return None
    symm = None
    buffer = None
    kernel = None
    local_ok = False
    try:
        import torch.distributed._symmetric_memory as symm

        local_ok = (
            torch.cuda.get_device_capability(device) == (12, 0)
            and symm.get_backend("cuda") == "CUDA"
            and all(
                peer == device.index
                or torch.cuda.can_device_access_peer(device.index, peer)
                for peer in range(8)
            )
        )
        if local_ok:
            from rtp_llm.models_py.triton_kernels.peer_all_reduce import (
                _canonical_peer_sum,
            )

            kernel = _canonical_peer_sum
            buffer = symm.empty(_MAX_ELEMENTS, dtype=torch.bfloat16, device=device)
    except Exception as error:
        logging.warning("Canonical P2P allocation unavailable: %s", error)
        local_ok = False
    if not _agree_across_group(group, local_ok, "canonical_p2p_alloc"):
        return None
    communicator = None
    try:
        handle = symm.rendezvous(buffer, group)
        peers = [
            handle.get_buffer(r, (_MAX_ELEMENTS,), torch.bfloat16) for r in range(8)
        ]
        communicator = CanonicalPeerAllReduce(
            group, device, buffer, handle, peers, kernel
        )
        # N is a runtime scalar: warm full-tile and masked specializations.
        communicator.all_reduce(torch.zeros_like(buffer))
        communicator.all_reduce(torch.zeros(257, dtype=buffer.dtype, device=device))
        torch.cuda.synchronize(device)
    except Exception as error:
        logging.warning("Canonical P2P initialization unavailable: %s", error)
        communicator = None
    if not _agree_across_group(group, communicator is not None, "canonical_p2p_ready"):
        return None
    logging.info("Canonical P2P all-reduce ready: BF16 TP8, at most 40 KiB")
    return communicator
