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
    def __init__(
        self,
        group,
        device,
        buffer,
        handle,
        peers,
        kernel,
        max_elements,
        slot=None,
        pingpong_kernels=None,
    ):
        self.group = group
        self.device = device
        self.buffer = buffer
        self.handle = handle
        self.peers = peers
        self.kernel = kernel
        self.max_elements = max_elements
        self.slot = slot
        self.pingpong_kernels = pingpong_kernels
        self.rank = dist.get_rank(group)
        self.signals = (
            [handle.get_signal_pad(r, (16,), torch.int32) for r in range(8)]
            if slot is not None
            else None
        )

    def should_use(self, tensor: torch.Tensor) -> bool:
        return (
            tensor.is_cuda
            and tensor.device == self.device
            and tensor.dtype == torch.bfloat16
            and tensor.is_contiguous()
            and 0 < tensor.numel() <= self.max_elements
            # Scalar peer reads regress for larger partial tiles. The extended
            # capacity is for full-tile verification rows; retain the backend
            # fallback for larger ragged messages.
            and (tensor.numel() <= 40960 or tensor.numel() % 256 == 0)
        )

    def all_reduce(self, tensor: torch.Tensor, *, inplace: bool = True):
        if not self.should_use(tensor):
            raise ValueError("unsupported canonical peer all-reduce tensor")
        out = tensor if inplace else torch.empty_like(tensor)
        count = tensor.numel()
        if self.slot is not None:
            stage, reduce, barrier = self.pingpong_kernels
            stage[((count + 255) // 256,)](
                tensor,
                self.buffer,
                self.slot,
                count,
                CAPACITY=self.max_elements,
                BLOCK=256,
                ALIGNED=count % 256 == 0,
                num_warps=4,
            )
            # Stage into the next buffer while peers can still read the old one.
            # This barrier advances the slot and uses its own signal channel;
            # the preceding call's barrier protected reuse of the older buffer.
            barrier[(1,)](*self.signals, self.slot, RANK=self.rank, num_warps=1)
            reduce[((count + 255) // 256,)](
                *self.peers,
                out,
                self.slot,
                count,
                CAPACITY=self.max_elements,
                BLOCK=256,
                ALIGNED=count % 256 == 0,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return out
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

    def close(self):
        # A last call may still read remote staging after this rank finishes.
        # Normal serial calls drain the older slot through their next barrier;
        # teardown needs an explicit drain before releasing peer allocations.
        if self.slot is not None:
            torch.cuda.synchronize(self.device)
            self.handle.barrier(channel=2, timeout_ms=10000)
            torch.cuda.synchronize(self.device)


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
    large = _agree_across_group(
        group, os.getenv("RTP_LLM_CANONICAL_P2P_LARGE") == "1", "canonical_p2p_large"
    )
    verify_batch = _agree_across_group(
        group,
        os.getenv("RTP_LLM_CANONICAL_P2P_VERIFY_BATCH") == "1",
        "canonical_p2p_verify_batch",
    )
    pingpong = _agree_across_group(
        group,
        os.getenv("RTP_LLM_CANONICAL_P2P_PINGPONG") == "1",
        "canonical_p2p_pingpong",
    )
    # TP8 verification of eight requests with three drafts has 32 hidden rows.
    max_elements = 81920 if verify_batch else (40960 if large else _MAX_ELEMENTS)
    buffer_elements = max_elements * (2 if pingpong else 1)
    slot = None
    pingpong_kernels = None
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
            buffer = symm.empty(buffer_elements, dtype=torch.bfloat16, device=device)
            if pingpong:
                from rtp_llm.models_py.triton_kernels.peer_all_reduce import (
                    _advance_pingpong_peer_barrier,
                    _canonical_pingpong_sum,
                    _stage_peer_pingpong,
                )

                pingpong_kernels = (
                    _stage_peer_pingpong,
                    _canonical_pingpong_sum,
                    _advance_pingpong_peer_barrier,
                )
                slot = torch.zeros(1, dtype=torch.int32, device=device)
    except Exception as error:
        logging.warning("Canonical P2P allocation unavailable: %s", error)
        local_ok = False
    if not _agree_across_group(group, local_ok, "canonical_p2p_alloc"):
        return None
    communicator = None
    try:
        handle = symm.rendezvous(buffer, group)
        peers = [
            handle.get_buffer(r, (buffer_elements,), torch.bfloat16) for r in range(8)
        ]
        communicator = CanonicalPeerAllReduce(
            group,
            device,
            buffer,
            handle,
            peers,
            kernel,
            max_elements,
            slot,
            pingpong_kernels,
        )
        if pingpong:
            x = torch.empty(max_elements, dtype=buffer.dtype, device=device)
            stage, reduce, barrier = pingpong_kernels
            barrier.warmup(
                *communicator.signals,
                slot,
                RANK=communicator.rank,
                num_warps=1,
                grid=(1,),
            )
            for count, aligned in ((max_elements, True), (2561, False)):
                stage.warmup(
                    x,
                    buffer,
                    slot,
                    count,
                    CAPACITY=max_elements,
                    BLOCK=256,
                    ALIGNED=aligned,
                    num_warps=4,
                    grid=((count + 255) // 256,),
                )
                reduce.warmup(
                    *peers,
                    x,
                    slot,
                    count,
                    CAPACITY=max_elements,
                    BLOCK=256,
                    ALIGNED=aligned,
                    num_warps=4,
                    enable_fp_fusion=False,
                    grid=((count + 255) // 256,),
                )
    except Exception as error:
        logging.warning("Canonical P2P compilation unavailable: %s", error)
        communicator = None
    # No rank may start the bounded GPU handshake while peers compile JIT.
    if not _agree_across_group(
        group, communicator is not None, "canonical_p2p_compiled"
    ):
        return None
    try:
        # N is a runtime scalar: warm full-tile and masked specializations.
        communicator.all_reduce(
            torch.zeros(max_elements, dtype=buffer.dtype, device=device)
        )
        communicator.all_reduce(torch.zeros(257, dtype=buffer.dtype, device=device))
        if pingpong:
            communicator.all_reduce(
                torch.zeros(2560, dtype=buffer.dtype, device=device)
            )
            # Also compile the masked large-message path before any capture.
            communicator.all_reduce(
                torch.zeros(2561, dtype=buffer.dtype, device=device)
            )
        torch.cuda.synchronize(device)
    except Exception as error:
        logging.warning("Canonical P2P initialization unavailable: %s", error)
        communicator = None
    if not _agree_across_group(group, communicator is not None, "canonical_p2p_ready"):
        return None
    logging.info(
        "Canonical P2P all-reduce ready: BF16 TP8, at most %d KiB, pingpong=%s",
        max_elements // 512,
        pingpong,
    )
    return communicator
