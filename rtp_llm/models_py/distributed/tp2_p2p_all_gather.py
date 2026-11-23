# SPDX-License-Identifier: Apache-2.0
"""Multicast-free TP2 CUDA IPC P2P AllGather for SM90 dual-GPU topologies.

Provides a single-kernel P2P all-gather over persistent 2-slot ping-pong CUDA
IPC buffers when NVSwitch hardware multicast (``multimem_all_gather_out``) is
unavailable (e.g. 2xL20X/H200 P2P) or when shard dtypes/shapes fall outside
``TorchSymmMemCommunicator.should_torch_symm_mem_allgather`` (e.g. raw FP8
vocabulary rows or >1024-token prefills).
"""

import ctypes
import logging
import os
from pathlib import Path
import socket
from typing import Optional

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.symm_mem import (
    _agree_across_group,
    _custom_all_reduce_enabled,
)

_SHARED_IPC_BUFFERS = {}
_H200_MIN_TOTAL_MEMORY_BYTES = 110 * (1024 ** 3)


def default_tp2_ipc_max_single_rows(device: torch.device, total_memory_bytes: Optional[int] = None) -> int:
    raw = os.getenv("RTP_LLM_TP2_IPC_MAX_ROWS", "").strip()
    if raw:
        val = max(128, min(12288, (int(raw) // 128) * 128))
        return val
    if total_memory_bytes is None:
        total_memory_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    # 140 GiB H200/L20X allocates the 120 MiB pool (12288 single / 6144 packed rows, 60 MiB/shard);
    # 95 GiB H20 keeps the exact 40 MiB pool (4096 single / 2048 packed rows, 20 MiB/shard) so
    # explicit kv_cache_mem_mb reservations retain identical headroom.
    return 12288 if total_memory_bytes >= _H200_MIN_TOTAL_MEMORY_BYTES else 4096


def acquire_tp2_shared_ipc_buffers(group, device: torch.device):
    from flashinfer.comm import cuda_ipc

    from rtp_llm.ops.compute_ops import rtp_llm_ops

    dev = torch.device(device)
    key = (id(group), int(dev.index))
    entry = _SHARED_IPC_BUFFERS.get(key)
    if entry is not None:
        entry["refs"] += 1
        return entry["buffers"], entry["buffer_bytes"], entry["max_single_rows"]

    local_rows = default_tp2_ipc_max_single_rows(dev)
    rows_t = torch.tensor([local_rows], device=dev, dtype=torch.int64)
    dist.all_reduce(rows_t, op=dist.ReduceOp.MIN, group=group)
    max_single_rows = int(rows_t.item())
    buffer_bytes = int(rtp_llm_ops.tp2_ar_hc_buffer_size(max_single_rows))
    with torch.cuda.device(dev):
        buffers = cuda_ipc.create_shared_buffer(buffer_bytes, group)
    _SHARED_IPC_BUFFERS[key] = {
        "group": group,
        "buffers": buffers,
        "buffer_bytes": buffer_bytes,
        "max_single_rows": max_single_rows,
        "refs": 1,
    }
    return buffers, buffer_bytes, max_single_rows


def release_tp2_shared_ipc_buffers(group, device: torch.device) -> None:
    from flashinfer.comm import cuda_ipc

    dev = torch.device(device)
    key = (id(group), int(dev.index))
    entry = _SHARED_IPC_BUFFERS.get(key)
    if entry is None:
        return
    entry["refs"] -= 1
    if entry["refs"] <= 0:
        _SHARED_IPC_BUFFERS.pop(key, None)
        cuda_ipc.free_shared_buffer(entry["buffers"], group)


def _tp2_peers_can_access(group, device: torch.device) -> bool:
    if dist.get_world_size(group) != 2:
        return False
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    boot_id = boot_path.read_text().strip() if boot_path.is_file() else ""
    info = (socket.gethostname(), boot_id, os.getuid(), int(device.index))
    peers = [None, None]
    dist.all_gather_object(peers, info, group=group)
    (h0, b0, u0, d0), (h1, b1, u1, d1) = peers
    return (
        (h0, b0, u0) == (h1, b1, u1)
        and d0 != d1
        and 0 <= d0 < torch.cuda.device_count()
        and 0 <= d1 < torch.cuda.device_count()
        and torch.cuda.can_device_access_peer(d0, d1)
        and torch.cuda.can_device_access_peer(d1, d0)
    )


class Tp2P2pAllGather:
    def __init__(self, group, device: torch.device) -> None:
        from flashinfer.comm import cuda_ipc

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group)
        self.ipc = cuda_ipc
        self.ops = rtp_llm_ops
        self.disabled = False
        self._pdl = (
            os.getenv("RTP_LLM_SM90_PDL", "1") == "1"
            and torch.cuda.get_device_capability(self.device) == (9, 0)
        )
        with torch.cuda.device(self.device):
            meta_bytes = self.ops.tp2_ar_hc_meta_size()
            self.signals = cuda_ipc.create_shared_buffer(meta_bytes, group)
            self.buffers, self.buffer_bytes, _ = acquire_tp2_shared_ipc_buffers(
                group, self.device
            )
            self.max_shard_bytes = self.buffer_bytes // 2
            cuda_ipc.cudart.cudaMemset(
                ctypes.c_void_p(self.signals[self.rank]),
                0,
                meta_bytes,
            )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=group)
            probe = torch.zeros((8,), device=self.device, dtype=torch.bfloat16)
            self.ops.tp2_p2p_all_gather(
                probe, self.buffers, self.signals, self.rank, False, self.buffer_bytes
            )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=group)

    def should_use(self, shard: torch.Tensor) -> bool:
        if self.disabled or not isinstance(shard, torch.Tensor):
            return False
        if (
            not shard.is_cuda
            or shard.device != self.device
            or shard.ndim < 1
            or not shard.is_contiguous()
            or shard.requires_grad
        ):
            return False
        shard_bytes = shard.numel() * shard.element_size()
        return (
            0 < shard_bytes <= self.max_shard_bytes
            and shard_bytes % 16 == 0
            and shard.data_ptr() % 16 == 0
        )

    def all_gather(self, shard: torch.Tensor) -> torch.Tensor:
        return self.ops.tp2_p2p_all_gather(
            shard, self.buffers, self.signals, self.rank, self._pdl, self.buffer_bytes
        )

    def close(self) -> None:
        if self.disabled:
            return
        self.disabled = True
        torch.cuda.synchronize(self.device)
        if dist.is_initialized():
            dist.barrier(group=self.group)
        self.ipc.free_shared_buffer(self.signals, self.group)
        release_tp2_shared_ipc_buffers(self.group, self.device)


def init_tp2_p2p_allgather(
    group,
    device: torch.device,
    *,
    disable_custom_all_reduce: Optional[bool] = None,
) -> Optional[Tp2P2pAllGather]:
    if os.getenv("RTP_LLM_DISABLE_TP2_P2P_ALL_GATHER", "0") == "1":
        return None
    if not _custom_all_reduce_enabled(disable_custom_all_reduce):
        return None
    if (
        not torch.cuda.is_available()
        or torch.version.hip is not None
        or not dist.is_initialized()
    ):
        return None
    non_member = getattr(dist.GroupMember, "NON_GROUP_MEMBER", None)
    if group is None or group is non_member or dist.get_rank(group) < 0:
        return None
    if dist.get_world_size(group) != 2:
        return None

    local_ok = False
    try:
        from flashinfer.comm import cuda_ipc  # noqa: F401

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        local_ok = (
            torch.cuda.device_count() >= 2
            and torch.cuda.get_device_capability(device) == (9, 0)
            and hasattr(rtp_llm_ops, "tp2_p2p_all_gather")
        )
    except Exception:
        local_ok = False

    if not _agree_across_group(group, local_ok, "tp2_p2p_ag_prereq"):
        return None
    if not _tp2_peers_can_access(group, device):
        return None

    comm = Tp2P2pAllGather(group, device)
    logging.info(
        "TP2 P2P IPC AllGather ready: device=%s max_shard_mib=%d",
        device,
        comm.max_shard_bytes // (1024 * 1024),
    )
    return comm
