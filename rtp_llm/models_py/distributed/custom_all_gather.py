"""Multicast AG workspaces, configured collectively before warmup/capture.

TP2/4/8/16 / K=7168 / SM100 or SM103 use one NVLink domain, including
cross-host fabric cliques with CUDA FABRIC allocations.
FP8 communicates values and packed UE8M0 scales together, without requantizing.
"""

from __future__ import annotations

import logging
import socket
from bisect import bisect_left

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.push_reduce_scatter import (
    _all_ready,
    _is_fabric_allocation,
    _nvlink_fabric,
    _nvlink_peers,
)

STAGING_MAX_LOCAL_M = 128  # Prefill only; exclusive per-rank input rows.
# Kernel launch tuning uses the gathered (global) row count.
_STAGING_ROWS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
_BF16_STAGING = (
    (96, 128),
    (64, 256),
    (128, 256),
    (128, 256),
    (96, 512),
    (128, 512),
    (96, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
)
_FP8_STAGING = (
    (128, 256),
    (64, 256),
    (64, 256),
    (64, 256),
    (96, 256),
    (128, 256),
    (128, 512),
    (128, 512),
    (96, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
)
_DIRECT_ROWS = (1024, 2048, 4096, 8192, 16384, 32768)
_BF16_DIRECT = ((2, 1024), (8, 256), (2, 1024), (8, 512), (8, 1024), (4, 1024))
_FP8_DIRECT = ((2, 1024), (2, 1024), (8, 512), (2, 1024), (64, 1024), (4, 1024))


def _load_kernels(fp8):
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    prefix = "custom_all_gather_fp8" if fp8 else "custom_all_gather"
    return (
        getattr(rtp_llm_ops, prefix + "_staging"),
        getattr(rtp_llm_ops, prefix + "_direct"),
    )


def create_custom_all_gather(
    group, device, *, max_m: int, k: int, fp8: bool, staging_only: bool = False
):
    """Return a workspace or collectively retain the existing AG backend.

    All capability, allocation and multicast mapping checks finish here. Forward
    must never rendezvous or make a rank-local choice to fall back to NCCL.
    max_m is the gathered output capacity (TP times the per-rank input rows).
    """
    world_size = group.size()
    if world_size not in (2, 4, 8, 16) or k != 7168:
        return None
    if max_m <= 0 or max_m % world_size:
        raise ValueError("custom AG capacity must be positive and divisible by TP size")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Custom AG must be configured before CUDA Graph capture")
    failure = ""
    try:
        import torch.distributed._symmetric_memory as symm

        kernels = _load_kernels(fp8)
        props = torch.cuda.get_device_properties(device)
        if (props.major, props.minor) not in ((10, 0), (10, 3)):
            raise RuntimeError("requires SM100/SM103")
        if symm.get_backend(device) != "CUDA":
            raise RuntimeError("requires the CUDA symmetric-memory backend")
        metadata = (
            socket.gethostname(),
            str(props.uuid),
            max_m,
            k,
            fp8,
            staging_only,
            props.multi_processor_count,
            _nvlink_fabric(str(props.uuid)),
        )
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.info(
            "[CUSTOM_AG] existing backend retained: %s", failure or "unsupported peer"
        )
        return None

    peers_metadata = [None] * world_size
    dist.all_gather_object(peers_metadata, metadata, group=group)
    try:
        uuids = [item[1] for item in peers_metadata]
        hostnames = [item[0] for item in peers_metadata]
        cross_host = len(set(hostnames)) > 1
        if (
            any(item[2:-1] != metadata[2:-1] for item in peers_metadata)
            or len(set(uuids)) != world_size
            or not _nvlink_peers(
                uuids,
                hostnames=hostnames,
                fabrics=[item[-1] for item in peers_metadata],
            )
        ):
            raise RuntimeError(
                "requires distinct NVLink peers and matching capacities/precision/SM counts"
            )
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.info(
            "[CUSTOM_AG] existing backend retained: %s",
            failure or "unsupported topology",
        )
        return None

    # Decode has no row-dependent dispatch: reserve staging for its configured
    # token capacity. Prefill switches to direct at STAGING_MAX_LOCAL_M input rows.
    local_staging_m = max_m // world_size
    if not staging_only:
        local_staging_m = min(local_staging_m, STAGING_MAX_LOCAL_M - 1)
    groups = (k + 511) // 512
    if fp8:
        # Four payload vectors plus one vector containing their original sign
        # bits and ready markers. Each phase owns one slot per source rank.
        vectors = local_staging_m * k // 16 + groups * ((local_staging_m + 3) // 4)
        slot_bytes = (vectors + 3) // 4 * 80
    else:
        slot_bytes = local_staging_m * k * 2
    storage = {}
    try:
        storage["staging"] = symm.empty(
            2 * world_size * slot_bytes, dtype=torch.uint8, device=device
        )
        storage["semaphores"] = symm.empty(
            (props.multi_processor_count, 128), dtype=torch.uint8, device=device
        )
        storage["values"] = symm.empty(
            (max_m, k), dtype=torch.uint8 if fp8 else torch.bfloat16, device=device
        )
        if fp8:
            storage["scales"] = symm.empty(
                ((max_m + 3) // 4 * 4) * groups, dtype=torch.int32, device=device
            )
        if cross_host and not all(_is_fabric_allocation(t) for t in storage.values()):
            raise RuntimeError("cross-host NVLink requires CUDA FABRIC allocations")
        counters = torch.zeros(
            props.multi_processor_count, dtype=torch.int32, device=device
        )
    except Exception as exc:
        failure = str(exc)
    # Agree before entering any rendezvous, including when only one rank OOMs.
    if not _all_ready(not failure, group, device):
        logging.warning(
            "[CUSTOM_AG] workspace allocation failed: %s",
            failure or "peer allocation failed",
        )
        return None
    handles = {}
    for name, tensor in storage.items():
        try:
            handle = symm.rendezvous(tensor, group=group)
            if not handle.multicast_ptr:
                raise RuntimeError(f"{name} has no multicast mapping")
            handles[name] = handle
        except Exception as exc:
            failure = str(exc)
        if not _all_ready(not failure, group, device):
            logging.warning(
                "[CUSTOM_AG] multicast mapping failed: %s",
                failure or "peer mapping failed",
            )
            return None
    storage["staging"].zero_()
    storage["semaphores"].zero_()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    workspace = CustomAllGather(
        kernels, storage, handles, counters, group.rank(), fp8, world_size
    )
    logging.info(
        "[CUSTOM_AG] enabled TP%d max_m=%d k=%d fp8=%s workspace=%.3f MiB",
        world_size,
        max_m,
        k,
        fp8,
        sum(t.numel() * t.element_size() for t in storage.values()) / (1 << 20),
    )
    return workspace


class CustomAllGather:
    """Same-order, serialized collective calls, including graph replay.

    Results alias reusable workspace and must be consumed before the next AG
    with this state. RTP's AG/GEMM caller completes all GEMMs on the same stream.
    Own the symmetric allocations and handles for the entire model/graph lifetime.
    """

    def __init__(self, kernels, storage, handles, counters, rank, fp8, world_size):
        self.staging_kernel, self.direct_kernel = kernels
        self.storage = storage
        self.handles = handles
        self.counters = counters
        self.rank = rank
        self.fp8 = fp8
        self.world_size = world_size
        self.pointers = {
            name: int(handle.multicast_ptr) for name, handle in handles.items()
        }

    def _launch(self, m, staging):
        rows = _STAGING_ROWS if staging else _DIRECT_ROWS
        configs = (
            (_FP8_STAGING if self.fp8 else _BF16_STAGING)
            if staging
            else (_FP8_DIRECT if self.fp8 else _BF16_DIRECT)
        )
        blocks, threads = configs[min(bisect_left(rows, m), len(rows) - 1)]
        return min(blocks, self.counters.numel() - 1), threads

    @staticmethod
    def _aligned(tensor):
        # A view may be contiguous with a rank-local, unaligned storage offset.
        # Copy it without changing the group's selected communication backend.
        return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()

    def all_gather(self, local_input, *, staging):
        if self.fp8:
            raise TypeError("BF16 AG requires a BF16 workspace")
        m = local_input.shape[0] * self.world_size
        if not 0 < m <= self.storage["values"].shape[0]:
            raise ValueError("BF16 AG exceeds configured capacity")
        local_input = self._aligned(local_input)
        output = self.storage["values"][:m]
        blocks, threads = self._launch(m, staging)
        if staging:
            self.staging_kernel(
                local_input,
                output,
                self.storage["staging"],
                self.counters,
                self.pointers["staging"],
                self.rank,
                blocks,
                threads,
            )
        else:
            self.direct_kernel(
                local_input,
                output,
                self.storage["semaphores"],
                self.pointers["values"],
                self.pointers["semaphores"],
                self.rank,
                blocks,
                threads,
            )
        return output

    def all_gather_fp8(self, values, wire, *, staging):
        if not self.fp8:
            raise TypeError("FP8 AG requires an FP8 workspace")
        m, k = values.shape
        m *= self.world_size
        if not 0 < m <= self.storage["values"].shape[0]:
            raise ValueError("FP8 AG exceeds configured capacity")
        values = self._aligned(values.view(torch.uint8))
        wire = self._aligned(wire)
        output = self.storage["values"][:m]
        scale_rows = (m + 3) // 4 * 4
        scales = self.storage["scales"][: ((k + 511) // 512) * scale_rows].view(
            -1, scale_rows
        )
        blocks, threads = self._launch(m, staging)
        if staging:
            self.staging_kernel(
                values,
                wire,
                output,
                scales,
                self.storage["staging"],
                self.counters,
                self.pointers["staging"],
                self.rank,
                blocks,
                threads,
            )
        else:
            self.direct_kernel(
                values,
                wire,
                output,
                scales,
                self.storage["semaphores"],
                self.pointers["values"],
                self.pointers["scales"],
                self.pointers["semaphores"],
                self.rank,
                blocks,
                threads,
            )
        return output.view(torch.float8_e4m3fn), scales[:, :m].T
