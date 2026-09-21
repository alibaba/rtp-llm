"""TP2/4/8/16 push RS workspace, initialized collectively before warmup/capture.

BF16 partials are accumulated in FP32 in rank order, then rounded once to BF16.
This need not be bitwise identical to NCCL's BF16 reduction tree.
"""

from __future__ import annotations

import ctypes
import logging
import socket
from bisect import bisect_left

import torch
import torch.distributed as dist

# Measured TP8 / BF16 / hidden=7168 launch configurations. Intermediate M uses
# the next measured bucket; decode above 1024 keeps the largest configuration.
_ROWS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
_LAUNCHES = (
    (128, 128),
    (96, 128),
    (96, 128),
    (128, 128),
    (64, 128),
    (96, 128),
    (96, 128),
    (128, 128),
    (128, 256),
    (128, 256),
    (96, 512),
    (128, 256),
    (128, 256),
    (128, 512),
)


def _load_kernel():
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops.push_reduce_scatter


def _nvlink_fabric(uuid):
    """Return the local GPU's usable fabric identity; old drivers may lack it."""
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + uuid)
        info = pynvml.c_nvmlGpuFabricInfoV_t()
        pynvml.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(info))
        # Older NVML bindings decode c_char arrays as text and truncate at NUL.
        field = type(info).clusterUuid
        cluster = ctypes.string_at(ctypes.addressof(info) + field.offset, field.size)
        if (
            info.state == pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
            and info.status == pynvml.NVML_SUCCESS
            and any(cluster)
        ):
            return cluster, int(info.cliqueId)
    except (AttributeError, pynvml.NVMLError):
        pass
    finally:
        pynvml.nvmlShutdown()
    return None


def _is_fabric_allocation(tensor):
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops.is_cuda_fabric_allocation(tensor)


def _nvlink_peers(uuids: list[str], *, hostnames=None, fabrics=None) -> bool:
    if hostnames is not None and len(set(hostnames)) > 1:
        # Remote UUIDs cannot be looked up using this host's NVML. A nonzero
        # cluster UUID plus clique ID identifies a shared multi-node NVLink domain.
        return (
            fabrics is not None
            and len(fabrics) == len(uuids)
            and fabrics[0] is not None
            and all(f == fabrics[0] for f in fabrics)
        )
    # UUIDs remain unambiguous with CUDA_VISIBLE_DEVICES remapping. NVML also
    # handles NVSwitch paths, unlike testing only direct NVLink remote ports.
    import pynvml

    pynvml.nvmlInit()
    try:
        handles = [pynvml.nvmlDeviceGetHandleByUUID("GPU-" + u) for u in uuids]
        return all(
            pynvml.nvmlDeviceGetP2PStatus(a, b, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
            == pynvml.NVML_P2P_STATUS_OK
            for i, a in enumerate(handles)
            for j, b in enumerate(handles)
            if i != j
        )
    finally:
        pynvml.nvmlShutdown()


def is_same_host_group(group):
    """Agree on allocation ordering before any prefill workspace rendezvous."""
    hosts = [None] * group.size()
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    return len(set(hosts)) == 1


def _all_ready(ready: bool, group, device) -> bool:
    flag = torch.tensor([int(ready)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def create_push_reduce_scatter(group, device, *, max_m: int, n: int):
    """Return a workspace, or collectively select NCCL on unsupported setups.

    TP2/4/8/16 / BF16 / N=7168 use direct NVLink peer mappings.
    Cross-host peers additionally require one fabric clique and FABRIC allocations.
    No JIT, allocation, rendezvous, topology query or backend change is allowed
    in forward/capture.
    """
    world_size = group.size()
    if world_size not in (2, 4, 8, 16) or n != 7168:
        return None
    if max_m <= 0 or max_m % world_size:
        raise ValueError("Push RS capacity must be positive and divisible by TP size")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Push RS must be configured before CUDA Graph capture")
    failure = ""
    try:
        import torch.distributed._symmetric_memory as symm

        kernel = _load_kernel()
        props = torch.cuda.get_device_properties(device)
        if (props.major, props.minor) not in ((10, 0), (10, 3)):
            raise RuntimeError("requires SM100/SM103")
        if symm.get_backend(device) != "CUDA":
            raise RuntimeError("requires the CUDA symmetric-memory backend")
        metadata = (
            socket.gethostname(),
            str(props.uuid),
            max_m,
            n,
            props.multi_processor_count,
            _nvlink_fabric(str(props.uuid)),
        )
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.info("[PUSH_RS] NCCL fallback: %s", failure or "unsupported peer")
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
                "requires distinct GPUs in one NVLink domain with matching capacities and SM counts"
            )
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.info(
            "[PUSH_RS] NCCL fallback: %s", failure or "unsupported peer topology"
        )
        return None

    # Two phases, each with one output shard slot per producer = 2 full inputs.
    # All ranks must finish allocating before any of them enters rendezvous.
    storage = counters = None
    try:
        storage = symm.empty(2 * max_m * n * 2, dtype=torch.uint8, device=device)
        if cross_host and not _is_fabric_allocation(storage):
            raise RuntimeError("cross-host NVLink requires a CUDA FABRIC allocation")
        counters = torch.zeros(
            props.multi_processor_count, dtype=torch.int32, device=device
        )
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.warning(
            "[PUSH_RS] NCCL fallback allocating workspace: %s",
            failure or "peer allocation failed",
        )
        return None
    handle = None
    try:
        handle = symm.rendezvous(storage, group=group)
        buffers = [
            handle.get_buffer(i, [storage.numel()], torch.uint8)
            for i in range(world_size)
        ]
        storage.zero_()
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.warning(
            "[PUSH_RS] NCCL fallback mapping workspace: %s",
            failure or "peer mapping failed",
        )
        return None
    # No rank may poll or write a peer whose initialization has not completed.
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    state = PushReduceScatter(kernel, storage, handle, buffers, counters, group.rank())
    logging.info(
        "[PUSH_RS] enabled TP%d max_m=%d n=%d workspace=%.3f MiB",
        world_size,
        max_m,
        n,
        storage.numel() / (1 << 20),
    )
    return state


class PushReduceScatter:
    """Shared workspace with the same ordering contract as the native kernel.

    The caller must serialize eager calls and graph replays using this state.
    RTP's graph runner synchronizes the device between warmup and capture and
    runs inference on its compute stream. Do not insert stream waits here:
    replay bypasses Python, so Python cannot track the last execution stream.
    """

    def __init__(self, kernel, storage, handle, peers, counters, rank):
        self.kernel = kernel
        # Own all mappings for the entire model/graph lifetime.
        self.storage = storage
        self.handle = handle
        self.peers = peers
        self.counters = counters
        self.rank = rank

    def reduce_scatter(self, partial: torch.Tensor, output: torch.Tensor) -> None:
        # Use the existing TP8 / hidden=7168 buckets as the initial launch profile
        # for each supported TP size. The bumper
        # CTA keeps phases consistent when either blocks or threads changes.
        index = min(bisect_left(_ROWS, partial.shape[0]), len(_ROWS) - 1)
        blocks, threads = _LAUNCHES[index]
        blocks = min(blocks, self.counters.numel() - 1)
        self.kernel(
            partial, output, self.peers, self.counters, self.rank, blocks, threads
        )
