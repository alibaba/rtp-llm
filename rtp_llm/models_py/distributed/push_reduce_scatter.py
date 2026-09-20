"""TP8 push RS workspace, initialized collectively before warmup/capture.

BF16 partials are accumulated in FP32 in rank order, then rounded once to BF16.
This need not be bitwise identical to NCCL's BF16 reduction tree.
"""

from __future__ import annotations

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


def _nvlink_peers(uuids: list[str]) -> bool:
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


def _all_ready(ready: bool, group, device) -> bool:
    flag = torch.tensor([int(ready)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def create_push_reduce_scatter(group, device, *, max_m: int, n: int):
    """Return a workspace, or collectively select NCCL on unsupported setups.

    Only the measured TP8 / BF16 / N=7168 shape and NVLink topology are enabled.
    No JIT, allocation, rendezvous, topology query or backend change is allowed
    in forward/capture.
    """
    if group.size() != 8 or n != 7168:
        return None
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
        metadata = (socket.gethostname(), str(props.uuid), max_m, n)
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.info("[PUSH_RS] NCCL fallback: %s", failure or "unsupported peer")
        return None

    peers_metadata = [None] * 8
    dist.all_gather_object(peers_metadata, metadata, group=group)
    try:
        uuids = [item[1] for item in peers_metadata]
        if (
            any(
                (item[0], item[2:]) != (metadata[0], metadata[2:])
                for item in peers_metadata
            )
            or len(set(uuids)) != 8
            or not _nvlink_peers(uuids)
        ):
            raise RuntimeError(
                "requires eight distinct GPUs in one NVLink domain and matching capacities"
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
            handle.get_buffer(i, [storage.numel()], torch.uint8) for i in range(8)
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
        "[PUSH_RS] enabled TP8 max_m=%d n=%d workspace=%.3f MiB",
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
        # Rounded buckets from the existing TP8 / hidden=7168 tuning. The bumper
        # CTA keeps phases consistent when either blocks or threads changes.
        index = min(bisect_left(_ROWS, partial.shape[0]), len(_ROWS) - 1)
        blocks, threads = _LAUNCHES[index]
        blocks = min(blocks, self.counters.numel() - 1)
        self.kernel(
            partial, output, self.peers, self.counters, self.rank, blocks, threads
        )
