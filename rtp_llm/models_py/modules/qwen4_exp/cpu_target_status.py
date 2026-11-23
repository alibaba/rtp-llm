"""Qwen4 transaction control uses CPU communication on every TP topology."""

import datetime
import os
import socket
import threading
import uuid
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.cpu_status import create_cpu_tp_status


class _GlooTargetStatus:
    def __init__(self, group):
        self.group = group
        self.value = torch.empty(1, device="cpu", dtype=torch.int32)
        self.lock = threading.Lock()

    def reduce(self, local_ok):
        with self.lock:
            self.value.fill_(int(bool(local_ok)))
            self._wait("allreduce", self.value, dist.ReduceOp.MIN)
            return bool(self.value.item())

    def broadcast(self, lengths):
        with self.lock:
            self._wait("broadcast", lengths, 0)

    def _wait(self, operation, *args):
        if self.group is None:
            raise RuntimeError("CPU target status is closed or failed")
        try:
            getattr(self.group, operation)(*args).wait()
        except BaseException:
            self.group = None
            raise

    def close(self, cancel=False):
        with self.lock:
            self.group = None


def _new_cpu_tp_control_group(group, rendezvous):
    if not dist.is_gloo_available():
        raise RuntimeError(
            "Distributed Qwen4 target transactions require Gloo CPU control"
        )
    # The parent TP group already agrees on the rendezvous. This avoids
    # deriving a group name from each rank's process-group registration count.
    store = dist.PrefixStore(
        f"rtp_llm/qwen4/target_control/{rendezvous}",
        dist.distributed_c10d._get_default_store(),
    )
    return dist.ProcessGroupGloo(
        store,
        dist.get_rank(group),
        dist.get_world_size(group),
        timeout=datetime.timedelta(seconds=60),
    )


def prepare_cpu_target_status(config, parallelism, *, broadcast_accept_lengths=False):
    role = str(getattr(parallelism, "role_type", "")).rsplit(".", 1)[-1]
    if not (
        role in ("DECODE", "PDFUSION")
        and not getattr(config, "is_mtp", False)
        and getattr(config, "enable_qwen4_ple", False)
        and parallelism.tp_size > 1
    ):
        return None
    from rtp_llm.models_py.distributed.collective_torch import Group, _get_group
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    group = _get_group(Group.TP)
    world_size = dist.get_world_size(group)
    status_size = getattr(rtp_llm_ops, "cpu_status_size", None)
    native_available = callable(status_size) and bool(status_size(world_size))
    if broadcast_accept_lengths:
        native_available = native_available and callable(
            getattr(rtp_llm_ops, "cpu_status_broadcast", None)
        )
    identity = (
        socket.gethostname(),
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        os.getuid(),
        os.stat("/dev/shm").st_dev,
        native_available,
    )
    peers = [None] * world_size
    rendezvous = uuid.uuid4().hex if dist.get_rank(group) == 0 else None
    dist.all_gather_object(peers, (identity, rendezvous), group=group)
    if native_available and all(peer_identity == identity for peer_identity, _ in peers):
        return create_cpu_tp_status(
            group, rtp_llm_ops, broadcast_accept_lengths=broadcast_accept_lengths
        )
    return _GlooTargetStatus(_new_cpu_tp_control_group(group, peers[0][1]))
