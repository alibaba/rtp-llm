"""Qwen4 transaction control uses CPU communication on every TP topology."""

import datetime
import os
import socket
import threading
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.cpu_status import create_cpu_tp_status


class _GlooTargetStatus:
    def __init__(self, group):
        self.group = group
        self.value = torch.empty(1, device="cpu", dtype=torch.int32)
        self.lock = threading.Lock()
        self.closed = False

    def reduce(self, local_ok):
        with self.lock:
            if self.closed:
                raise RuntimeError("CPU target status is closed or failed")
            self.value.fill_(int(bool(local_ok)))
            try:
                dist.all_reduce(self.value, op=dist.ReduceOp.MIN, group=self.group)
            except BaseException:
                self.closed = True
                raise
            return bool(self.value.item())

    def close(self, cancel=False):
        with self.lock:
            self.closed = True


def prepare_cpu_target_status(config, parallelism):
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
    identity = (
        socket.gethostname(),
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        os.getuid(),
        os.stat("/dev/shm").st_dev,
        native_available,
    )
    peers = [None] * world_size
    dist.all_gather_object(peers, identity, group=group)
    if native_available and all(peer == identity for peer in peers):
        return create_cpu_tp_status(group, rtp_llm_ops)
    if not dist.is_gloo_available():
        raise RuntimeError(
            "Distributed Qwen4 target transactions require Gloo CPU control"
        )
    ranks = [dist.get_global_rank(group, rank) for rank in range(len(peers))]
    control = dist.new_group(
        ranks=ranks,
        backend="gloo",
        timeout=datetime.timedelta(seconds=60),
        use_local_synchronization=True,
    )
    return _GlooTargetStatus(control)
