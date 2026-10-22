"""Qwen4 admission policy for the shared TP2 host status collective."""

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.cpu_status import create_cpu_tp_status


def prepare_cpu_target_status(config, parallelism):
    role = str(getattr(parallelism, "role_type", "")).rsplit(".", 1)[-1]
    if not (
        role == "DECODE"
        and not getattr(config, "is_mtp", False)
        and getattr(config, "enable_qwen4_ple", False)
        and parallelism.tp_size == 2
        and parallelism.dp_size == 1
        and parallelism.local_world_size == 2
    ):
        return None
    from rtp_llm.models_py.distributed.collective_torch import Group, _get_group
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    group = _get_group(Group.TP)
    eligible = (
        torch.version.hip is None
        and torch.cuda.get_device_capability() == (12, 0)
        and torch.cuda.device_count() == 2
    )
    identity = (
        eligible,
        socket.gethostname(),
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        os.getuid(),
    )
    peers = [None, None]
    dist.all_gather_object(peers, identity, group=group)
    if peers[0] != peers[1]:
        raise RuntimeError("CPU target status requires identical local TP2 eligibility")
    if not eligible:
        return None
    return create_cpu_tp_status(group, rtp_llm_ops)
