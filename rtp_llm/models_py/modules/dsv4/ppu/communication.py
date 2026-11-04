"""Startup-only PCCL workspace preparation for the validated TP4 prefill path."""

import logging
import os

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.ops import RoleType


def maybe_warmup_ppu_tp_communication(parallelism_config, *, options=None) -> bool:
    """Opt-in before model/KV allocation; never called from a forward or replay.

    On PPU, the medium-message BF16 all-reduce protocol allocates additional
    non-PyTorch workspace lazily. A scalar barrier and large-message traffic do
    not initialize it. Doing so after the Torch cache fills device memory can
    fail despite unused cached blocks, without incrementing Torch's OOM count.

    Reuse the real TP communicator and allocate its workspace before memory
    budgeting. The 256KiB message exercises the measured medium protocol. This
    is not an all-shape memory guarantee, nor permission to retry NCCL errors.
    """
    options = os.environ if options is None else options
    if options.get("DSV4_PPU_TP_COMM_WARMUP", "0") != "1":
        return False
    if (
        parallelism_config.role_type not in (RoleType.PREFILL, RoleType.PDFUSION)
        or parallelism_config.tp_size != 4
        or parallelism_config.world_size != 4
        or parallelism_config.prefill_cp_config.prefill_cp_size != 1
        or get_device_type() != DeviceType.Ppu
    ):
        return False
    device = torch.device("cuda", parallelism_config.local_rank)
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PPU TP communication warmup must run before capture")
        # Zeros avoid consuming sampling RNG or touching model buffers.
        buffer = torch.zeros(128 * 1024, dtype=torch.bfloat16, device=device)
        all_reduce(buffer, Group.TP)
        # Startup only: establish completion before weights and KV budgeting.
        # Fail startup on error; continuing with a failed communicator is unsafe.
        torch.cuda.current_stream(device).synchronize()
    logging.info(
        "[DSV4_PPU_TP_COMM_WARMUP] prepared BF16 medium-message workspace "
        "before model/KV allocation on %s",
        device,
    )
    return True
