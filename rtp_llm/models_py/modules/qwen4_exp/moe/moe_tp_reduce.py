"""Worker-owned TRTLLM communicator for the qualified small MoE branch only.

Reuse RTP's FlashInfer integration, including completion-at-end for PDL.
Initialize before capture; keep it alive until the worker and its graphs exit.
Attention, other FFNs and serial diagnostic references retain their backend.
"""

import logging
import os

import torch
import torch.distributed as dist

_COMMUNICATORS = {}
_DEVICE_COMMUNICATORS = {}


def get_prepared_moe_communicator(device):
    index = getattr(device, "index", None)
    return None if index is None else _DEVICE_COMMUNICATORS.get(index)


def prepare_moe_communicator(device: torch.device, max_decode_rows: int = 128):
    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.distributed.flashinfer_all_reduce import FlashInferAllReduce

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("MoE communicator must be prepared before graph capture")
    if max_decode_rows not in (64, 128):
        raise ValueError("MoE decode row capacity must be 64 or 128")
    if os.getenv("FT_DISABLE_CUSTOM_AR", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    ):
        return None
    if not dist.is_initialized() or not collective_torch._initialized:
        return None
    try:
        group = collective_torch._get_group(collective_torch.Group.TP)
    except (RuntimeError, ValueError):
        return None
    if dist.get_world_size(group) != 2:
        return None
    key = (id(group), device.index, max_decode_rows)
    if key in _COMMUNICATORS:
        return _COMMUNICATORS[key]
    comm = FlashInferAllReduce(
        group,
        device,
        single_node=True,
        disable_custom_all_reduce=False,
        h20_max_num_tokens=2 * max_decode_rows,
    )
    if comm.disabled:
        _COMMUNICATORS[key] = None
        return None
    with torch.cuda.device(device):
        probe = torch.zeros(
            (2 * max_decode_rows, 2560), device=device, dtype=torch.bfloat16
        )
        if not comm.should_use(probe):
            comm.destroy()
            _COMMUNICATORS[key] = None
            return None
        comm.all_reduce(probe)  # Initialize/JIT on both ranks before capture.
    _COMMUNICATORS[key] = comm
    _DEVICE_COMMUNICATORS[device.index] = comm
    logging.getLogger(__name__).info(
        "MoE private custom AR ready: backend=FlashInfer/TRTLLM device=%s "
        "packed_rows=1..%s hidden=2560; other module backends unchanged",
        device,
        2 * max_decode_rows,
    )
    return comm
