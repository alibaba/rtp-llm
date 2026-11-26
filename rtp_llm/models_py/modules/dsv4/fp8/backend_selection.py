"""Device-stable selection for DSv4 attention and projection implementations."""

from functools import cache
from typing import Optional

import torch

from rtp_llm.device.device_type import is_cuda
from rtp_llm.models_py.utils.arch import is_sm120


@cache
def _is_cuda_runtime() -> bool:
    return is_cuda()


def cuda_device_index(device: torch.device) -> Optional[int]:
    if device.type != "cuda" or not _is_cuda_runtime():
        return None
    # Never cache the alias: current_device can change between model warmups.
    return device.index if device.index is not None else torch.cuda.current_device()


@cache
def _is_sm120_index(index: int) -> bool:
    return is_sm120(torch.device("cuda", index))


def is_sm120_device(device: torch.device) -> bool:
    index = cuda_device_index(device)
    return index is not None and _is_sm120_index(index)
