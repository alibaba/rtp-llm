import functools
from typing import Tuple

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type, is_cuda, is_hip


def get_num_device_sms() -> int:
    if is_cuda():
        assert torch.cuda.is_available()
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return props.multi_processor_count
    else:
        raise NotImplementedError("Only cuda is supported get_num_device_sms yet")


def get_sm(device_id: int = 0) -> Tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(device_id)
    return major, minor


@functools.cache
def is_sm12x() -> bool:
    """SM 12.x consumer Blackwell (RTX PRO 5000 / 6000, RTX 5090)."""
    if not is_cuda():
        return False
    return get_sm()[0] == 12


@functools.cache
def is_blackwell() -> bool:
    """Blackwell-class: SM 10.x datacenter (B200/GB200) or SM 12.x consumer."""
    if not is_cuda():
        return False
    return get_sm()[0] in (10, 12)


@functools.cache
def is_blackwell_datacenter() -> bool:
    """SM 10.x datacenter Blackwell only (B200/GB200, NOT consumer sm12x)."""
    if not is_cuda():
        return False
    return get_sm()[0] == 10
