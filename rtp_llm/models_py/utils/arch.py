import functools
from typing import Optional, Tuple, Union

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type, is_cuda, is_hip


def _canonical_cuda_device(device_id: Optional[Union[int, torch.device]]) -> int:
    if device_id is None:
        return torch.cuda.current_device()
    if isinstance(device_id, torch.device):
        if device_id.index is None:
            return torch.cuda.current_device()
        return device_id.index
    return int(device_id)


@functools.cache
def _get_sm_for_device(device_id: int) -> Tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(device_id)
    return major, minor


def is_sm90(device_id: Optional[Union[int, torch.device]] = None) -> bool:
    """SM 9.x Hopper (H100 / H200 / H800 / H20)."""
    if not is_cuda():
        return False
    return get_sm(device_id)[0] == 9


def is_sm10x(device_id: Optional[Union[int, torch.device]] = None) -> bool:
    """SM 10.x datacenter Blackwell (B200 / GB200)."""
    if not is_cuda():
        return False
    return get_sm(device_id)[0] == 10


def get_num_device_sms() -> int:
    if is_cuda():
        assert torch.cuda.is_available()
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return props.multi_processor_count
    else:
        raise NotImplementedError("Only cuda is supported get_num_device_sms yet")


def get_sm(device_id: Optional[Union[int, torch.device]] = None) -> Tuple[int, int]:
    device_id = _canonical_cuda_device(device_id)
    return _get_sm_for_device(device_id)


def is_sm12x(device_id: Optional[Union[int, torch.device]] = None) -> bool:
    """SM 12.x consumer Blackwell (RTX PRO 5000 / 6000, RTX 5090)."""
    if not is_cuda():
        return False
    return get_sm(device_id)[0] == 12


def is_blackwell(device_id: Optional[Union[int, torch.device]] = None) -> bool:
    """Blackwell-class: SM 10.x datacenter (B200/GB200) or SM 12.x consumer."""
    if not is_cuda():
        return False
    return get_sm(device_id)[0] in (10, 12)
