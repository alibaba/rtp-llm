from typing import Tuple

import torch

from rtp_llm.ops.compute_ops import DeviceType, get_device


def is_cuda():
    return get_device().get_device_type() == DeviceType.Cuda


def is_hip():
    return get_device().get_device_type() == DeviceType.ROCm


def is_ppu():
    return get_device().get_device_type() == DeviceType.Ppu


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
