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
