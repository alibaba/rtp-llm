from typing import Tuple

import torch


def is_cuda():
    from rtp_llm.device import get_current_device

    return get_current_device().is_cuda()


def is_hip():
    from rtp_llm.device import get_current_device

    return get_current_device().is_rocm()


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
