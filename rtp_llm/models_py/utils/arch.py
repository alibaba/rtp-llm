import functools
from typing import Optional, Tuple, Union

import torch

from rtp_llm.device.device_type import (
    DeviceType,
    get_device_type,
    is_cuda,
    is_hip,
    is_ppu,
)


_NON_CUDA_DEVICE_TYPES = frozenset(
    {"cpu", "meta", "mps", "xpu", "hip", "ppu", "npu", "hpu", "privateuse1"}
)


def _resolve_device_spec(
    device_id: Optional[Union[int, str, torch.device]],
) -> Tuple[bool, Optional[int]]:
    """Classify a device spec into `(is_explicit_non_cuda, optional_cuda_index)`."""
    if device_id is None:
        return False, None
    if isinstance(device_id, torch.device):
        if device_id.type != "cuda":
            return True, None
        return False, device_id.index
    if isinstance(device_id, str):
        spec = device_id.strip()
        if not spec:
            raise ValueError(f"invalid device specification: {device_id!r}")
        if spec.isascii() and spec.isdigit():
            return False, int(spec)
        if spec == "cuda" or spec.startswith("cuda:"):
            dev = torch.device(spec)
            return False, dev.index
        dev_type, sep, idx_part = spec.partition(":")
        if dev_type in _NON_CUDA_DEVICE_TYPES:
            if sep and not (idx_part.isascii() and idx_part.isdigit()):
                raise ValueError(f"invalid device specification: {device_id!r}")
            return True, None
        raise ValueError(f"invalid device specification: {device_id!r}")
    if isinstance(device_id, int) and not isinstance(device_id, bool):
        if device_id < 0:
            raise ValueError(f"CUDA device index must be non-negative, got {device_id}")
        return False, device_id
    raise ValueError(f"invalid device specification: {device_id!r}")


def _canonical_cuda_device(
    device_id: Optional[Union[int, str, torch.device]],
) -> int:
    is_non_cuda, cuda_index = _resolve_device_spec(device_id)
    if is_non_cuda:
        raise ValueError(f"expected a CUDA device, got {device_id}")
    if cuda_index is None:
        return torch.cuda.current_device()
    return cuda_index


@functools.cache
def _get_sm_for_device(device_id: int) -> Tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(device_id)
    return major, minor


def is_sm90(device_id: Optional[Union[int, str, torch.device]] = None) -> bool:
    """SM 9.x Hopper (H100 / H200 / H800 / H20)."""
    if _is_explicit_non_cuda_device(device_id) or not is_cuda():
        return False
    return get_sm(device_id)[0] == 9


def is_sm10x(device_id: Optional[Union[int, str, torch.device]] = None) -> bool:
    """SM 10.x datacenter Blackwell (B200 / GB200)."""
    if _is_explicit_non_cuda_device(device_id) or not is_cuda():
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


def get_sm(
    device_id: Optional[Union[int, str, torch.device]] = None
) -> Tuple[int, int]:
    device_id = _canonical_cuda_device(device_id)
    return _get_sm_for_device(device_id)


def is_sm12x(device_id: Optional[Union[int, str, torch.device]] = None) -> bool:
    """SM 12.x consumer Blackwell (RTX PRO 5000 / 6000, RTX 5090)."""
    if _is_explicit_non_cuda_device(device_id) or not is_cuda():
        return False
    return get_sm(device_id)[0] == 12


def is_sm120(device_id: Optional[Union[int, str, torch.device]] = None) -> bool:
    """Return whether the device is SM 12.0."""
    if _is_explicit_non_cuda_device(device_id) or not is_cuda():
        return False
    return get_sm(device_id) == (12, 0)


def is_blackwell(device_id: Optional[Union[int, str, torch.device]] = None) -> bool:
    """Blackwell-class: SM 10.x datacenter (B200/GB200) or SM 12.x consumer."""
    if _is_explicit_non_cuda_device(device_id) or not is_cuda():
        return False
    return get_sm(device_id)[0] in (10, 12)


def _is_explicit_non_cuda_device(
    device_id: Optional[Union[int, str, torch.device]],
) -> bool:
    return _resolve_device_spec(device_id)[0]
