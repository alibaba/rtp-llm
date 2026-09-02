import functools
import os
from typing import Optional, Tuple, Union

import torch

from rtp_llm.device.device_type import (
    DeviceType,
    get_device_type,
    is_cuda,
    is_hip,
    is_ppu,
)


def _canonical_cuda_device(
    device_id: Optional[Union[int, str, torch.device]],
) -> int:
    if device_id is None:
        return torch.cuda.current_device()
    if isinstance(device_id, str):
        # A few model/device plumbing paths still pass ``"cuda[:N]"`` as a
        # string.  Normalize it before handling torch.device instances so
        # architecture checks remain valid for both representations.
        if device_id.startswith("cuda"):
            device_id = torch.device(device_id)
        else:
            return int(device_id)
    if isinstance(device_id, torch.device):
        if device_id.type != "cuda":
            raise ValueError(f"expected a CUDA device, got {device_id}")
        if device_id.index is None:
            return torch.cuda.current_device()
        return device_id.index
    return int(device_id)


@functools.cache
def _get_sm_for_device(device_id: int) -> Tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(device_id)
    return major, minor


def is_sm90(device_id: Optional[Union[int, str, torch.device]] = None) -> bool:
    """SM 9.x Hopper (H100 / H200 / H800 / H20)."""
    if not is_cuda() or _is_explicit_non_cuda_device(device_id):
        return False
    return get_sm(device_id)[0] == 9


def is_sm10x(device_id: Optional[Union[int, str, torch.device]] = None) -> bool:
    """SM 10.x datacenter Blackwell (B200 / GB200)."""
    if not is_cuda() or _is_explicit_non_cuda_device(device_id):
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


def get_sm(device_id: Optional[Union[int, str, torch.device]] = None) -> Tuple[int, int]:
    device_id = _canonical_cuda_device(device_id)
    return _get_sm_for_device(device_id)


def is_sm12x(device_id: Optional[Union[int, str, torch.device]] = None) -> bool:
    """SM 12.x consumer Blackwell (RTX PRO 5000 / 6000, RTX 5090)."""
    if not is_cuda() or _is_explicit_non_cuda_device(device_id):
        return False
    return get_sm(device_id)[0] == 12


def is_sm120(device_id: Optional[Union[int, str, torch.device]] = None) -> bool:
    """Exact SM 12.0 device supported by the SM120 blockwise kernels."""
    if not is_cuda() or _is_explicit_non_cuda_device(device_id):
        return False
    return get_sm(device_id) == (12, 0)


def mhc_pre_gemm_backend(device_id: Optional[Union[int, str, torch.device]] = None) -> str:
    """Resolve the mHC prenorm GEMM backend for an explicit device."""
    requested = os.environ.get("DSV4_MHC_PRE_GEMM_BACKEND", "").strip().lower()
    aliases = {
        "dg": "deepgemm",
        "tilelang": "tilelang_single",
        "single": "tilelang_single",
    }
    if requested not in ("", "auto"):
        return aliases.get(requested, requested)
    return "tilelang_single" if is_sm12x(device_id) else "deepgemm"


def is_blackwell(device_id: Optional[Union[int, str, torch.device]] = None) -> bool:
    """Blackwell-class: SM 10.x datacenter (B200/GB200) or SM 12.x consumer."""
    if not is_cuda() or _is_explicit_non_cuda_device(device_id):
        return False
    return get_sm(device_id)[0] in (10, 12)


def _is_explicit_non_cuda_device(
    device_id: Optional[Union[int, str, torch.device]],
) -> bool:
    if isinstance(device_id, str):
        return not device_id.startswith("cuda")
    return isinstance(device_id, torch.device) and device_id.type != "cuda"
