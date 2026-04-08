from typing import Tuple

import torch

from rtp_llm.ops.compute_ops import DeviceType, get_exec_ctx


def is_cuda():
    device_type = get_exec_ctx().get_device_type()
    if device_type == DeviceType.Cuda:
        return True
    else:
        return False


def is_hip():
    device_type = get_exec_ctx().get_device_type()
    if device_type == DeviceType.ROCm:
        return True
    else:
        return False


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


_flashinfer_gdn_available: bool | None = None


def is_flashinfer_gdn_available() -> bool:
    """Check if FlashInfer GDN kernels are available (requires SM90+ and flashinfer >= 0.6)."""
    global _flashinfer_gdn_available
    if _flashinfer_gdn_available is not None:
        return _flashinfer_gdn_available
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            _flashinfer_gdn_available = False
            return False
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose  # noqa: F401
        _flashinfer_gdn_available = True
    except (ImportError, Exception):
        _flashinfer_gdn_available = False
    return _flashinfer_gdn_available
