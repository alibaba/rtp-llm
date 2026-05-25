import logging
import os

import torch

VISION_ATTN_IMPL_ENV = "RTP_LLM_VISION_ATTN_IMPL"
_VALID_VISION_ATTN_IMPLS = {"auto", "sdpa", "eager", "flash_attention_2"}


def can_use_flash_attn(device_id=0):
    """Check if a GPU supports FlashAttention."""
    major, minor = torch.cuda.get_device_capability(device_id)
    device_full_name = torch.cuda.get_device_name(device_id)
    device_name = device_full_name.split()[-1]

    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0
    if "MI308X" in device_name:
        is_sm90 = major == 9 and minor >= 0

    return is_sm8x or is_sm90


def _can_import_flash_attn() -> bool:
    try:
        __import__("flash_attn")
        return True
    except Exception as e:
        logging.info(
            f"flash_attn is not available for vision attention, using sdpa instead: {e}"
        )
        return False


def get_default_vision_attention_impl(device_id=0) -> str:
    override = os.environ.get(VISION_ATTN_IMPL_ENV, "auto").strip().lower()
    if override not in _VALID_VISION_ATTN_IMPLS:
        raise ValueError(
            f"{VISION_ATTN_IMPL_ENV} must be one of "
            f"{sorted(_VALID_VISION_ATTN_IMPLS)}, got {override!r}"
        )

    if override != "auto":
        return override

    try:
        if can_use_flash_attn(device_id) and _can_import_flash_attn():
            return "flash_attention_2"
    except Exception as e:
        logging.info(
            f"initialize flash_attn failed for vision attention, using sdpa instead: {e}"
        )
    return "sdpa"
