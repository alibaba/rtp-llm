import functools

import torch


@functools.cache
def _flash_attn_2_available():
    try:
        import flash_attn_2_cuda  # noqa: F401
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input
    except (ImportError, OSError):
        return False
    return all(
        callable(function)
        for function in (
            flash_attn_func,
            flash_attn_varlen_func,
            pad_input,
            unpad_input,
        )
    )


def can_use_flash_attn(device_id=0):
    """Check if a GPU supports FlashAttention."""
    if not _flash_attn_2_available():
        return False
    major, minor = torch.cuda.get_device_capability(device_id)
    device_full_name = torch.cuda.get_device_name(device_id)
    device_name = device_full_name.split()[-1]

    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0
    if "MI308X" in device_name:
        is_sm90 = major == 9 and minor >= 0

    return is_sm8x or is_sm90
