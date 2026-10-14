import functools
import re

import torch


@functools.cache
def _flash_attn_2_available():
    # Same extension name on both the CUDA and ROCm flash-attn builds, so the
    # probe is platform-symmetric; RuntimeError covers extension loading
    # failures (e.g. a wheel built for a different torch ABI).  The result is
    # cached for the process lifetime: package/toolchain layout is fixed at
    # startup, so a later dynamic install is not expected to be picked up.
    try:
        import flash_attn_2_cuda  # noqa: F401
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input
    except (ImportError, OSError, RuntimeError):
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


def _flash_attn_has_sm120_kernels():
    """flash-attn builds before 2.8.4 carry no sm_120 cubins.

    Importability only proves the extension loads, not that kernels exist
    for the arch: the 2.8.3.post1 wheel imports cleanly on sm_120 and then
    raises cudaErrorNoKernelImageForDevice at the first launch.
    """
    try:
        import flash_attn

        return _flash_attn_version(flash_attn.__version__) >= (2, 8, 4)
    except (ImportError, AttributeError, ValueError):
        return False


def _flash_attn_version(version_str):
    """Parse the first three numeric components of a flash-attn version.

    Handles local build suffixes, e.g. "2.8.4+b27cdc3.cu132" -> (2, 8, 4)
    and "2.8.3.post1+cu13torch2.11" -> (2, 8, 3).
    """
    parts = []
    for component in version_str.split(".")[:3]:
        match = re.match(r"^(\d+)", component)
        parts.append(int(match.group(1)) if match else 0)
    return tuple(parts)


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
    # SM 12.x (RTX PRO Blackwell): supported since flash-attn 2.8.4
    # (sm_120 cubins; validated end-to-end on RTX PRO 5000 in CI run
    # 79742649). Older wheels must keep their original fallback routing.
    is_sm12x = major == 12 and _flash_attn_has_sm120_kernels()
    if "MI308X" in device_name:
        is_sm90 = major == 9 and minor >= 0

    return is_sm8x or is_sm90 or is_sm12x
