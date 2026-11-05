import logging
from typing import Tuple

import torch

from rtp_llm.models_py.utils.arch import get_sm, is_cuda

logger = logging.getLogger(__name__)

_NVFP4_READY = False
_NVFP4_IMPORT_ERROR = None

if is_cuda() and get_sm()[0] >= 10:
    try:
        from rtp_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant

        _NVFP4_READY = True
    except ImportError as exc:
        # `rtp_kernel.nvfp4` only ships in the cuda13 rtp-kernel wheel pinned
        # by deps/requirements_lock_torch_gpu_cuda13.txt.  A stale wheel stays
        # importable (`import rtp_kernel` succeeds) while missing the
        # submodule, and the CUDA linear registration imports fp4_linear
        # *before* registering the FP8 backends -- so an eager failure here
        # silently unregisters FP8 as well.  Keep this module importable and
        # raise a clear error from the wrappers instead.
        _NVFP4_IMPORT_ERROR = exc
        logging.warning(
            "rtp_kernel.nvfp4 unavailable (%s); FP4 kernels will raise on use",
            exc,
        )
else:
    logging.info("skip import fp4 kernel from rtp_kernel.nvfp4 for non cuda platform")


def _require_nvfp4() -> None:
    if _NVFP4_READY:
        return
    if _NVFP4_IMPORT_ERROR is not None:
        raise RuntimeError(
            "rtp_kernel.nvfp4 is unavailable in this environment "
            f"({_NVFP4_IMPORT_ERROR}); install the cuda13 rtp-kernel wheel "
            "pinned by deps/requirements_lock_torch_gpu_cuda13.txt"
        )
    raise RuntimeError(
        "rtp_kernel.nvfp4 is only available on CUDA devices with sm >= 10"
    )


def cutlass_scaled_fp4_mm_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    _require_nvfp4()
    return cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b, alpha, out_dtype)


def scaled_fp4_quant_wrapper(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dim to NVFP4 with swizzled UE4M3 scales."""
    _require_nvfp4()
    return scaled_fp4_quant(input, input_global_scale)
