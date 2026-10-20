"""Quack fused W1/SwiGLU for the DeepSeek V4.1 vision FFN.

The checkpoint keeps its original concatenated [gate; up] weight layout.
Quack pairs the accumulators inside the GEMM epilogue and writes only the
activated intermediate. W2, checkpoint keys and tensor ownership are unchanged.

This deliberately retains FP32 GEMM accumulators through SwiGLU, rather than
reproducing the old intermediate BF16 rounding boundaries. It is not a
byte-preserving replacement.

The optional dependency set is pinned in deps/requirements_dsv41_vision_quack.txt.
Resolve/JIT during normal eager warmup before CUDA Graph capture. Missing
dependencies and supported-launch failures are errors, never silent fallbacks.
"""

import inspect
import sys
from functools import lru_cache
from typing import Optional

import torch


def _supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not torch.is_grad_enabled()
        and x.layout == torch.strided
        and weight.layout == torch.strided
        and x.device.type == "cuda"
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] == 1024
        and weight.shape == (5632, 1024)
        and x.is_contiguous()
        and weight.is_contiguous()
        and torch.cuda.get_device_capability(x.device)[0] == 10
    )


@lru_cache(maxsize=1)
def _linear_gated():
    try:
        import quack

        if quack.__version__ != "0.6.5":
            raise RuntimeError(
                "DeepSeek V4.1 fused vision FFN requires the screened Quack "
                "0.6.5 commit pinned in deps/requirements_dsv41_vision_quack.txt; "
                f"found {quack.__version__}"
            )
        from quack.gemm_config import SplitKMode
        from quack.rounding import RoundingMode

        if sys.version_info < (3, 11):
            # Python 3.10 formats IntEnum defaults as "RoundingMode.RN", which
            # torch.library cannot parse as a numeric schema default. Match
            # Python 3.11 for these two host-side classes only; values and
            # device arithmetic are unchanged.
            RoundingMode.__str__ = int.__repr__
            SplitKMode.__str__ = int.__repr__

        from quack.linear import linear_gated_func
    except ImportError as error:
        raise RuntimeError(
            "DeepSeek V4.1 fused vision FFN requires the pinned Quack dependency "
            "set in deps/requirements_dsv41_vision_quack.txt"
        ) from error
    if "concat_layout" not in inspect.signature(linear_gated_func).parameters:
        raise RuntimeError(
            "Installed Quack lacks concat_layout; use the pinned dependency set "
            "in deps/requirements_dsv41_vision_quack.txt"
        )
    return linear_gated_func


def apply_vision_swiglu(
    x: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """Return a fresh [rows,2816] tensor, or None for unsupported inputs."""
    if not _supported(x, weight):
        return None
    with torch.cuda.device(x.device):
        _, intermediate = _linear_gated()(
            x,
            weight,
            "swiglu",
            store_preact=False,
            concat_layout=True,
            # Keep the exact official default configuration screened on SM10x.
            # No full autotuning sweep in model startup or graph capture.
            tuned=False,
        )
    return intermediate
