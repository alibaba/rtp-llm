import logging
from typing import Tuple

import torch

from rtp_llm.models_py.utils.arch import is_cuda

if is_cuda():
    from rtp_llm.ops.compute_ops import (
        per_token_group_quant_fp4 as _per_token_group_quant_fp4_op,
    )

    try:
        from rtp_llm.ops.compute_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
    except ImportError:
        cutlass_scaled_fp4_mm = None  # type: ignore[misc, assignment]
        scaled_fp4_quant = None  # type: ignore[misc, assignment]
else:
    logging.info("skip import fp4 kernel from rtp_llm_ops for non cuda platform")
    _per_token_group_quant_fp4_op = None  # type: ignore[misc, assignment]
    cutlass_scaled_fp4_mm = None  # type: ignore[misc, assignment]
    scaled_fp4_quant = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

_FP4_SCALES_PACK_FACTOR = 4


def create_per_token_group_quant_fp4_output_scale(
    x_shape: torch.Size,
    device: torch.device,
    group_size: int,
    use_packed_ue8m0: bool = True,
) -> torch.Tensor:
    """Allocate scale tensor for ``per_token_group_quant_fp4`` output."""
    n = x_shape[-1]
    assert n % group_size == 0
    num_groups = n // group_size
    if use_packed_ue8m0:
        assert num_groups % _FP4_SCALES_PACK_FACTOR == 0, (
            f"num_groups={num_groups} must be divisible by "
            f"{_FP4_SCALES_PACK_FACTOR} for packed UE8M0"
        )
        scale_shape = x_shape[:-1] + (num_groups // _FP4_SCALES_PACK_FACTOR,)
        return torch.empty(scale_shape, device=device, dtype=torch.int32)
    return torch.empty(
        x_shape[:-1] + (num_groups,),
        device=device,
        dtype=torch.float32,
    )


def per_token_group_quant_fp4(
    x: torch.Tensor,
    group_size: int = 32,
    eps: float = 1e-4,
    use_packed_ue8m0: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group FP4 e2m1 quant with UE8M0 scales (indexer layout).

    CUDA implementation matching ``deep_gemm.utils.per_token_cast_to_fp4``
    (``use_ue8m0=True``, ``gran_k=32``, ``use_packed_ue8m0=True``).

    Args:
        x: Input tensor [..., K] bf16/fp16, contiguous.
        group_size: Quant group size (indexer default 32).
        eps: Floor for per-group amax before scale computation.
        use_packed_ue8m0: Pack 4 UE8M0 exponent bytes into one int32 scale slot.

    Returns:
        (x_fp4, x_scale):
            x_fp4:   int8  [..., K // 2]  (2 FP4 nibbles per byte)
            x_scale: int32 [..., K // group_size // 4] when ``use_packed_ue8m0``
                     else float32 [..., K // group_size]
    """
    assert (
        x.shape[-1] % group_size == 0
    ), f"last dim {x.shape[-1]} must be divisible by group_size={group_size}"
    assert x.is_contiguous(), "`x` must be contiguous"

    n_in = x.shape[-1]
    out_shape = x.shape[:-1] + (n_in // 2,)
    x_q = torch.empty(out_shape, device=x.device, dtype=torch.int8)
    x_s = create_per_token_group_quant_fp4_output_scale(
        x_shape=x.shape,
        device=x.device,
        group_size=group_size,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    if x.numel() > 0:
        _per_token_group_quant_fp4_op(x, x_q, x_s, group_size, eps, use_packed_ue8m0)
    return x_q, x_s


def cutlass_scaled_fp4_mm_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if cutlass_scaled_fp4_mm is None:
        raise RuntimeError("cutlass_scaled_fp4_mm is not available in this build")
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b, alpha)
    return out


def scaled_fp4_quant_wrapper(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in a sizzled layout.
    """
    if scaled_fp4_quant is None:
        raise RuntimeError("scaled_fp4_quant is not available in this build")
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Then, the scaling
    # factors in float8_e4m3fn are packed into an int32 for every 4 values.
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // block_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    # padded part should be zeroed out
    if rounded_n > scale_n:
        output_scale = torch.zeros(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )
    else:
        output_scale = torch.empty(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )

    scaled_fp4_quant(output, input, output_scale, input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale
