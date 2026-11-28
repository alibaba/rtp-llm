"""Physical dense layouts owned by the parameter's quantization method.

No checkpoint-name or DeviceImpl dependency: tensors use [out, in] coordinates.
The ROCm executor consumes their transpose, matching the existing [in, out]
hipBLASLt contract. Permutations run on CPU too for transactional preparation.
"""

import torch

from rtp_llm.utils.swizzle_utils import calculate_k_for_swizzling, swizzle_tensor


def can_swizzle_dense(weight: torch.Tensor) -> bool:
    if weight.ndim != 2:
        return False
    try:
        mik, _, pack = calculate_k_for_swizzling(weight.dtype)
    except ValueError:
        return False
    rows, columns = weight.shape
    return rows % 16 == 0 and columns % (mik * pack) == 0


def encode_dense_weight(weight: torch.Tensor) -> torch.Tensor:
    if not can_swizzle_dense(weight):
        raise ValueError(
            f"Dense swizzle requires aligned [out, in] shape: {weight.shape}"
        )
    return swizzle_tensor(weight.contiguous(), False)


def decode_dense_weight(weight: torch.Tensor) -> torch.Tensor:
    if not can_swizzle_dense(weight):
        raise ValueError(
            f"Dense swizzle requires aligned [out, in] shape: {weight.shape}"
        )
    rows, columns = weight.shape
    mik, mikv, pack = calculate_k_for_swizzling(weight.dtype)
    return (
        weight.contiguous()
        .reshape(rows // 16, columns // (mik * pack), mik // mikv, 16, mikv * pack)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(rows, columns)
    )


def rocm_dense_executor_type(swizzled: bool):
    from rtp_llm.models_py.modules.factory.linear.impl.rocm.f16_linear import (
        RocmF16LinearNoSwizzle,
        RocmF16LinearWithSwizzle,
    )

    return RocmF16LinearWithSwizzle if swizzled else RocmF16LinearNoSwizzle
