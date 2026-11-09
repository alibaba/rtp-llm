"""Scale convention conversion, independent of native GPU extensions."""

import torch


def reciprocal_global_scale(tensors):
    if not tensors:
        raise ValueError("NVFP4 global scales must not be empty")
    for value in tensors:
        if (
            value.numel() != 1
            or not torch.isfinite(value).all()
            or not (value > 0).all()
        ):
            raise ValueError("NVFP4 global scales must be finite positive scalars")
    first = tensors[0].float()
    if any(
        not torch.equal(first.reshape(()), t.float().reshape(())) for t in tensors[1:]
    ):
        raise ValueError("Fused NVFP4 projections must have identical global scales")
    result = first.reciprocal()
    if not torch.isfinite(result).all():
        raise ValueError("NVFP4 reciprocal global scale overflow")
    return result
