import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


def atex_rmsnorm(
    x: torch.Tensor, w: torch.Tensor, eps: float, normailize_shape: int
) -> torch.Tensor:
    """
    Applies Root Mean Square Layer Normalization (RMSNorm) to the input tensor.

    RMSNorm normalizes the activations by scaling them with the reciprocal of the root mean square (RMS),
    without subtracting the mean (in contrast to standard LayerNorm).  A learnable weight vector `w`
    is then applied to re-scale the normalized activations.

    For a 2-D input tensor `x` of shape (N, D), the computation for each row x_i is:

        rms(x_i) = sqrt( (1/D) * sum(x_i[j]^2) + eps )
        y_i = (x_i / rms(x_i)) * w

    where:
        * x_i[j] is the j-th element of the i-th row,
        * w is the learnable weight vector of shape (D,),
        * eps is a small constant for numerical stability,
        * y_i is the normalized output for the i-th row.

    Args:
        x (torch.Tensor): Input tensor to be normalized.  Must be either float16 or bfloat16.
        w (torch.Tensor): Learnable weight tensor of shape matching the last dimension of `x`.
        eps (float): Small constant to avoid division by zero.
        normailize_shape (int): Expected size of the last dimension of `x`.  Used to validate tensor shapes.

    Returns:
        torch.Tensor: RMS-normalized tensor of the same dtype and shape as `x`.

    Raises:
        ValueError: If the last dimension of `x` does not match `normailize_shape`.

    Notes:
        This function dispatches to custom CUDA kernels (`rmsnorm_fp16` or `rmsnorm_bf16`) provided
        by `rtp_llm_ops.atex` for optimized execution on GPU.
    """

    if normailize_shape != x.shape[-1]:
        raise ValueError("...")

    # kernel dispatch
    if x.dtype == torch.float16:
        return rtp_llm_ops.atex_rmsnorm_fp16(x, w, eps)

    elif x.dtype == torch.bfloat16:
        return rtp_llm_ops.atex_rmsnorm_bf16(x, w, eps)

    else:
        raise TypeError(f"Unsupported dtype {x.dtype}")


def atex_skiprmsnorm(
    x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float, normailize_shape: int
) -> list[torch.Tensor]:
    """
    Applies an optimized fused operation: Residual Sum and Root Mean Square Normalization (SkipRMSNorm).

    This function fuses the residual addition (`x + r`) with the RMSNorm operation.
    It computes the sum, normalizes the sum, and returns both the normalized result and the unnormalized sum,
    which is often used in subsequent layers (e.g., as the next residual input).

    The computations are:

        sum = x + r
        o1 = rmsnorm(sum, w, eps)
        o2 = sum

    Args:
        x (torch.Tensor): The primary input tensor.
        r (torch.Tensor): The residual input tensor to be added to `x`. Must be the same shape and dtype as `x`.
        w (torch.Tensor): Learnable weight tensor for the RMSNorm, matching the last dimension of `x`.
        eps (float): Small constant for numerical stability in RMSNorm.
        normailize_shape (int): Expected size of the last dimension of `x`. Used to validate tensor shapes.

    Returns:
        list[torch.Tensor]: A tuple containing two tensors:
            * o1 (torch.Tensor): The RMS-normalized result of (x + r).
            * o2 (torch.Tensor): The unnormalized sum (x + r).

    Raises:
        ValueError: If the last dimension of `x` does not match `normailize_shape` or if `x` and `r` shapes do not match.

    Notes:
        This function dispatches to custom fused CUDA kernels (`skiprmsnorm_fp16` or `skiprmsnorm_bf16`)
        for optimized execution on GPU.
    """

    if normailize_shape != x.shape[-1]:
        raise ValueError("...")

    # kernel dispatch
    if x.dtype == torch.float16:
        return rtp_llm_ops.atex_skiprmsnorm_fp16(x, r, w, eps)

    elif x.dtype == torch.bfloat16:
        return rtp_llm_ops.atex_skiprmsnorm_bf16(x, r, w, eps)

    else:
        raise TypeError(f"Unsupported dtype {x.dtype}")
