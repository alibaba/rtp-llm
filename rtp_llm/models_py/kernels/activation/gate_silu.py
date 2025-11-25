import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


def atex_gate_silu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the Gated SiLU activation function to the input tensor.

    This operator assumes that the last dimension of the input tensor `x`
    contains **two concatenated parts of equal size**:

        x = [value_part, gate_part]

    where both value_part and gate_part have shape (..., D),
    so the input last dimension is 2 * D.

    For each element pair (v, g), the Gated SiLU activation is computed as:

        y = v * sigmoid(g)

    More explicitly, for each position j in the last dimension:

        let v_j = x[..., j]
        let g_j = x[..., j + D]

        y_j = v_j * sigmoid(g_j)
            = v_j * 1 / (1 + exp(-g_j))

    The output tensor has the same leading dimensions as the input, but its
    last dimension becomes D (half of the input's last dimension), since only
    the value_part is transformed.

    Shape:
        Input:  (..., 2 * D)
        Output: (..., D)

    Args:
        x (torch.Tensor): The input tensor. Must be FP16 or BF16, CUDA, contiguous.
                          The last dimension must be even, since it represents
                          concatenated (value, gate) halves.

    Returns:
        torch.Tensor: The result of applying gate-silu activation, with the same
                      dtype and device as `x`, and with last dimension D.

    Raises:
        ValueError: If the last dimension is not even.
        TypeError:  If the dtype is not supported.
    """
    # kernel dispatch
    if x.dtype == torch.float16:
        return rtp_llm_ops.atex_gate_silu_fp16(x)

    elif x.dtype == torch.bfloat16:
        return rtp_llm_ops.atex_gate_silu_fp16(x)

    else:
        raise TypeError(f"Unsupported dtype {x.dtype}")
