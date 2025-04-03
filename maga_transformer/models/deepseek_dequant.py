import triton
import torch
import triton.language as tl
from triton import Config

from maga_transformer.utils.util import check_with_info

def weight_dequant_kernel_cpu(x: torch.Tensor, s: torch.Tensor, y: torch.Tensor, M: int , N: int, BLOCK_SIZE: int):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    for i in range(0, M, BLOCK_SIZE):
        for j in range(0, N, BLOCK_SIZE):
            y[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = x[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] * s[i // BLOCK_SIZE, j // BLOCK_SIZE]


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    check_with_info(x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous')
    check_with_info(x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions')
    M, N = x.size()  
    check_with_info(M % block_size == 0 and N % block_size == 0, 'Weight tensor must be divisible by block size, M: {}, N: {}, block_size: {}'.format(M, N, block_size))
    check_with_info(M // block_size == s.size(0) and N // block_size == s.size(1), 'Scale tensor multi block size not equal to the weight tensor, M: {}, N: {}, s.size(0): {}, s.size(1): {}'.format(M, N, s.size(0), s.size(1)))
    y = torch.empty_like(x, dtype=torch.float32)
    weight_dequant_kernel_cpu(x.float(), s.float(), y, M, N, block_size)
    return y
