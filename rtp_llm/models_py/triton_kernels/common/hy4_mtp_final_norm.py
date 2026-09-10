"""HY4 MTP residual RMSNorm with a BF16 boundary before normalization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _hy4_mtp_final_norm_kernel(
    hidden,
    residual,
    weight,
    hidden_stride,
    residual_stride,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    h_ptr = hidden + row * hidden_stride + cols
    r_ptr = residual + row * residual_stride + cols
    h = tl.load(h_ptr, mask, 0).to(tl.float32)
    r = tl.load(r_ptr, mask, 0).to(tl.float32)
    # The rounded sum must feed both the variance and normalized output.
    rounded = (h + r).to(tl.bfloat16)
    x = rounded.to(tl.float32)
    variance = tl.sum(x * x, axis=0) / WIDTH
    inv_rms = tl.rsqrt(variance + EPS)
    w = tl.load(weight + cols, mask, 0).to(tl.float32)
    output = (x * inv_rms) * w
    tl.store(r_ptr, rounded, mask)
    tl.store(h_ptr, output, mask)


def hy4_mtp_final_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize in place; return the original hidden and residual buffers."""
    assert hidden_states.is_cuda
    assert hidden_states.device == residual.device == weight.device
    assert hidden_states.dtype == residual.dtype == torch.bfloat16
    assert hidden_states.ndim == 2 and hidden_states.shape == residual.shape
    assert hidden_states.stride(-1) == residual.stride(-1) == 1
    rows, width = hidden_states.shape
    assert 0 < width <= 8192
    assert weight.ndim == 1 and weight.numel() == width and weight.is_contiguous()
    if rows:
        _hy4_mtp_final_norm_kernel[(rows,)](
            hidden_states,
            residual,
            weight,
            hidden_states.stride(0),
            residual.stride(0),
            width,
            eps,
            triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return hidden_states, residual
