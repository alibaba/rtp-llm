"""Single-launch residual addition for vision tensors beyond 32-bit offsets.

No normalization or GEMM arithmetic is changed. Small tensors retain ATen,
whose eager launch path is faster. Outputs never alias either residual input.
"""

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton = None


if triton is not None:

    @triton.jit
    def _vision_residual_add64_kernel(X, R, Y, N: tl.constexpr):
        offsets = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
        x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
        r = tl.load(R + offsets, offsets < N, other=0).to(tl.float32)
        tl.store(Y + offsets, x + r, offsets < N)


def apply_vision_ffn_residual_add(x, residual):
    """Return a fresh sum, or None when the ordinary residual path is better."""
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(residual, torch.Tensor)
        and not torch.is_grad_enabled()
        and x.layout == residual.layout == torch.strided
        and x.is_cuda
        and x.device == residual.device
        and x.dtype == residual.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape == residual.shape
        and x.shape[1] == 1024
        and x.is_contiguous()
        and residual.is_contiguous()
        and x.numel() * x.element_size() > 2**31 - 1
        and torch.cuda.get_device_capability(x.device)[0] == 10
    ):
        return None
    if triton is None:
        return None
    output = torch.empty_like(x)
    _vision_residual_add64_kernel[(triton.cdiv(x.numel(), 1024),)](
        x, residual, output, x.numel(), num_warps=4, enable_fp_fusion=False
    )
    return output
