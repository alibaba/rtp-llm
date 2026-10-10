"""SiTU activation kernels and launch adapters."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _situ_and_mul_kernel(G, U, O, N: tl.constexpr, W: tl.constexpr,
          G0: tl.constexpr, G1: tl.constexpr, U0: tl.constexpr, U1: tl.constexpr,
          BETA: tl.constexpr, UP_BETA: tl.constexpr,
          BLOCK: tl.constexpr):
    # Full prefill projections can exceed 2**31 elements, even in BF16.
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // W, i % W
    g = tl.load(G + row * G0 + col * G1, i < N, other=0).to(tl.float32)
    u = tl.load(U + row * U0 + col * U1, i < N, other=0).to(tl.float32)
    # vLLM c3b484463 situ_activation: preserve its FP32 evaluation order.
    inv_beta = tl.div_rn(1.0, BETA)
    g = (
        (0.5 * BETA) * libdevice.tanh(g * inv_beta)
        * (1.0 + libdevice.tanh(g * 0.5))
    )
    if UP_BETA is not None:
        inv_up_beta = tl.div_rn(1.0, UP_BETA)
        u = UP_BETA * libdevice.tanh(u * inv_up_beta)
    tl.store(O + i, g * u, i < N)


def situ_and_mul(gate, up, beta, linear_beta=None, *, inplace=False):
    """Apply SiTU to gate, optionally saturate up, and multiply in FP32.

    Split gate/up views are read without copying. CUDA in-place output
    requires an owned contiguous gate with storage independent of up.
    """
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError("SiTU requires matching gate/up tensors")
    if not gate.is_cuda:
        g, u = gate.float(), up.float()
        g = beta * torch.tanh(g / beta) * torch.sigmoid(g)
        if linear_beta is not None:
            u = linear_beta * torch.tanh(u / linear_beta)
        result = (g * u).to(gate.dtype)
        return gate.copy_(result) if inplace else result
    if gate.ndim != 2:
        raise ValueError("CUDA SiTU requires two-dimensional gate/up tensors")
    if gate.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Unsupported SiTU dtype")
    if inplace:
        # Only an owned, dense gate projection may be consumed. Split views
        # retain the allocating path so stores cannot overwrite another input.
        if not gate.is_contiguous() or gate._base is not None:
            raise ValueError("In-place SiTU requires an owned contiguous gate")
        if gate.numel() and (
            gate.untyped_storage().data_ptr() == up.untyped_storage().data_ptr()
        ):
            raise ValueError("In-place SiTU requires independent gate/up storage")
        output = gate
    else:
        output = torch.empty_like(gate, memory_format=torch.contiguous_format)
    if gate.numel():
        _situ_and_mul_kernel[(triton.cdiv(gate.numel(), 256),)](
            gate, up, output, gate.numel(), gate.shape[1],
            gate.stride(0), gate.stride(1), up.stride(0), up.stride(1),
            beta, linear_beta, 256, enable_fp_fusion=False,
        )
    return output
