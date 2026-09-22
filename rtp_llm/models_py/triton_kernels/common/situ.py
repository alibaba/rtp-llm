"""SiTU with FP32 intermediates held in registers, including split gate/up views."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _situ(G, U, O, N: tl.constexpr, W: tl.constexpr,
          G0: tl.constexpr, G1: tl.constexpr, U0: tl.constexpr, U1: tl.constexpr,
          BETA: tl.constexpr, UP_BETA: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // W, i % W
    g = tl.load(G + row * G0 + col * G1, i < N, other=0).to(tl.float32)
    u = tl.load(U + row * U0 + col * U1, i < N, other=0).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-g))
    g = BETA * libdevice.tanh(tl.div_rn(g, BETA)) * sigmoid
    if UP_BETA is not None:
        u = UP_BETA * libdevice.tanh(tl.div_rn(u, UP_BETA))
    tl.store(O + i, g * u, i < N)


def situ(gate, up, beta, linear_beta):
    if gate.ndim != 2 or gate.shape != up.shape:
        raise ValueError("SiTU requires matching two-dimensional gate/up tensors")
    if not gate.is_cuda or up.device != gate.device or up.dtype != gate.dtype:
        raise ValueError("SiTU requires matching CUDA devices and dtypes")
    if gate.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Unsupported SiTU dtype")
    output = torch.empty_like(gate, memory_format=torch.contiguous_format)
    if gate.numel():
        _situ[(triton.cdiv(gate.numel(), 256),)](
            gate, up, output, gate.numel(), gate.shape[1],
            gate.stride(0), gate.stride(1), up.stride(0), up.stride(1),
            beta, linear_beta, 256, enable_fp_fusion=False,
        )
    return output
