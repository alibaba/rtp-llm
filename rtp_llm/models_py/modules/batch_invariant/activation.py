# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SwiGLU with vLLM 0.29.0's intermediate BF16 rounding.

Arithmetic adapted from csrc/libtorch_stable/activation_kernels.cu.
The SiLU result rounds to the operand dtype before multiplication by up.
Keeping that intermediate in FP32 is numerically different. See UPSTREAM.md.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(X, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + row * 2 * D + columns, columns < D, 0).to(tl.float32)
    up = tl.load(X + row * 2 * D + D + columns, columns < D, 0).to(tl.float32)
    activated = gate / (1.0 + tl.exp(-gate))
    # This cast is part of vLLM's numerical contract, even inside one kernel.
    activated = activated.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + columns, activated * up, columns < D)


def silu_and_mul(x):
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("batch invariant SwiGLU requires CUDA BF16 inputs")
    if x.ndim == 0 or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError("batch invariant SwiGLU requires paired gate/up features")
    shape = x.shape[:-1]
    d = x.shape[-1] // 2
    x = x.reshape(-1, d * 2).contiguous()
    out = torch.empty((x.shape[0], d), device=x.device, dtype=x.dtype)
    if x.shape[0]:
        _silu_mul_kernel[(x.shape[0], triton.cdiv(d, 512))](
            x, out, d, 512, enable_fp_fusion=False
        )
    return out.view(*shape, d)
