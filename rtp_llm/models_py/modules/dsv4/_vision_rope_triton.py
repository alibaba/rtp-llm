# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""V4.1 Q/K specialization of FlashAttention's BSD-3-Clause rotary kernel.

Source: Dao-AILab/flash-attention a9a3170fc98cbd22a4cc870937b390f3d483f1eb,
flash_attn/ops/triton/rotary.py. Keep the official FP32 multiply/add rounding.
"""



import torch
import triton
import triton.language as tl


@triton.jit
def _vision_qk_rotary_kernel(
    QK, COS, SIN, OUT, N: tl.constexpr, ROW_STRIDE: tl.constexpr
):
    heads = tl.program_id(0) * 2 + tl.arange(0, 2)
    rows = tl.program_id(1) * 8 + tl.arange(0, 8)
    dims = tl.arange(0, 32)
    cos = tl.load(
        COS + rows[:, None] * 32 + dims[None, :], rows[:, None] < N, other=1.0
    )
    sin = tl.load(
        SIN + rows[:, None] * 32 + dims[None, :], rows[:, None] < N, other=0.0
    )
    offsets = (
        heads[:, None, None] * 64
        + rows[None, :, None] * ROW_STRIDE
        + dims[None, None, :]
    )
    valid = rows[None, :, None] < N
    x0 = tl.load(QK + offsets, valid, other=0.0).to(tl.float32)
    x1 = tl.load(QK + offsets + 32, valid, other=0.0).to(tl.float32)
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    out_offsets = (
        heads[:, None, None] * 64 + rows[None, :, None] * 2048 + dims[None, None, :]
    )
    tl.store(OUT + out_offsets, out0, valid)
    tl.store(OUT + out_offsets + 32, out1, valid)


def _enabled_or_supported(q, k, cos, sin):
    return (
        not torch.is_grad_enabled()
        and q.is_cuda
        and q.dtype == k.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape == k.shape
        and q.shape[0] > 0
        and q.shape[1:] == (16, 64)
        and q.stride() == k.stride() == (3072, 64, 1)
        and cos.shape == sin.shape == (q.shape[0], 1, 32)
        and cos.dtype == sin.dtype == torch.float32
        and cos.is_contiguous()
        and sin.is_contiguous()
        and q.device == k.device == cos.device == sin.device
        and torch.cuda.get_device_capability(q.device)[0] == 10
        and q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
        and k.data_ptr() == q.data_ptr() + 1024 * q.element_size()
    )


def apply_vision_qk_rope(q, k, cos, sin):
    """Return fresh Q/K with the original V buffer untouched, or None."""
    if not _enabled_or_supported(q, k, cos, sin):
        return None
    n = q.shape[0]
    qk = q.as_strided((n, 32, 64), (3072, 64, 1))
    output = torch.empty((n, 32, 64), dtype=q.dtype, device=q.device)
    _vision_qk_rotary_kernel[(16, triton.cdiv(n, 8))](
        qk, cos, sin, output, n, 3072, enable_fp_fusion=False
    )
    return output[:, :16], output[:, 16:]
