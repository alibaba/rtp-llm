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
"""Packed V4.1 rotary, preserving the existing FP32 multiply/add rounding.

The BHSD writer below changes only the output address calculation of
``_vision_rope_triton._vision_qk_rotary_kernel`` (itself derived from
Dao-AILab/flash-attention a9a3170fc98cbd22a4cc870937b390f3d483f1eb,
flash_attn/ops/triton/rotary.py). No attention backend or arithmetic changes.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41._vision_rope_triton import (
    _enabled_or_supported,
    _vision_qk_rotary_kernel,
)


def apply_packed_vision_qk_rope(q, k, cos, sin):
    """Return shared-buffer Q/K for the original specialization, or None."""
    if not _enabled_or_supported(q, k, cos, sin):
        return None
    rows = q.shape[0]
    qk = q.as_strided((rows, 32, 64), (3072, 64, 1))
    output = torch.empty((rows, 32, 64), dtype=q.dtype, device=q.device)
    # Eight rows per grid.y block: preserve both the CUDA grid.y limit and
    # the original kernel's 32-bit row offsets. Sliced pointers make every
    # launch use local row indices while retaining absolute rotary positions.
    limit = 65535 * 8
    for start in range(0, rows, limit):
        stop = min(start + limit, rows)
        _vision_qk_rotary_kernel[(16, triton.cdiv(stop - start, 8))](
            qk[start:stop],
            cos[start:stop],
            sin[start:stop],
            output[start:stop],
            stop - start,
            3072,
            enable_fp_fusion=False,
        )
    return output[:, :16], output[:, 16:]


@triton.jit
def _vision_qk_rotary_bhsd_kernel(
    QK,
    COS,
    SIN,
    OUT,
    N: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    START_ROW,
    TOTAL_ROWS: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
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
    # [2, B, H, S, D]. Promote BEFORE multiplication: B128's combined
    # Q/K allocation contains more than INT_MAX BF16 elements. Input pointers
    # and positional tables are already sliced; output rows remain absolute.
    absolute_rows = rows.to(tl.int64) + START_ROW
    absolute_heads = heads.to(tl.int64)
    out_offsets = (
        (absolute_heads[:, None, None] // 16) * TOTAL_ROWS * 1024
        + (
            (absolute_rows[None, :, None] // SEQUENCE_LENGTH) * 16
            + absolute_heads[:, None, None] % 16
        )
        * SEQUENCE_LENGTH
        * 64
        + (absolute_rows[None, :, None] % SEQUENCE_LENGTH) * 64
        + dims[None, None, :]
    )
    tl.store(OUT + out_offsets, out0, valid)
    tl.store(OUT + out_offsets + 32, out1, valid)


def apply_packed_vision_qk_rope_bhsd(q, k, cos, sin, batch_size, sequence_length):
    """Return fresh contiguous [B,16,S,64] Q/K, or None if unsupported.

    Explicit B/S describe one homogeneous geometry group; callers must not
    flatten mixed geometries into this interface. The original packed-QKV
    input contract and rotary arithmetic are unchanged. Output is per-call
    owned storage, not reused scratch. Exceptions from supported launches
    propagate: None is only a pre-launch unsupported-contract decision.
    """
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or batch_size <= 0
        or sequence_length <= 0
        or q.ndim != 3
        or q.shape[0] != batch_size * sequence_length
        or not _enabled_or_supported(q, k, cos, sin)
    ):
        return None
    rows = q.shape[0]
    qk = q.as_strided((rows, 32, 64), (3072, 64, 1))
    output = torch.empty(
        (2, batch_size, 16, sequence_length, 64), dtype=q.dtype, device=q.device
    )
    limit = 65535 * 8
    for start in range(0, rows, limit):
        stop = min(start + limit, rows)
        _vision_qk_rotary_bhsd_kernel[(16, triton.cdiv(stop - start, 8))](
            qk[start:stop],
            cos[start:stop],
            sin[start:stop],
            output,
            stop - start,
            3072,
            start,
            rows,
            sequence_length,
            enable_fp_fusion=False,
        )
    return output[0], output[1]
