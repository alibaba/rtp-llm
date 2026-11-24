# SPDX-License-Identifier: MIT
# Copyright (c) 2023 DeepSeek
"""Activation recipe adapted from DeepSeek-V4.1-Flash inference/kernel.py.

Checkpoint source revision: 2bc89ac599031fa673cab993f1df02fc4a98c673.
See fp8/_v41_cache_triton.LICENSE for the source license.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_group32_kernel(X, Q, S, groups, BLOCK_G: tl.constexpr):
    group = (tl.program_id(0) * BLOCK_G + tl.arange(0, BLOCK_G)).to(tl.int64)
    offsets = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + offsets, group[:, None] < groups, 0).to(tl.float32)
    tl.device_assert(
        tl.sum((tl.abs(x) < float("inf")).to(tl.int32), 1) == 32,
        "V4.1 activation quantization requires finite inputs",
    )
    maximum = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-4)
    scaled = maximum * (1.0 / 448.0)
    bits = scaled.to(tl.uint32, bitcast=True)
    encoded = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    scale = (encoded << 23).to(tl.float32, bitcast=True)
    normalized = tl.clamp(tl.div_rn(x, scale[:, None]), -448.0, 448.0)
    payload = normalized.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(
        tl.uint8, bitcast=True
    )
    tl.store(Q + offsets, payload, group[:, None] < groups)
    tl.store(S + group, encoded.to(tl.uint8), group < groups)


def quantize_rows(x):
    values = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (x.numel() // x.shape[-1], x.shape[-1] // 32),
        dtype=torch.uint8,
        device=x.device,
    )
    groups = x.numel() // 32
    if groups:
        _quantize_group32_kernel[(triton.cdiv(groups, 16),)](
            x,
            values.view(torch.uint8),
            scales,
            groups,
            BLOCK_G=16,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return values, scales
