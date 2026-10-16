"""Decode-only G128 FP8 quantization with direct Humming promotion scales.

Quantization is always G128 with RTP's eps=1e-4. Promotion G32 repeats
metadata four times; it never recomputes absmax or changes FP8 payloads.
The original packed-scale forward_quantized ABI is not part of this API.
"""

import torch
import triton
import triton.language as tl

from ._sm90_quant_scale_fused import _input, _output, _quantize, use_v2


@triton.jit(do_not_specialize=["input_stride"])
def _moe_quant_kernel(
    x,
    q,
    scales,
    input_stride,
    K: tl.constexpr,
    PROMOTION: tl.constexpr,
    V2: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    groups = tl.program_id(1) * 4 + tl.arange(0, 4)
    columns = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    values = tl.load(x + row * input_stride + columns, columns < K, other=0).to(
        tl.float32
    )
    payload, scale = _quantize(values, V2, 1.0e-4)
    tl.store(q + row * K + columns, payload, columns < K)
    if PROMOTION == 32:
        destinations = groups[:, None] * 4 + tl.arange(0, 4)[None, :]
        tl.store(
            scales + row * (K // 32) + destinations,
            tl.broadcast_to(scale[:, None], (4, 4)),
            destinations < K // 32,
        )
    else:
        tl.store(scales + row * (K // 128) + groups, scale, groups < K // 128)


def quantize_moe_g128(x, *, promotion_group_size=32, out_q=None, out_scale=None):
    """BF16 [M,K] -> contiguous E4M3 [M,K], FP32 [M,K/promotion].

    Bounded to decode M<=256, K%128==0. Row-strided BF16 inputs are accepted
    without a materialized contiguous copy. Metadata validation never reads
    a device value or synchronizes; preallocated outputs support graph replay.
    Other dtypes/prefill must use the caller's established original path.
    """
    _input(x)
    rows, width = x.shape
    if rows > 256 or width % 128 or promotion_group_size not in (32, 128):
        raise ValueError("Expected decode M<=256, K%128==0 and promotion 32 or 128")
    v2 = use_v2(rows, width)
    q = _output(out_q, x.shape, torch.float8_e4m3fn, x.device, (x,))
    scale = _output(
        out_scale,
        (rows, width // promotion_group_size),
        torch.float32,
        x.device,
        (x, q),
    )
    if rows:
        _moe_quant_kernel[(rows, triton.cdiv(width, 512))](
            x,
            q,
            scale,
            x.stride(0),
            width,
            promotion_group_size,
            v2,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, scale
