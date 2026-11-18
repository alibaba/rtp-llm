"""Per-token NVFP4 activation packing for DeepGEMM symmetric input buffers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_row(X, Packed, Scale, Global, stride, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * stride + col, mask=col < H, other=0).to(tl.float32)
    blocks = tl.reshape(x, (B // 16, 16))
    maxima = tl.max(tl.abs(blocks), axis=1)
    gsf = tl.maximum(tl.max(maxima, axis=0), 1e-30) * (1.0 / 2688.0)
    sf = tl.div_rn(maxima * (1.0 / 6.0), gsf)
    sf = tl.minimum(tl.maximum(sf, 1.0 / 512.0), 448.0).to(tl.float8e4nv)
    inv = tl.div_rn(tl.div_rn(1.0, gsf), sf.to(tl.float32))
    scaled = tl.reshape(blocks * inv[:, None], (B,))
    mag = tl.abs(scaled)
    # E2M1 midpoint ties select the even code. Extract the sign including -0.
    code = (mag > 0.25).to(tl.uint8) + (mag >= 0.75).to(tl.uint8)
    code += (mag > 1.25).to(tl.uint8) + (mag >= 1.75).to(tl.uint8)
    code += (mag > 2.5).to(tl.uint8) + (mag >= 3.5).to(tl.uint8)
    code += (mag > 5.0).to(tl.uint8)
    code |= ((scaled.to(tl.uint32, bitcast=True) >> 31) << 3).to(tl.uint8)
    lo, hi = tl.split(tl.reshape(code, (B // 2, 2)))
    pair = tl.arange(0, B // 2)
    group = tl.arange(0, B // 16)
    tl.store(Packed + row * (H // 2) + pair, lo | (hi << 4), mask=pair < H // 2)
    tl.store(
        Scale + row * (H // 16) + group,
        sf.to(tl.uint8, bitcast=True),
        mask=group < H // 16,
    )
    tl.store(Global + row, gsf)


def quantize_input(x):
    rows, hidden = x.shape
    packed = torch.empty((rows, hidden // 2), dtype=torch.int8, device=x.device)
    scale = torch.empty((rows, hidden // 16), dtype=torch.uint8, device=x.device)
    global_scale = torch.empty(rows, dtype=torch.float32, device=x.device)
    if rows:
        _quantize_row[(rows,)](
            x,
            packed,
            scale,
            global_scale,
            x.stride(0),
            hidden,
            triton.next_power_of_2(hidden),
        )
    return packed, scale.view(torch.int32), global_scale
