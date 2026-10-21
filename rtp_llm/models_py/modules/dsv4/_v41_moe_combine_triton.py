"""SM120 sequential FP32 combination of already weighted expert rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rn(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _combine_kernel(
    DOWN,
    MAP,
    OUT,
    D: tl.constexpr,
    K: tl.constexpr,
    MS0: tl.constexpr,
    MS1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    feature = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    accumulated = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(K):
        destination = tl.load(MAP + row * MS0 + slot * MS1)
        value = tl.load(
            DOWN + tl.maximum(destination, 0) * D + feature,
            mask=(destination >= 0) & (feature < D),
            other=0,
        ).to(tl.float32)
        accumulated = _add_rn(accumulated, value)
    tl.store(OUT + row * D + feature, accumulated, mask=feature < D)


def combine_ordered_rows(down, route_map):
    if not (
        down.device.type == "cuda"
        and torch.cuda.get_device_capability(down.device) == (12, 0)
        and down.ndim == 2
        and down.shape[0] > 0
        and down.shape[1] > 0
        and down.dtype == torch.bfloat16
        and down.is_contiguous()
        and route_map.device == down.device
        and route_map.ndim == 2
        and 0 < route_map.shape[0] <= 128
        and 0 < route_map.shape[1] <= 32
        and route_map.dtype in (torch.int32, torch.int64)
    ):
        return None
    rows, topk = route_map.shape
    dim = down.shape[1]
    output = torch.empty((rows, dim), device=down.device, dtype=torch.float32)
    _combine_kernel[(rows, triton.cdiv(dim, 256))](
        down,
        route_map,
        output,
        dim,
        topk,
        route_map.stride(0),
        route_map.stride(1),
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
