"""CP indexed cache IO. Callers own valid indices and unique destination slots.

The write kernels preserve the existing byte layouts and quantization steps;
collectives and stream/event ordering remain at the call site.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _div(x, y):
    return tl.inline_asm_elementwise(
        "div.rn.ftz.f32 $0, $1, $2;",
        "=f,f,f",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@tr.jit
def _write(
    K,
    P,
    IDS,
    SLOTS,
    CACHE,
    ROWS,
    KS: tl.constexpr,
    PS: tl.constexpr,
    BS: tl.constexpr,
    CS: tl.constexpr,
    ES: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row).to(tl.int64)
    if slot >= 0:
        src = tl.load(IDS + row).to(tl.int64)
        src = tl.where(src < 0, src + ROWS, src)
        c = tl.arange(0, 512).reshape(4, 128)
        x = tl.load(K + src * KS + c).to(tl.float32)
        amax = tl.max(tl.where(x == x, tl.abs(x), 0.0), 1)
        scale = tl.maximum(amax * (1.0 / 448.0), 1.1754943508222875e-38)
        base = (slot // BS) * CS + (slot % BS) * ES
        scaled = _div(x, scale[:, None])
        quant = tl.where(
            scaled == scaled, tl.minimum(tl.maximum(scaled, -448.0), 448.0), scaled
        ).to(tl.float8e4nv)
        tl.store(CACHE + base + c, quant.to(tl.uint8, bitcast=True))
        tl.store(
            (CACHE + base + 512).to(tl.pointer_type(tl.float32)) + tl.arange(0, 4),
            scale,
        )
        r = tl.arange(0, 64)
        rope = tl.load(P + src * PS + r)
        tl.store((CACHE + base + 528).to(tl.pointer_type(tl.bfloat16)) + r, rope)


def _ids(x, device):
    return (
        x is not None
        and x.device == device
        and x.dtype in (torch.int32, torch.int64)
        and x.is_contiguous()
    )


def try_indexed_cache_write(k, rope, ids, slots, cache):
    """Return False without writing if dtype/layout unsupported.

    Negative restore indices follow advanced indexing. Negative slots skip
    writes. All other indices must be in bounds, as in the CUDA caller.
    """
    if not (
        k.is_cuda
        and k.dtype == torch.bfloat16
        and k.ndim == 2
        and k.shape[1] == 512
        and k.stride(1) == 1
        and not k.requires_grad
        and _ids(ids, k.device)
        and ids.ndim == 1
        and _ids(slots, k.device)
        and slots.numel() == ids.numel()
        and cache.device == k.device
        and cache.dtype == torch.uint8
        and cache.is_contiguous()
    ):
        return False
    if not (
        cache.ndim == 3
        and cache.shape[2] == 656
        and rope.dtype == k.dtype
        and rope.device == k.device
        and rope.shape == (k.shape[0], 64)
        and rope.stride(1) == 1
        and not rope.requires_grad
    ):
        return False
    ps = rope.stride(0)
    if ids.numel():
        _write[(ids.numel(),)](
            k,
            rope,
            ids,
            slots,
            cache,
            k.shape[0],
            k.stride(0),
            ps,
            cache.shape[1],
            cache.stride(0),
            cache.stride(1),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return True
