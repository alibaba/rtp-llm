"""CP indexed cache IO. Callers own valid indices and unique destination slots.

The write kernels preserve the existing byte layouts and quantization steps;
collectives and stream/event ordering remain at the call site.
"""

import torch
import triton as tr
import triton.language as tl
from rtp_llm.models_py.triton_kernels.common import fast_launch as _cpu_launch

_kernels = {}


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


@tr.jit(do_not_specialize=["N"])
def _read_zero(CACHE, SLOTS, OUT, N, STRIDE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(SLOTS + row, row < N, other=0).to(tl.int64)
    c = tl.arange(0, 512)
    value = tl.load(CACHE + slot * STRIDE + c, row < N, other=0)
    scale = tl.load(
        (CACHE + slot * STRIDE + 512).to(tl.pointer_type(tl.float32)) + c // 128,
        row < N,
        other=0,
    )
    x = value.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    tl.store(OUT + row * 576 + c, x)
    r = tl.arange(0, 64)
    rope = tl.load(
        (CACHE + slot * STRIDE + 528).to(tl.pointer_type(tl.bfloat16)) + r,
        row < N,
        other=0,
    )
    tl.store(OUT + row * 576 + 512 + r, rope)


def _launch(jit, grid, args):
    grid = (grid[0], 1, 1)
    # Cache only compilation, never tensors. Runtime sizes remain runtime args.
    key = (
        jit,
        tuple(
            (
                (a.dtype, a.device.index, a.data_ptr() % 16)
                if isinstance(a, torch.Tensor)
                else a
            )
            for a in args
        ),
    )
    kernel = _kernels.get(key)
    if kernel is None:
        kernel = jit[grid](*args, num_warps=4, enable_fp_fusion=False)
        if len(_kernels) >= 128:
            _kernels.pop(next(iter(_kernels)))
        _kernels[key] = kernel
    else:
        _cpu_launch.launch_compiled(kernel, grid, args)
    return kernel


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
        _launch(
            _write,
            (ids.numel(),),
            (
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
            ),
        )
    return True


_native_readers = {}


def try_read_new_kv(cache, slots, padded_rows):
    """Flat token cache slots -> zero-padded BF16 rows, without a zeroing launch."""
    native_key = None
    from rtp_llm.models_py.kernels.cuda import prefill_cpu_ops

    if (
        prefill_cpu_ops.NATIVE_LAUNCH
        and _cpu_launch.ENABLED
        and _cpu_launch._raw_stream is not None
        and _cpu_launch.hooks_idle()
        and cache.is_cuda
        and slots is not None
    ):
        native_key = (cache.device.index, slots.dtype, min(slots.numel() // 2**31, 2))
        prepared = _native_readers.get(native_key)
        if prepared is not None:
            if torch.cuda.current_device() != cache.device.index:
                with torch.cuda.device(cache.device):
                    return try_read_new_kv(cache, slots, padded_rows)
            out = prepared.read_kv(
                cache, slots, padded_rows, _cpu_launch._raw_stream(cache.device.index)
            )
            if out is not None:
                return out
    if not (
        cache.is_cuda
        and cache.dtype == torch.uint8
        and cache.ndim == 3
        and cache.shape[1:] == (1, 656)
        and cache.is_contiguous()
        and _ids(slots, cache.device)
        and slots.numel() <= padded_rows
    ):
        return None
    out = torch.empty((padded_rows, 576), device=cache.device, dtype=torch.bfloat16)
    if padded_rows:
        kernel = _launch(
            _read_zero,
            (padded_rows,),
            (cache, slots, out, slots.numel(), cache.stride(0)),
        )
        if (
            native_key is not None
            and cache.data_ptr() % 16 == slots.data_ptr() % 16 == 0
        ):
            prepared = _cpu_launch.native_launcher(kernel)
            if prepared:
                if len(_native_readers) >= 128:
                    _native_readers.pop(next(iter(_native_readers)))
                _native_readers[native_key] = prepared
    return out
