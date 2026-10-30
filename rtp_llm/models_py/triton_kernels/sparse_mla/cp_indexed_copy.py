"""Exact byte-moving CP forward fusions; no arithmetic on KV values."""

import torch
import triton as tr
import triton.language as tl
from rtp_llm.models_py.kernels.cuda import prefill_cpu_ops as _cpu_ops
from rtp_llm.models_py.triton_kernels.common import fast_launch as _cpu_launch

_copy_kernels = {}
_native_copies = {}


def _launch(cache, key, jit, grid, args):
    kernel = cache.get(key)
    if kernel is None:
        kernel = jit[grid](*args)
        if len(cache) >= 128:
            cache.pop(next(iter(cache)))
        cache[key] = kernel
    else:
        _cpu_launch.launch_compiled(kernel, grid, args)


def _matrix(x):
    return (
        x.is_cuda
        and x.ndim == 2
        and x.stride(1) == 1
        and x.stride(0) >= x.shape[1]
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and not x.requires_grad
    )


def _indices(x, device):
    return (
        x.device == device
        and x.ndim == 1
        and x.is_contiguous()
        and x.dtype in (torch.int32, torch.int64)
    )


@tr.jit(do_not_specialize=["N"])
def _indexed_copy(
    Src,
    SrcIdx,
    DstIdx,
    Out,
    N,
    W: tl.constexpr,
    SS: tl.constexpr,
    DS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = p // W, p % W
    s = tl.load(SrcIdx + row, row < N, other=0).to(tl.int64)
    d = tl.load(DstIdx + row, row < N, other=0).to(tl.int64)
    value = tl.load(Src + s * SS + col, row < N, other=0)
    tl.store(Out + d * DS + col, value, row < N)


def try_indexed_copy(src, src_indices, dst_indices, out) -> bool:
    """out[dst_indices] = src[src_indices], with unique, in-bounds destinations.

    The CP prefix plan guarantees index bounds/uniqueness. Source and output
    must not alias; the caller must have joined the history producer event.
    """
    native_key = None
    use_native = (
        _cpu_ops.NATIVE_LAUNCH
        and _cpu_launch.ENABLED
        and _cpu_launch._raw_stream is not None
        and _cpu_launch.hooks_idle()
        and src.is_cuda
        and src.ndim == out.ndim == 2
    )
    if use_native:
        native_key = (
            src.device.index,
            src.dtype,
            src_indices.dtype,
            dst_indices.dtype,
            src.shape[1],
            src.stride(0),
            out.stride(0),
            min(src_indices.numel() // 2**31, 2),
        )
        prepared = _native_copies.get(native_key)
        if prepared is not None and torch.cuda.current_device() != src.device.index:
            with torch.cuda.device(src.device):
                return try_indexed_copy(src, src_indices, dst_indices, out)
        if prepared is not None and prepared.indexed_copy(
            src,
            src_indices,
            dst_indices,
            out,
            _cpu_launch._raw_stream(src.device.index),
        ):
            return True
    if not (
        _matrix(src)
        and _matrix(out)
        and src.device == out.device
        and src.dtype == out.dtype
        and src.shape[1] == out.shape[1]
        and src.shape[1] > 0
        and _indices(src_indices, src.device)
        and _indices(dst_indices, src.device)
        and src_indices.shape == dst_indices.shape
        and out.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
        and out.untyped_storage().data_ptr() != src_indices.untyped_storage().data_ptr()
        and out.untyped_storage().data_ptr() != dst_indices.untyped_storage().data_ptr()
    ):
        return False
    n = src_indices.numel()
    if n:
        key = (
            src.device.index,
            src.dtype,
            src_indices.dtype,
            dst_indices.dtype,
            src.shape[1],
            src.stride(0),
            out.stride(0),
            min(n // 2**31, 2),
            src.data_ptr() % 16,
            src_indices.data_ptr() % 16,
            dst_indices.data_ptr() % 16,
            out.data_ptr() % 16,
        )
        with torch.cuda.device(src.device):
            _launch(
                _copy_kernels,
                key,
                _indexed_copy,
                (tr.cdiv(n * src.shape[1], 1024), 1, 1),
                (
                    src,
                    src_indices,
                    dst_indices,
                    out,
                    n,
                    src.shape[1],
                    src.stride(0),
                    out.stride(0),
                    1024,
                ),
            )
            if native_key is not None and all(
                t.data_ptr() % 16 == 0 for t in (src, src_indices, dst_indices, out)
            ):
                prepared = _cpu_launch.native_launcher(_copy_kernels[key])
                if prepared:
                    if len(_native_copies) >= 128:
                        _native_copies.pop(next(iter(_native_copies)))
                    _native_copies[native_key] = prepared
    return True
