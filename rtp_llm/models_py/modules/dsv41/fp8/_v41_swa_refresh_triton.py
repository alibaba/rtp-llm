"""Bitwise ragged SWA refresh; dynamic row counts share one warmed kernel."""

import threading

import torch
import triton
import triton.language as tl

_READY = {}
_LOCK = threading.Lock()
_BLOCK = 2048


@triton.jit(do_not_specialize=["WIDTH", "SOURCE_STRIDE", "BATCH"])
def _copy_swa_rows(
    source, output, layout, WIDTH, SOURCE_STRIDE, BATCH, BLOCK: tl.constexpr
):
    tile = tl.program_id(0).to(tl.int64)
    # Host layout contains cumulative *live* tile counts. A compact grid avoids
    # B*max(length) masked CTAs for a batch with one long request and many tails.
    lower = tl.full((), 0, tl.int32)
    upper = BATCH
    while lower < upper:
        middle = (lower + upper) // 2
        end = tl.load(layout + middle * 4 + 3)
        passed = tile >= end
        lower = tl.where(passed, middle + 1, lower)
        upper = tl.where(passed, upper, middle)
    request = lower
    first_tile = tl.load(layout + (request - 1) * 4 + 3, request > 0, other=0)
    source_row = tl.load(layout + request * 4)
    output_row = tl.load(layout + request * 4 + 1)
    rows = tl.load(layout + request * 4 + 2)
    element = (tile - first_tile) * BLOCK + tl.arange(0, BLOCK)
    valid = element < rows * WIDTH
    if SOURCE_STRIDE == WIDTH:
        src = source_row * SOURCE_STRIDE + element
    else:
        src = (
            source_row * SOURCE_STRIDE
            + element // WIDTH * SOURCE_STRIDE
            + element % WIDTH
        )
    dst = output_row * WIDTH + element
    # No floating-point conversion: preserve signed zero and all NaN payloads.
    value = tl.load(source.to(tl.pointer_type(tl.uint64)) + src, valid, other=0)
    tl.store(output.to(tl.pointer_type(tl.uint64)) + dst, value, valid)


def _key(device):
    return device.index if device.index is not None else torch.cuda.current_device()


def is_supported(source):
    return (
        source.is_cuda
        and torch.version.hip is None
        and source.dtype == torch.bfloat16
        and source.ndim == 2
        and source.shape[1] > 0
        and source.shape[1] % 4 == 0
        and source.stride(1) == 1
        and source.stride(0) >= source.shape[1]
        and source.stride(0) % 4 == 0
        and source.data_ptr() % 8 == 0
        and not source.requires_grad
        and not source.is_conj()
        and not source.is_neg()
        and _key(source.device) in _READY
    )


def copy_rows(source, output, layout, tiles):
    """Inputs have already been checked by the per-forward workspace owner."""
    if tiles == 0:
        return
    kernel = _READY[_key(source.device)]
    kernel[(tiles, 1, 1)](
        source,
        output,
        layout,
        source.shape[1] // 4,
        source.stride(0) // 4,
        layout.shape[0],
        _BLOCK,
    )


def warmup(device):
    """Compile/load once before requests; never retain private tensor storage."""
    device = torch.device(device)
    if device.type != "cuda" or torch.version.hip is not None:
        return False
    with torch.cuda.device(device), _LOCK:
        key = torch.cuda.current_device()
        if key in _READY:
            return True
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SWA refresh warmup must precede CUDA graph capture")
        # Minimal alignment avoids depending on request storage offsets. Width
        # and source stride remain runtime values even for a partial final row.
        source = torch.zeros(4 + 2 * 68, device=device, dtype=torch.bfloat16)[4:].view(
            2, 68
        )
        output = torch.empty_like(source)
        host = torch.tensor([[0, 0, 2, 1]], dtype=torch.int64, pin_memory=True)
        layout = host.to(device=device, non_blocking=True)
        args = (source, output, layout, 17, 17, 1, _BLOCK)
        kernel = _copy_swa_rows.warmup(*args, grid=(1, 1, 1), num_warps=4)
        kernel[(1, 1, 1)](*args)
        torch.cuda.current_stream().synchronize()
        _READY[key] = kernel
    return True
