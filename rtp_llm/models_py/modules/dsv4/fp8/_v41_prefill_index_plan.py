"""Single-request FlashMLA index plans without a bool argsort or host reads.

The caller still owns per-forward plan caching and CED/layout invalidation.
Unsupported metadata returns None before allocation/launch; execution errors
propagate.
"""

from functools import lru_cache

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name not in ("triton", "triton.language"):
        raise
    triton = None


_MAX_ROWS = 131072
_MAX_WIDTH = 2048
_INT32_MAX = 2**31 - 1


@lru_cache(maxsize=16)
def _device_supported(device):
    return torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))


if triton is not None:

    @triton.jit(do_not_specialize=["OFFSET", "GLOBAL_COUNT", "SWA_START"])
    def _prefill_index_plan_kernel(
        selected,
        positions,
        indices,
        lengths,
        OFFSET,
        GLOBAL_COUNT,
        SWA_START,
        SELECTED_STRIDE: tl.constexpr,
        POSITION_STRIDE: tl.constexpr,
        K: tl.constexpr,
        WINDOW: tl.constexpr,
        PADDED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        column = tl.arange(0, BLOCK)
        picked = tl.load(
            selected + row * SELECTED_STRIDE + column, column < K, other=-1
        ).to(tl.int64)
        position = tl.load(positions + row * POSITION_STRIDE).to(tl.int64)
        offset = OFFSET.to(tl.int64)
        size = GLOBAL_COUNT.to(tl.int64)
        start = SWA_START.to(tl.int64)
        swpos = position - WINDOW + 1 + (column - K).to(tl.int64)
        global_index = tl.where(picked >= 0, offset + picked, -1)
        swa_index = tl.where(swpos >= start, offset + size + swpos - start, -1)
        # The eager path casts to int32 BEFORE sorting validity and summing.
        value = tl.where(column < K, global_index, swa_index).to(tl.int32)
        active = column < K + WINDOW
        valid = active & (value >= 0)
        prefix = tl.cumsum(valid.to(tl.int32), axis=0)
        count = tl.sum(valid.to(tl.int32), axis=0)
        # Valid columns scatter to [0,count) in original order. Invalid
        # columns fill [count,K+WINDOW), normally with -1. Preserve wrapped
        # negative int32 payloads as well, matching eager .int()+stable sort.
        destination = tl.where(valid, prefix - 1, count + column - prefix)
        tl.store(indices + row * PADDED + destination, value, active)
        # Logical invalids and alignment padding occupy disjoint addresses.
        tl.store(
            indices + row * PADDED + column,
            -1,
            (column >= K + WINDOW) & (column < PADDED),
        )
        tl.store(lengths + row, count)

else:
    _prefill_index_plan_kernel = None


def is_supported(selected, positions, offset, global_count, swa_start, window_size):
    """Metadata-only gate; tensor-valued/batched offsets deliberately fall back."""
    if not (
        _prefill_index_plan_kernel is not None
        and torch.version.hip is None
        and all(type(x) is int for x in (offset, global_count, swa_start, window_size))
        and 0 <= offset <= _INT32_MAX
        and 0 <= global_count <= _INT32_MAX - offset
        and -(2**31) <= swa_start <= _INT32_MAX
        and 1 <= window_size <= _MAX_WIDTH
        and selected.is_cuda
        and selected.dtype in (torch.int32, torch.int64)
        and selected.ndim == 2
        and 0 <= selected.shape[0] <= _MAX_ROWS
        and 1 <= selected.shape[1] <= _MAX_WIDTH - window_size
        and selected.stride(1) == 1
        and selected.stride(0) >= selected.shape[1]
        and positions.device == selected.device
        and positions.ndim == 1
        and positions.shape[0] == selected.shape[0]
        and positions.dtype in (torch.int32, torch.int64)
        and positions.stride(0) > 0
    ):
        return False
    return _device_supported(selected.device)


def try_build_index_plan(
    selected,
    positions,
    offset,
    global_count,
    swa_start,
    window_size=128,
    *,
    out=None,
    lengths_out=None,
):
    """Return contiguous int32 (indices[M,pad64(K+window)], lengths[M]).

    Scalar metadata follows _prefill_chunk_meta's single-request contract.
    Negative selected entries are normalized to -1 before offsetting. SWA
    validity uses exactly the eager lower bound, with no extra upper clamp.
    Optional outputs avoid allocation during graph capture/replay. No tensor
    is cached, synchronized, or retained by the module.
    """
    if not is_supported(
        selected, positions, offset, global_count, swa_start, window_size
    ):
        return None
    rows, k = selected.shape
    padded = ((k + window_size + 63) // 64) * 64
    for output, shape in ((out, (rows, padded)), (lengths_out, (rows,))):
        if output is not None and not (
            output.device == selected.device
            and output.dtype == torch.int32
            and output.shape == shape
            and output.is_contiguous()
        ):
            return None
    # Reject any output sharing storage with an input or the other output.
    # Storage metadata is host-visible and never requires a CUDA synchronize.
    provided = [x for x in (out, lengths_out) if x is not None and x.numel()]
    sources = [x for x in (selected, positions) if x.numel()]
    storage = {x.untyped_storage().data_ptr() for x in sources}
    for output in provided:
        pointer = output.untyped_storage().data_ptr()
        if pointer in storage:
            return None
        storage.add(pointer)
    if out is None:
        out = torch.empty((rows, padded), dtype=torch.int32, device=selected.device)
    if lengths_out is None:
        lengths_out = torch.empty((rows,), dtype=torch.int32, device=selected.device)
    if rows:
        _prefill_index_plan_kernel[(rows,)](
            selected,
            positions,
            out,
            lengths_out,
            offset,
            global_count,
            swa_start,
            selected.stride(0),
            positions.stride(0),
            k,
            window_size,
            padded,
            triton.next_power_of_2(padded),
            num_warps=4,
        )
    return out, lengths_out
