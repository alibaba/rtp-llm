"""Copy rank-major new KV rows directly into unique logical destinations."""

import torch
import triton
import triton.language as tl


@triton.jit
def _restore_patch_kernel(
    Source, Restore, Dest, Output, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    src = tl.load(Restore + row).to(tl.int64)
    dst = tl.load(Dest + row).to(tl.int64)
    col = tl.arange(0, BLOCK)
    value = tl.load(Source + src * WIDTH + col, mask=col < WIDTH, other=0)
    tl.store(Output + dst * WIDTH + col, value, mask=col < WIDTH)


def restore_patch_kv(source, restore, dest, output):
    """Exact index_select + index_copy without the selected intermediate.

    Caller supplies in-range indices with unique destination rows from
    prefix_plan. Source and output are distinct contiguous BF16 allocations.
    Untouched history rows are preserved. No host/device synchronization.
    """
    if (
        source.ndim != 2
        or output.ndim != 2
        or source.size(1) != output.size(1)
        or source.dtype != torch.bfloat16
        or output.dtype != source.dtype
        or source.device != output.device
        or not source.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError("invalid BF16 KV restore/patch buffers")
    for index in (restore, dest):
        if (
            index.ndim != 1
            or index.dtype != torch.int64
            or index.device != source.device
            or not index.is_contiguous()
        ):
            raise ValueError("invalid KV restore/patch indices")
    if restore.numel() != dest.numel():
        raise ValueError("KV restore/patch row mismatch")
    if not restore.numel():
        return
    if not source.is_cuda:
        output.index_copy_(0, dest, source.index_select(0, restore))
        return
    # Use int16 views to retain all bits, including NaN payload and signed zero.
    _restore_patch_kernel[(restore.numel(),)](
        source.view(torch.int16),
        restore,
        dest,
        output.view(torch.int16),
        source.size(1),
        triton.next_power_of_2(source.size(1)),
    )
