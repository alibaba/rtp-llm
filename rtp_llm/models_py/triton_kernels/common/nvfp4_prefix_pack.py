"""Pack both opaque M3.1 prefix pools without changing collective layout."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["PAGES"])
def _pack_prefix_pools(
    MAIN, SIDE, IDS, MAIN_OUT, SIDE_OUT, PAGES,
    MAIN_STRIDE: tl.constexpr, SIDE_STRIDE: tl.constexpr,
    TILE: tl.constexpr = 4096,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    source = tl.load(IDS + row).to(tl.int64)
    tl.device_assert((source >= 0) & (source < PAGES), "prefix page ID out of bounds")
    offset = tile * TILE + tl.arange(0, TILE)
    if tile * TILE < 65536:
        value = tl.load(MAIN + source * MAIN_STRIDE + offset, offset < 65536, 0)
        tl.store(MAIN_OUT + row * 65536 + offset, value, offset < 65536)
    else:
        offset -= 65536
        value = tl.load(SIDE + source * SIDE_STRIDE + offset, offset < 17408, 0)
        tl.store(SIDE_OUT + row * 17408 + offset, value, offset < 17408)


def pack_prefix_pools(main, side, page_ids, *, out=None):
    """One gather kernel, two independent contiguous uint8 outputs.

    The caller supplies valid physical IDs (including repeated IDs and padded
    zero pages). Bounds assertions preserve the old index_select failure rather
    than silently reading invalid pages. Optional nonaliasing ``out`` buffers
    allow the caller to reuse stream-ordered communication scratch. The caller
    owns buffer readiness and must not alias either input or the other output.
    This helper adds no cache; collective layouts and payload widths are unchanged.
    """
    if (
        main.ndim != 2 or side.ndim != 2
        or main.dtype != torch.uint8 or side.dtype != torch.uint8
        or main.shape[1] != 65536 or side.shape != (main.shape[0], 17408)
        or main.stride(1) != 1 or side.stride(1) != 1
        or main.stride(0) < 65536 or side.stride(0) < 17408
        or not main.is_cuda or side.device != main.device
        or page_ids.device != main.device or page_ids.ndim != 1
        or page_ids.dtype not in (torch.int32, torch.int64)
        or not page_ids.is_contiguous()
    ):
        raise ValueError("M3.1 prefix pack requires two uint8 page pools and CUDA IDs")
    rows = page_ids.numel()
    if out is None:
        main_out = torch.empty((rows, 65536), dtype=torch.uint8, device=main.device)
        side_out = torch.empty((rows, 17408), dtype=torch.uint8, device=main.device)
    else:
        if len(out) != 2:
            raise ValueError("prefix pack out requires two tensors")
        main_out, side_out = out
        for tensor, width in ((main_out, 65536), (side_out, 17408)):
            if (tensor.shape != (rows, width) or tensor.dtype != torch.uint8
                    or tensor.device != main.device or not tensor.is_contiguous()):
                raise ValueError("prefix pack out has incompatible shape/dtype/device/stride")
    if rows:
        _pack_prefix_pools[(rows, triton.cdiv(65536 + 17408, 4096))](
            main, side, page_ids, main_out, side_out, main.shape[0],
            main.stride(0), side.stride(0), num_warps=4, debug=True,
        )
    return main_out, side_out
