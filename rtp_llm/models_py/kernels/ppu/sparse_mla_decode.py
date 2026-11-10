"""Stable sparse MLA decode with FP8 storage and BF16 attention arithmetic."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.flash_mla_legacy import flash_mla_sparse_fwd


@triton.jit
def _gather_selected(
    cache,
    indices,
    output,
    mapped,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    MAX_SLOT: tl.constexpr,
):
    row = tl.program_id(0)
    physical = tl.load(indices + row).to(tl.int64)
    valid = (physical >= 0) & (physical < MAX_SLOT)
    physical = tl.where(valid, physical, 0)
    source = (
        cache
        + physical // BLOCK_SIZE * BLOCK_STRIDE
        + physical % BLOCK_SIZE * ROW_STRIDE
    )
    cols = tl.arange(0, 1024)
    bits = tl.load(source + cols, valid & (cols < 512), other=0)
    values = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scales = tl.load(
        (source + 512).to(tl.pointer_type(tl.float32)) + cols // 128,
        valid & (cols < 512),
        other=0,
    )
    rope = tl.load(
        (source + 528).to(tl.pointer_type(tl.bfloat16)) + cols - 512,
        valid & (cols >= 512) & (cols < 576),
        other=0,
    ).to(tl.float32)
    restored = tl.where(cols < 512, values * scales, rope).to(tl.bfloat16)
    tl.store(output + row * 576 + cols, restored, cols < 576)
    tl.store(mapped + row, tl.where(valid, row, -1))


@triton.jit
def _zero_empty_rows(
    output,
    indices,
    WIDTH: tl.constexpr,
    MAX_SLOT: tl.constexpr,
    HEADS: tl.constexpr,
    STRIDE_ROW: tl.constexpr,
    STRIDE_HEAD: tl.constexpr,
    STRIDE_COL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    physical = tl.load(indices + row * WIDTH + k, k < WIDTH, other=-1)
    count = tl.sum(((physical >= 0) & (physical < MAX_SLOT)).to(tl.int32), 0)
    if count == 0:
        col = tl.arange(0, BLOCK_OUT)
        tl.store(
            output
            + row * STRIDE_ROW
            + col // 512 * STRIDE_HEAD
            + col % 512 * STRIDE_COL,
            0,
            col < HEADS * 512,
        )


def sparse_mla_fp8_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attend to selected 656-byte MLA rows, retaining the FP8 cache format.

    The PPU paged FP8 kernel can return different BF16 values on identical
    inputs. Gather only the selected keys and use the repeatable BF16 sparse
    kernel instead. All work stays on the current stream, including graph
    replay; query/indices/cache remain dynamic graph inputs.

    Workspace is local to the forward. The caching allocator can reuse it
    between layers, while graph capture retains its storage for replay.
    """
    rows, heads, dim = q.shape
    if q.dtype != torch.bfloat16 or dim != 576:
        raise ValueError("PPU sparse MLA decode requires BF16 queries with dim 576")
    if cache.shape[-1] != 656 or indices.shape[:2] != (rows, 1):
        raise ValueError(
            "PPU sparse MLA decode requires a 656-byte cache and [T,1,K] indices"
        )
    width = indices.shape[-1]
    if rows == 0 or width == 0:
        return q.new_zeros((rows, heads, 512))

    packed = cache if cache.dtype == torch.uint8 else cache.view(torch.uint8)
    physical = indices.contiguous()
    workspace = q.new_empty((rows * width, 1, 576))
    mapped = torch.empty((rows, 1, width), device=q.device, dtype=torch.int32)
    _gather_selected[(rows * width,)](
        packed,
        physical,
        workspace,
        mapped,
        cache.shape[1],
        packed.stride(0),
        packed.stride(1),
        cache.shape[0] * cache.shape[1],
        num_warps=4,
    )
    output, _, _ = flash_mla_sparse_fwd(q, workspace, mapped, scale, d_v=512)
    # Padded graph rows may contain no selected keys. Never expose an undefined
    # softmax result from those rows to a subsequent residual/GEMM.
    # Fuse the validity reduction and zero-fill. Rows with keys need no output
    # read/write, and there is no temporary mask or separate comparison kernel.
    _zero_empty_rows[(rows,)](
        output,
        physical,
        width,
        cache.shape[0] * cache.shape[1],
        heads,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        triton.next_power_of_2(width),
        triton.next_power_of_2(heads * 512),
        num_warps=4,
    )
    return output
