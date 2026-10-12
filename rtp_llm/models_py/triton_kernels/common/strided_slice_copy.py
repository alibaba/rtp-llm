"""Fast strided slice copy: ``dst[..., col_offset : col_offset + R] = src``.

Targets the F5 site in ``_apply_input_bmm`` where we copy
``q_pe`` (contiguous ``[T, H, R]``) into the suffix of an interleaved
``[T, H, KV+R]`` buffer. PyTorch's default elementwise kernel runs at
~15% of peak BW for this src→interleaved-dst pattern; a vectorized
Triton kernel reaches near-peak BW.

Layout:
  - src ``[T, H, R]``, strides ``(H*R, R, 1)`` — last 2 dims contig
  - dst ``[T, H, KV+R]``, strides ``(H*(KV+R), KV+R, 1)`` — interleaved
  - copy writes to ``dst[t, h, KV : KV+R]`` for all (t, h)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _strided_slice_copy_kernel(
    src_ptr,
    dst_ptr,
    n_rows,  # total = T * H
    R: tl.constexpr,  # source last-dim and slice width
    dst_col_offset,  # column offset into dst's last dim
    src_stride_row,  # stride between consecutive (T*H+i) rows in src
    dst_stride_row,  # stride between consecutive rows in dst
    BLOCK_ROWS: tl.constexpr,  # rows per program
    BLOCK_R: tl.constexpr,  # power-of-2 padded R
):
    """Each program copies BLOCK_ROWS rows of R elements each.

    Both src and dst can have arbitrary row strides (last dim must be contig).
    Used to copy a strided slice from one interleaved buffer to another.
    """
    # Long prefill can make row * stride exceed int32 range (for GLM5,
    # T~112K,H=64,D=576 is >4B elements). Keep address arithmetic in int64.
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * BLOCK_ROWS
    row_offs = row_start + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    row_mask = row_offs < n_rows

    col_offs = tl.arange(0, BLOCK_R)
    col_mask = col_offs < R
    col_offs_i64 = col_offs.to(tl.int64)
    src_stride_row_i64 = src_stride_row.to(tl.int64)
    dst_stride_row_i64 = dst_stride_row.to(tl.int64)
    dst_col_offset_i64 = dst_col_offset.to(tl.int64)

    # 2D load: [BLOCK_ROWS, BLOCK_R]
    src_addrs = src_ptr + row_offs[:, None] * src_stride_row_i64 + col_offs_i64[None, :]
    src_data = tl.load(src_addrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    dst_addrs = (
        dst_ptr
        + row_offs[:, None] * dst_stride_row_i64
        + dst_col_offset_i64
        + col_offs_i64[None, :]
    )
    tl.store(dst_addrs, src_data, mask=row_mask[:, None] & col_mask[None, :])


def strided_slice_copy_(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_col_offset: int,
) -> None:
    """In-place copy of ``src`` into ``dst[..., dst_col_offset : dst_col_offset + R]``.

    Args:
        dst: destination tensor of shape ``[T, H, D]`` (D >= dst_col_offset + R)
        src: source tensor of shape ``[T, H, R]`` — strided OK, last dim must be contig
        dst_col_offset: column offset where src is placed inside dst's last dim

    Equivalent semantics to:
        ``dst[..., dst_col_offset : dst_col_offset + R].copy_(src)``

    Constraints:
        - both src and dst must be 3-D bf16/fp16/fp32
        - src last dim must be contiguous (stride[-1] == 1) — middle dim can be strided
        - dst last dim must be contiguous (stride[-1] == 1)
        - both must satisfy ``stride(0) == H * stride(1)`` (flat-row trick)

    NOTE: src does NOT need to be ``.contiguous()`` — the kernel reads via
    ``stride(1)``. This is the whole point: callers like
    ``q_pe = q.split(...)`` produce strided views, and forcing
    ``q_pe.contiguous()`` would re-introduce the very copy kernel we're
    trying to fuse away.
    """
    assert src.dim() == 3 and dst.dim() == 3
    assert src.dtype == dst.dtype
    assert src.stride(-1) == 1, "src last dim must be contiguous"
    assert dst.stride(-1) == 1, "dst last dim must be contiguous"
    T, H, R = src.shape
    assert dst.shape[0] == T and dst.shape[1] == H
    assert dst_col_offset + R <= dst.shape[2]
    if T == 0:
        return

    # The flattening (t, h) -> i = t*H + h must respect each tensor's strides.
    # For a tensor with stride(0) == H * stride(1), the flat-row index walks
    # via stride(1) for both h++ within a token AND t++ at h=0 boundary.
    assert src.stride(0) == H * src.stride(1), (
        f"src must have stride(0) == H * stride(1) (got {src.stride()}), "
        "i.e. T axis contiguous in per-head blocks"
    )
    assert dst.stride(0) == H * dst.stride(
        1
    ), f"dst must have stride(0) == H * stride(1) (got {dst.stride()})"

    # In CUDA Graph mode (the production decode path), Triton wins for all T:
    # bench shows 1.06x-3.83x across T=1..2048. In pure eager mode (no graph)
    # Triton has ~30us launch overhead which can lose to torch's native
    # elementwise kernel for very small workloads, but production MLA decode
    # runs under CUDA Graph so launch overhead is amortized.
    n_rows = T * H
    src_stride_row = src.stride(1)
    dst_stride_row = dst.stride(1)

    # Pick BLOCK_ROWS so each program handles ~64 KB to amortize launch.
    BLOCK_ROWS = 256
    BLOCK_R = triton.next_power_of_2(R)

    grid = (triton.cdiv(n_rows, BLOCK_ROWS),)
    _strided_slice_copy_kernel[grid](
        src,
        dst,
        n_rows,
        R,
        dst_col_offset,
        src_stride_row,
        dst_stride_row,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_R=BLOCK_R,
        num_warps=4,
    )
