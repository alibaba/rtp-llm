"""E7 (Batch-4b part 2): fused UE8M0 scale-layout kernels.

The SM120 MXFP8 linear chain round-trips activation scales through two
pure-layout transforms per call:

1. ``utils.V41MXFP8Linear.forward_quantized`` converts the sgl quantizer's
   int32-packed column-major TMA scales to linear uint8 via
   ``clone -> view(uint8) -> slice -> contiguous`` (two full copies);
2. ``v41_mxfp8.V41MXFP8Linear.forward_quantized`` then calls flashinfer's
   ``block_scale_interleave`` (another full copy) into the 128x4-swizzled
   flat layout the CUTLASS GEMM consumes.

Under ``DSV41_E7_SCALE_SWIZZLE`` these collapse into one Triton kernel per
call that reads the source layout and writes the swizzled flat buffer
directly. Layout only — no arithmetic: the emitted bytes must be identical
to ``block_scale_interleave`` of the linear view (GPU byte-compare gated).

Swizzle (NVIDIA 128x4 SF layout, matching flashinfer sm100 op):
    in[M, K] uint8 (zero-padded to ceil128(M) x ceil4(K))
    out[((m//128)*K4 + k//4)*512 + (m%32)*16 + ((m%128)//32)*4 + k%4]
        = in[m, k]
"""

import os
import sys

import torch
import triton
import triton.language as tl

_E7_FLAG = "DSV41_E7_SCALE_SWIZZLE"
_E7_ANNOUNCED = False


def e7_scale_swizzle_enabled() -> bool:
    """E7: emit swizzled scales in one kernel. Default ON since Batch-5
    (R16): byte-gate 0/1,822,720 vs the compiled flashinfer
    block_scale_interleave (packed-int32 AND linear-uint8 entries, padded
    shapes), GPU parity byte-exact with engagement
    proven x8, A/B median −8.3 ms 2/3. ``=0`` restores the
    clone/view/slice/contiguous copy + per-call interleave."""
    return os.environ.get(_E7_FLAG, "1") == "1"


def _announce_e7(where: str) -> None:
    global _E7_ANNOUNCED
    if not _E7_ANNOUNCED:
        _E7_ANNOUNCED = True
        print(f"[E7SWZ] scale-swizzle engaged ({where})", file=sys.stderr, flush=True)


def swizzled_scale_size(rows: int, k32: int) -> int:
    """Flat byte size of the 128x4-swizzled buffer (matches _check_packed)."""
    return ((rows + 127) // 128 * 128) * ((k32 + 3) // 4 * 4)


def swizzle_reference(src: torch.Tensor) -> torch.Tensor:
    """Torch reference of the swizzle: [M, K] uint8 -> flat padded buffer.

    CPU-testable ground truth of the index math the Triton kernels implement.
    """
    if src.dtype != torch.uint8 or src.ndim != 2:
        raise ValueError("swizzle reference requires uint8 [M, K]")
    m, k = src.shape
    pm, pk = (m + 127) // 128 * 128, (k + 3) // 4 * 4
    padded = torch.zeros((pm, pk), dtype=torch.uint8, device=src.device)
    padded[:m, :k] = src
    return (
        padded.reshape(pm // 128, 4, 32, pk // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .flatten()
        .contiguous()
    )


@triton.jit
def _swizzle_kernel(
    SRC,
    OUT,
    rows,
    src_row_stride,
    K32: tl.constexpr,
    K4: tl.constexpr,
    PACKED_INT32: tl.constexpr,
):
    """One program per (128-row tile, 4-col tile) -> 512 output bytes.

    PACKED_INT32: SRC is the int32 column-major buffer (stride between
    packed columns = src_row_stride int32 elements); byte col of the
    linear view = 4*col32 + b lives at little-endian byte b of column
    col32. Otherwise SRC is the linear uint8 [rows, K32] view.
    """
    m_tile = tl.program_id(0)
    k_tile = tl.program_id(1)
    lanes = tl.arange(0, 128)
    sub = tl.arange(0, 4)
    rows_idx = m_tile * 128 + lanes
    cols_idx = k_tile * 4 + sub
    row_mask = rows_idx < rows
    col_mask = cols_idx < K32
    if PACKED_INT32:
        word = tl.load(
            SRC + k_tile.to(tl.int64) * src_row_stride + rows_idx,
            row_mask,
            0,
        )
        shifts = (sub * 8)[None, :]
        block = ((word[:, None] >> shifts) & 0xFF).to(tl.uint8)
    else:
        block = tl.load(
            SRC + rows_idx.to(tl.int64)[:, None] * (K32) + cols_idx[None, :],
            row_mask[:, None] & col_mask[None, :],
            0,
        ).to(tl.uint8)
    block = tl.where(row_mask[:, None] & col_mask[None, :], block, 0).to(tl.uint8)
    out_off = (
        (m_tile * K4 + k_tile).to(tl.int64) * 512
        + (lanes % 32)[:, None] * 16
        + (lanes // 32)[:, None] * 4
        + sub[None, :]
    )
    tl.store(OUT + out_off, block)


def _launch(src_ptr, out, rows, k32, src_row_stride, packed):
    k4 = (k32 + 3) // 4
    grid = ((rows + 127) // 128, k4)
    _swizzle_kernel[grid](
        src_ptr,
        out,
        rows,
        src_row_stride,
        K32=k32,
        K4=k4,
        PACKED_INT32=packed,
        num_warps=4,
    )
    return out


def swizzle_linear(scales: torch.Tensor) -> torch.Tensor:
    """[rows, k32] contiguous uint8 -> flat swizzled buffer."""
    if (
        scales.dtype != torch.uint8
        or scales.ndim != 2
        or not scales.is_contiguous()
        or not scales.is_cuda
        or scales.shape[0] == 0
    ):
        raise ValueError("E7 linear swizzle requires contiguous CUDA uint8 [rows,k32]")
    rows, k32 = scales.shape
    out = torch.empty(
        (swizzled_scale_size(rows, k32),), dtype=torch.uint8, device=scales.device
    )
    _announce_e7("linear")
    return _launch(scales, out, rows, k32, 0, False)


def swizzle_from_packed_int32(x_s: torch.Tensor, rows: int, k32: int) -> torch.Tensor:
    """sgl int32 column-major TMA scales -> flat swizzled buffer.

    ``x_s`` is the sgl_per_token_group_quant_fp8 output view: int32 with
    shape (rows, ceil(k32/4)) and strides (1, aligned_rows); byte b of
    column c is the linear scale for group 4c+b. Replaces the
    clone->view(uint8)->slice->contiguous copy plus the separate
    block_scale_interleave pass.
    """
    if (
        x_s.dtype != torch.int32
        or x_s.ndim != 2
        or x_s.shape[0] != rows
        or x_s.stride(0) != 1
        or x_s.shape[1] * 4 < k32
        or not x_s.is_cuda
        or rows == 0
    ):
        raise ValueError("E7 packed swizzle requires int32 column-major sgl scales")
    out = torch.empty(
        (swizzled_scale_size(rows, k32),), dtype=torch.uint8, device=x_s.device
    )
    _announce_e7("packed")
    return _launch(x_s, out, rows, k32, x_s.stride(1), True)
