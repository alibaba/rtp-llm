"""MiniMax-M3.1 NVFP4 paged-cache layout, writers, and reference helpers.

The persistent representation follows the MiniMax-M3.1 contract:

* each contiguous group of 16 values has an E4M3 scale ``clamp(amax / 6,
  1/512, 448)`` (rounded by the E4M3 cast);
* scaled values use E2M1 with round-to-nearest-even boundaries;
* two E2M1 nibbles are stored per byte, with the lower-dimension value in the
  low nibble;
* there is no tensor-level scale (it is identically one).

Production prefill/decode readers live in the sparse-MSA modules and consume
the packed value/scale planes directly.  The BF16 conversion helpers retained
in this module are offline correctness utilities; they are not a runtime
fallback for MiniMax-M3.1.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

NVFP4_GROUP_SIZE = 16
NVFP4_VALUES_PER_BYTE = 2
NVFP4_SCALE_MIN = 1.0 / 512.0
NVFP4_SCALE_MAX = 448.0
NVFP4_E2M1_MAX = 6.0

# Triton rejects ordinary Python globals referenced from a JIT function.  Keep
# public integer/float constants above for host-side layout code and explicit
# constexpr twins for device code.
_TL_GROUP_SIZE = tl.constexpr(16)
_TL_VALUES_PER_BYTE = tl.constexpr(2)
_TL_SCALE_MIN = tl.constexpr(1.0 / 512.0)
_TL_SCALE_MAX = tl.constexpr(448.0)
_TL_E2M1_MAX = tl.constexpr(6.0)


@triton.jit(
    do_not_specialize=["N", "COLS", "TS0"],
    do_not_specialize_on_alignment=["N", "COLS", "TS0"],
)
def _decode_physical_slots_kernel(
    LENS,
    TABLE,
    OUT,
    N,
    COLS,
    LS: tl.constexpr,
    TS0,
    TS1: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = row < N
    if COLS == 0:
        tl.store(OUT + row, -1, mask=active)
    else:
        pos = tl.load(LENS + row * LS, mask=active, other=0).to(tl.int64) - 1
        col = tl.maximum(pos, 0) // PAGE
        valid = active & (pos >= 0) & (col < COLS)
        page = tl.load(
            TABLE + row.to(tl.int64) * TS0 + col * TS1,
            mask=valid,
            other=0,
        ).to(tl.int64)
        slot = page * PAGE + pos % PAGE
        tl.store(OUT + row, tl.where(valid & (page > 0), slot, -1), mask=active)


def build_decode_physical_slots(seq_lens, block_table, page_size=128):
    """Map token rows to int64 cache slots; invalid/padded rows become -1.

    The writer retains responsibility for physical pool bounds. Empty tables
    explicitly yield invalid slots. Allocation during capture belongs to the
    existing graph pool, avoiding mutable cross-layer workspace ownership.
    """
    if seq_lens.ndim != 1 or block_table.ndim != 2:
        raise ValueError("expected lens[N] and table[N,C]")
    n = seq_lens.numel()
    if block_table.shape[0] != n:
        raise ValueError("token dimensions must match")
    if seq_lens.dtype not in (torch.int32, torch.int64) or block_table.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("lens/table must be int32 or int64")
    if not seq_lens.is_cuda or seq_lens.device != block_table.device:
        raise ValueError("lens/table must use the same CUDA device")
    if not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    out = torch.empty((n,), dtype=torch.int64, device=seq_lens.device)
    if n:
        _decode_physical_slots_kernel[(triton.cdiv(n, 128),)](
            seq_lens,
            block_table,
            out,
            n,
            block_table.shape[1],
            seq_lens.stride(0),
            block_table.stride(0),
            block_table.stride(1),
            page_size,
            128,
            num_warps=4,
        )
    return out


@triton.jit
def _scale_128x4_offset(row, group):
    """Offset inside one 128-row cuBLAS/cuDNN block-scale tile."""
    return (group // 4) * 512 + (row % 32) * 16 + (row // 32) * 4 + group % 4


@dataclass(frozen=True)
class NVFP4CacheLayout:
    """Byte views for one layer of the persistent paged cache."""

    packed_main: torch.Tensor
    main_scales: torch.Tensor
    num_blocks: int
    num_heads: int
    page_size: int
    head_dim: int
    main_packed_bytes: int
    main_scale_bytes: int
    side_bytes: torch.Tensor

    def logical_views(self, indexer_dim: int) -> "NVFP4LogicalViews":
        """Expose the two-region cache as four native-kernel logical planes.

        The first-stage ABI still owns only ``kv_cache_base`` and
        ``kv_scale_base``.  Native kernels should consume these explicit views
        instead of re-deriving the side-region offsets independently.
        """
        main_k, main_k_scale = self.main_plane(0)
        main_v, main_v_scale = self.main_plane(1)
        idx_k, idx_k_scale = self.indexer(indexer_dim)
        return NVFP4LogicalViews(
            main_k_fp4=main_k,
            main_v_fp4=main_v,
            main_k_scale=main_k_scale,
            main_v_scale=main_v_scale,
            idx_k_fp4=idx_k,
            idx_k_scale=idx_k_scale,
        )

    def main_plane(self, kv_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_index not in (0, 1):
            raise ValueError(f"kv_index must be 0 or 1, got {kv_index}")
        packed_plane_bytes = self.num_heads * self.page_size * self.head_dim // 2
        scale_plane_bytes = (
            self.num_heads * self.page_size * (self.head_dim // NVFP4_GROUP_SIZE)
        )
        p0 = kv_index * packed_plane_bytes
        s0 = kv_index * scale_plane_bytes
        return (
            self.packed_main[:, p0 : p0 + packed_plane_bytes],
            self.main_scales[:, s0 : s0 + scale_plane_bytes],
        )

    def indexer(self, indexer_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_grouped_dim(indexer_dim, "NVFP4 indexer")
        value_bytes = self.page_size * indexer_dim // NVFP4_VALUES_PER_BYTE
        scale_bytes = self.page_size * indexer_dim // NVFP4_GROUP_SIZE
        begin = self.main_scale_bytes
        end = begin + value_bytes + scale_bytes
        if int(self.side_bytes.shape[1]) < end:
            raise RuntimeError(
                "NVFP4 indexer side region is too small: "
                f"got {self.side_bytes.shape[1]} bytes, need {end} "
                f"(main_scales={self.main_scale_bytes}, indexer_dim={indexer_dim})"
            )
        values = self.side_bytes[:, begin : begin + value_bytes]
        scales = self.side_bytes[:, begin + value_bytes : end].view(torch.float8_e4m3fn)
        return values, scales


@dataclass(frozen=True)
class NVFP4LogicalViews:
    """Four logical planes backed by the first-stage two-region allocation."""

    main_k_fp4: torch.Tensor
    main_v_fp4: torch.Tensor
    main_k_scale: torch.Tensor
    main_v_scale: torch.Tensor
    idx_k_fp4: torch.Tensor
    idx_k_scale: torch.Tensor


def _validate_grouped_dim(dim: int, name: str) -> None:
    if dim <= 0 or dim % NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"{name} dimension must be a positive multiple of "
            f"{NVFP4_GROUP_SIZE}, got {dim}"
        )


def cache_layout(
    kv_cache_base: torch.Tensor,
    kv_scale_base: torch.Tensor,
    num_heads: int,
    page_size: int,
    head_dim: int,
) -> NVFP4CacheLayout:
    """Validate and split a layer's raw persistent NVFP4 allocation."""
    _validate_grouped_dim(head_dim, "NVFP4 KV head")
    if kv_cache_base is None or kv_cache_base.dim() != 2:
        raise RuntimeError(
            "NVFP4 kv_cache_base must be a raw 2-D byte tensor, got "
            f"{None if kv_cache_base is None else tuple(kv_cache_base.shape)}"
        )
    if kv_cache_base.dtype != torch.uint8:
        raise RuntimeError(
            f"NVFP4 kv_cache_base must use torch.uint8, got {kv_cache_base.dtype}"
        )
    if kv_scale_base is None or kv_scale_base.dim() != 2:
        raise RuntimeError(
            "NVFP4 kv_scale_base must be a raw 2-D byte tensor, got "
            f"{None if kv_scale_base is None else tuple(kv_scale_base.shape)}"
        )
    if kv_scale_base.dtype != torch.uint8:
        raise RuntimeError(
            f"NVFP4 kv_scale_base must use torch.uint8, got {kv_scale_base.dtype}"
        )
    if kv_cache_base.stride(1) != 1 or kv_scale_base.stride(1) != 1:
        raise RuntimeError("NVFP4 cache byte rows must be contiguous")
    if int(kv_cache_base.shape[0]) != int(kv_scale_base.shape[0]):
        raise RuntimeError(
            "NVFP4 value/side block counts differ: "
            f"{kv_cache_base.shape[0]} vs {kv_scale_base.shape[0]}"
        )

    num_blocks = int(kv_cache_base.shape[0])
    main_packed_bytes = 2 * num_heads * page_size * head_dim // 2
    main_scale_bytes = 2 * num_heads * page_size * (head_dim // NVFP4_GROUP_SIZE)
    if int(kv_cache_base.shape[1]) < main_packed_bytes:
        raise RuntimeError(
            "NVFP4 value block stride is too small: "
            f"got {kv_cache_base.shape[1]}, need {main_packed_bytes}"
        )
    if int(kv_scale_base.shape[1]) < main_scale_bytes:
        raise RuntimeError(
            "NVFP4 scale block stride is too small: "
            f"got {kv_scale_base.shape[1]}, need {main_scale_bytes}"
        )
    packed_main = kv_cache_base[:, :main_packed_bytes]
    side_bytes = kv_scale_base.view(torch.uint8)
    main_scales = side_bytes[:, :main_scale_bytes].view(torch.float8_e4m3fn)
    return NVFP4CacheLayout(
        packed_main=packed_main,
        main_scales=main_scales,
        num_blocks=num_blocks,
        num_heads=num_heads,
        page_size=page_size,
        head_dim=head_dim,
        main_packed_bytes=main_packed_bytes,
        main_scale_bytes=main_scale_bytes,
        side_bytes=side_bytes,
    )


@triton.jit
def _e2m1_encode(values, scale):
    """Exact RNE ladder, compared in the original value domain.

    Dividing by ``scale`` can move a mathematical midpoint (for example 2.5)
    a few ulps above the threshold on GPU. Comparing ``abs(x)`` with the exact
    binary threshold multiplied by the stored E4M3 scale preserves ties-to-even.
    """
    magnitude = tl.abs(values)
    code = tl.where(
        magnitude > 5.0 * scale,
        7,
        tl.where(
            magnitude >= 3.5 * scale,
            6,
            tl.where(
                magnitude > 2.5 * scale,
                5,
                tl.where(
                    magnitude >= 1.75 * scale,
                    4,
                    tl.where(
                        magnitude > 1.25 * scale,
                        3,
                        tl.where(
                            magnitude >= 0.75 * scale,
                            2,
                            tl.where(magnitude > 0.25 * scale, 1, 0),
                        ),
                    ),
                ),
            ),
        ),
    ).to(tl.uint8)
    # Canonicalize both +0 and -0 to the single positive-zero encoding.
    sign = tl.where((values < 0.0) & (code != 0), 8, 0).to(tl.uint8)
    return code | sign


@triton.jit
def _e2m1_decode(code):
    magnitude_code = code & 7
    magnitude = tl.where(
        magnitude_code == 0,
        0.0,
        tl.where(
            magnitude_code == 1,
            0.5,
            tl.where(
                magnitude_code == 2,
                1.0,
                tl.where(
                    magnitude_code == 3,
                    1.5,
                    tl.where(
                        magnitude_code == 4,
                        2.0,
                        tl.where(
                            magnitude_code == 5,
                            3.0,
                            tl.where(magnitude_code == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 8) != 0, -magnitude, magnitude)


@triton.jit
def _to_e4m3_compute_grid(values):
    """Round NVFP4 dequantized values to the SM100 Q8KV4 compute grid.

    The packed cache remains E2M1 plus an E4M3 block scale.  MiniMax's native
    Q8KV4 kernels multiply in FP16 and then use a saturating E4M3 conversion
    before the MMA.  BF16 working pages are only a carrier here: rounding to
    E4M3 and widening back to FP32/BF16 reproduces that input value grid while
    retaining the existing BF16 attention implementation.
    """
    return values.to(tl.float8e4nv).to(tl.float32)


@triton.jit
def _quantize_rows_kernel(
    src_ptr,
    slots_ptr,
    packed_ptr,
    scales_ptr,
    N,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    SRC_S2: tl.constexpr,
    PACKED_S0: tl.constexpr,
    SCALE_S0: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    hg = tl.program_id(1)
    head = hg // GROUPS
    group = hg - head * GROUPS
    pair = tl.arange(0, _TL_GROUP_SIZE // _TL_VALUES_PER_BYTE)
    valid_row = row < N
    slot = tl.load(slots_ptr + row, mask=valid_row, other=-1).to(tl.int64)
    valid = valid_row & (slot >= 0) & (slot < NUM_BLOCKS * PAGE_SIZE)
    block = slot // PAGE_SIZE
    page_offset = slot - block * PAGE_SIZE

    even_values = tl.load(
        src_ptr
        + row * SRC_S0
        + head * SRC_S1
        + (group * _TL_GROUP_SIZE + 2 * pair) * SRC_S2,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    odd_values = tl.load(
        src_ptr
        + row * SRC_S0
        + head * SRC_S1
        + (group * _TL_GROUP_SIZE + 2 * pair + 1) * SRC_S2,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.maximum(tl.abs(even_values), tl.abs(odd_values)), axis=0)
    raw_scale = tl.minimum(
        tl.maximum(amax / _TL_E2M1_MAX, _TL_SCALE_MIN),
        _TL_SCALE_MAX,
    )
    # Casting first is material: E2M1 quantization divides by the stored E4M3
    # scale, not the unrounded FP32 candidate.
    stored_scale = raw_scale.to(tl.float8e4nv)
    scale = stored_scale.to(tl.float32)
    low_codes = _e2m1_encode(even_values, scale)
    high_codes = _e2m1_encode(odd_values, scale)
    packed_codes = low_codes | (high_codes << 4)

    packed_row_bytes = HEAD_DIM // _TL_VALUES_PER_BYTE
    packed_offset = (
        block * PACKED_S0
        + (head * PAGE_SIZE + page_offset) * packed_row_bytes
        + group * (_TL_GROUP_SIZE // _TL_VALUES_PER_BYTE)
        + pair
    )
    scale_offset = block * SCALE_S0 + (head * PAGE_SIZE + page_offset) * GROUPS + group
    tl.store(packed_ptr + packed_offset, packed_codes, mask=valid)
    tl.store(scales_ptr + scale_offset, stored_scale, mask=valid)


@triton.jit
def _quantize_main_index_rows_kernel(
    k_ptr,
    v_ptr,
    idx_ptr,
    slots_ptr,
    k_packed_ptr,
    k_scales_ptr,
    v_packed_ptr,
    v_scales_ptr,
    idx_packed_ptr,
    idx_scales_ptr,
    N,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    K_S2: tl.constexpr,
    V_S0: tl.constexpr,
    V_S1: tl.constexpr,
    V_S2: tl.constexpr,
    IDX_S0: tl.constexpr,
    IDX_S2: tl.constexpr,
    MAIN_PACKED_S0: tl.constexpr,
    MAIN_SCALE_S0: tl.constexpr,
    IDX_PACKED_S0: tl.constexpr,
    IDX_SCALE_S0: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAIN_GROUPS: tl.constexpr,
    IDX_DIM: tl.constexpr,
    IDX_GROUPS: tl.constexpr,
    MMA_SCALE_LAYOUT: tl.constexpr,
):
    """Quantize K, V and indexer-K rows with one launch.

    Program 1 enumerates all 16-value groups from the three logical planes.
    Keeping one group per CTA preserves the exact reduction and rounding order
    of ``_quantize_rows_kernel`` while removing two launch boundaries from the
    decode cache-write path.
    """
    row = tl.program_id(0)
    plane_group = tl.program_id(1)
    main_plane_groups = NUM_HEADS * MAIN_GROUPS
    is_k = plane_group < main_plane_groups
    is_v = (plane_group >= main_plane_groups) & (plane_group < 2 * main_plane_groups)
    is_idx = plane_group >= 2 * main_plane_groups

    local_group = tl.where(
        is_k,
        plane_group,
        tl.where(
            is_v,
            plane_group - main_plane_groups,
            plane_group - 2 * main_plane_groups,
        ),
    )
    main_head = local_group // MAIN_GROUPS
    main_group = local_group - main_head * MAIN_GROUPS
    idx_group = local_group
    pair = tl.arange(0, _TL_GROUP_SIZE // _TL_VALUES_PER_BYTE)

    valid_row = row < N
    slot = tl.load(slots_ptr + row, mask=valid_row, other=-1).to(tl.int64)
    valid_slot = valid_row & (slot >= 0) & (slot < NUM_BLOCKS * PAGE_SIZE)
    block = slot // PAGE_SIZE
    page_offset = slot - block * PAGE_SIZE

    main_even_offset = (
        row * K_S0 + main_head * K_S1 + (main_group * _TL_GROUP_SIZE + 2 * pair) * K_S2
    )
    main_odd_offset = main_even_offset + K_S2
    v_even_offset = (
        row * V_S0 + main_head * V_S1 + (main_group * _TL_GROUP_SIZE + 2 * pair) * V_S2
    )
    v_odd_offset = v_even_offset + V_S2
    k_even = tl.load(k_ptr + main_even_offset, mask=valid_slot & is_k, other=0.0)
    k_odd = tl.load(k_ptr + main_odd_offset, mask=valid_slot & is_k, other=0.0)
    v_even = tl.load(v_ptr + v_even_offset, mask=valid_slot & is_v, other=0.0)
    v_odd = tl.load(v_ptr + v_odd_offset, mask=valid_slot & is_v, other=0.0)
    idx_even = tl.load(
        idx_ptr
        + row * IDX_S0
        + idx_group * _TL_GROUP_SIZE * IDX_S2
        + 2 * pair * IDX_S2,
        mask=valid_slot & is_idx,
        other=0.0,
    )
    idx_odd = tl.load(
        idx_ptr
        + row * IDX_S0
        + idx_group * _TL_GROUP_SIZE * IDX_S2
        + (2 * pair + 1) * IDX_S2,
        mask=valid_slot & is_idx,
        other=0.0,
    )
    even_values = tl.where(is_k, k_even, tl.where(is_v, v_even, idx_even)).to(
        tl.float32
    )
    odd_values = tl.where(is_k, k_odd, tl.where(is_v, v_odd, idx_odd)).to(tl.float32)
    amax = tl.max(tl.maximum(tl.abs(even_values), tl.abs(odd_values)), axis=0)
    raw_scale = tl.minimum(
        tl.maximum(amax / _TL_E2M1_MAX, _TL_SCALE_MIN),
        _TL_SCALE_MAX,
    )
    stored_scale = raw_scale.to(tl.float8e4nv)
    scale = stored_scale.to(tl.float32)
    packed_codes = _e2m1_encode(even_values, scale) | (
        _e2m1_encode(odd_values, scale) << 4
    )

    main_packed_row_bytes = HEAD_DIM // _TL_VALUES_PER_BYTE
    main_packed_offset = (
        block * MAIN_PACKED_S0
        + (main_head * PAGE_SIZE + page_offset) * main_packed_row_bytes
        + main_group * (_TL_GROUP_SIZE // _TL_VALUES_PER_BYTE)
        + pair
    )
    if MMA_SCALE_LAYOUT:
        main_scale_offset = (
            block * MAIN_SCALE_S0
            + main_head * PAGE_SIZE * MAIN_GROUPS
            + _scale_128x4_offset(page_offset, main_group)
        )
    else:
        main_scale_offset = (
            block * MAIN_SCALE_S0
            + (main_head * PAGE_SIZE + page_offset) * MAIN_GROUPS
            + main_group
        )
    idx_packed_row_bytes = IDX_DIM // _TL_VALUES_PER_BYTE
    idx_packed_offset = (
        block * IDX_PACKED_S0
        + page_offset * idx_packed_row_bytes
        + idx_group * (_TL_GROUP_SIZE // _TL_VALUES_PER_BYTE)
        + pair
    )
    if MMA_SCALE_LAYOUT:
        idx_scale_offset = block * IDX_SCALE_S0 + _scale_128x4_offset(
            page_offset, idx_group
        )
    else:
        idx_scale_offset = block * IDX_SCALE_S0 + page_offset * IDX_GROUPS + idx_group
    tl.store(
        k_packed_ptr + main_packed_offset,
        packed_codes,
        mask=valid_slot & is_k,
    )
    tl.store(
        k_scales_ptr + main_scale_offset,
        stored_scale,
        mask=valid_slot & is_k,
    )
    tl.store(
        v_packed_ptr + main_packed_offset,
        packed_codes,
        mask=valid_slot & is_v,
    )
    tl.store(
        v_scales_ptr + main_scale_offset,
        stored_scale,
        mask=valid_slot & is_v,
    )
    tl.store(
        idx_packed_ptr + idx_packed_offset,
        packed_codes,
        mask=valid_slot & is_idx,
    )
    tl.store(
        idx_scales_ptr + idx_scale_offset,
        stored_scale,
        mask=valid_slot & is_idx,
    )


# Counts/capacities only bound accesses; reuse the binary across Prefill shapes.
@triton.jit(do_not_specialize=["N", "NUM_BLOCKS", "PERSIST_NUM_BLOCKS"])
def _quantize_main_index_rows_d128_kernel(
    k_ptr,
    v_ptr,
    idx_ptr,
    slots_ptr,
    unpad_ptr,
    owned_rows_ptr,
    k_packed_ptr,
    k_scales_ptr,
    v_packed_ptr,
    v_scales_ptr,
    idx_packed_ptr,
    idx_scales_ptr,
    N,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    K_S2: tl.constexpr,
    V_S0: tl.constexpr,
    V_S1: tl.constexpr,
    V_S2: tl.constexpr,
    IDX_S0: tl.constexpr,
    IDX_S2: tl.constexpr,
    MAIN_PACKED_S0: tl.constexpr,
    MAIN_SCALE_S0: tl.constexpr,
    IDX_PACKED_S0: tl.constexpr,
    IDX_SCALE_S0: tl.constexpr,
    NUM_BLOCKS,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MMA_SCALE_LAYOUT: tl.constexpr,
    MAP_SOURCE_ROWS: tl.constexpr = False,
    MAP_OWNED_ROWS: tl.constexpr = False,
    persist_slots_ptr=None,
    persist_k_packed_ptr=None,
    persist_k_scales_ptr=None,
    persist_v_packed_ptr=None,
    persist_v_scales_ptr=None,
    persist_idx_packed_ptr=None,
    persist_idx_scales_ptr=None,
    PERSIST_MAIN_PACKED_S0: tl.constexpr = 0,
    PERSIST_MAIN_SCALE_S0: tl.constexpr = 0,
    PERSIST_IDX_PACKED_S0: tl.constexpr = 0,
    PERSIST_IDX_SCALE_S0: tl.constexpr = 0,
    PERSIST_NUM_BLOCKS=0,
    WRITE_PERSISTENT: tl.constexpr = False,
    FI_WORKING_LAYOUT: tl.constexpr = False,
):
    """One row/plane CTA, eight independent groups by eight nibble pairs.

    Reduction axis 1 preserves each group of sixteen. Scale stores are 1D;
    packed byte stores are 2D. Row and slot offsets promote to int64.
    """
    row = tl.program_id(0).to(tl.int64)
    plane = tl.program_id(1)
    is_k = plane < NUM_HEADS
    is_v = (plane >= NUM_HEADS) & (plane < 2 * NUM_HEADS)
    is_idx = plane == 2 * NUM_HEADS
    head = tl.where(is_k, plane, tl.where(is_v, plane - NUM_HEADS, 0))

    valid_row = row < N
    source_row = row
    if MAP_SOURCE_ROWS:
        logical_row = row
        if MAP_OWNED_ROWS:
            logical_row = tl.load(owned_rows_ptr + row, mask=valid_row, other=0).to(
                tl.int64
            )
        source_row = tl.load(unpad_ptr + logical_row, mask=valid_row, other=0).to(
            tl.int64
        )
    slot = tl.load(slots_ptr + row, mask=valid_row, other=-1).to(tl.int64)
    valid_slot = valid_row & (slot >= 0) & (slot < NUM_BLOCKS * PAGE_SIZE)
    input_valid = valid_slot
    if WRITE_PERSISTENT:
        persist_slot = tl.load(persist_slots_ptr + row, mask=valid_row, other=-1).to(
            tl.int64
        )
        persist_valid = (
            valid_row
            & (persist_slot >= 0)
            & (persist_slot < PERSIST_NUM_BLOCKS * PAGE_SIZE)
        )
        input_valid = valid_slot | persist_valid
    block = slot // PAGE_SIZE
    page_offset = slot - block * PAGE_SIZE
    pair = tl.arange(0, 8)[None, :]
    groups = tl.arange(0, 8)
    group = groups[:, None]
    element = group * 16 + 2 * pair
    main_even_offset = source_row * K_S0 + head * K_S1 + element * K_S2
    main_odd_offset = main_even_offset + K_S2
    v_even_offset = source_row * V_S0 + head * V_S1 + element * V_S2
    v_odd_offset = v_even_offset + V_S2
    idx_even_offset = source_row * IDX_S0 + element * IDX_S2
    idx_odd_offset = idx_even_offset + IDX_S2

    k_even = tl.load(k_ptr + main_even_offset, mask=input_valid & is_k, other=0.0)
    k_odd = tl.load(k_ptr + main_odd_offset, mask=input_valid & is_k, other=0.0)
    v_even = tl.load(v_ptr + v_even_offset, mask=input_valid & is_v, other=0.0)
    v_odd = tl.load(v_ptr + v_odd_offset, mask=input_valid & is_v, other=0.0)
    idx_even = tl.load(idx_ptr + idx_even_offset, mask=input_valid & is_idx, other=0.0)
    idx_odd = tl.load(idx_ptr + idx_odd_offset, mask=input_valid & is_idx, other=0.0)
    even_values = tl.where(is_k, k_even, tl.where(is_v, v_even, idx_even)).to(
        tl.float32
    )
    odd_values = tl.where(is_k, k_odd, tl.where(is_v, v_odd, idx_odd)).to(tl.float32)
    amax = tl.max(tl.maximum(tl.abs(even_values), tl.abs(odd_values)), axis=1)
    raw_scale = tl.minimum(
        tl.maximum(amax / _TL_E2M1_MAX, _TL_SCALE_MIN), _TL_SCALE_MAX
    )
    stored_scale = raw_scale.to(tl.float8e4nv)
    scale = stored_scale.to(tl.float32)[:, None]
    packed_codes = _e2m1_encode(even_values, scale) | (
        _e2m1_encode(odd_values, scale) << 4
    )

    main_packed_offset = (
        block * MAIN_PACKED_S0
        + (head * PAGE_SIZE + page_offset) * 64
        + group * 8
        + pair
    )
    if MMA_SCALE_LAYOUT:
        main_scale_offset = (
            block * MAIN_SCALE_S0
            + head * PAGE_SIZE * 8
            + _scale_128x4_offset(page_offset, groups)
        )
    else:
        main_scale_offset = (
            block * MAIN_SCALE_S0 + (head * PAGE_SIZE + page_offset) * 8 + groups
        )
    if FI_WORKING_LAYOUT:
        # K scales are token-major; V scales use FI's 4-token permutation.
        # Only working stores change. Quantized codes and persistent MMA stores
        # are shared with the original writer.
        main_scale_offset = block * MAIN_SCALE_S0 + head * 1024 + page_offset * 8 + groups
        v_scale_offset = (block * MAIN_SCALE_S0 + head * 1024
                          + ((page_offset // 4) * 4 + groups // 2) * 8
                          + (groups % 2) * 4 + page_offset % 4)
    else:
        v_scale_offset = main_scale_offset
    idx_packed_offset = block * IDX_PACKED_S0 + page_offset * 64 + group * 8 + pair
    if MMA_SCALE_LAYOUT:
        idx_scale_offset = block * IDX_SCALE_S0 + _scale_128x4_offset(
            page_offset, groups
        )
    else:
        idx_scale_offset = block * IDX_SCALE_S0 + page_offset * 8 + groups

    tl.store(
        k_packed_ptr + main_packed_offset,
        packed_codes,
        mask=valid_slot & is_k,
    )
    tl.store(
        k_scales_ptr + main_scale_offset,
        stored_scale,
        mask=valid_slot & is_k,
    )
    tl.store(
        v_packed_ptr + main_packed_offset,
        packed_codes,
        mask=valid_slot & is_v,
    )
    tl.store(
        v_scales_ptr + v_scale_offset,
        stored_scale,
        mask=valid_slot & is_v,
    )
    tl.store(
        idx_packed_ptr + idx_packed_offset,
        packed_codes,
        mask=valid_slot & is_idx,
    )
    tl.store(
        idx_scales_ptr + idx_scale_offset,
        stored_scale,
        mask=valid_slot & is_idx,
    )

    if WRITE_PERSISTENT:
        persist_block = persist_slot // PAGE_SIZE
        persist_page_offset = persist_slot - persist_block * PAGE_SIZE
        persist_main_packed_offset = (
            persist_block * PERSIST_MAIN_PACKED_S0
            + (head * PAGE_SIZE + persist_page_offset) * 64
            + group * 8
            + pair
        )
        persist_main_scale_offset = (
            persist_block * PERSIST_MAIN_SCALE_S0
            + head * PAGE_SIZE * 8
            + _scale_128x4_offset(persist_page_offset, groups)
        )
        persist_idx_packed_offset = (
            persist_block * PERSIST_IDX_PACKED_S0
            + persist_page_offset * 64
            + group * 8
            + pair
        )
        persist_idx_scale_offset = (
            persist_block * PERSIST_IDX_SCALE_S0
            + _scale_128x4_offset(persist_page_offset, groups)
        )
        tl.store(
            persist_k_packed_ptr + persist_main_packed_offset,
            packed_codes,
            mask=persist_valid & is_k,
        )
        tl.store(
            persist_k_scales_ptr + persist_main_scale_offset,
            stored_scale,
            mask=persist_valid & is_k,
        )
        tl.store(
            persist_v_packed_ptr + persist_main_packed_offset,
            packed_codes,
            mask=persist_valid & is_v,
        )
        tl.store(
            persist_v_scales_ptr + persist_main_scale_offset,
            stored_scale,
            mask=persist_valid & is_v,
        )
        tl.store(
            persist_idx_packed_ptr + persist_idx_packed_offset,
            packed_codes,
            mask=persist_valid & is_idx,
        )
        tl.store(
            persist_idx_scales_ptr + persist_idx_scale_offset,
            stored_scale,
            mask=persist_valid & is_idx,
        )


# Counts/capacities only bound accesses; reuse the binary across Prefill shapes.
@triton.jit(do_not_specialize=["N", "NUM_BLOCKS", "PERSIST_NUM_BLOCKS"])
def _quantize_main_index_rows_d128_multirow_kernel(
    k_ptr,
    v_ptr,
    idx_ptr,
    slots_ptr,
    unpad_ptr,
    owned_rows_ptr,
    k_packed_ptr,
    k_scales_ptr,
    v_packed_ptr,
    v_scales_ptr,
    idx_packed_ptr,
    idx_scales_ptr,
    N,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    K_S2: tl.constexpr,
    V_S0: tl.constexpr,
    V_S1: tl.constexpr,
    V_S2: tl.constexpr,
    IDX_S0: tl.constexpr,
    IDX_S2: tl.constexpr,
    MAIN_PACKED_S0: tl.constexpr,
    MAIN_SCALE_S0: tl.constexpr,
    IDX_PACKED_S0: tl.constexpr,
    IDX_SCALE_S0: tl.constexpr,
    NUM_BLOCKS,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MMA_SCALE_LAYOUT: tl.constexpr,
    MAP_SOURCE_ROWS: tl.constexpr = False,
    MAP_OWNED_ROWS: tl.constexpr = False,
    persist_slots_ptr=None,
    persist_k_packed_ptr=None,
    persist_k_scales_ptr=None,
    persist_v_packed_ptr=None,
    persist_v_scales_ptr=None,
    persist_idx_packed_ptr=None,
    persist_idx_scales_ptr=None,
    PERSIST_MAIN_PACKED_S0: tl.constexpr = 0,
    PERSIST_MAIN_SCALE_S0: tl.constexpr = 0,
    PERSIST_IDX_PACKED_S0: tl.constexpr = 0,
    PERSIST_IDX_SCALE_S0: tl.constexpr = 0,
    PERSIST_NUM_BLOCKS=0,
    WRITE_PERSISTENT: tl.constexpr = False,
    FI_WORKING_LAYOUT: tl.constexpr = False,
    ROWS_PER_CTA: tl.constexpr = 4,
):
    """Experimental row-group CTA with the original eight-pair reduction.

    Each of ROWS_PER_CTA rows retains eight independent groups of sixteen.
    Reduction axis 1 and FP8/E2M1 rounding match the one-row writer exactly.
    """
    row_group = tl.arange(0, ROWS_PER_CTA * 8)
    row = tl.program_id(0).to(tl.int64) * ROWS_PER_CTA + row_group // 8
    plane = tl.program_id(1)
    is_k = plane < NUM_HEADS
    is_v = (plane >= NUM_HEADS) & (plane < 2 * NUM_HEADS)
    is_idx = plane == 2 * NUM_HEADS
    head = tl.where(is_k, plane, tl.where(is_v, plane - NUM_HEADS, 0))

    valid_row = row < N
    source_row = row
    if MAP_SOURCE_ROWS:
        logical_row = row
        if MAP_OWNED_ROWS:
            logical_row = tl.load(owned_rows_ptr + row, mask=valid_row, other=0).to(
                tl.int64
            )
        source_row = tl.load(unpad_ptr + logical_row, mask=valid_row, other=0).to(
            tl.int64
        )
    slot = tl.load(slots_ptr + row, mask=valid_row, other=-1).to(tl.int64)
    valid_slot = valid_row & (slot >= 0) & (slot < NUM_BLOCKS * PAGE_SIZE)
    input_valid = valid_slot
    if WRITE_PERSISTENT:
        persist_slot = tl.load(persist_slots_ptr + row, mask=valid_row, other=-1).to(
            tl.int64
        )
        persist_valid = (
            valid_row
            & (persist_slot >= 0)
            & (persist_slot < PERSIST_NUM_BLOCKS * PAGE_SIZE)
        )
        input_valid = valid_slot | persist_valid
    block = slot // PAGE_SIZE
    page_offset = slot - block * PAGE_SIZE
    pair = tl.arange(0, 8)[None, :]
    groups = row_group % 8
    group = groups[:, None]
    element = group * 16 + 2 * pair
    main_even_offset = source_row[:, None] * K_S0 + head * K_S1 + element * K_S2
    main_odd_offset = main_even_offset + K_S2
    v_even_offset = source_row[:, None] * V_S0 + head * V_S1 + element * V_S2
    v_odd_offset = v_even_offset + V_S2
    idx_even_offset = source_row[:, None] * IDX_S0 + element * IDX_S2
    idx_odd_offset = idx_even_offset + IDX_S2

    k_even = tl.load(k_ptr + main_even_offset, mask=input_valid[:, None] & is_k, other=0.0)
    k_odd = tl.load(k_ptr + main_odd_offset, mask=input_valid[:, None] & is_k, other=0.0)
    v_even = tl.load(v_ptr + v_even_offset, mask=input_valid[:, None] & is_v, other=0.0)
    v_odd = tl.load(v_ptr + v_odd_offset, mask=input_valid[:, None] & is_v, other=0.0)
    idx_even = tl.load(idx_ptr + idx_even_offset, mask=input_valid[:, None] & is_idx, other=0.0)
    idx_odd = tl.load(idx_ptr + idx_odd_offset, mask=input_valid[:, None] & is_idx, other=0.0)
    even_values = tl.where(is_k, k_even, tl.where(is_v, v_even, idx_even)).to(
        tl.float32
    )
    odd_values = tl.where(is_k, k_odd, tl.where(is_v, v_odd, idx_odd)).to(tl.float32)
    amax = tl.max(tl.maximum(tl.abs(even_values), tl.abs(odd_values)), axis=1)
    raw_scale = tl.minimum(
        tl.maximum(amax / _TL_E2M1_MAX, _TL_SCALE_MIN), _TL_SCALE_MAX
    )
    stored_scale = raw_scale.to(tl.float8e4nv)
    scale = stored_scale.to(tl.float32)[:, None]
    packed_codes = _e2m1_encode(even_values, scale) | (
        _e2m1_encode(odd_values, scale) << 4
    )

    main_packed_offset = (
        block[:, None] * MAIN_PACKED_S0
        + (head * PAGE_SIZE + page_offset[:, None]) * 64
        + group * 8
        + pair
    )
    if MMA_SCALE_LAYOUT:
        main_scale_offset = (
            block * MAIN_SCALE_S0
            + head * PAGE_SIZE * 8
            + _scale_128x4_offset(page_offset, groups)
        )
    else:
        main_scale_offset = (
            block * MAIN_SCALE_S0 + (head * PAGE_SIZE + page_offset) * 8 + groups
        )
    if FI_WORKING_LAYOUT:
        # K scales are token-major; V scales use FI's 4-token permutation.
        # Only working stores change. Quantized codes and persistent MMA stores
        # are shared with the original writer.
        main_scale_offset = block * MAIN_SCALE_S0 + head * 1024 + page_offset * 8 + groups
        v_scale_offset = (block * MAIN_SCALE_S0 + head * 1024
                          + ((page_offset // 4) * 4 + groups // 2) * 8
                          + (groups % 2) * 4 + page_offset % 4)
    else:
        v_scale_offset = main_scale_offset
    idx_packed_offset = block[:, None] * IDX_PACKED_S0 + page_offset[:, None] * 64 + group * 8 + pair
    if MMA_SCALE_LAYOUT:
        idx_scale_offset = block * IDX_SCALE_S0 + _scale_128x4_offset(
            page_offset, groups
        )
    else:
        idx_scale_offset = block * IDX_SCALE_S0 + page_offset * 8 + groups

    tl.store(
        k_packed_ptr + main_packed_offset,
        packed_codes,
        mask=valid_slot[:, None] & is_k,
    )
    tl.store(
        k_scales_ptr + main_scale_offset,
        stored_scale,
        mask=valid_slot & is_k,
    )
    tl.store(
        v_packed_ptr + main_packed_offset,
        packed_codes,
        mask=valid_slot[:, None] & is_v,
    )
    tl.store(
        v_scales_ptr + v_scale_offset,
        stored_scale,
        mask=valid_slot & is_v,
    )
    tl.store(
        idx_packed_ptr + idx_packed_offset,
        packed_codes,
        mask=valid_slot[:, None] & is_idx,
    )
    tl.store(
        idx_scales_ptr + idx_scale_offset,
        stored_scale,
        mask=valid_slot & is_idx,
    )

    if WRITE_PERSISTENT:
        persist_block = persist_slot // PAGE_SIZE
        persist_page_offset = persist_slot - persist_block * PAGE_SIZE
        persist_main_packed_offset = (
            persist_block[:, None] * PERSIST_MAIN_PACKED_S0
            + (head * PAGE_SIZE + persist_page_offset[:, None]) * 64
            + group * 8
            + pair
        )
        persist_main_scale_offset = (
            persist_block * PERSIST_MAIN_SCALE_S0
            + head * PAGE_SIZE * 8
            + _scale_128x4_offset(persist_page_offset, groups)
        )
        persist_idx_packed_offset = (
            persist_block[:, None] * PERSIST_IDX_PACKED_S0
            + persist_page_offset[:, None] * 64
            + group * 8
            + pair
        )
        persist_idx_scale_offset = (
            persist_block * PERSIST_IDX_SCALE_S0
            + _scale_128x4_offset(persist_page_offset, groups)
        )
        tl.store(
            persist_k_packed_ptr + persist_main_packed_offset,
            packed_codes,
            mask=persist_valid[:, None] & is_k,
        )
        tl.store(
            persist_k_scales_ptr + persist_main_scale_offset,
            stored_scale,
            mask=persist_valid & is_k,
        )
        tl.store(
            persist_v_packed_ptr + persist_main_packed_offset,
            packed_codes,
            mask=persist_valid[:, None] & is_v,
        )
        tl.store(
            persist_v_scales_ptr + persist_main_scale_offset,
            stored_scale,
            mask=persist_valid & is_v,
        )
        tl.store(
            persist_idx_packed_ptr + persist_idx_packed_offset,
            packed_codes,
            mask=persist_valid[:, None] & is_idx,
        )
        tl.store(
            persist_idx_scales_ptr + persist_idx_scale_offset,
            stored_scale,
            mask=persist_valid & is_idx,
        )


@triton.jit
def _quantize_query_rows_mma_kernel(
    src_ptr,
    packed_ptr,
    scales_ptr,
    N,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    SRC_S2: tl.constexpr,
    PACKED_S0: tl.constexpr,
    PACKED_S1: tl.constexpr,
    SCALE_HEAD_STRIDE: tl.constexpr,
    SCALE_TILE_STRIDE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """Quantize decode Q and directly emit SM100 MMA scale storage."""
    row = tl.program_id(0)
    head_group = tl.program_id(1)
    head = head_group // GROUPS
    group = head_group - head * GROUPS
    pair = tl.arange(0, _TL_GROUP_SIZE // _TL_VALUES_PER_BYTE)
    valid = (row < N) & (head < NUM_HEADS)
    src_base = row * SRC_S0 + head * SRC_S1 + group * _TL_GROUP_SIZE * SRC_S2
    even = tl.load(
        src_ptr + src_base + 2 * pair * SRC_S2,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    odd = tl.load(
        src_ptr + src_base + (2 * pair + 1) * SRC_S2,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.maximum(tl.abs(even), tl.abs(odd)), axis=0)
    raw_scale = tl.minimum(
        tl.maximum(amax / _TL_E2M1_MAX, _TL_SCALE_MIN),
        _TL_SCALE_MAX,
    )
    stored_scale = raw_scale.to(tl.float8e4nv)
    scale = stored_scale.to(tl.float32)
    packed_codes = _e2m1_encode(even, scale) | (_e2m1_encode(odd, scale) << 4)
    packed_offset = (
        row * PACKED_S0
        + head * PACKED_S1
        + group * (_TL_GROUP_SIZE // _TL_VALUES_PER_BYTE)
        + pair
    )
    scale_offset = (
        head * SCALE_HEAD_STRIDE
        + (row // 128) * SCALE_TILE_STRIDE
        + _scale_128x4_offset(row % 128, group)
    )
    tl.store(packed_ptr + packed_offset, packed_codes, mask=valid)
    tl.store(scales_ptr + scale_offset, stored_scale, mask=valid)


@triton.jit
def _gather_rows_kernel(
    packed_ptr,
    scales_ptr,
    physical_slots_ptr,
    destination_slots_ptr,
    out_ptr,
    N,
    PACKED_S0: tl.constexpr,
    SCALE_S0: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUPS: tl.constexpr,
    OUT_HND: tl.constexpr,
    E4M3_COMPUTE_GRID: tl.constexpr,
):
    row = tl.program_id(0)
    hg = tl.program_id(1)
    head = hg // GROUPS
    group = hg - head * GROUPS
    lane = tl.arange(0, _TL_GROUP_SIZE)
    valid_row = row < N
    source_slot = tl.load(physical_slots_ptr + row, mask=valid_row, other=-1).to(
        tl.int64
    )
    destination_slot = tl.load(
        destination_slots_ptr + row, mask=valid_row, other=-1
    ).to(tl.int64)
    valid = (
        valid_row
        & (source_slot >= 0)
        & (source_slot < NUM_BLOCKS * PAGE_SIZE)
        & (destination_slot >= 0)
    )
    block = source_slot // PAGE_SIZE
    page_offset = source_slot - block * PAGE_SIZE
    packed_row_bytes = HEAD_DIM // _TL_VALUES_PER_BYTE
    byte_offset = lane // _TL_VALUES_PER_BYTE
    packed_value = tl.load(
        packed_ptr
        + block * PACKED_S0
        + (head * PAGE_SIZE + page_offset) * packed_row_bytes
        + group * (_TL_GROUP_SIZE // _TL_VALUES_PER_BYTE)
        + byte_offset,
        mask=valid,
        other=0,
    ).to(tl.uint8)
    code = tl.where((lane & 1) == 0, packed_value & 15, (packed_value >> 4) & 15)
    scale = tl.load(
        scales_ptr
        + block * SCALE_S0
        + (head * PAGE_SIZE + page_offset) * GROUPS
        + group,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    values = _e2m1_decode(code) * scale
    if E4M3_COMPUTE_GRID:
        values = _to_e4m3_compute_grid(values)
    dim = group * _TL_GROUP_SIZE + lane
    if OUT_HND:
        dst_page = destination_slot // PAGE_SIZE
        dst_offset = destination_slot - dst_page * PAGE_SIZE
        out_offset = (
            dst_page * NUM_HEADS * PAGE_SIZE * HEAD_DIM
            + head * PAGE_SIZE * HEAD_DIM
            + dst_offset * HEAD_DIM
            + dim
        )
    else:
        out_offset = destination_slot * NUM_HEADS * HEAD_DIM + head * HEAD_DIM + dim
    tl.store(out_ptr + out_offset, values, mask=valid)


@triton.jit
def _materialize_q8kv4_rows_kernel(
    src_ptr,
    destination_slots_ptr,
    out_ptr,
    N,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    SRC_S2: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUPS: tl.constexpr,
    OUT_HND: tl.constexpr,
):
    """FP4 QDQ + E4M3-grid materialization without a persistent-cache read.

    CP prefill has the complete suffix only in its all-gathered activation
    tensor; each rank's persistent pool intentionally contains just its owned
    pages.  This kernel makes the transient suffix numerically identical to a
    native Q8KV4 cache read without requiring non-owned persistent pages.
    """
    row = tl.program_id(0)
    hg = tl.program_id(1)
    head = hg // GROUPS
    group = hg - head * GROUPS
    lane = tl.arange(0, _TL_GROUP_SIZE)
    valid_row = row < N
    destination_slot = tl.load(
        destination_slots_ptr + row, mask=valid_row, other=-1
    ).to(tl.int64)
    valid = valid_row & (destination_slot >= 0)
    dim = group * _TL_GROUP_SIZE + lane
    values = tl.load(
        src_ptr + row * SRC_S0 + head * SRC_S1 + dim * SRC_S2,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    raw_scale = tl.minimum(
        tl.maximum(amax / _TL_E2M1_MAX, _TL_SCALE_MIN),
        _TL_SCALE_MAX,
    )
    stored_scale = raw_scale.to(tl.float8e4nv)
    scale = stored_scale.to(tl.float32)
    codes = _e2m1_encode(values, scale)
    q8kv4_values = _to_e4m3_compute_grid(_e2m1_decode(codes) * scale)
    if OUT_HND:
        dst_page = destination_slot // PAGE_SIZE
        dst_offset = destination_slot - dst_page * PAGE_SIZE
        out_offset = (
            dst_page * NUM_HEADS * PAGE_SIZE * HEAD_DIM
            + head * PAGE_SIZE * HEAD_DIM
            + dst_offset * HEAD_DIM
            + dim
        )
    else:
        out_offset = destination_slot * NUM_HEADS * HEAD_DIM + head * HEAD_DIM + dim
    tl.store(out_ptr + out_offset, q8kv4_values, mask=valid)


@triton.jit
def _round_to_e4m3_compute_grid_kernel(values_ptr, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    values = tl.load(values_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    tl.store(
        values_ptr + offsets,
        _to_e4m3_compute_grid(values),
        mask=valid,
    )


@triton.jit
def _convert_pages_kernel(
    block_table_ptr,
    packed_ptr,
    scales_ptr,
    working_ptr,
    TABLE_ENTRIES,
    PACKED_S0: tl.constexpr,
    SCALE_S0: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUPS: tl.constexpr,
    TOTAL_GROUPS: tl.constexpr,
    GROUP_BATCH: tl.constexpr,
    QUANTIZE: tl.constexpr,
):
    table_entry = tl.program_id(0)
    group_chunk = tl.program_id(1)
    linear_group = group_chunk * GROUP_BATCH + tl.arange(0, GROUP_BATCH)
    valid_group = linear_group < TOTAL_GROUPS
    group = linear_group % GROUPS
    row = linear_group // GROUPS
    page_offset = row % PAGE_SIZE
    head_kv = row // PAGE_SIZE
    head = head_kv % NUM_HEADS
    kv_index = head_kv // NUM_HEADS
    valid_entry = table_entry < TABLE_ENTRIES
    block = tl.load(block_table_ptr + table_entry, mask=valid_entry, other=-1).to(
        tl.int64
    )
    # Cache block zero is the framework's zero/sentinel page and must remain
    # immutable.  Real allocations begin at block one.
    valid_page = valid_entry & (block > 0) & (block < NUM_BLOCKS)
    plane_packed_bytes = NUM_HEADS * PAGE_SIZE * HEAD_DIM // 2
    plane_scale_bytes = NUM_HEADS * PAGE_SIZE * GROUPS
    packed_row_bytes = HEAD_DIM // 2
    packed_base = (
        block * PACKED_S0
        + kv_index * plane_packed_bytes
        + (head * PAGE_SIZE + page_offset) * packed_row_bytes
        + group * (_TL_GROUP_SIZE // _TL_VALUES_PER_BYTE)
    )
    scale_offset = (
        block * SCALE_S0
        + kv_index * plane_scale_bytes
        + (head * PAGE_SIZE + page_offset) * GROUPS
        + group
    )
    working_group_base = (
        block * 2 * NUM_HEADS * PAGE_SIZE * HEAD_DIM
        + kv_index * NUM_HEADS * PAGE_SIZE * HEAD_DIM
        + head * PAGE_SIZE * HEAD_DIM
        + page_offset * HEAD_DIM
        + group * _TL_GROUP_SIZE
    )

    if QUANTIZE:
        pair = tl.arange(0, _TL_GROUP_SIZE // _TL_VALUES_PER_BYTE)
        even_values = tl.load(
            working_ptr + working_group_base[:, None] + 2 * pair[None, :],
            mask=valid_page & valid_group[:, None],
            other=0.0,
        ).to(tl.float32)
        odd_values = tl.load(
            working_ptr + working_group_base[:, None] + 2 * pair[None, :] + 1,
            mask=valid_page & valid_group[:, None],
            other=0.0,
        ).to(tl.float32)
        amax = tl.max(tl.maximum(tl.abs(even_values), tl.abs(odd_values)), axis=1)
        raw_scale = tl.minimum(
            tl.maximum(amax / _TL_E2M1_MAX, _TL_SCALE_MIN),
            _TL_SCALE_MAX,
        )
        stored_scale = raw_scale.to(tl.float8e4nv)
        scale = stored_scale.to(tl.float32)[:, None]
        low_codes = _e2m1_encode(even_values, scale)
        high_codes = _e2m1_encode(odd_values, scale)
        packed_codes = low_codes | (high_codes << 4)
        tl.store(
            packed_ptr + packed_base[:, None] + pair[None, :],
            packed_codes,
            mask=valid_page & valid_group[:, None],
        )
        tl.store(
            scales_ptr + scale_offset,
            stored_scale,
            mask=valid_page & valid_group,
        )
    else:
        lane = tl.arange(0, _TL_GROUP_SIZE)
        byte_offset = lane // 2
        packed_value = tl.load(
            packed_ptr + packed_base[:, None] + byte_offset[None, :],
            mask=valid_page & valid_group[:, None],
            other=0,
        ).to(tl.uint8)
        code = tl.where(
            (lane[None, :] & 1) == 0,
            packed_value & 15,
            (packed_value >> 4) & 15,
        )
        scale = tl.load(
            scales_ptr + scale_offset,
            mask=valid_page & valid_group,
            other=0.0,
        ).to(tl.float32)[:, None]
        tl.store(
            working_ptr + working_group_base[:, None] + lane[None, :],
            _e2m1_decode(code) * scale,
            mask=valid_page & valid_group[:, None],
        )


@triton.jit
def _scatter_hnd_rows_kernel(
    src_ptr,
    slots_ptr,
    dst_ptr,
    N,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    SRC_S2: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, BLOCK_D)
    valid_row = row < N
    slot = tl.load(slots_ptr + row, mask=valid_row, other=-1).to(tl.int64)
    valid = valid_row & (slot >= 0)
    page = slot // PAGE_SIZE
    page_offset = slot - page * PAGE_SIZE
    values = tl.load(
        src_ptr + row * SRC_S0 + head * SRC_S1 + dim * SRC_S2,
        mask=valid & (dim < HEAD_DIM),
        other=0.0,
    )
    dst_offset = (
        page * NUM_HEADS * PAGE_SIZE * HEAD_DIM
        + head * PAGE_SIZE * HEAD_DIM
        + page_offset * HEAD_DIM
        + dim
    )
    tl.store(dst_ptr + dst_offset, values, mask=valid & (dim < HEAD_DIM))


@triton.jit
def _clear_hnd_tails_kernel(
    k_ptr,
    v_ptr,
    idx_ptr,
    kv_lens_ptr,
    BATCH_SIZE,
    SCRATCH_SEQ_LEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IDX_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    item = tl.program_id(0)
    head = tl.program_id(1)
    batch = item // PAGE_SIZE
    tail_offset = item - batch * PAGE_SIZE
    valid_batch = batch < BATCH_SIZE
    kv_len = tl.load(kv_lens_ptr + batch, mask=valid_batch, other=0).to(tl.int64)
    tail = (PAGE_SIZE - kv_len % PAGE_SIZE) % PAGE_SIZE
    slot = batch * SCRATCH_SEQ_LEN + kv_len + tail_offset
    valid = valid_batch & (tail_offset < tail) & (slot < (batch + 1) * SCRATCH_SEQ_LEN)
    page = slot // PAGE_SIZE
    page_offset = slot - page * PAGE_SIZE
    dim = tl.arange(0, BLOCK_D)
    main_offset = (
        page * NUM_HEADS * PAGE_SIZE * HEAD_DIM
        + head * PAGE_SIZE * HEAD_DIM
        + page_offset * HEAD_DIM
        + dim
    )
    tl.store(k_ptr + main_offset, 0.0, mask=valid & (dim < HEAD_DIM))
    tl.store(v_ptr + main_offset, 0.0, mask=valid & (dim < HEAD_DIM))
    tl.store(
        idx_ptr + slot * IDX_DIM + dim,
        0.0,
        mask=valid & (head == 0) & (dim < IDX_DIM),
    )


def _quantize_rows(
    src: torch.Tensor,
    physical_slots: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    page_size: int,
) -> None:
    if src.dim() != 3 or not src.is_contiguous():
        raise ValueError(
            f"NVFP4 row source must be contiguous [token,head,dim], got "
            f"shape={tuple(src.shape)} strides={tuple(src.stride())}"
        )
    rows, heads, dim = map(int, src.shape)
    _validate_grouped_dim(dim, "NVFP4 row")
    if int(physical_slots.numel()) != rows:
        raise ValueError(
            f"NVFP4 row/slot count mismatch: {rows} vs {physical_slots.numel()}"
        )
    groups = dim // NVFP4_GROUP_SIZE
    expected_packed = heads * page_size * dim // 2
    expected_scales = heads * page_size * groups
    if packed.dim() != 2 or int(packed.shape[1]) < expected_packed:
        raise ValueError(
            f"NVFP4 packed plane needs at least {expected_packed} bytes/block, "
            f"got {tuple(packed.shape)}"
        )
    if scales.dim() != 2 or int(scales.shape[1]) < expected_scales:
        raise ValueError(
            f"NVFP4 scale plane needs at least {expected_scales} bytes/block, "
            f"got {tuple(scales.shape)}"
        )
    if rows == 0:
        return
    _quantize_rows_kernel[(rows, heads * groups)](
        src,
        physical_slots,
        packed,
        scales,
        rows,
        SRC_S0=int(src.stride(0)),
        SRC_S1=int(src.stride(1)),
        SRC_S2=int(src.stride(2)),
        PACKED_S0=int(packed.stride(0)),
        SCALE_S0=int(scales.stride(0)),
        NUM_BLOCKS=int(packed.shape[0]),
        NUM_HEADS=heads,
        PAGE_SIZE=page_size,
        HEAD_DIM=dim,
        GROUPS=groups,
        num_warps=1,
    )


def _gather_rows(
    packed: torch.Tensor,
    scales: torch.Tensor,
    physical_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    out: torch.Tensor,
    page_size: int,
    *,
    out_hnd: bool = False,
    e4m3_compute_grid: bool = False,
) -> None:
    if int(physical_slots.numel()) != int(destination_slots.numel()):
        raise ValueError("NVFP4 gather source/destination slot counts differ")
    if out.dim() not in (3, 4):
        raise ValueError(f"NVFP4 gather output must be 3-D or 4-D, got {out.dim()}D")
    if out_hnd:
        _, heads, out_page, dim = map(int, out.shape)
        if out_page != page_size:
            raise ValueError(
                f"NVFP4 HND output page mismatch: {out_page} vs {page_size}"
            )
    else:
        _, heads, dim = map(int, out.shape)
    _validate_grouped_dim(dim, "NVFP4 gather")
    groups = dim // NVFP4_GROUP_SIZE
    rows = int(physical_slots.numel())
    if rows == 0:
        return
    _gather_rows_kernel[(rows, heads * groups)](
        packed,
        scales,
        physical_slots,
        destination_slots,
        out,
        rows,
        PACKED_S0=int(packed.stride(0)),
        SCALE_S0=int(scales.stride(0)),
        NUM_BLOCKS=int(packed.shape[0]),
        NUM_HEADS=heads,
        PAGE_SIZE=page_size,
        HEAD_DIM=dim,
        GROUPS=groups,
        OUT_HND=out_hnd,
        E4M3_COMPUTE_GRID=e4m3_compute_grid,
        num_warps=1,
    )


def materialize_q8kv4_rows(
    src: torch.Tensor,
    destination_slots: torch.Tensor,
    out: torch.Tensor,
    page_size: int,
    *,
    out_hnd: bool = False,
) -> None:
    """Write Q8KV4-equivalent values into a BF16 working allocation.

    This intentionally does not touch persistent storage.  It is for the
    all-gathered CP suffix whose non-owned rows cannot be read from the local
    sharded cache.
    """
    if src.dim() == 2:
        src = src[:, None, :]
    if src.dim() != 3 or src.stride(2) != 1:
        raise ValueError(
            "Q8KV4 working source must have contiguous innermost dim, got "
            f"shape={tuple(src.shape)} strides={tuple(src.stride())}"
        )
    rows, heads, dim = map(int, src.shape)
    _validate_grouped_dim(dim, "Q8KV4 working row")
    if int(destination_slots.numel()) != rows:
        raise ValueError(
            f"Q8KV4 working row/slot count mismatch: {rows} vs "
            f"{destination_slots.numel()}"
        )
    if out_hnd:
        if out.dim() != 4:
            raise ValueError(f"Q8KV4 HND output must be 4-D, got {out.dim()}D")
        _, out_heads, out_page, out_dim = map(int, out.shape)
        if out_page != page_size:
            raise ValueError(f"Q8KV4 HND page mismatch: {out_page} vs {page_size}")
    else:
        if out.dim() != 3:
            raise ValueError(f"Q8KV4 flat output must be 3-D, got {out.dim()}D")
        _, out_heads, out_dim = map(int, out.shape)
    if out_heads != heads or out_dim != dim:
        raise ValueError(
            "Q8KV4 working source/output shape mismatch: "
            f"src heads/dim=({heads},{dim}) vs out=({out_heads},{out_dim})"
        )
    if rows == 0:
        return
    groups = dim // NVFP4_GROUP_SIZE
    _materialize_q8kv4_rows_kernel[(rows, heads * groups)](
        src,
        destination_slots,
        out,
        rows,
        SRC_S0=int(src.stride(0)),
        SRC_S1=int(src.stride(1)),
        SRC_S2=int(src.stride(2)),
        NUM_HEADS=heads,
        PAGE_SIZE=page_size,
        HEAD_DIM=dim,
        GROUPS=groups,
        OUT_HND=out_hnd,
        num_warps=1,
    )


def round_to_e4m3_compute_grid_(values: torch.Tensor) -> torch.Tensor:
    """In-place saturating E4M3 round-trip for BF16 Q/idx-Q carriers."""
    if not values.is_cuda or not values.is_contiguous():
        raise ValueError("E4M3 compute-grid input must be contiguous CUDA storage")
    if values.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            f"E4M3 compute-grid input must be BF16/FP16/FP32, got {values.dtype}"
        )
    count = int(values.numel())
    if count == 0:
        return values
    block = 256
    _round_to_e4m3_compute_grid_kernel[(triton.cdiv(count, block),)](
        values, count, BLOCK=block, num_warps=4
    )
    return values


def quantize_main_rows(
    k: torch.Tensor,
    v: torch.Tensor,
    physical_slots: torch.Tensor,
    layout: NVFP4CacheLayout,
) -> None:
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError(f"NVFP4 K/V shape mismatch: {k.shape} vs {v.shape}")
    for kv_index, values in enumerate((k, v)):
        packed, scales = layout.main_plane(kv_index)
        _quantize_rows(
            values.contiguous(), physical_slots, packed, scales, layout.page_size
        )


def quantize_query_rows_mma(
    values: torch.Tensor,
    packed: torch.Tensor,
    scales_mma: torch.Tensor,
) -> None:
    """Quantize decode Q into packed E2M1 plus preordered SM100 scales.

    ``scales_mma`` is contiguous storage with one padded 128-row 128x4 tile
    per head.  For D=128 it can be shaped as ``[H,1,2,32,4,4]``, matching
    ``fp4_indexer_mma_scale_storage_shape(N, H, fp4_format="nvfp4")``.
    The output is head-major contiguous MMA storage.  Unlike the initial
    decode-only helper, this supports arbitrary prefill row counts by adding
    the missing 128-row tile term to the scale address.
    """
    if values.dim() != 3 or not values.is_contiguous():
        raise ValueError(
            "NVFP4 query source must be contiguous [token,head,dim], got "
            f"shape={tuple(values.shape)} strides={tuple(values.stride())}"
        )
    rows, heads, dim = map(int, values.shape)
    _validate_grouped_dim(dim, "NVFP4 query")
    groups = dim // NVFP4_GROUP_SIZE
    if rows <= 0 or groups % 4 != 0:
        raise ValueError(
            "NVFP4 MMA query quantization requires positive rows and a scale-"
            f"group count divisible by 4, got rows={rows}, groups={groups}"
        )
    expected_packed = (rows, heads, dim // NVFP4_VALUES_PER_BYTE)
    if (
        packed.dtype != torch.uint8
        or tuple(packed.shape) != expected_packed
        or not packed.is_contiguous()
    ):
        raise ValueError(
            "NVFP4 packed query output must be contiguous uint8 with shape "
            f"{expected_packed}, got shape={tuple(packed.shape)} dtype={packed.dtype}"
        )
    scale_tile_stride = 128 * groups
    scale_head_stride = ((rows + 127) // 128) * scale_tile_stride
    if (
        scales_mma.dtype != torch.float8_e4m3fn
        or not scales_mma.is_contiguous()
        or int(scales_mma.numel()) != heads * scale_head_stride
    ):
        raise ValueError(
            "NVFP4 query MMA scales must be contiguous E4M3 storage with "
            f"{heads * scale_head_stride} elements"
        )
    _quantize_query_rows_mma_kernel[(rows, heads * groups)](
        values,
        packed,
        scales_mma,
        rows,
        SRC_S0=int(values.stride(0)),
        SRC_S1=int(values.stride(1)),
        SRC_S2=int(values.stride(2)),
        PACKED_S0=int(packed.stride(0)),
        PACKED_S1=int(packed.stride(1)),
        SCALE_HEAD_STRIDE=scale_head_stride,
        SCALE_TILE_STRIDE=scale_tile_stride,
        NUM_HEADS=heads,
        HEAD_DIM=dim,
        GROUPS=groups,
        num_warps=1,
    )


def quantize_main_index_rows(
    k: torch.Tensor,
    v: torch.Tensor,
    index_k: torch.Tensor,
    physical_slots: torch.Tensor,
    layout: NVFP4CacheLayout,
    *,
    mma_scale_layout: bool = False,
) -> None:
    """Fused decode writer for Main K/V and the shared Indexer-K row.

    This is the production decode/target-verify cache writer and is also kept
    directly callable for focused correctness and performance tests.

    ``mma_scale_layout=True`` stores E4M3 scales directly in the common
    cuBLAS/cuDNN 128x4 layout used by the SM100 Triton IndexScore and CuTe
    sparse-attention kernels. It changes only scale placement, not allocation
    size or packed E2M1 values.
    """
    if tuple(k.shape) != tuple(v.shape) or k.dim() != 3:
        raise ValueError(
            f"NVFP4 fused K/V need matching [token,head,dim] tensors, got "
            f"{tuple(k.shape)} and {tuple(v.shape)}"
        )
    rows, heads, head_dim = map(int, k.shape)
    if (
        k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or index_k.dtype != torch.bfloat16
    ):
        raise ValueError("NVFP4 fused writer inputs must use BF16")
    if heads != layout.num_heads or head_dim != layout.head_dim:
        raise ValueError(
            "NVFP4 fused K/V shape does not match cache layout: "
            f"tensor heads/dim={heads}/{head_dim}, "
            f"layout={layout.num_heads}/{layout.head_dim}"
        )
    if int(physical_slots.numel()) != rows:
        raise ValueError(
            f"NVFP4 fused row/slot count mismatch: {rows} vs "
            f"{physical_slots.numel()}"
        )
    if physical_slots.dtype != torch.int64 or not physical_slots.is_contiguous():
        raise ValueError("NVFP4 fused writer slots must be contiguous int64")
    if index_k.dim() == 2:
        index_k = index_k[:, None, :]
    if index_k.dim() != 3 or int(index_k.shape[0]) != rows:
        raise ValueError(
            "NVFP4 fused Indexer-K must be [token,dim] or [token,1,dim], "
            f"got {tuple(index_k.shape)}"
        )
    if int(index_k.shape[1]) != 1:
        raise ValueError(
            "NVFP4 cache layout stores one shared Indexer-K row per token, "
            f"got {index_k.shape[1]} heads"
        )
    index_dim = int(index_k.shape[2])
    _validate_grouped_dim(index_dim, "NVFP4 fused indexer")
    if mma_scale_layout and (
        layout.page_size != 128
        or (head_dim // NVFP4_GROUP_SIZE) % 4 != 0
        or (index_dim // NVFP4_GROUP_SIZE) % 4 != 0
    ):
        raise ValueError(
            "NVFP4 MMA scale layout requires page_size=128 and scale-group "
            "counts divisible by 4"
        )
    if rows == 0:
        return
    # The kernels consume explicit input strides. This intentionally accepts
    # K/V/index-K column views of one fused projection buffer so prefill does
    # not copy the same suffix into three temporary contiguous tensors.
    if any(int(tensor.stride(-1)) != 1 for tensor in (k, v, index_k)):
        raise ValueError("NVFP4 fused writer inputs require unit inner stride")

    cache_tensors = (layout.packed_main, layout.side_bytes)
    if not k.is_cuda or any(
        tensor.device != k.device
        for tensor in (v, index_k, physical_slots, *cache_tensors)
    ):
        raise ValueError("NVFP4 fused writer tensors must share one CUDA device")

    k_packed, k_scales = layout.main_plane(0)
    v_packed, v_scales = layout.main_plane(1)
    idx_packed, idx_scales = layout.indexer(index_dim)
    main_groups = head_dim // NVFP4_GROUP_SIZE
    index_groups = index_dim // NVFP4_GROUP_SIZE
    total_groups = 2 * heads * main_groups + index_groups
    if head_dim == 128 and index_dim == 128 and layout.page_size == 128:
        _quantize_main_index_rows_d128_kernel[(rows, 2 * heads + 1)](
            k,
            v,
            index_k,
            physical_slots,
            physical_slots,
            physical_slots,
            k_packed,
            k_scales,
            v_packed,
            v_scales,
            idx_packed,
            idx_scales,
            rows,
            K_S0=int(k.stride(0)),
            K_S1=int(k.stride(1)),
            K_S2=int(k.stride(2)),
            V_S0=int(v.stride(0)),
            V_S1=int(v.stride(1)),
            V_S2=int(v.stride(2)),
            IDX_S0=int(index_k.stride(0)),
            IDX_S2=int(index_k.stride(2)),
            MAIN_PACKED_S0=int(k_packed.stride(0)),
            MAIN_SCALE_S0=int(k_scales.stride(0)),
            IDX_PACKED_S0=int(idx_packed.stride(0)),
            IDX_SCALE_S0=int(idx_scales.stride(0)),
            NUM_BLOCKS=layout.num_blocks,
            NUM_HEADS=heads,
            PAGE_SIZE=layout.page_size,
            MMA_SCALE_LAYOUT=mma_scale_layout,
            num_warps=1,
        )
        return

    _quantize_main_index_rows_kernel[(rows, total_groups)](
        k,
        v,
        index_k,
        physical_slots,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        idx_packed,
        idx_scales,
        rows,
        K_S0=int(k.stride(0)),
        K_S1=int(k.stride(1)),
        K_S2=int(k.stride(2)),
        V_S0=int(v.stride(0)),
        V_S1=int(v.stride(1)),
        V_S2=int(v.stride(2)),
        IDX_S0=int(index_k.stride(0)),
        IDX_S2=int(index_k.stride(2)),
        MAIN_PACKED_S0=int(k_packed.stride(0)),
        MAIN_SCALE_S0=int(k_scales.stride(0)),
        IDX_PACKED_S0=int(idx_packed.stride(0)),
        IDX_SCALE_S0=int(idx_scales.stride(0)),
        NUM_BLOCKS=layout.num_blocks,
        NUM_HEADS=heads,
        PAGE_SIZE=layout.page_size,
        HEAD_DIM=head_dim,
        MAIN_GROUPS=main_groups,
        IDX_DIM=index_dim,
        IDX_GROUPS=index_groups,
        MMA_SCALE_LAYOUT=mma_scale_layout,
        num_warps=1,
    )


def _validate_disjoint_cp_planes(planes, pages):
    """Reject overlapping destinations using host metadata only.

    FI working and persistent ABI planes can occupy disjoint regions of the
    same allocation. Their common page pitch lets us compare regions modulo
    that pitch without iterating over pages or reading device values.
    """
    if not pages:
        return
    regions = {}
    for tensor in planes:
        storage = tensor.untyped_storage().data_ptr()
        pitch = int(tensor.stride(0))
        start = int(tensor.storage_offset())  # uint8/E4M3 both use one-byte elements.
        width = tensor.numel() // pages
        end = start + (pages - 1) * pitch + width
        siblings = regions.setdefault(storage, [])
        for other_start, other_end, other_pitch, other_width in siblings:
            if end <= other_start or other_end <= start:
                continue
            if pitch != other_pitch:
                raise ValueError("shared CP plane storage requires a common page stride")
            distance = (start - other_start) % pitch
            if distance < other_width or distance + width > pitch:
                raise ValueError("CP writer destination planes must not overlap")
        siblings.append((start, end, pitch, width))


def _validate_cp_writer_planes(planes):
    if len(planes) != 6 or planes[0].ndim < 2:
        raise ValueError("M3.1 CP writer requires six paged planes")
    pages = int(planes[0].shape[0])
    for tensor, elements, dtype in zip(
        planes,
        (32768, 4096, 32768, 4096, 8192, 1024),
        (torch.uint8, torch.float8_e4m3fn) * 3,
    ):
        stride = 1
        if tensor.ndim < 2 or int(tensor.shape[0]) != pages or tensor.dtype != dtype:
            raise ValueError("M3.1 CP writer plane geometry or dtype mismatch")
        for size, actual_stride in zip(
            reversed(tensor.shape[1:]), reversed(tensor.stride()[1:])
        ):
            if int(actual_stride) != stride:
                raise ValueError("M3.1 CP writer plane rows must be contiguous")
            stride *= int(size)
        if stride != elements or int(tensor.stride(0)) < elements:
            raise ValueError("M3.1 CP writer requires head4/dim128/page128 planes")
    if planes[0].stride(0) != planes[2].stride(0) or planes[1].stride(0) != planes[
        3
    ].stride(0):
        raise ValueError("M3.1 CP writer K/V plane row strides must match")
    _validate_disjoint_cp_planes(planes, pages)
    return pages


def quantize_cp_main_index_rows_to_planes(
    packed: torch.Tensor,
    unpad_indices: torch.Tensor,
    slots: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
    idx_packed: torch.Tensor,
    idx_scales: torch.Tensor,
    *,
    owned_rows: Optional[torch.Tensor] = None,
    persistent_slots: Optional[torch.Tensor] = None,
    persistent_planes: Optional[tuple[torch.Tensor, ...]] = None,
    rows_per_cta: int = 1,
    fi_working_layout: bool = False,
) -> None:
    """M3.1 CP suffix writer reading the padded packed projection directly.

    Working rows read ``unpad_indices[row]``; rank-owned persistent rows read
    ``unpad_indices[owned_rows[row]]``. The maps must contain valid source
    indices, as required by the former index_select chain. Their values remain
    on device. Output planes may be independent working allocations or views
    of the persistent value/side ABI. No tensor or device metadata is allocated.
    Optional persistent planes reuse the quantized values with a full logical
    slot map, never a compressed owned-row map, and independent storage.
    ``rows_per_cta=4/8`` explicitly opts into the experimental multirow writer;
    the default retains the original one-row launch. ``fi_working_layout``
    selects linear K/FI-swizzled V scales for working stores only; persistent
    and index stores retain the original MMA byte layout.
    """
    if type(rows_per_cta) is not int or rows_per_cta not in (1, 4, 8):
        raise ValueError("rows_per_cta must be 1 (original), 4, or 8")
    if (
        packed.ndim != 2
        or int(packed.shape[1]) != 1152
        or packed.dtype != torch.bfloat16
        or packed.stride(1) != 1
    ):
        raise ValueError("M3.1 CP writer requires BF16 packed[padded_rows,1152]")
    dual = persistent_planes is not None
    if dual != (persistent_slots is not None) or (dual and owned_rows is not None):
        raise ValueError(
            "dual CP writer requires full persistent slots and no owned-row map"
        )
    persistent = tuple(persistent_planes) if dual else ()
    maps = (unpad_indices, slots) + (() if owned_rows is None else (owned_rows,))
    if dual:
        maps += (persistent_slots,)
    if any(
        tensor.ndim != 1 or tensor.dtype != torch.int64 or not tensor.is_contiguous()
        for tensor in maps
    ):
        raise ValueError("M3.1 CP writer maps and slots require contiguous int64")
    rows = int(slots.numel())
    if dual and persistent_slots.numel() != rows:
        raise ValueError("dual CP writer persistent slots must cover every logical row")
    if (owned_rows is None and unpad_indices.numel() != rows) or (
        owned_rows is not None and owned_rows.numel() != rows
    ):
        raise ValueError("M3.1 CP writer row-map/slot count mismatch")
    outputs = (k_packed, k_scales, v_packed, v_scales, idx_packed, idx_scales)
    if not packed.is_cuda or any(
        not tensor.is_cuda or tensor.device != packed.device
        for tensor in (*maps, *outputs, *persistent)
    ):
        raise ValueError("M3.1 CP writer tensors must share one CUDA device")
    pages = _validate_cp_writer_planes(outputs)
    if dual:
        persistent_pages = _validate_cp_writer_planes(persistent)
        if rows:
            working_storage = {
                x.untyped_storage().data_ptr() for x in (*outputs, packed, *maps)
            }
            if working_storage.intersection(
                x.untyped_storage().data_ptr() for x in persistent
            ):
                raise ValueError(
                    "dual CP writer destinations must have independent non-input storage"
                )
    if rows == 0:
        return
    k = packed[:, :512].view(packed.shape[0], 4, 128)
    v = packed[:, 512:1024].view(packed.shape[0], 4, 128)
    index_k = packed[:, 1024:].view(packed.shape[0], 1, 128)
    kernel = (_quantize_main_index_rows_d128_kernel if rows_per_cta == 1
              else _quantize_main_index_rows_d128_multirow_kernel)
    launch_options = {} if rows_per_cta == 1 else {"ROWS_PER_CTA": rows_per_cta}
    kernel[(triton.cdiv(rows, rows_per_cta), 9)](
        k,
        v,
        index_k,
        slots,
        unpad_indices,
        unpad_indices if owned_rows is None else owned_rows,
        *outputs,
        rows,
        K_S0=int(k.stride(0)),
        K_S1=int(k.stride(1)),
        K_S2=int(k.stride(2)),
        V_S0=int(v.stride(0)),
        V_S1=int(v.stride(1)),
        V_S2=int(v.stride(2)),
        IDX_S0=int(index_k.stride(0)),
        IDX_S2=int(index_k.stride(2)),
        MAIN_PACKED_S0=int(k_packed.stride(0)),
        MAIN_SCALE_S0=int(k_scales.stride(0)),
        IDX_PACKED_S0=int(idx_packed.stride(0)),
        IDX_SCALE_S0=int(idx_scales.stride(0)),
        NUM_BLOCKS=pages,
        NUM_HEADS=4,
        PAGE_SIZE=128,
        MMA_SCALE_LAYOUT=True,
        MAP_SOURCE_ROWS=True,
        MAP_OWNED_ROWS=owned_rows is not None,
        persist_slots_ptr=persistent_slots,
        persist_k_packed_ptr=persistent[0] if dual else None,
        persist_k_scales_ptr=persistent[1] if dual else None,
        persist_v_packed_ptr=persistent[2] if dual else None,
        persist_v_scales_ptr=persistent[3] if dual else None,
        persist_idx_packed_ptr=persistent[4] if dual else None,
        persist_idx_scales_ptr=persistent[5] if dual else None,
        PERSIST_MAIN_PACKED_S0=int(persistent[0].stride(0)) if dual else 0,
        PERSIST_MAIN_SCALE_S0=int(persistent[1].stride(0)) if dual else 0,
        PERSIST_IDX_PACKED_S0=int(persistent[4].stride(0)) if dual else 0,
        PERSIST_IDX_SCALE_S0=int(persistent[5].stride(0)) if dual else 0,
        PERSIST_NUM_BLOCKS=persistent_pages if dual else 0,
        WRITE_PERSISTENT=dual,
        FI_WORKING_LAYOUT=fi_working_layout,
        num_warps=1 if rows_per_cta == 1 else 4,
        **launch_options,
    )


def quantize_main_index_rows_to_planes(
    k: torch.Tensor,
    v: torch.Tensor,
    index_k: torch.Tensor,
    slots: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
    idx_packed: torch.Tensor,
    idx_scales: torch.Tensor,
    *,
    page_size: int = 128,
) -> None:
    """Quantize rows into independent contiguous native-kernel planes."""
    if tuple(k.shape) != tuple(v.shape) or k.dim() != 3:
        raise ValueError("NVFP4 working K/V must be matching [token,head,dim]")
    rows, heads, head_dim = map(int, k.shape)
    if index_k.dim() == 2:
        index_k = index_k[:, None, :]
    if index_k.dim() != 3 or tuple(index_k.shape[:2]) != (rows, 1):
        raise ValueError("NVFP4 working index K must be [token,1,dim]")
    index_dim = int(index_k.shape[2])
    main_groups = head_dim // NVFP4_GROUP_SIZE
    index_groups = index_dim // NVFP4_GROUP_SIZE
    expected_main = (k_packed.shape[0], heads, page_size, head_dim // 2)
    expected_idx = (k_packed.shape[0], 1, page_size, index_dim // 2)
    if tuple(k_packed.shape) != expected_main or tuple(v_packed.shape) != expected_main:
        raise ValueError("NVFP4 working main packed plane shape mismatch")
    if tuple(idx_packed.shape) != expected_idx:
        raise ValueError("NVFP4 working index packed plane shape mismatch")
    tensors = (
        k,
        v,
        index_k,
        slots,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        idx_packed,
        idx_scales,
    )
    if any(not tensor.is_cuda or tensor.device != k.device for tensor in tensors):
        raise ValueError("NVFP4 working writer tensors must share one CUDA device")
    if any(int(tensor.stride(-1)) != 1 for tensor in (k, v, index_k)):
        raise ValueError("NVFP4 working writer inputs require unit inner stride")
    outputs = (k_packed, k_scales, v_packed, v_scales, idx_packed, idx_scales)
    if any(not tensor.is_contiguous() for tensor in outputs):
        raise ValueError("NVFP4 working writer outputs must be contiguous")
    if slots.dtype != torch.int64 or int(slots.numel()) != rows:
        raise ValueError("NVFP4 working slots must be contiguous int64 per row")
    if rows == 0:
        return
    total_groups = 2 * heads * main_groups + index_groups
    if head_dim == 128 and index_dim == 128 and page_size == 128:
        _quantize_main_index_rows_d128_kernel[(rows, 2 * heads + 1)](
            k,
            v,
            index_k,
            slots,
            slots,
            slots,
            k_packed,
            k_scales,
            v_packed,
            v_scales,
            idx_packed,
            idx_scales,
            rows,
            K_S0=int(k.stride(0)),
            K_S1=int(k.stride(1)),
            K_S2=int(k.stride(2)),
            V_S0=int(v.stride(0)),
            V_S1=int(v.stride(1)),
            V_S2=int(v.stride(2)),
            IDX_S0=int(index_k.stride(0)),
            IDX_S2=int(index_k.stride(2)),
            MAIN_PACKED_S0=int(k_packed.stride(0)),
            MAIN_SCALE_S0=int(k_scales.stride(0)),
            IDX_PACKED_S0=int(idx_packed.stride(0)),
            IDX_SCALE_S0=int(idx_scales.stride(0)),
            NUM_BLOCKS=int(k_packed.shape[0]),
            NUM_HEADS=heads,
            PAGE_SIZE=page_size,
            MMA_SCALE_LAYOUT=True,
            num_warps=1,
        )
        return

    _quantize_main_index_rows_kernel[(rows, total_groups)](
        k,
        v,
        index_k,
        slots,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        idx_packed,
        idx_scales,
        rows,
        K_S0=int(k.stride(0)),
        K_S1=int(k.stride(1)),
        K_S2=int(k.stride(2)),
        V_S0=int(v.stride(0)),
        V_S1=int(v.stride(1)),
        V_S2=int(v.stride(2)),
        IDX_S0=int(index_k.stride(0)),
        IDX_S2=int(index_k.stride(2)),
        MAIN_PACKED_S0=int(k_packed.stride(0)),
        MAIN_SCALE_S0=int(k_scales.stride(0)),
        IDX_PACKED_S0=int(idx_packed.stride(0)),
        IDX_SCALE_S0=int(idx_scales.stride(0)),
        NUM_BLOCKS=int(k_packed.shape[0]),
        NUM_HEADS=heads,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        MAIN_GROUPS=main_groups,
        IDX_DIM=index_dim,
        IDX_GROUPS=index_groups,
        MMA_SCALE_LAYOUT=True,
        num_warps=1,
    )


def gather_main_rows(
    layout: NVFP4CacheLayout,
    physical_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
    *,
    out_hnd: bool = False,
    e4m3_compute_grid: bool = False,
) -> None:
    for kv_index, output in enumerate((out_k, out_v)):
        packed, scales = layout.main_plane(kv_index)
        _gather_rows(
            packed,
            scales,
            physical_slots,
            destination_slots,
            output,
            layout.page_size,
            out_hnd=out_hnd,
            e4m3_compute_grid=e4m3_compute_grid,
        )


def quantize_index_rows(
    values: torch.Tensor,
    physical_slots: torch.Tensor,
    layout: NVFP4CacheLayout,
) -> None:
    if values.dim() == 2:
        values = values[:, None, :]
    packed, scales = layout.indexer(int(values.shape[-1]))
    _quantize_rows(
        values.contiguous(), physical_slots, packed, scales, layout.page_size
    )


def gather_index_rows(
    layout: NVFP4CacheLayout,
    indexer_dim: int,
    physical_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    output: torch.Tensor,
    *,
    e4m3_compute_grid: bool = False,
) -> None:
    packed, scales = layout.indexer(indexer_dim)
    out = output if output.dim() == 3 else output[:, None, :]
    _gather_rows(
        packed,
        scales,
        physical_slots,
        destination_slots,
        out,
        layout.page_size,
        e4m3_compute_grid=e4m3_compute_grid,
    )


def scatter_main_rows_to_hnd(
    k: torch.Tensor,
    v: torch.Tensor,
    destination_slots: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
) -> None:
    rows, heads, dim = map(int, k.shape)
    if tuple(v.shape) != tuple(k.shape):
        raise ValueError("BF16 working K/V shapes differ")
    page_size = int(out_k.shape[2])
    for source, output in ((k, out_k), (v, out_v)):
        _scatter_hnd_rows_kernel[(rows, heads)](
            source,
            destination_slots,
            output,
            rows,
            SRC_S0=int(source.stride(0)),
            SRC_S1=int(source.stride(1)),
            SRC_S2=int(source.stride(2)),
            NUM_HEADS=heads,
            PAGE_SIZE=page_size,
            HEAD_DIM=dim,
            BLOCK_D=triton.next_power_of_2(dim),
            num_warps=1,
        )


@triton.jit
def _clear_packed_working_tail_scales_kernel(
    k_scale,
    v_scale,
    idx_scale,
    lengths,
    K_PAGE_STRIDE: tl.constexpr,
    V_PAGE_STRIDE: tl.constexpr,
    I_PAGE_STRIDE: tl.constexpr,
    SCRATCH_SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    IDX_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
    FI_WORKING_LAYOUT: tl.constexpr = False,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    length = tl.load(lengths + batch).to(tl.int64)
    tail_start = length % 128
    page = (batch * SCRATCH_SEQ_LEN + length) // 128
    offsets = tl.arange(0, BLOCK)
    row, group = offsets // GROUPS, offsets % GROUPS
    valid = (tail_start != 0) & (row < 128) & (row >= tail_start)
    swizzle = group // 4 * 512 + row % 32 * 16 + row // 32 * 4 + group % 4
    k_offset = swizzle
    v_offset = swizzle
    if FI_WORKING_LAYOUT:
        k_offset = row * GROUPS + group
        v_offset = ((row // 4) * 4 + group // 2) * 8 + (group % 2) * 4 + row % 4
    tl.store(k_scale + page * K_PAGE_STRIDE + head * 128 * GROUPS + k_offset, 0.0, valid)
    tl.store(v_scale + page * V_PAGE_STRIDE + head * 128 * GROUPS + v_offset, 0.0, valid)
    idx_row, idx_group = offsets // IDX_GROUPS, offsets % IDX_GROUPS
    idx_valid = (
        (head == 0) & (tail_start != 0) & (idx_row < 128) & (idx_row >= tail_start)
    )
    idx_swizzle = (
        idx_group // 4 * 512 + idx_row % 32 * 16 + idx_row // 32 * 4 + idx_group % 4
    )
    tl.store(idx_scale + page * I_PAGE_STRIDE + idx_swizzle, 0.0, idx_valid)


def clear_packed_working_tail_scales(
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    idx_scales: torch.Tensor,
    kv_lens: torch.Tensor,
    scratch_seq_len: int,
    heads: int,
    head_dim: int,
    index_dim: int,
    *,
    fi_working_layout: bool = False,
) -> None:
    """Make only the unread last-page rows finite for whole-page FP8 MMA.

    Native prefill masks QK but reads full-page V. Zero probability multiplied
    by an uninitialized E4M3 NaN still poisons PV. Zero scales make the finite
    E2M1 payload decode to zero, without touching any live row or clearing the
    full historical working set. Main scales follow the selected working
    layout; index scales always use the writer's 128x4 MMA swizzle.
    """
    if scratch_seq_len % 128 or head_dim % 64 or index_dim % 64:
        raise ValueError("packed tail scales require page128 and MMA group4 alignment")
    groups, idx_groups = head_dim // NVFP4_GROUP_SIZE, index_dim // NVFP4_GROUP_SIZE
    _clear_packed_working_tail_scales_kernel[(kv_lens.numel(), heads)](
        k_scales,
        v_scales,
        idx_scales,
        kv_lens,
        K_PAGE_STRIDE=k_scales.stride(0),
        V_PAGE_STRIDE=v_scales.stride(0),
        I_PAGE_STRIDE=idx_scales.stride(0),
        SCRATCH_SEQ_LEN=scratch_seq_len,
        HEADS=heads,
        GROUPS=groups,
        IDX_GROUPS=idx_groups,
        BLOCK=triton.next_power_of_2(128 * max(groups, idx_groups)),
        FI_WORKING_LAYOUT=fi_working_layout,
        num_warps=4,
    )


def clear_working_tails(
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    idx_rows: torch.Tensor,
    kv_lens: torch.Tensor,
    scratch_seq_len: int,
) -> None:
    batch_size = int(kv_lens.numel())
    if batch_size == 0:
        return
    heads = int(k_pages.shape[1])
    page_size = int(k_pages.shape[2])
    head_dim = int(k_pages.shape[3])
    idx_dim = int(idx_rows.shape[-1])
    _clear_hnd_tails_kernel[(batch_size * page_size, heads)](
        k_pages,
        v_pages,
        idx_rows,
        kv_lens,
        batch_size,
        SCRATCH_SEQ_LEN=scratch_seq_len,
        PAGE_SIZE=page_size,
        NUM_HEADS=heads,
        HEAD_DIM=head_dim,
        IDX_DIM=idx_dim,
        BLOCK_D=triton.next_power_of_2(max(head_dim, idx_dim)),
        num_warps=1,
    )


def convert_active_pages(
    layout: NVFP4CacheLayout,
    block_table: torch.Tensor,
    working: torch.Tensor,
    *,
    quantize: bool,
) -> None:
    """Convert active physical pages between persistent NVFP4 and BF16 HND."""
    if tuple(working.shape) != (
        layout.num_blocks,
        2,
        layout.num_heads,
        layout.page_size,
        layout.head_dim,
    ):
        raise ValueError(f"invalid NVFP4 BF16 workspace shape {tuple(working.shape)}")
    if working.dtype != torch.bfloat16 or not working.is_contiguous():
        raise ValueError("NVFP4 attention workspace must be contiguous BF16")
    table = block_table.contiguous().view(-1)
    groups = layout.head_dim // NVFP4_GROUP_SIZE
    total_groups = 2 * layout.num_heads * layout.page_size * groups
    group_batch = 32
    _convert_pages_kernel[(int(table.numel()), triton.cdiv(total_groups, group_batch))](
        table,
        layout.packed_main,
        layout.main_scales,
        working,
        int(table.numel()),
        PACKED_S0=int(layout.packed_main.stride(0)),
        SCALE_S0=int(layout.main_scales.stride(0)),
        NUM_BLOCKS=layout.num_blocks,
        NUM_HEADS=layout.num_heads,
        PAGE_SIZE=layout.page_size,
        HEAD_DIM=layout.head_dim,
        GROUPS=groups,
        TOTAL_GROUPS=total_groups,
        GROUP_BATCH=group_batch,
        QUANTIZE=quantize,
        num_warps=1,
    )


def reference_quantize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Small torch reference used by unit tests and format debugging."""
    if values.shape[-1] != NVFP4_GROUP_SIZE:
        raise ValueError("reference_quantize expects groups of exactly 16 values")
    x = values.float()
    scale = torch.clamp(
        x.abs().amax(dim=-1) / NVFP4_E2M1_MAX,
        NVFP4_SCALE_MIN,
        NVFP4_SCALE_MAX,
    ).to(torch.float8_e4m3fn)
    scale_f = scale.float().unsqueeze(-1)
    magnitude = x.abs()
    code = torch.where(
        magnitude > 5.0 * scale_f,
        7,
        torch.where(
            magnitude >= 3.5 * scale_f,
            6,
            torch.where(
                magnitude > 2.5 * scale_f,
                5,
                torch.where(
                    magnitude >= 1.75 * scale_f,
                    4,
                    torch.where(
                        magnitude > 1.25 * scale_f,
                        3,
                        torch.where(
                            magnitude >= 0.75 * scale_f,
                            2,
                            torch.where(magnitude > 0.25 * scale_f, 1, 0),
                        ),
                    ),
                ),
            ),
        ),
    ).to(torch.uint8)
    code |= torch.where((x < 0) & (code != 0), 8, 0).to(torch.uint8)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed, scale


def reference_dequantize(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`reference_quantize` for tests and inspection."""
    code = torch.empty(
        *packed.shape[:-1],
        packed.shape[-1] * 2,
        dtype=torch.uint8,
        device=packed.device,
    )
    code[..., 0::2] = packed & 15
    code[..., 1::2] = (packed >> 4) & 15
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    values = magnitudes[(code & 7).long()]
    values = torch.where((code & 8) != 0, -values, values)
    return values * scale.float().unsqueeze(-1)
