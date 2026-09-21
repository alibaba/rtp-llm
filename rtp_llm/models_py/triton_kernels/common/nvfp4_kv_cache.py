"""NVFP4 paged-KV storage and BF16 attention adapters.

The persistent representation follows the MiniMax-M3.1 contract:

* each contiguous group of 16 values has an E4M3 scale ``clamp(amax / 6,
  1/512, 448)`` (rounded by the E4M3 cast);
* scaled values use E2M1 with round-to-nearest-even boundaries;
* two E2M1 nibbles are stored per byte, with the lower-dimension value in the
  low nibble;
* there is no tensor-level scale (it is identically one).

There is intentionally no NVFP4 attention implementation here.  The public
helpers materialize BF16 working pages immediately around an existing attention
operator.  That boundary is the single replacement point for a future native
NVFP4 attention kernel.
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
def _e2m1_encode(values):
    """Exact RNE decision ladder required by the M3.1 NVFP4 format."""
    magnitude = tl.minimum(tl.abs(values), _TL_E2M1_MAX)
    code = tl.where(
        magnitude > 5.0,
        7,
        tl.where(
            magnitude >= 3.5,
            6,
            tl.where(
                magnitude > 2.5,
                5,
                tl.where(
                    magnitude >= 1.75,
                    4,
                    tl.where(
                        magnitude > 1.25,
                        3,
                        tl.where(
                            magnitude >= 0.75,
                            2,
                            tl.where(magnitude > 0.25, 1, 0),
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
    low_codes = _e2m1_encode(even_values / scale)
    high_codes = _e2m1_encode(odd_values / scale)
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
        low_codes = _e2m1_encode(even_values / scale)
        high_codes = _e2m1_encode(odd_values / scale)
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
        num_warps=1,
    )


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


def gather_main_rows(
    layout: NVFP4CacheLayout,
    physical_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
    *,
    out_hnd: bool = False,
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


class _BF16PageWorkspace:
    def __init__(self) -> None:
        self._buffers: dict[tuple, torch.Tensor] = {}

    def acquire(self, layout: NVFP4CacheLayout, device: torch.device) -> torch.Tensor:
        key = (
            device.type,
            device.index,
            layout.num_blocks,
            layout.num_heads,
            layout.page_size,
            layout.head_dim,
        )
        result = self._buffers.get(key)
        shape = (
            layout.num_blocks,
            2,
            layout.num_heads,
            layout.page_size,
            layout.head_dim,
        )
        if result is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "NVFP4 BF16 attention workspace must be allocated during "
                    "CUDA-graph warmup before capture"
                )
            # Physical block zero is the framework's immutable sentinel page.
            # Conversion deliberately skips it, so initialize the shared
            # workspace once to preserve the sentinel's all-zero semantics.
            result = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            self._buffers[key] = result
        return result


_BF16_PAGE_WORKSPACE = _BF16PageWorkspace()


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


def _attention_block_table(fmha_impl) -> torch.Tensor:
    attn_inputs = getattr(fmha_impl, "attn_inputs", None)
    if attn_inputs is None:
        raise RuntimeError("NVFP4 attention adapter requires prepared attention inputs")
    table = getattr(attn_inputs, "kv_cache_block_id_device", None)
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        raise RuntimeError("NVFP4 attention adapter could not resolve a block table")
    return table


def dense_attention_forward(
    fmha_impl,
    qkv: torch.Tensor,
    kv_cache,
    layer_idx: int,
    num_heads: int,
    page_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Run an existing BF16 attention operator over persistent NVFP4 pages.

    The temporary cache swap is deliberately local to this call.  Cache-store
    transfer is delayed until after the BF16 pages (including newly written K/V)
    have been requantized into the persistent allocation.
    """
    if kv_cache is None:
        return fmha_impl.forward(qkv, kv_cache, layer_idx)
    layout = cache_layout(
        kv_cache.kv_cache_base,
        kv_cache.kv_scale_base,
        num_heads,
        page_size,
        head_dim,
    )
    block_table = _attention_block_table(fmha_impl)
    working = _BF16_PAGE_WORKSPACE.acquire(layout, qkv.device)
    convert_active_pages(layout, block_table, working, quantize=False)

    persistent_base = kv_cache.kv_cache_base
    write_cache_store = getattr(fmha_impl, "write_cache_store_impl", None)
    has_write_member = hasattr(fmha_impl, "write_cache_store_impl")
    if has_write_member:
        fmha_impl.write_cache_store_impl = None
    kv_cache.kv_cache_base = working
    succeeded = False
    try:
        output = fmha_impl.forward(qkv, kv_cache, layer_idx)
        convert_active_pages(layout, block_table, working, quantize=True)
        succeeded = True
    finally:
        kv_cache.kv_cache_base = persistent_base
        if has_write_member:
            fmha_impl.write_cache_store_impl = write_cache_store

    if succeeded and write_cache_store is not None:
        from rtp_llm.models_py.modules.factory.attention import common

        common.apply_write_cache_store(
            write_cache_store, fmha_impl.attn_inputs, kv_cache
        )
    return output


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
    scaled = x / scale.float().unsqueeze(-1)
    magnitude = scaled.abs().clamp(max=NVFP4_E2M1_MAX)
    code = torch.where(
        magnitude > 5.0,
        7,
        torch.where(
            magnitude >= 3.5,
            6,
            torch.where(
                magnitude > 2.5,
                5,
                torch.where(
                    magnitude >= 1.75,
                    4,
                    torch.where(
                        magnitude > 1.25,
                        3,
                        torch.where(
                            magnitude >= 0.75,
                            2,
                            torch.where(magnitude > 0.25, 1, 0),
                        ),
                    ),
                ),
            ),
        ),
    ).to(torch.uint8)
    code |= torch.where((scaled < 0) & (code != 0), 8, 0).to(torch.uint8)
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
