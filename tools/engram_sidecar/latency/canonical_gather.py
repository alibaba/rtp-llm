"""Experimental canonical Engram gather; never selected by the production path.

The owner must have computed the exact lookup-dequant -> BF16 -> fixed-variant
group32 activation quantization before publishing these tables. Raw checkpoint
E4M3/E8M0 tables are NOT this representation. Metadata is a trusted control-plane
contract, not proof that arbitrary bytes have been converted correctly.

Call allocate_outputs during initialization. gather_into creates no torch
tensors, performs no host/device transfers or synchronization, and launches one
kernel (zero for M=0). Callers own storage/graph lifetime and stream ordering.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


CANONICAL_REPRESENTATION = "engram.activation-e4m3-unpacked-e8m0.group32.v1"
LEGACY_QUANT_ABI = "rtpllm-492ecf45.lookup-bf16.sgl-legacy-group32-f32tiny-floor1e-10.v1"
V2_QUANT_ABI = "rtpllm-492ecf45.lookup-bf16.sgl-v2-group32-f32tiny.v1"


@dataclass(frozen=True)
class CanonicalMetadata:
    representation: str
    quant_abi: str

    def __post_init__(self):
        if self.representation != CANONICAL_REPRESENTATION:
            raise ValueError("expected converted canonical activation tables, not raw checkpoint bytes")
        if self.quant_abi not in (LEGACY_QUANT_ABI, V2_QUANT_ABI):
            raise ValueError("canonical quantization ABI must explicitly select legacy or v2")

    @property
    def zero_scale(self):
        # Legacy rounds max(amax/448,1e-10) up to 2^-33: exponent 94.
        # Current v2 with eps=FP32 tiny emits exponent 1 for an all-zero group.
        # These are the existing implementations' behavior, not generic E8M0.
        return 94 if self.quant_abi == LEGACY_QUANT_ABI else 1


def _geometry(rows, heads, head_dim):
    if type(rows) is not int or rows < 0:
        raise ValueError("rows must be a nonnegative integer")
    if type(heads) is not int or heads <= 0:
        raise ValueError("heads must be a positive integer")
    if type(head_dim) is not int or head_dim < 32 or head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two >=32")
    width = heads * head_dim
    return width, (width + 127) // 128, (rows + 3) // 4 * 4


def allocate_outputs(*, rows, heads, head_dim, device):
    """Allocate local HBM scratch once; return (FP8 payload, packed int32 scales).

    Scales have logical [M,ceil(K/128)] and stride (1,align_up(M,4)). The
    backing includes all aligned rows; its row padding is zeroed here. Reuse
    these tensors only with the same shape, after prior work on them completes.
    """
    width, packs, aligned_rows = _geometry(rows, heads, head_dim)
    device = torch.device("cuda", device) if type(device) is int else torch.device(device)
    if device.type != "cuda":
        raise ValueError("canonical outputs must be allocated on a CUDA consumer")
    with torch.cuda.device(device):
        q = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=device)
        backing = torch.zeros((packs, aligned_rows), dtype=torch.int32, device=device)
        scales = backing.as_strided((rows, packs), (1, aligned_rows))
    return q, scales


@triton.jit
def _canonical_gather_kernel(
    QTable, STable, Ids, QOut, SOut,
    M: tl.constexpr, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    ID_STRIDE_M: tl.constexpr, ID_STRIDE_H: tl.constexpr,
    SCALE_STRIDE: tl.constexpr, ZERO_SCALE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_PACKS: tl.constexpr,
):
    # Each tile writes disjoint complete int32 scale words, including when an
    # int32 packs groups from multiple hash rows (D=32 or D=64).
    token = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    first_pack = tl.program_id(1).to(tl.int64) * BLOCK_PACKS
    column = first_pack * 128 + tl.arange(0, BLOCK_PACKS * 128)
    head = column // D
    index = tl.load(
        Ids + token[:, None] * ID_STRIDE_M + head[None, :] * ID_STRIDE_H,
        (token[:, None] < M) & (head[None, :] < H), other=-1,
    ).to(tl.int64)
    valid = (token[:, None] < M) & (head[None, :] < H) & (index >= 0) & (index < R)
    safe_index = tl.where(valid, index, 0)
    raw = tl.load(QTable + safe_index * D + column[None, :] % D, valid, other=0)
    tl.store(
        QOut + token[:, None] * (H * D) + column[None, :],
        raw.to(tl.float8e4nv, bitcast=True),
        (token[:, None] < M) & (column[None, :] < H * D),
    )

    pack = first_pack + tl.arange(0, BLOCK_PACKS)
    word = tl.full((BLOCK_ROWS, BLOCK_PACKS), 0, tl.uint32)
    for byte in tl.static_range(4):
        group = pack * 4 + byte
        scale_head = group // (D // 32)
        scale_index = tl.load(
            Ids + token[:, None] * ID_STRIDE_M + scale_head[None, :] * ID_STRIDE_H,
            (token[:, None] < M) & (scale_head[None, :] < H), other=-1,
        ).to(tl.int64)
        valid_scale = ((token[:, None] < M) & (scale_head[None, :] < H)
                       & (scale_index >= 0) & (scale_index < R))
        safe_scale_index = tl.where(valid_scale, scale_index, 0)
        exponent = tl.load(
            STable + safe_scale_index * (D // 32) + group[None, :] % (D // 32),
            valid_scale, other=ZERO_SCALE,
        )
        # A logical invalid ID requires the quantizer's zero-group encoding.
        # Padding past K has no logical group and is deterministically zero.
        exponent = tl.where(group[None, :] < H * (D // 32), exponent, 0)
        word = word | (exponent.to(tl.uint32) << (byte * 8))
    tl.store(
        SOut + token[:, None] + pack[None, :] * SCALE_STRIDE,
        word.to(tl.int32),
        (token[:, None] < M) & (pack[None, :] < (H * D + 127) // 128),
    )


def _span(tensor):
    """Conservative byte address span; metadata-only, including strided IDs."""
    if tensor.numel() == 0:
        return (0, 0)
    elements = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    begin = tensor.data_ptr()
    return begin, begin + elements * tensor.element_size()


def gather_into(q_table, scale_table, ids, out_q, out_scales, *, metadata,
                block_rows=1, block_packs=4, num_warps=4):
    """Gather into initialization-time local outputs; one asynchronous kernel.

    Inputs must be sealed canonical uint8 matrices on this consumer's CUDA
    device; the pointers may have shared HOST_NUMA backing. Output tensors must
    be disjoint local HBM scratch from allocate_outputs. IDs may be strided.
    Never call this while owner/storage is draining; graph users must retain
    mappings through graph reset. This function does not acquire a storage lease.
    """
    if not isinstance(metadata, CanonicalMetadata):
        raise TypeError("explicit CanonicalMetadata is required")
    # Recheck even manually constructed/deserialized dataclasses. This validates
    # the declared ABI, not the truth of a caller-controlled metadata label.
    metadata.__post_init__()
    tensors = (q_table, scale_table, ids, out_q, out_scales)
    if any(not isinstance(t, torch.Tensor) or t.ndim != 2 for t in tensors):
        raise ValueError("all inputs and outputs must be 2D torch tensors")
    device = q_table.device
    if device.type != "cuda" or any(t.device != device for t in tensors):
        raise ValueError("all tensors must use the same CUDA consumer device")
    if q_table.dtype != torch.uint8 or scale_table.dtype != torch.uint8:
        raise ValueError("canonical shared tables must preserve bytes as uint8")
    rows, heads = ids.shape
    table_rows, dim = q_table.shape
    width, packs, aligned_rows = _geometry(rows, heads, dim)
    if table_rows <= 0 or scale_table.shape != (table_rows, dim // 32):
        raise ValueError("canonical table shapes do not match")
    if not q_table.is_contiguous() or not scale_table.is_contiguous():
        raise ValueError("canonical shared tables must be contiguous")
    if ids.dtype not in (torch.int32, torch.int64) or any(s < 0 for s in ids.stride()):
        raise ValueError("IDs must be int32/int64 with nonnegative strides")
    if (out_q.dtype != torch.float8_e4m3fn or out_q.shape != (rows, width)
            or not out_q.is_contiguous()):
        raise ValueError("payload output must be contiguous E4M3 [M,H*D]")
    if (out_scales.dtype != torch.int32 or out_scales.shape != (rows, packs)
            or out_scales.stride() != (1, aligned_rows)):
        raise ValueError("scale output must use packed int32 [M,ceil(K/128)] with TMA strides")
    if (type(block_rows) is not int or block_rows not in (1, 2, 4, 8)
            or type(block_packs) is not int or block_packs not in (1, 2, 4, 8)
            or type(num_warps) is not int or num_warps not in (1, 2, 4, 8)):
        raise ValueError("unsupported tile/warp configuration")
    # Catch ordinary aliasing before writing. Separate VAs mapping the same
    # physical object cannot be inferred from Tensor metadata; callers must use
    # allocate_outputs, not manufacture output views of shared backing.
    spans = [_span(t) for t in tensors]
    for output in (3, 4):
        start, end = spans[output]
        for other, (begin, finish) in enumerate(spans):
            if other != output and start < finish and begin < end:
                raise ValueError("output scratch must not alias inputs or each other")
    if rows == 0:
        return
    grid = (triton.cdiv(rows, block_rows), triton.cdiv(packs, block_packs))
    if grid[0] > 2147483647 or grid[1] > 65535:
        raise ValueError("requested tile grid exceeds CUDA launch dimensions")
    with torch.cuda.device(device):
        _canonical_gather_kernel[grid](
            q_table, scale_table, ids, out_q, out_scales,
            M=rows, H=heads, D=dim, R=table_rows,
            ID_STRIDE_M=ids.stride(0), ID_STRIDE_H=ids.stride(1),
            SCALE_STRIDE=aligned_rows, ZERO_SCALE=metadata.zero_scale,
            BLOCK_ROWS=block_rows, BLOCK_PACKS=block_packs, num_warps=num_warps,
        )
