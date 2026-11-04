"""Pool-tag keyed KV codecs for the V4.1 PPU backend.

``encode_kv``/``decode_kv`` implement the ``V41BackendOps`` codec contract on
contiguous row blocks: ring position mapping and block-table traversal stay
with the caller (``V41CacheView.block_tables``); this module converts between
raw BF16 KV rows and the packed byte rows defined by the pool tag:

* ``global_kv_1`` / ``global_kv_2`` — 288B FP4 rows, token-interleaved
  ``[256B E2M1 payload | 32B E4M3 scales]``, ratio 1/2 from the descriptor.
* ``indexer_kv`` — 68B MX FP4 rows (``[64B E2M1 payload | 4B packed UE8M0
  scales]``); the *pool* is planar (payload plane then scale plane per block)
  — use :func:`planar_block_to_rows` / :func:`rows_to_planar_block` to move
  row blocks in and out of pool blocks.
* ``swa_kv`` — 528B MXFP8 rows, token-interleaved
  ``[512B E4M3 payload | 16B UE8M0 scales]``, sliding window 128 (ring).

``decode_kv`` fills the caller-owned bounded ``out`` tensor: shape and dtype
are validated, strides are respected, and the same object is returned — no
silent dtype or stride changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rtp_llm.models_py.modules.dsv41.ppu import _quant

GLOBAL_KV_TAGS = ("global_kv_1", "global_kv_2")
INDEXER_KV_TAG = "indexer_kv"
SWA_KV_TAG = "swa_kv"
DECODER_SWA_KV_TAG = "decoder_swa_kv"

DEFAULT_BLOCK_ENTRIES = 128  # entries per pool block (tokens per block)
DEFAULT_SWA_BLOCK_ENTRIES = 512
SWA_WINDOW = 128


@dataclass(frozen=True)
class PoolDescriptor:
    """Byte layout of one V4.1 KV pool entry row.

    ``planar`` describes the *pool block* layout: token-interleaved pools
    store each entry's ``[payload | scales]`` bytes contiguously, planar pools
    store all payloads of a block first and all scale bytes after.
    """

    entry_bytes: int
    ratio: int = 1
    planar: bool = False
    block_entries: int = DEFAULT_BLOCK_ENTRIES
    window: int | None = None
    kind: str = "fp4_e2m1_e4m3"  # fp4_e2m1_e4m3 | fp8_e4m3_ue8m0 | fp4_e2m1_ue8m0

    @property
    def row_dim(self) -> int:
        """Dimension of the raw (unquantized) rows this pool stores."""
        if self.kind == "fp4_e2m1_ue8m0":
            return _quant.INDEXER_HEAD_DIM
        return _quant.KV_DIM


def _descriptor_attr(descriptor, *names, default=None):
    for name in names:
        value = getattr(descriptor, name, None)
        if value is not None:
            return value
    return default


def default_descriptor(pool_tag: str, descriptor: object = None) -> PoolDescriptor:
    """Build the :class:`PoolDescriptor` for ``pool_tag``.

    Numeric fields are read from the caller's descriptor object when present
    (``compression_ratio``, ``entry_elems``, ``tokens_per_block`` /
    ``block_entries``), otherwise the documented defaults apply.
    """
    ratio = int(_descriptor_attr(descriptor, "compression_ratio", default=1) or 1)
    entries = int(
        _descriptor_attr(
            descriptor,
            "tokens_per_block",
            "block_entries",
            default=DEFAULT_BLOCK_ENTRIES,
        )
        or DEFAULT_BLOCK_ENTRIES
    )
    entry_elems = int(_descriptor_attr(descriptor, "entry_elems", default=0) or 0)
    if pool_tag in GLOBAL_KV_TAGS:
        entry_bytes = entry_elems or _quant.FP4_GLOBAL_ENTRY_BYTES
        return PoolDescriptor(
            entry_bytes=entry_bytes,
            ratio=ratio,
            planar=False,
            block_entries=entries,
            kind="fp4_e2m1_e4m3",
        )
    if pool_tag == INDEXER_KV_TAG:
        entry_bytes = entry_elems or _quant.INDEXER_K_ENTRY_BYTES
        return PoolDescriptor(
            entry_bytes=entry_bytes,
            ratio=ratio,
            planar=True,
            block_entries=entries,
            kind="fp4_e2m1_ue8m0",
        )
    if pool_tag in (SWA_KV_TAG, DECODER_SWA_KV_TAG):
        entry_bytes = entry_elems or _quant.SWA_KV_ENTRY_BYTES
        return PoolDescriptor(
            entry_bytes=entry_bytes,
            ratio=1,
            planar=False,
            block_entries=entries or DEFAULT_SWA_BLOCK_ENTRIES,
            window=SWA_WINDOW,
            kind="fp8_e4m3_ue8m0",
        )
    raise ValueError(f"unknown V4.1 KV pool tag {pool_tag!r}")


# ---------------------------------------------------------------------------
# Planar (indexer) block <-> row conversion. Block-table traversal and ring
# position mapping stay with the caller; these helpers move one contiguous
# block of rows at a time.
# ---------------------------------------------------------------------------


def rows_to_planar_block(packed_rows: torch.Tensor) -> torch.Tensor:
    """Token-interleaved 68B rows ``[E,68]`` -> planar block bytes
    ``[E*64 payload | E*4 scales]`` (uint8, flat)."""
    rows = packed_rows.shape[0]
    payload = packed_rows[:, :64]
    scales = packed_rows[:, 64:]
    return torch.cat((payload.reshape(-1), scales.reshape(-1)), dim=0).contiguous()


def planar_block_to_rows(block_bytes: torch.Tensor) -> torch.Tensor:
    """Planar block bytes ``[E*68]`` -> token-interleaved rows ``[E,68]``."""
    total = block_bytes.shape[-1]
    rows = total // _quant.INDEXER_K_ENTRY_BYTES
    payload = block_bytes[: rows * 64].reshape(rows, 64)
    scales = block_bytes[rows * 64 :].reshape(rows, 4)
    return torch.cat((payload, scales), dim=1).contiguous()


# ---------------------------------------------------------------------------
# The V41BackendOps codec entry points.
# ---------------------------------------------------------------------------


def encode_kv(kv: torch.Tensor, *, pool_tag: str, descriptor: object = None) -> Tensor:
    """Raw BF16 KV rows -> packed uint8 rows ``[N, entry_bytes]``.

    ``kv`` is ``[N, 512]`` for the global/SWA pools and ``[N, 128]`` for
    ``indexer_kv``. Rows are encoded independently and contiguously — the
    caller owns request grouping, compression-ratio semantics, ring position
    mapping and block-table traversal.
    """
    desc = default_descriptor(pool_tag, descriptor)
    if desc.kind == "fp4_e2m1_e4m3":
        if tuple(kv.shape[-1:]) != (_quant.KV_DIM,):
            raise ValueError(f"global kv rows must be [N, {_quant.KV_DIM}]")
        return _quant.quantize_global_kv_rows(kv.reshape(-1, _quant.KV_DIM))
    if desc.kind == "fp8_e4m3_ue8m0":
        if tuple(kv.shape[-1:]) != (_quant.KV_DIM,):
            raise ValueError(f"swa kv rows must be [N, {_quant.KV_DIM}]")
        return _quant.quantize_swa_kv_rows(kv.reshape(-1, _quant.KV_DIM))
    payload, sf = _quant.quantize_indexer_k_rows(kv.reshape(-1, desc.row_dim))
    return _quant.indexer_k_packed_rows(payload, sf)


def decode_kv(
    packed: torch.Tensor, *, pool_tag: str, descriptor: object = None, out: torch.Tensor
) -> Tensor:
    """Packed uint8 rows ``[N, entry_bytes]`` -> fill caller-owned ``out``.

    ``out`` must be BF16 (or FP32) with shape ``[N, row_dim]``; arbitrary
    strides are allowed as long as the last dim is contiguous. The function
    writes only the value region (stride padding is untouched), makes no
    dtype/stride changes of its own, and returns the same ``out`` object.
    """
    desc = default_descriptor(pool_tag, descriptor)
    rows = packed.shape[0]
    expected = (rows, desc.row_dim)
    if tuple(out.shape) != expected:
        raise ValueError(f"{pool_tag} out shape {tuple(out.shape)} != {expected}")
    if out.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"{pool_tag} out dtype must be bf16/fp32, got {out.dtype}")
    if out.stride(-1) != 1:
        raise ValueError(f"{pool_tag} out last dim must be contiguous")
    packed = packed.view(torch.uint8)
    if desc.kind == "fp4_e2m1_e4m3":
        return _quant.dequantize_global_kv_rows(packed, out=out)
    if desc.kind == "fp8_e4m3_ue8m0":
        return _quant.dequantize_swa_kv_rows(packed, out=out)
    sf = _quant._bytes_to_sf(packed[:, 64:].contiguous())
    values = _quant.dequantize_mx_fp4_rows(packed[:, :64].view(torch.int8), sf)
    out.copy_(values.to(out.dtype))
    return out


def gather_rows(
    pool: torch.Tensor, pool_tag: str, descriptor: object, slots: torch.Tensor
) -> torch.Tensor:
    """Gather packed entry rows by absolute slot id from a 2D pool.

    The pool is ``[blocks, entries*entry_bytes]`` uint8; slot =
    ``block*entries + offset``. Sentinel (negative) slots zero-fill. Planar
    pools are gathered through the planar layout.
    """
    desc = default_descriptor(pool_tag, descriptor)
    entries = desc.block_entries
    rows = slots.shape[0]
    if desc.planar:
        payload, sf = _quant.gather_indexer_k_planar(pool, slots)
        return _quant.indexer_k_packed_rows(payload, sf)
    out = torch.zeros((rows, desc.entry_bytes), dtype=torch.uint8, device=pool.device)
    for row in range(rows):
        slot = int(slots[row])
        if slot < 0:
            continue
        block, offset = divmod(slot, entries)
        begin = offset * desc.entry_bytes
        out[row] = pool[block, begin : begin + desc.entry_bytes]
    return out


def insert_rows(
    pool: torch.Tensor,
    pool_tag: str,
    descriptor: object,
    slots: torch.Tensor,
    packed: torch.Tensor,
):
    """Scatter packed entry rows into a 2D pool by absolute slot id.

    Sentinel (negative) slots are skipped; other pool bytes stay untouched.
    """
    desc = default_descriptor(pool_tag, descriptor)
    entries = desc.block_entries
    if desc.planar:
        payload = packed[:, :64].view(torch.int8)
        sf = _quant._bytes_to_sf(packed[:, 64:].contiguous())
        _quant.store_indexer_k_planar(pool, slots, payload, sf)
        return
    for row in range(packed.shape[0]):
        slot = int(slots[row])
        if slot < 0:
            continue
        block, offset = divmod(slot, entries)
        begin = offset * desc.entry_bytes
        pool[block, begin : begin + desc.entry_bytes] = packed[row]
