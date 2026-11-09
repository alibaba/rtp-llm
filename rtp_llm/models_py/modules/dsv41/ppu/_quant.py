"""Shared low-level V4.1 codecs for the PPU backend (pure-torch reference).

Every format here is byte-exact against rtp-llm's own codec oracles
(``fp8/test/test_v41_fp4_codec.py`` and ``fp8/test/test_v41_swa_codec.py``):

* GLOBAL 288B: 256B E2M1 payload (low nibble = element 2i) + 32B FP8-E4M3
  scales (group 16, ``e4m3(clamp(amax/6, 2**-9, 448))``; a zero group gets
  scale byte 1), token-interleaved ``[256B payload | 32B scales]``.
* SWA 528B: ``[512B E4M3 payload | 16B UE8M0 scales]`` (group 32,
  ``scale = 2**ceil(log2(amax/448))`` with the amax floored at 1e-4, scale
  byte = exponent + 127), token-interleaved.
* INDEXER 68B: 64B E2M1 payload + 4B packed UE8M0 scales (group 32, MX form
  with rtp-llm's ``_v41_fp4_triton``/``_v41_decode_indexer`` convention: the
  amax is floored at ``6 * 2**-126`` — NOT sglang's 1e-4 — before dividing
  by 6; the four scale bytes pack little-endian into one int32).

Triton fast paths are copied from sglang-for-sail (commit 512b03d,
Apache-2.0) and are only used for accelerator tensors on a PPU host; the
pure-torch paths are the portable ground truth.
"""

# Adapted from sglang-for-sail python/sglang/kernels/ops/quantization/fp4_indexer_quant.py,
# python/sglang/kernels/ops/kvcache/dsv4_fp4_index_k_cache.py and
# python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py (512b03d), Apache-2.0.

from __future__ import annotations

import types
from functools import lru_cache

import torch

# ---------------------------------------------------------------------------
# Format constants (documented contract; tests re-derive them independently).
# ---------------------------------------------------------------------------

KV_DIM = 512
INDEXER_HEAD_DIM = 128

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_MAX = 6.0

E4M3_MAX = 448.0
E4M3_MIN_SCALE = 2.0**-9  # zero-group scale byte is 1
UE8M0_BIAS = 127

FP4_GROUP = 16  # values per E4M3 scale (global pool)
FP8_GROUP = 32  # values per UE8M0 scale (SWA pool)
MX_GROUP = 32  # values per UE8M0 scale (indexer MX FP4)

FP4_GLOBAL_ENTRY_BYTES = 288  # 256B payload + 32B scales
SWA_KV_ENTRY_BYTES = 528  # 512B payload + 16B scales
INDEXER_K_ENTRY_BYTES = 68  # 64B payload + 4B scales

SWA_AMAX_FLOOR = 1e-4
MX_AMAX_FLOOR = 6.0 * (2.0**-126)  # rtp-llm convention (see module docstring)


@lru_cache(maxsize=None)
def _e2m1_codebooks(device: torch.device):
    """Retain immutable codebooks for every actual device used by this process.

    Tensor construction from a Python tuple copies host data. Do that only
    during eager preparation, and keep strong references for every captured
    Graph's lifetime. A cold capture fails before attempting an H2D copy.
    """
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "E2M1 codebooks must be prepared before Graph capture; call "
            "prepare_e2m1_codebooks(device) during eager initialization"
        )
    return (
        torch.tensor(E2M1_MIDPOINTS, dtype=torch.float32, device=device),
        torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=device),
    )


def prepare_e2m1_codebooks(device) -> None:
    """Materialize a device's immutable FP4 lookup tensors before capture."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _e2m1_codebooks(device)


@lru_cache(maxsize=1)
def is_ppu() -> bool:
    """True when torch runs on a T-Head SAIL PPU (device name carries 'ZW').

    Checked at call time; importing this module never needs a device.
    """
    try:
        return torch.cuda.is_available() and "ZW" in torch.cuda.get_device_name()
    except Exception:  # pragma: no cover - defensive
        return False


def _ppu_fast_path(tensor: torch.Tensor) -> bool:
    """True for accelerator tensors when the PPU Triton kernels are usable."""
    if not tensor.is_cuda or not is_ppu():
        return False
    try:
        _triton_kernels()
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# E2M1 (4-bit) element codec.
# ---------------------------------------------------------------------------


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round to E2M1 codes with RNE ties (tie order [0,2,4,6,1,3,5,7]).

    Halfway values round to the even-magnitude code (0.25->0, 0.75->1.0,
    2.5->2.0, 5.0->4.0, ...). Sign is ``signbit(x)`` shifted into bit 3, so
    negative values that round to zero keep the -0.0 code 0x8. NaN inputs map
    to code 0 (matching sglang's reference encoder).
    """
    x = x.float()
    ax = torch.nan_to_num(x.abs(), nan=0.0, posinf=E2M1_MAX).clamp_max(E2M1_MAX)
    mids, _ = _e2m1_codebooks(x.device)
    idx = (ax.unsqueeze(-1) >= mids).sum(-1).to(torch.uint8)
    on_tie = (ax.unsqueeze(-1) == mids).any(-1)
    # A tie counts both neighbours with ``>=``; drop the odd code back down.
    idx = torch.where(on_tie & (idx & 1 == 1), idx - 1, idx)
    sign = torch.signbit(x).to(torch.uint8) << 3
    return idx | sign


def _e2m1_decode_fp32(packed: torch.Tensor) -> torch.Tensor:
    """Packed uint8 nibbles [..., K/2] -> fp32 values [..., K]."""
    packed = packed.view(torch.uint8) if packed.dtype != torch.uint8 else packed
    low = (packed & 0xF).long()
    high = (packed >> 4).long()
    _, levels = _e2m1_codebooks(packed.device)
    # flatten(-2): reshape(..., -1) is ambiguous for 0-element tensors.
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    values = levels[codes & 7]
    return torch.where(codes.ge(8), -values, values)


def encode_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """``x [..., K]`` (even K) -> packed int8 ``[..., K/2]``.

    Low nibble holds element 2i, high nibble element 2i+1.
    """
    codes = _e2m1_codes(x)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.int8)


def decode_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Packed int8/uint8 ``[..., K/2]`` -> bf16 ``[..., K]``."""
    return _e2m1_decode_fp32(packed).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Scale byte codecs.
# ---------------------------------------------------------------------------


def encode_e4m3_scale(amax: torch.Tensor) -> torch.Tensor:
    """Group-16 scale bytes: ``e4m3(clamp(amax/6, 2**-9, 448))`` as uint8.

    A zero group clamps to ``2**-9`` which is E4M3 byte 1. NaN amax keeps the
    NaN scale byte (0x7F), matching the "NaN poisons the tile" convention.
    """
    scale = (amax.float() / E2M1_MAX).clamp(E4M3_MIN_SCALE, E4M3_MAX)
    return scale.to(torch.float8_e4m3fn).view(torch.uint8)


def decode_e4m3_scale(scale_bytes: torch.Tensor) -> torch.Tensor:
    return scale_bytes.view(torch.float8_e4m3fn).float()


def ue8m0_byte_from_scale(s: torch.Tensor) -> torch.Tensor:
    """UE8M0 byte for a positive finite fp32 ``s``: ``ceil(log2(s)) + 127``.

    Computed on the IEEE bits so it is exact at (and just above) powers of
    two: biased exponent, plus one when the mantissa is non-zero.
    """
    s = s.float().contiguous()
    bits = s.view(torch.int32)
    biased = (bits >> 23) & 0xFF
    byte = biased + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    return byte.to(torch.uint8)


def ue8m0_scale_from_byte(scale_bytes: torch.Tensor) -> torch.Tensor:
    """UE8M0 bytes -> fp32 ``2**(byte-127)`` (byte 255 is NaN, as e8m0fnu)."""
    b = scale_bytes.to(torch.int32)
    values = torch.exp2((b - UE8M0_BIAS).float())
    return torch.where(b == 255, torch.full_like(values, float("nan")), values)


def mx_group_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-32 MX FP4 scale (indexer): rtp-llm convention.

    ``amax`` floored at ``6 * 2**-126``; scale = ``2**ceil_bits(amax/6)``
    derived from the fp32 bits. Returns ``(scale fp32, byte uint8)`` where the
    byte is the biased exponent used directly in the packed int32 SF.
    """
    amax = amax.float().clamp_min(MX_AMAX_FLOOR)
    scaled = amax * (1.0 / E2M1_MAX)
    bits = scaled.contiguous().view(torch.int32)
    byte = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    scale = torch.exp2((byte - UE8M0_BIAS).float())
    return scale, byte.to(torch.uint8)


def global_group_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-16 E4M3 scale for the 288B global pool."""
    byte = encode_e4m3_scale(amax)
    return decode_e4m3_scale(byte), byte


def swa_group_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-32 UE8M0 scale for the 528B SWA pool.

    ``amax`` floored at 1e-4, then ``scale = 2**ceil(log2(amax/448))`` and the
    byte is the unbiased exponent + 127 (rtp oracle convention; the floor
    applies to the amax, not to amax/448).
    """
    v = amax.float().clamp_min(SWA_AMAX_FLOOR) / E4M3_MAX
    byte = ue8m0_byte_from_scale(v)
    return ue8m0_scale_from_byte(byte), byte


# ---------------------------------------------------------------------------
# GLOBAL 288B rows (token-interleaved [256B payload | 32B E4M3 scales]).
# ---------------------------------------------------------------------------


def quantize_global_kv_rows(kv: torch.Tensor) -> torch.Tensor:
    """``kv bf16 [N,512]`` -> uint8 ``[N,288]``.

    NaN elements poison their group: NaN E4M3 scale byte and zero codes.
    Preserve the NV cache kernel's FP32 reciprocal followed by multiplication;
    direct division can land on a different E2M1 rounding boundary.
    """
    rows = kv.shape[0]
    x = kv.float().reshape(rows, KV_DIM // FP4_GROUP, FP4_GROUP)
    amax = x.abs().amax(dim=-1)
    scale, scale_bytes = global_group_scale(amax)
    inverse_scale = torch.reciprocal(scale)
    codes = _e2m1_codes(x * inverse_scale.unsqueeze(-1))
    codes = codes.reshape(rows, KV_DIM)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat((payload, scale_bytes), dim=1).contiguous()


def dequantize_global_kv_rows(packed: torch.Tensor, out: torch.Tensor = None):
    """uint8 ``[N,288]`` -> bf16/fp32 ``[N,512]``.

    With ``out`` given, fills the caller-owned tensor (any strides whose last
    dim is contiguous) and returns it unchanged in dtype/strides.
    """
    packed = packed.view(torch.uint8)
    rows = packed.shape[0]
    payload = packed[:, : KV_DIM // 2]
    scales = packed[:, KV_DIM // 2 :]
    values = _e2m1_decode_fp32(payload).reshape(rows, KV_DIM // FP4_GROUP, FP4_GROUP)
    values = values * decode_e4m3_scale(scales).unsqueeze(-1)
    values = values.reshape(rows, KV_DIM)
    if out is None:
        return values.to(torch.bfloat16)
    if tuple(out.shape) != (rows, KV_DIM):
        raise ValueError(f"global out shape {tuple(out.shape)} != {(rows, KV_DIM)}")
    out.copy_(values.to(out.dtype))
    return out


def insert_global_kv_rows(pool: torch.Tensor, slots: torch.Tensor, kv: torch.Tensor):
    """Scatter 288B rows into a 2D pool ``[blocks, entries*288]`` uint8.

    ``slots`` are absolute entry ids (``slot = block*entries + offset``);
    negative slots are skipped (sentinel). Caller owns block-table traversal.
    """
    entries = pool.shape[1] // FP4_GLOBAL_ENTRY_BYTES
    packed = quantize_global_kv_rows(kv)
    for row in range(kv.shape[0]):
        slot = int(slots[row])
        if slot < 0:
            continue
        block, offset = divmod(slot, entries)
        pool[
            block,
            offset * FP4_GLOBAL_ENTRY_BYTES : (offset + 1) * FP4_GLOBAL_ENTRY_BYTES,
        ] = packed[row]


def gather_global_kv_rows(pool: torch.Tensor, slots: torch.Tensor):
    """Gather 288B rows by entry id; sentinel (negative) slots zero-fill."""
    entries = pool.shape[1] // FP4_GLOBAL_ENTRY_BYTES
    rows = slots.shape[0]
    out = torch.zeros(
        (rows, FP4_GLOBAL_ENTRY_BYTES), dtype=torch.uint8, device=pool.device
    )
    for row in range(rows):
        slot = int(slots[row])
        if slot < 0:
            continue
        block, offset = divmod(slot, entries)
        out[row] = pool[
            block,
            offset * FP4_GLOBAL_ENTRY_BYTES : (offset + 1) * FP4_GLOBAL_ENTRY_BYTES,
        ]
    return out


def dequantize_global_kv_slots(
    pool: torch.Tensor, slots: torch.Tensor, out: torch.Tensor = None
):
    """Gather + dequantize 288B pool entries by absolute slot id.

    Sentinel slots (negative) zero-fill, matching rtp's byte-gather contract.
    """
    if (
        out is not None
        and _ppu_fast_path(pool)
        and out.device == pool.device
        and slots.device == pool.device
        and slots.numel()
        and bool((slots >= 0).all())
    ):
        return dequantize_global_kv_slots_ppu(pool, slots, out)
    return dequantize_global_kv_rows(gather_global_kv_rows(pool, slots), out=out)


# ---------------------------------------------------------------------------
# SWA 528B rows (token-interleaved [512B E4M3 payload | 16B UE8M0 scales]).
# ---------------------------------------------------------------------------


def quantize_swa_kv_rows(kv: torch.Tensor) -> torch.Tensor:
    """``kv bf16 [N,512]`` -> uint8 ``[N,528]``."""
    rows = kv.shape[0]
    x = kv.float().reshape(rows, KV_DIM // FP8_GROUP, FP8_GROUP)
    amax = x.abs().amax(dim=-1)
    scale, scale_bytes = swa_group_scale(amax)
    payload = (x / scale.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)
    payload = payload.to(torch.float8_e4m3fn).view(torch.uint8).reshape(rows, KV_DIM)
    return torch.cat((payload, scale_bytes), dim=1).contiguous()


def dequantize_swa_kv_rows(packed: torch.Tensor, out: torch.Tensor = None):
    """uint8 ``[N,528]`` -> bf16 ``[N,512]`` (E4M3 * 2**k is exact in bf16).

    With a strided ``out`` (e.g. a ``[:, :512]`` view of ``[N,528]`` backing)
    only the 512 value columns are written; any stride padding stays
    untouched. Returns ``out`` itself when provided.
    """
    packed = packed.view(torch.uint8)
    rows = packed.shape[0]
    payload = packed[:, :KV_DIM].reshape(rows, KV_DIM // FP8_GROUP, FP8_GROUP)
    scales = packed[:, KV_DIM:]
    values = payload.view(torch.float8_e4m3fn).float()
    values = values * ue8m0_scale_from_byte(scales).unsqueeze(-1)
    values = values.reshape(rows, KV_DIM)
    if out is None:
        if _ppu_fast_path(packed):
            # The paged kernel with one entry per "page" handles plain rows.
            out = torch.empty(
                (rows, KV_DIM), dtype=torch.bfloat16, device=packed.device
            )
            return dequantize_swa_kv_slots_ppu(
                packed, torch.arange(rows, device=packed.device), out
            )
        return values.to(torch.bfloat16)
    if tuple(out.shape) != (rows, KV_DIM):
        raise ValueError(f"swa out shape {tuple(out.shape)} != {(rows, KV_DIM)}")
    out.copy_(values.to(out.dtype))
    return out


# ---------------------------------------------------------------------------
# INDEXER MX FP4 (group-32 UE8M0): Q rows and the 68B planar pool.
# ---------------------------------------------------------------------------


def _quantize_mx_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[R,128]`` -> (payload int8 ``[R,64]``, sf int32 ``[R]``)."""
    rows = x.shape[0]
    grouped = x.float().reshape(rows, INDEXER_HEAD_DIM // MX_GROUP, MX_GROUP)
    scale, scale_bytes = mx_group_scale(grouped.abs().amax(-1))
    codes = _e2m1_codes(grouped / scale.unsqueeze(-1))
    codes = codes.reshape(rows, INDEXER_HEAD_DIM)
    payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.int8)
    sf = (
        scale_bytes[:, 0].to(torch.int32)
        | (scale_bytes[:, 1].to(torch.int32) << 8)
        | (scale_bytes[:, 2].to(torch.int32) << 16)
        | (scale_bytes[:, 3].to(torch.int32) << 24)
    )
    return payload, sf


def quantize_indexer_q(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q bf16 [M,H,128]`` -> (payload int8 ``[M,H,64]``, sf int32 ``[M,H]``).

    Group-32 UE8M0 MX quantization with rtp-llm's convention (amax floored at
    ``6*2**-126``, not sglang's 1e-4; see module docstring). The four scale
    bytes pack little-endian into one int32 per (token, head), group-0 scale
    in the low byte.
    """
    shape = q.shape
    if _ppu_fast_path(q):
        return quantize_indexer_q_ppu(q)
    rows = q.numel() // INDEXER_HEAD_DIM
    payload, sf = _quantize_mx_rows(q.reshape(rows, INDEXER_HEAD_DIM))
    return (
        payload.view(*shape[:-1], INDEXER_HEAD_DIM // 2),
        sf.view(*shape[:-1]),
    )


def _sf_to_bytes(sf: torch.Tensor) -> torch.Tensor:
    """int32 sf ``[R]`` -> uint8 ``[R,4]`` (little-endian, group 0 low byte)."""
    rows = sf.numel()
    return (
        sf.reshape(rows).to(torch.int32).contiguous().view(torch.uint8).reshape(rows, 4)
    )


def _bytes_to_sf(scale_bytes: torch.Tensor) -> torch.Tensor:
    """uint8 ``[R,4]`` little-endian -> int32 sf ``[R]``."""
    rows = scale_bytes.shape[0]
    return (
        scale_bytes.contiguous()
        .view(torch.uint8)
        .reshape(rows, 4)
        .contiguous()
        .view(torch.int32)
        .reshape(rows)
    )


def dequantize_mx_fp4_rows(payload: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Payload int8 ``[...,64]`` + sf int32 ``[...]`` -> fp32 ``[...,128]``.

    Nibble pair (2i, 2i+1) of payload byte i lives in scale group ``i // 16``
    (32 elements per group). Values and scales are powers of two times E2M1
    levels, so the products are exact in fp32/bf16.
    """
    shape = payload.shape
    rows = payload.numel() // (INDEXER_HEAD_DIM // 2)
    flat_payload = payload.reshape(rows, INDEXER_HEAD_DIM // 2)
    scale_bytes = _sf_to_bytes(sf)
    scale = ue8m0_scale_from_byte(scale_bytes)  # [rows, 4]
    values = _e2m1_decode_fp32(flat_payload).reshape(
        rows, INDEXER_HEAD_DIM // MX_GROUP, MX_GROUP
    )
    values = values * scale.unsqueeze(-1)
    return values.reshape(*shape[:-1], INDEXER_HEAD_DIM)


def quantize_indexer_k_rows(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``k bf16 [N,128]`` -> (payload int8 ``[N,64]``, sf int32 ``[N]``)."""
    # Explicit row count: reshape(-1, D) is ambiguous for 0-element tensors.
    return _quantize_mx_rows(k.reshape(k.shape[0], INDEXER_HEAD_DIM))


def indexer_k_packed_rows(payload: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Token-interleaved 68B rows ``[N,68]`` = [64B payload | 4B sf bytes]."""
    payload_u8 = payload.reshape(payload.shape[0], INDEXER_HEAD_DIM // 2).view(
        torch.uint8
    )
    return torch.cat((payload_u8, _sf_to_bytes(sf)), dim=1).contiguous()


def store_indexer_k_planar(
    pool: torch.Tensor, slots: torch.Tensor, payload: torch.Tensor, sf: torch.Tensor
):
    """Scatter 68B entries into the planar indexer pool.

    Pool layout per block (``[blocks, entries*68]`` uint8): a payload plane of
    ``entries*64`` bytes followed by a scale plane of ``entries*4`` bytes.
    Negative (sentinel) slots are skipped.
    """
    entries = pool.shape[1] // INDEXER_K_ENTRY_BYTES
    # Explicit width: reshape(rows, -1) is ambiguous for 0-entry writes.
    payload_u8 = payload.reshape(payload.shape[0], INDEXER_HEAD_DIM // 2).view(
        torch.uint8
    )
    sf_bytes = _sf_to_bytes(sf)
    for row in range(payload.shape[0]):
        slot = int(slots[row])
        if slot < 0:
            continue
        block, offset = divmod(slot, entries)
        pool[block, offset * 64 : (offset + 1) * 64] = payload_u8[row]
        begin = entries * 64 + offset * 4
        pool[block, begin : begin + 4] = sf_bytes[row]


def gather_indexer_k_planar(
    pool: torch.Tensor, slots: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather (payload int8 ``[S,64]``, sf int32 ``[S]``) from the planar pool.

    Sentinel slots zero-fill both outputs without touching the pool.
    """
    entries = pool.shape[1] // INDEXER_K_ENTRY_BYTES
    rows = slots.shape[0]
    payload = torch.zeros(
        (rows, INDEXER_HEAD_DIM // 2), dtype=torch.uint8, device=pool.device
    )
    sf_bytes = torch.zeros((rows, 4), dtype=torch.uint8, device=pool.device)
    for row in range(rows):
        slot = int(slots[row])
        if slot < 0:
            continue
        block, offset = divmod(slot, entries)
        payload[row] = pool[block, offset * 64 : (offset + 1) * 64]
        begin = entries * 64 + offset * 4
        sf_bytes[row] = pool[block, begin : begin + 4]
    return payload.view(torch.int8), _bytes_to_sf(sf_bytes)


# ---------------------------------------------------------------------------
# PPU Triton fast paths (copied from sglang-for-sail, 512b03d, Apache-2.0).
#
# Kernels are defined lazily inside ``_triton_kernels()`` so this module never
# imports triton on a CPU host. Every fast path is guarded by ``is_ppu()`` at
# call time and silently falls back to the pure-torch reference above when
# triton is unavailable; the reference is the byte-exact ground truth.
# ---------------------------------------------------------------------------

_KERNELS: object | None = None


def _triton_kernels():
    """Build (once) and return the copied sglang-for-sail Triton kernels."""
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    ns = types.SimpleNamespace(triton=triton, tl=tl)

    # Copied from fp4_indexer_quant.py; the scale floor is rtp-llm's
    # 6*2**-126 amax clamp (see module docstring), not sglang's 1e-4.
    @triton.jit
    def _ceil_ue8m0_exp(x):
        bits = x.to(tl.int32, bitcast=True)
        exp = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF
        exp += mantissa != 0
        return tl.minimum(tl.maximum(exp, 1), 254)

    @triton.jit
    def _fp4_e2m1_code_rne(x):
        ax = tl.minimum(tl.abs(x), 6.0)
        idx = (ax >= 0.25).to(tl.uint8)
        idx += (ax >= 0.75).to(tl.uint8)
        idx += (ax >= 1.25).to(tl.uint8)
        idx += (ax >= 1.75).to(tl.uint8)
        idx += (ax >= 2.5).to(tl.uint8)
        idx += (ax >= 3.5).to(tl.uint8)
        idx += (ax >= 5.0).to(tl.uint8)
        is_boundary = (
            (ax == 0.25)
            | (ax == 0.75)
            | (ax == 1.25)
            | (ax == 1.75)
            | (ax == 2.5)
            | (ax == 3.5)
            | (ax == 5.0)
        )
        idx = tl.where(is_boundary & ((idx & 1) == 1), idx - 1, idx)
        # Preserve the input sign even when RNE maps its magnitude to zero.
        sign = ((x.to(tl.int32, bitcast=True) >> 31) & 1).to(tl.uint8)
        return idx | (sign << 3)

    @triton.jit
    def _quantize_fp4_indexer_kernel(
        x,
        x_fp4,
        x_sf,
        BLOCK_N: tl.constexpr,
        GROUP_N: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        pair_offsets = tl.arange(0, BLOCK_N // 2)
        row = x + token_id * BLOCK_N
        values = tl.load(row + offs).to(tl.float32)
        abs_values = tl.abs(values)
        amax0 = tl.max(tl.where(offs < GROUP_N, abs_values, 0.0), axis=0)
        amax1 = tl.max(
            tl.where((GROUP_N <= offs) & (offs < 2 * GROUP_N), abs_values, 0.0), axis=0
        )
        amax2 = tl.max(
            tl.where((2 * GROUP_N <= offs) & (offs < 3 * GROUP_N), abs_values, 0.0),
            axis=0,
        )
        amax3 = tl.max(tl.where(3 * GROUP_N <= offs, abs_values, 0.0), axis=0)
        # rtp-llm floor: amax >= 6 * 2**-126 (sglang floors the sf at 1e-4).
        sf0 = tl.maximum(amax0, 6.0 * (2.0**-126)) * (1.0 / 6.0)
        sf1 = tl.maximum(amax1, 6.0 * (2.0**-126)) * (1.0 / 6.0)
        sf2 = tl.maximum(amax2, 6.0 * (2.0**-126)) * (1.0 / 6.0)
        sf3 = tl.maximum(amax3, 6.0 * (2.0**-126)) * (1.0 / 6.0)
        exp0 = _ceil_ue8m0_exp(sf0)
        exp1 = _ceil_ue8m0_exp(sf1)
        exp2 = _ceil_ue8m0_exp(sf2)
        exp3 = _ceil_ue8m0_exp(sf3)
        packed_sf = exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24)
        group0 = pair_offsets * 2 // GROUP_N
        group1 = (pair_offsets * 2 + 1) // GROUP_N
        scale_exp0 = tl.where(
            group0 == 0,
            exp0,
            tl.where(group0 == 1, exp1, tl.where(group0 == 2, exp2, exp3)),
        )
        scale_exp1 = tl.where(
            group1 == 0,
            exp0,
            tl.where(group1 == 1, exp1, tl.where(group1 == 2, exp2, exp3)),
        )
        scale0 = (scale_exp0 << 23).to(tl.float32, bitcast=True)
        scale1 = (scale_exp1 << 23).to(tl.float32, bitcast=True)
        v0 = tl.load(row + pair_offsets * 2).to(tl.float32) / scale0
        v1 = tl.load(row + pair_offsets * 2 + 1).to(tl.float32) / scale1
        code0 = _fp4_e2m1_code_rne(v0)
        code1 = _fp4_e2m1_code_rne(v1)
        packed = (code0 & 0x0F) | ((code1 & 0x0F) << 4)
        tl.store(x_sf + token_id, packed_sf)
        tl.store(x_fp4 + token_id * (BLOCK_N // 2) + pair_offsets, packed)

    ns.quantize_fp4_indexer_kernel = _quantize_fp4_indexer_kernel

    # Copied from dsv4_fp4_index_k_cache.py.
    @triton.jit
    def _store_fp4_index_k_cache_kernel(
        k_fp4,
        k_sf,
        cache,
        loc,
        page_size: tl.constexpr,
        cache_stride: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        cache_loc = tl.load(loc + token_id)
        page = cache_loc // page_size
        page_offset = cache_loc - page * page_size
        k = tl.load(k_fp4 + token_id * BLOCK + offsets)
        tl.store(cache + page * cache_stride + page_offset * BLOCK + offsets, k)
        sf = tl.load(k_sf + token_id)
        sf_offsets = tl.arange(0, 4)
        sf_bytes = (sf >> (sf_offsets * 8)) & 0xFF
        tl.store(
            cache
            + page * cache_stride
            + page_size * BLOCK
            + page_offset * 4
            + sf_offsets,
            sf_bytes,
        )

    ns.store_fp4_index_k_cache_kernel = _store_fp4_index_k_cache_kernel

    @triton.jit
    def _gather_fp4_index_k_kernel(
        cache,
        cache_stride,
        slots,
        payload,
        scales,
        n,
        page_size: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid = rows < n
        slot = tl.load(slots + rows, valid, 0).to(tl.int64)
        page = slot // page_size
        offset = slot - page * page_size
        page_base = cache + page * cache_stride
        words = tl.arange(0, 16)
        payload_words = tl.load(
            page_base[:, None] + offset[:, None] * 16 + words[None, :], valid[:, None]
        )
        tl.store(
            payload + rows[:, None].to(tl.int64) * 16 + words[None, :],
            payload_words,
            valid[:, None],
        )
        tl.store(
            scales + rows, tl.load(page_base + page_size * 16 + offset, valid), valid
        )

    ns.gather_fp4_index_k_kernel = _gather_fp4_index_k_kernel

    # Copied from dequant_k_cache.py (V41 token-interleaved layouts only).
    @triton.jit
    def _ue8m0_to_fp32(scale_u8):
        normal = (scale_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        denormal = tl.full(scale_u8.shape, 0x00400000, tl.int32).to(
            tl.float32, bitcast=True
        )
        scale = tl.where(scale_u8 == 0, denormal, normal)
        return tl.where(scale_u8 == 255, float("nan"), scale)

    ns.ue8m0_to_fp32 = _ue8m0_to_fp32

    @triton.jit
    def _pack_v41_swa_row(kv, payload, packed):
        row = tl.program_id(0).to(tl.int64)
        column = tl.arange(0, 512)
        values = tl.load(kv + row * 512 + column).to(tl.float32).reshape((16, 32))
        maximum = tl.max(tl.abs(values), 1)
        poison = tl.sum((values != values).to(tl.int32), 1) > 0
        maximum = tl.where(poison, float("nan"), tl.maximum(maximum, 1.0e-4))
        bits = (maximum * (1.0 / 448.0)).to(tl.int32, bitcast=True)
        exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)).to(tl.uint8)
        normalized = values / _ue8m0_to_fp32(exponent)[:, None]
        clamped = tl.where(normalized == normalized, tl.clamp(normalized, -448.0, 448.0), normalized)
        tl.store(payload + row * 528 + column, clamped.reshape((512,)))
        tl.store(packed + row * 528 + 512 + tl.arange(0, 16), exponent)

    @triton.jit
    def _pack_v41_rotary_fp4_row(
        kv, frequencies, positions, packed, fp8, position_stride, max_position,
        DIM: tl.constexpr, GROUP: tl.constexpr, HALF_ROPE: tl.constexpr,
        GLOBAL: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        pair = tl.arange(0, DIM // 2)
        token = tl.minimum(tl.load(positions + row * position_stride), max_position)
        frequency = pair - (DIM // 2 - HALF_ROPE)
        rotates = frequency >= 0
        angle = frequencies + token * (2 * HALF_ROPE) + 2 * frequency
        cosine = tl.load(angle, rotates, other=1.0)
        sine = tl.load(angle + 1, rotates, other=0.0)
        real = tl.load(kv + row * DIM + 2 * pair).to(tl.float32)
        imag = tl.load(kv + row * DIM + 2 * pair + 1).to(tl.float32)
        # Preserve the PPU complex multiply and round to BF16 before quantizing.
        rotated_real = tl.fma(real, cosine, -(imag * sine))
        rotated_imag = tl.fma(imag, cosine, real * sine)
        real = tl.where(rotates, rotated_real, real).to(tl.bfloat16).to(tl.float32)
        imag = tl.where(rotates, rotated_imag, imag).to(tl.bfloat16).to(tl.float32)
        real = real.reshape((DIM // GROUP, GROUP // 2))
        imag = imag.reshape((DIM // GROUP, GROUP // 2))
        maximum = tl.max(tl.maximum(tl.abs(real), tl.abs(imag)), 1)
        poison = tl.sum(((real != real) | (imag != imag)).to(tl.int32), 1) > 0
        maximum = tl.where(poison, float("nan"), maximum)
        base = row * (DIM // 2 + DIM // GROUP)
        groups = tl.arange(0, DIM // GROUP)
        if GLOBAL:
            scale = maximum * (1.0 / 6.0)
            scale = tl.where(poison, scale, tl.clamp(scale, 2.0**-9, 448.0))
            encoded = scale.to(fp8.dtype.element_ty, fp_downcast_rounding="rtne")
            inverse = tl.div_rn(1.0, encoded.to(tl.float32))
            real *= inverse[:, None]
            imag *= inverse[:, None]
            tl.store(fp8 + base + DIM // 2 + groups, encoded)
        else:
            scaled = tl.maximum(maximum, 6.0 * (2.0**-126)) * (1.0 / 6.0)
            bits = scaled.to(tl.int32, bitcast=True)
            exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)
            scale = tl.exp2((exponent - 127).to(tl.float32))
            real /= scale[:, None]
            imag /= scale[:, None]
            tl.store(packed + base + DIM // 2 + groups, exponent.to(tl.uint8))
        low = _fp4_e2m1_code_rne(real)
        high = _fp4_e2m1_code_rne(imag)
        low = tl.where(real == real, low, ((real.to(tl.int32, bitcast=True) >> 28) & 8).to(tl.uint8))
        high = tl.where(imag == imag, high, ((imag.to(tl.int32, bitcast=True) >> 28) & 8).to(tl.uint8))
        tl.store(packed + base + pair, (low | (high << 4)).reshape((DIM // 2,)))

    ns.pack_v41_swa_row = _pack_v41_swa_row
    ns.pack_v41_rotary_fp4_row = _pack_v41_rotary_fp4_row

    @triton.jit
    def _e2m1_code_to_fp32(code):
        m = code & 7
        e = m >> 1
        f = (m & 1).to(tl.float32)
        mag = tl.where(
            e == 0, 0.5 * f, tl.exp2((e - 1).to(tl.float32)) * (1.0 + 0.5 * f)
        )
        sign = (code & 8).to(tl.int32) << 28
        return (mag.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)

    ns.e2m1_code_to_fp32 = _e2m1_code_to_fp32

    @triton.jit
    def _dequantize_k_cache_paged_v41_fp4_kernel(
        output_ptr,
        buf_uint8_ptr,
        buf_fp8_ptr,
        page_table_ptr,
        output_stride_0,
        BYTES_PER_PAGE: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DATA_BYTES: tl.constexpr,
        SCALE_BYTES: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        DATA_STRIDE: tl.constexpr,
        SCALE_BASE: tl.constexpr,
        SCALE_STRIDE: tl.constexpr,
    ):
        token_id = tl.program_id(0).to(tl.int64)
        if page_table_ptr is None:
            loc = token_id
        else:
            loc = tl.load(page_table_ptr + token_id).to(tl.int64)
        page_idx = loc // PAGE_SIZE
        in_page = loc % PAGE_SIZE
        page_byte_base = page_idx * BYTES_PER_PAGE
        token_data_base = page_byte_base + in_page * DATA_STRIDE
        token_scale_base = page_byte_base + SCALE_BASE + in_page * SCALE_STRIDE
        boffs = tl.arange(0, DATA_BYTES)
        packed = tl.load(buf_uint8_ptr + token_data_base + boffs)
        scale = tl.load(buf_fp8_ptr + token_scale_base + (2 * boffs) // TILE_SIZE).to(
            tl.float32
        )
        lo = _e2m1_code_to_fp32(packed & 0xF) * scale
        hi = _e2m1_code_to_fp32(packed >> 4) * scale
        out_base = output_ptr + token_id * output_stride_0
        tl.store(out_base + 2 * boffs, lo.to(tl.bfloat16))
        tl.store(out_base + 2 * boffs + 1, hi.to(tl.bfloat16))

    ns.dequant_v41_fp4_kernel = _dequantize_k_cache_paged_v41_fp4_kernel

    @triton.jit
    def _dequantize_k_cache_paged_v41_fp8_kernel(
        output_ptr,
        buf_fp8_ptr,
        buf_uint8_ptr,
        page_table_ptr,
        output_stride_0,
        BYTES_PER_PAGE: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DATA_BYTES: tl.constexpr,
        SCALE_BYTES: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        DATA_STRIDE: tl.constexpr,
        SCALE_BASE: tl.constexpr,
        SCALE_STRIDE: tl.constexpr,
    ):
        token_id = tl.program_id(0).to(tl.int64)
        if page_table_ptr is None:
            loc = token_id
        else:
            loc = tl.load(page_table_ptr + token_id).to(tl.int64)
        page_idx = loc // PAGE_SIZE
        in_page = loc % PAGE_SIZE
        page_byte_base = page_idx * BYTES_PER_PAGE
        token_data_base = page_byte_base + in_page * DATA_STRIDE
        token_scale_base = page_byte_base + SCALE_BASE + in_page * SCALE_STRIDE
        offs = tl.arange(0, DATA_BYTES)
        vals = tl.load(buf_fp8_ptr + token_data_base + offs).to(tl.float32)
        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + offs // TILE_SIZE)
        out = vals * _ue8m0_to_fp32(scale_u8)
        tl.store(output_ptr + token_id * output_stride_0 + offs, out.to(tl.bfloat16))

    ns.dequant_v41_fp8_kernel = _dequantize_k_cache_paged_v41_fp8_kernel

    @triton.jit
    def _dequantize_v41_swa_segments_kernel(
        tail, fresh, out, start, tail_rows, out_stride,
        WIDTH: tl.constexpr, DIM: tl.constexpr, GROUP: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        slot = row + start
        if slot < tail_rows:
            base = tail + slot * WIDTH
        else:
            base = fresh + (slot - tail_rows) * WIDTH
        column = tl.arange(0, DIM)
        value = tl.load(base + column).to(tl.float32)
        scale = tl.load(base.to(tl.pointer_type(tl.uint8)) + DIM + column // GROUP)
        tl.store(out + row * out_stride + column, (value * _ue8m0_to_fp32(scale)).to(tl.bfloat16))

    ns.dequant_v41_swa_segments = _dequantize_v41_swa_segments_kernel

    _KERNELS = ns
    return ns


def _quantize_mx_rows_ppu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PPU fast path for ``_quantize_mx_rows`` (sglang indexer quant kernel)."""
    ns = _triton_kernels()
    rows = x.shape[0]
    payload = torch.empty((rows, 64), dtype=torch.int8, device=x.device)
    sf = torch.empty((rows,), dtype=torch.int32, device=x.device)
    if rows:
        ns.quantize_fp4_indexer_kernel[(rows,)](
            x.contiguous(),
            payload,
            sf,
            BLOCK_N=128,
            GROUP_N=32,
        )
    return payload, sf


def quantize_indexer_q_ppu(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = q.shape
    rows = q.numel() // INDEXER_HEAD_DIM
    payload, sf = _quantize_mx_rows_ppu(
        q.reshape(rows, INDEXER_HEAD_DIM).float().contiguous()
    )
    return payload.view(*shape[:-1], 64), sf.view(*shape[:-1])


def dequantize_global_kv_slots_ppu(
    pool: torch.Tensor, slots: torch.Tensor | None, out: torch.Tensor
) -> torch.Tensor:
    """PPU fast path for ``dequantize_global_kv_slots`` (288B FP4 pool)."""
    ns = _triton_kernels()
    entries = pool.shape[1] // FP4_GLOBAL_ENTRY_BYTES
    buf_u8 = pool.reshape(-1)
    buf_fp8 = pool.view(torch.float8_e4m3fn).reshape(-1)
    count = pool.shape[0] * entries if slots is None else slots.numel()
    ns.dequant_v41_fp4_kernel[(count,)](
        out,
        buf_u8,
        buf_fp8,
        None if slots is None else slots.to(torch.int64),
        out.stride(0),
        BYTES_PER_PAGE=pool.shape[1],
        PAGE_SIZE=entries,
        DATA_BYTES=KV_DIM // 2,
        SCALE_BYTES=KV_DIM // FP4_GROUP,
        TILE_SIZE=FP4_GROUP,
        DATA_STRIDE=FP4_GLOBAL_ENTRY_BYTES,
        SCALE_BASE=KV_DIM // 2,
        SCALE_STRIDE=FP4_GLOBAL_ENTRY_BYTES,
    )
    return out


def dequantize_swa_kv_slots_ppu(
    pool: torch.Tensor, slots: torch.Tensor | None, out: torch.Tensor
) -> torch.Tensor:
    """PPU fast path for SWA 528B FP8 pool dequant (strided ``out`` ok)."""
    ns = _triton_kernels()
    entries = pool.shape[1] // SWA_KV_ENTRY_BYTES
    buf_fp8 = pool.view(torch.float8_e4m3fn).reshape(-1)
    buf_u8 = pool.reshape(-1)
    count = pool.shape[0] * entries if slots is None else slots.numel()
    ns.dequant_v41_fp8_kernel[(count,)](
        out,
        buf_fp8,
        buf_u8,
        None if slots is None else slots.to(torch.int64),
        out.stride(0),
        BYTES_PER_PAGE=pool.shape[1],
        PAGE_SIZE=entries,
        DATA_BYTES=KV_DIM,
        SCALE_BYTES=KV_DIM // FP8_GROUP,
        TILE_SIZE=FP8_GROUP,
        DATA_STRIDE=SWA_KV_ENTRY_BYTES,
        SCALE_BASE=KV_DIM,
        SCALE_STRIDE=SWA_KV_ENTRY_BYTES,
    )
    return out


def try_dequantize_prefill_rows(packed, out, *, global_pool):
    """Decode dense 288/528-byte rows into the existing BF16 workspace."""
    width = FP4_GLOBAL_ENTRY_BYTES if global_pool else SWA_KV_ENTRY_BYTES
    if not (
        packed.dtype == torch.uint8
        and packed.ndim == 2
        and packed.shape[1] == width
        and packed.is_contiguous()
        and out.device == packed.device
        and out.dtype == torch.bfloat16
        and out.shape == (packed.shape[0], KV_DIM)
        and out.stride(1) == 1
        and _ppu_fast_path(packed)
    ):
        return False
    if packed.shape[0]:
        decode = dequantize_global_kv_slots_ppu if global_pool else dequantize_swa_kv_slots_ppu
        decode(packed, None, out)
    return True


def try_dequantize_swa_segments(tail, fresh, start, out):
    """Write a covered interval of the two packed SWA segments into BF16."""
    if not (
        out.ndim == 2
        and out.shape[1] == KV_DIM
        and out.dtype == torch.bfloat16
        and out.stride(1) == 1
        and all(
            rows.ndim == 2
            and rows.shape[1] == SWA_KV_ENTRY_BYTES
            and rows.dtype == torch.uint8
            and rows.is_contiguous()
            and rows.device == out.device
            for rows in (tail, fresh)
        )
        and 0 <= start <= tail.shape[0] + fresh.shape[0] - out.shape[0]
        and _ppu_fast_path(fresh)
    ):
        return False
    if out.shape[0]:
        _triton_kernels().dequant_v41_swa_segments[(out.shape[0],)](
            tail.view(torch.float8_e4m3fn), fresh.view(torch.float8_e4m3fn),
            out, start, tail.shape[0], out.stride(0),
            SWA_KV_ENTRY_BYTES, KV_DIM, FP8_GROUP,
        )
    return True


def try_pack_swa_rows(kv):
    """Pack BF16 SWA rows without materializing scale and payload tensors."""
    if not (
        kv.ndim == 2 and kv.shape[1] == KV_DIM and kv.dtype == torch.bfloat16
        and kv.is_contiguous() and _ppu_fast_path(kv)
    ):
        return None
    packed = torch.empty((kv.shape[0], SWA_KV_ENTRY_BYTES), dtype=torch.uint8, device=kv.device)
    if kv.shape[0]:
        _triton_kernels().pack_v41_swa_row[(kv.shape[0],)](
            kv, packed.view(torch.float8_e4m3fn), packed, enable_fp_fusion=False,
        )
    return packed


def try_pack_rotary_fp4_rows(kv, freqs, positions, *, global_pool):
    """Fuse the producer's indexed RoPE and native interleaved FP4 codec."""
    dim = KV_DIM if global_pool else INDEXER_HEAD_DIM
    group = FP4_GROUP if global_pool else MX_GROUP
    if not (
        kv.ndim == 2 and kv.shape[1] == dim and kv.dtype == torch.bfloat16 and kv.is_contiguous()
        and positions.shape == (kv.shape[0],) and positions.dtype == torch.int64
        and freqs.ndim == 2 and freqs.shape[0] > 0 and 0 < freqs.shape[1] <= dim // 2
        and freqs.dtype == torch.complex64 and freqs.is_contiguous() and not freqs.is_conj()
        and kv.device == freqs.device == positions.device and _ppu_fast_path(kv)
    ):
        return None
    packed = torch.empty((kv.shape[0], dim // 2 + dim // group), dtype=torch.uint8, device=kv.device)
    if kv.shape[0]:
        _triton_kernels().pack_v41_rotary_fp4_row[(kv.shape[0],)](
            kv, torch.view_as_real(freqs), positions, packed, packed.view(torch.float8_e4m3fn),
            positions.stride(0), freqs.shape[0] - 1, dim, group, freqs.shape[1], global_pool,
            enable_fp_fusion=False,
        )
    return packed
