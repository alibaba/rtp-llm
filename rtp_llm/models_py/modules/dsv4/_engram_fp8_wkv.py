# SPDX-License-Identifier: Apache-2.0
"""Direct FP8 input to the Engram WKV (eliminate the BF16 intermediate).

DSV41_ENGRAM_WKV_INPUT=baseline|native_fp8 (default baseline). When native_fp8,
Engram._forward_rows gathers the canonical quantized operands (E4M3 payload +
group-32 UE8M0 packed scales) directly from the resident table and feeds the
UNCHANGED native production WKV via V41MXFP8Linear.forward_quantized, instead of
materializing BF16 rows and re-quantizing them.

fused_lookup_quant_rows fuses the baseline-equivalent decode (FP8 x 2^(s-127),
rounded through BF16) with the native v2 quantizer recipe (eps-floored group
amax, fast_log2_ceil power-of-2 scale, clamp +/-448, RTNE-satfinite E4M3) and
emits the packed column-major int32 TMA scale layout directly — bit-exact vs the
baseline lookup + native quantization.

Default off; no production-path change when off.
"""

import os

import torch
import triton
import triton.language as tl

HEADS = 24
DIM = 256
QB = 32
GROUPS_PER_HEAD = DIM // QB  # 8
WORDS_PER_TOKEN = HEADS * DIM // QB // 4  # 48 int32 words per token


def engram_wkv_input_native_fp8() -> bool:
    return os.environ.get("DSV41_ENGRAM_WKV_INPUT", "baseline") == "native_fp8"


def _inv448_f32():
    # fp32 value of 1.0f/448.0f from IEEE fp32 division (matches the C++
    # MAX_8BIT_INV constant bit-exactly). float() of an fp32 value is exact in
    # f64, and Triton passes the scalar back to the kernel as fp32 — no
    # double-rounding hazard.
    return float(
        torch.tensor(1.0, dtype=torch.float32)
        / torch.tensor(448.0, dtype=torch.float32)
    )


# One program = one table row (token, head): 256 payload bytes + 8 scale bytes
# -> 8 group-32 payloads + 2 packed int32 scale words. Fuses the
# baseline-equivalent decode with the native v2 quantization (no BF16
# materialization).
@triton.jit
def _fused_lookup_quant_row_kernel(
    weight,
    scales,
    indices,
    out_q,
    out_s,
    table_rows,
    rows,
    m4,
    ids_stride_t,
    ids_stride_h,
    INV448,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QB: tl.constexpr,
):
    """One program = one table row (token, head): 256 payload + 8 scale bytes ->
    8 group-32 quantized payloads + 2 packed int32 scale words."""
    r = tl.program_id(0)
    if r >= rows:
        return
    t = r // HEADS
    h = r % HEADS
    idx = tl.load(indices + t.to(tl.int64) * ids_stride_t + h * ids_stride_h).to(
        tl.int64
    )
    valid = (idx >= 0) & (idx < table_rows)
    idxc = tl.where(valid, idx, 0)
    cols = tl.arange(0, DIM)
    raw = tl.load(weight + idxc * DIM + cols, mask=valid & (cols >= 0), other=0)
    sc = tl.load(
        scales + idxc * (DIM // QB) + cols // QB, mask=valid & (cols >= 0), other=0
    )
    val = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale_f = (sc.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    v = (val * scale_f).to(tl.bfloat16).to(tl.float32)
    v2 = tl.reshape(v, (DIM // QB, QB))
    # native v2 quantizer recipe (per_token_group_quant_8bit_v2.cu), bit-exact:
    eps = 1.1754943508222875e-38  # torch.finfo(torch.float32).tiny
    amax = tl.maximum(tl.max(tl.abs(v2), axis=1), eps)
    prod = amax * INV448
    bits = prod.to(tl.int32, bitcast=True)
    # the baseline kernel is compiled with FTZ: a denormal product flushes to
    # +0.0 (this is what zero/eps-floor groups produce). Replicate the flush
    # before the bit decomposition so the scale byte matches (byte 0).
    is_denorm = ((bits >> 23) & 0xFF) == 0
    bits = tl.where(is_denorm, 0, bits)
    e = ((bits >> 23) & 0xFF) - 127 + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    qscale = ((-e + 127) << 23).to(tl.float32, bitcast=True)
    q = v2 * qscale[:, None]
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    qf8 = tl.reshape(q, (DIM,)).to(tl.float8e4nv)
    tl.store(out_q + r.to(tl.int64) * DIM + cols, qf8.to(tl.uint8, bitcast=True))
    sbyte = e + 127  # [DIM//QB] int32
    g8 = tl.arange(0, DIM // QB)
    word0 = tl.sum(tl.where(g8 < 4, sbyte << (g8 * 8), 0))
    word1 = tl.sum(tl.where(g8 >= 4, sbyte << ((g8 - 4) * 8), 0))
    tl.store(out_s + (2 * h) * m4 + t, word0)
    tl.store(out_s + (2 * h + 1) * m4 + t, word1)


def fused_lookup_quant_rows(weight_uva, scales_uva, hash_ids, table_rows):
    """Fused gather+decode+quantize. Returns (x_q [tokens,6144] E4M3 contiguous,
    x_s int32 [tokens, 48] strides (1, m4)) — the exact canonical operands the
    native quantizer would have produced from the BF16 lookup output."""
    device = weight_uva.device
    tokens, heads = hash_ids.shape
    assert heads == HEADS
    x_q = torch.empty((tokens, HEADS * DIM), dtype=torch.float8_e4m3fn, device=device)
    m4 = max(1, (tokens + 3) // 4 * 4)
    sbuf = torch.empty((WORDS_PER_TOKEN, m4), dtype=torch.int32, device=device)
    rows = tokens * HEADS
    if rows:
        _fused_lookup_quant_row_kernel[(rows,)](
            weight_uva,
            scales_uva,
            hash_ids,
            x_q.view(torch.uint8),
            sbuf,
            table_rows,
            rows,
            m4,
            hash_ids.stride(0),
            hash_ids.stride(1),
            INV448=_inv448_f32(),
            HEADS=HEADS,
            DIM=DIM,
            QB=QB,
            num_warps=4,
        )
    return x_q, sbuf.transpose(0, 1)[:tokens]


def operands_from_embedding(embedding, hash_ids):
    """Build the canonical quantized operands from the live embedding's table.
    Returns (x_q, x_s) or None when the embedding flavor is unsupported (caller
    falls back to the baseline path). Supports the model's _SharedTableEmbedding
    (UVA views under the table mutex) and the reference HostEngramEmbedding."""
    from rtp_llm.models_py.modules.dsv4.engram import HostEngramEmbedding

    if isinstance(embedding, HostEngramEmbedding):
        if embedding._uva is None:
            return None
        weight_uva, scales_uva = embedding._uva
        table_rows = embedding.weight.shape[0]
        return fused_lookup_quant_rows(weight_uva, scales_uva, hash_ids, table_rows)
    table = getattr(embedding, "_table", None)
    if table is None:
        return None
    with table._mutex:
        table._check_open()
        if hash_ids.device != table._device:
            return None
        weight_uva = table._cuda_weight
        scales_uva = table._cuda_scales
        table_rows = table._weight.shape[0]
        return fused_lookup_quant_rows(weight_uva, scales_uva, hash_ids, table_rows)
