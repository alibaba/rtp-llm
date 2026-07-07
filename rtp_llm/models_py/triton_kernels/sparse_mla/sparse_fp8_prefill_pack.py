"""Triton kernels for GLM5 sparse-FP8 prefill packing.

Two entry points used by ``SparseMlaFp8Op._forward_gather`` /
``SparseMlaFp8CPOp._attend_gather`` when
``RTP_LLM_GLM5_SPARSE_ATTN_DTYPE=fp8``:

  - ``paged_fp8_gather_pack``: consumes the paged FP8 cache directly (per-token
    layout: 4x128 fp8 nope + 4 fp32 per-group scales + 64 bf16 rope = 656B),
    gathers the requested tokens into a ragged workspace, and repacks the fp8
    values to a single per-tensor scale (max of all gathered group scales).
    Replaces ``cp_gather_and_upconvert_fp8_kv_cache_v2`` + BF16 workspace +
    Python bf16->fp8 repack, cutting one 576-dim BF16 round-trip.

  - ``pack_q_656``: takes ``[s_q, h_q, 576]`` bf16 Q and emits ``[s_q, h_q, 656]``
    uint8 with fp8 nope + bf16 rope pre-divided by ``qk_scale``. Q scale is a
    per-tensor amax computed via torch (well-optimized reduction).

Layout on disk (matches ``fused_qk_rope_cat_cache_mla_fp8_kernel`` write side):
  bytes [0:512]    4 x 128 fp8_e4m3fn nope (per-group quantized)
  bytes [512:528]  4 x fp32 per-group scales
  bytes [528:656]  64 x bf16 rope (unchanged)
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

BYTES_PER_TOKEN = 656
KV_LORA = 512
NUM_GROUPS = 4
GROUP_SIZE = 128
ROPE_DIM_BF16 = 64  # 128 bytes / 2

FP8_MAX = 448.0
SCALE_FLOOR = 1e-6

# fp32 offset (in fp32 elements) of the per-group scale block inside a slot.
SCALE_OFFSET_FP32 = KV_LORA // 4  # 128
# bf16 offset (in bf16 elements) of the rope block inside a slot.
ROPE_OFFSET_BF16 = (KV_LORA + 16) // 2  # 264


# ---------------------------------------------------------------------------
# KV path: paged FP8 -> per-tensor FP8 ragged workspace
# ---------------------------------------------------------------------------


@triton.jit
def _reduce_scales_kernel(
    PAGED_FP32,  # fp32 view of paged cache: [nb, bs, 164]
    TOKEN_BATCH,  # int32 [total_kv_len]
    WORKSPACE_STARTS,  # int32 [batch_size]
    BLOCK_TABLE,  # int32 [batch_size, max_blocks_per_seq]
    K_SCALE_OUT,  # fp32 [1] atomic max target
    total_kv_len,
    stride_paged_page,
    stride_paged_slot,
    stride_bt_batch,
    TOKENS_PER_BLOCK: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    NUM_GROUPS_C: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """One program per BLOCK_TOKENS-sized tile; one atomic_max per program."""
    pid = tl.program_id(0).to(tl.int64)
    t = pid * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    mask = t < total_kv_len

    b = tl.load(TOKEN_BATCH + t, mask=mask, other=0)
    ws_start = tl.load(WORKSPACE_STARTS + b, mask=mask, other=0)
    local = t - ws_start
    block_row = local // TOKENS_PER_BLOCK
    slot = local % TOKENS_PER_BLOCK
    block_id = tl.load(
        BLOCK_TABLE + b * stride_bt_batch + block_row, mask=mask, other=0
    )

    g_off = tl.arange(0, NUM_GROUPS_C)
    src = (
        block_id[:, None] * stride_paged_page
        + slot[:, None] * stride_paged_slot
        + SCALE_OFFSET
        + g_off[None, :]
    )
    scales = tl.load(PAGED_FP32 + src, mask=mask[:, None], other=0.0)
    tl.atomic_max(K_SCALE_OUT, tl.max(scales))


@triton.jit
def _gather_repack_kernel(
    PAGED_FP8,  # fp8_e4m3fn view: [nb, bs, 656]
    PAGED_FP32,  # fp32 view: [nb, bs, 164]
    PAGED_BF16,  # bf16 view: [nb, bs, 328]
    TOKEN_BATCH,  # int32 [total_kv_len]
    WORKSPACE_STARTS,  # int32 [batch_size]
    BLOCK_TABLE,  # int32 [batch_size, max_blocks_per_seq]
    K_SCALE_DEV,  # fp32 [1] (max group scale, from reduce kernel)
    OUT_FP8,  # fp8_e4m3fn view of output: [total_kv_len, 656]
    OUT_BF16,  # bf16 view of output: [total_kv_len, 328]
    total_kv_len,
    stride_paged_fp8_page,
    stride_paged_fp8_slot,
    stride_paged_fp32_page,
    stride_paged_fp32_slot,
    stride_paged_bf16_page,
    stride_paged_bf16_slot,
    stride_out_fp8_row,
    stride_out_bf16_row,
    stride_bt_batch,
    TOKENS_PER_BLOCK: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS_C: tl.constexpr,
    ROPE_DIM_BF16_C: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
):
    """One program per output token: read fp8+scales+rope, rescale fp8, write.

    ratio = group_scale / k_scale ∈ (0, 1] so re-cast to fp8 cannot overflow.
    """
    t = tl.program_id(0).to(tl.int64)
    if t >= total_kv_len:
        return

    b = tl.load(TOKEN_BATCH + t)
    ws_start = tl.load(WORKSPACE_STARTS + b)
    local = t - ws_start
    block_row = local // TOKENS_PER_BLOCK
    slot = local % TOKENS_PER_BLOCK
    block_id = tl.load(BLOCK_TABLE + b * stride_bt_batch + block_row)

    k_scale = tl.load(K_SCALE_DEV)

    fp8_src = block_id * stride_paged_fp8_page + slot * stride_paged_fp8_slot
    fp32_src = (
        block_id * stride_paged_fp32_page + slot * stride_paged_fp32_slot + SCALE_OFFSET
    )
    bf16_src = (
        block_id * stride_paged_bf16_page + slot * stride_paged_bf16_slot + ROPE_OFFSET
    )
    fp8_dst = t * stride_out_fp8_row
    bf16_dst = t * stride_out_bf16_row + ROPE_OFFSET

    elem = tl.arange(0, GROUP_SIZE_C)
    for g in tl.static_range(NUM_GROUPS_C):
        scale_g = tl.load(PAGED_FP32 + fp32_src + g)
        ratio = scale_g / k_scale
        fp8_vals = tl.load(PAGED_FP8 + fp8_src + g * GROUP_SIZE_C + elem)
        rescaled = fp8_vals.to(tl.float32) * ratio
        tl.store(
            OUT_FP8 + fp8_dst + g * GROUP_SIZE_C + elem,
            rescaled.to(tl.float8e4nv),
        )

    r_off = tl.arange(0, ROPE_DIM_BF16_C)
    rope = tl.load(PAGED_BF16 + bf16_src + r_off)
    tl.store(OUT_BF16 + bf16_dst + r_off, rope)


def paged_fp8_gather_pack(
    paged_u8: torch.Tensor,
    block_table: torch.Tensor,
    workspace_starts: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    total_kv_len: int,
    tokens_per_block: int,
    output_u8: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Gather+repack the paged FP8 cache into a per-tensor FP8 ragged buffer.

    Args:
      paged_u8:         [num_blocks, block_size, 656] uint8, contiguous.
      block_table:      [batch_size, max_blocks_per_seq] int32/int64.
      workspace_starts: [batch_size] int (indptr[:-1] into the ragged output).
      seq_lens:         [batch_size] int (indptr diff).
      batch_size, total_kv_len, tokens_per_block: scalars matching the above.
      output_u8:        Optional pre-allocated [total_kv_len, 656] uint8 buffer
                        to write into (per-forward workspace reused across
                        layers). Allocates a fresh buffer if None.

    Returns:
      packed_u8: [total_kv_len, 656] uint8 (fp8 nope rescaled to per-tensor,
                 4B*4 pad at [512:528] is zeroed, rope bf16 copied from source).
      k_scale:   Python float, the per-tensor scale (max over all gathered
                 per-group scales, floored at SCALE_FLOOR).
    """
    assert paged_u8.dtype == torch.uint8, f"paged dtype {paged_u8.dtype}"
    assert (
        paged_u8.shape[-1] == BYTES_PER_TOKEN
    ), f"paged last dim {paged_u8.shape[-1]} != {BYTES_PER_TOKEN}"
    assert paged_u8.is_contiguous(), "paged cache must be contiguous"

    device = paged_u8.device

    # Same buffer viewed as fp8/fp32/bf16 — Triton loads through matching dtype
    # pointers instead of doing bitcasts inside the kernel.
    paged_fp8 = paged_u8.view(torch.float8_e4m3fn)
    paged_fp32 = paged_u8.view(torch.float32)
    paged_bf16 = paged_u8.view(torch.bfloat16)

    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if workspace_starts.dtype != torch.int32:
        workspace_starts = workspace_starts.to(torch.int32)
    seq_lens_i32 = (
        seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
    )

    token_batch = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=device),
        seq_lens_i32,
    )

    if output_u8 is None:
        output_u8 = torch.empty(
            (total_kv_len, BYTES_PER_TOKEN), dtype=torch.uint8, device=device
        )
    else:
        assert output_u8.dtype == torch.uint8
        assert output_u8.shape == (
            total_kv_len,
            BYTES_PER_TOKEN,
        ), f"output_u8 shape {tuple(output_u8.shape)} != ({total_kv_len}, {BYTES_PER_TOKEN})"
        assert output_u8.is_contiguous(), "output_u8 must be contiguous"
    # Zero the 16B pad region [512:528] once; kernel doesn't touch it.
    output_u8[:, KV_LORA : KV_LORA + 16].zero_()

    output_fp8 = output_u8.view(torch.float8_e4m3fn)
    output_bf16 = output_u8.view(torch.bfloat16)

    k_scale_dev = torch.zeros(1, dtype=torch.float32, device=device)

    if total_kv_len == 0:
        return output_u8, SCALE_FLOOR

    reduce_block = 128
    reduce_grid = (triton.cdiv(total_kv_len, reduce_block),)
    _reduce_scales_kernel[reduce_grid](
        paged_fp32,
        token_batch,
        workspace_starts,
        block_table,
        k_scale_dev,
        total_kv_len,
        paged_fp32.stride(0),
        paged_fp32.stride(1),
        block_table.stride(0),
        TOKENS_PER_BLOCK=tokens_per_block,
        SCALE_OFFSET=SCALE_OFFSET_FP32,
        NUM_GROUPS_C=NUM_GROUPS,
        BLOCK_TOKENS=reduce_block,
        num_warps=2,
    )

    gather_grid = (total_kv_len,)
    _gather_repack_kernel[gather_grid](
        paged_fp8,
        paged_fp32,
        paged_bf16,
        token_batch,
        workspace_starts,
        block_table,
        k_scale_dev,
        output_fp8,
        output_bf16,
        total_kv_len,
        paged_fp8.stride(0),
        paged_fp8.stride(1),
        paged_fp32.stride(0),
        paged_fp32.stride(1),
        paged_bf16.stride(0),
        paged_bf16.stride(1),
        output_fp8.stride(0),
        output_bf16.stride(0),
        block_table.stride(0),
        TOKENS_PER_BLOCK=tokens_per_block,
        GROUP_SIZE_C=GROUP_SIZE,
        NUM_GROUPS_C=NUM_GROUPS,
        ROPE_DIM_BF16_C=ROPE_DIM_BF16,
        SCALE_OFFSET=SCALE_OFFSET_FP32,
        ROPE_OFFSET=ROPE_OFFSET_BF16,
        num_warps=4,
    )

    k_scale = max(k_scale_dev.item(), SCALE_FLOOR)
    return output_u8, k_scale


# ---------------------------------------------------------------------------
# Q path: bf16 [s_q, h_q, 576] -> per-tensor FP8 [s_q, h_q, 656]
# ---------------------------------------------------------------------------


@triton.jit
def _pack_q_kernel(
    Q_BF16,
    Q_SCALE_DEV,  # fp32 [1] (max|q_nope|/448, precomputed)
    K_SCALE_DEV,  # fp32 [1] (from KV pack)
    OUT_FP8,
    OUT_BF16,
    stride_q_sq,
    stride_q_hq,
    stride_out_fp8_sq,
    stride_out_fp8_hq,
    stride_out_bf16_sq,
    stride_out_bf16_hq,
    KV_LORA_C: tl.constexpr,
    ROPE_DIM_BF16_C: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
):
    """Grid: (s_q, h_q). Per (token, head) quantize nope + pre-divide rope."""
    sq = tl.program_id(0).to(tl.int64)
    hq = tl.program_id(1)

    q_scale = tl.load(Q_SCALE_DEV)
    k_scale = tl.load(K_SCALE_DEV)
    qk_scale = q_scale * k_scale

    q_base = sq * stride_q_sq + hq * stride_q_hq
    out_fp8_base = sq * stride_out_fp8_sq + hq * stride_out_fp8_hq
    out_bf16_base = sq * stride_out_bf16_sq + hq * stride_out_bf16_hq + ROPE_OFFSET

    n_off = tl.arange(0, KV_LORA_C)
    nope = tl.load(Q_BF16 + q_base + n_off).to(tl.float32)
    tl.store(OUT_FP8 + out_fp8_base + n_off, (nope / q_scale).to(tl.float8e4nv))

    r_off = tl.arange(0, ROPE_DIM_BF16_C)
    rope = tl.load(Q_BF16 + q_base + KV_LORA_C + r_off).to(tl.float32)
    tl.store(OUT_BF16 + out_bf16_base + r_off, (rope / qk_scale).to(tl.bfloat16))


def pack_q_656(q_bf16: torch.Tensor, k_scale: float) -> Tuple[torch.Tensor, float]:
    """Pack ``q_bf16`` ``[s_q, h_q, 576]`` into the 656B per-tensor FP8 layout.

    q_scale is computed by torch amax on the nope slice (fast, well-optimized);
    the Triton kernel only does the per-(token,head) pack + rope pre-divide.
    """
    assert q_bf16.dtype == torch.bfloat16, f"q dtype {q_bf16.dtype}"
    assert (
        q_bf16.shape[-1] == KV_LORA + ROPE_DIM_BF16
    ), f"q last dim {q_bf16.shape[-1]} != {KV_LORA + ROPE_DIM_BF16}"
    assert q_bf16.stride(-1) == 1, "q last dim must be unit-strided"

    s_q, h_q, _ = q_bf16.shape
    device = q_bf16.device

    q_scale_dev = (
        (q_bf16[..., :KV_LORA].abs().amax().to(torch.float32) / FP8_MAX)
        .clamp_min(SCALE_FLOOR)
        .reshape(1)
    )
    k_scale_dev = torch.tensor([k_scale], dtype=torch.float32, device=device)

    output_u8 = torch.empty(
        (s_q, h_q, BYTES_PER_TOKEN), dtype=torch.uint8, device=device
    )
    output_u8[:, :, KV_LORA : KV_LORA + 16].zero_()

    output_fp8 = output_u8.view(torch.float8_e4m3fn)
    output_bf16 = output_u8.view(torch.bfloat16)

    if s_q == 0 or h_q == 0:
        return output_u8, q_scale_dev.item()

    grid = (s_q, h_q)
    _pack_q_kernel[grid](
        q_bf16,
        q_scale_dev,
        k_scale_dev,
        output_fp8,
        output_bf16,
        q_bf16.stride(0),
        q_bf16.stride(1),
        output_fp8.stride(0),
        output_fp8.stride(1),
        output_bf16.stride(0),
        output_bf16.stride(1),
        KV_LORA_C=KV_LORA,
        ROPE_DIM_BF16_C=ROPE_DIM_BF16,
        ROPE_OFFSET=ROPE_OFFSET_BF16,
        num_warps=4,
    )

    q_scale = max(q_scale_dev.item(), SCALE_FLOOR)
    return output_u8, q_scale
