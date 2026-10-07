"""MiniMax-M3/M3.1 sparse attention (MSA) module.

The module keeps legacy M3 BF16/FP8 sparse paths and the M3.1 NVFP4 path.
For M3.1 with ``nvfp4_kv_cache`` enabled, Prefill IndexScore consumes packed
idx_K4 plus its E4M3 scales, Prefill sparse attention consumes packed main
KV4 plus scales, and Decode/target-verify use the direct paged Q8K4 reader.
Those M3.1 routes do not gather/dequantize the full history into BF16 working
pages. The BF16/FP8 branches below remain for non-NVFP4 M3 compatibility and
are not fallback routes for an M3.1 NVFP4 request.

All persistent KV/index data uses the cache-manager's paged pools; no
per-layer side cache is introduced. Main K/V and index-K plus scales use the
cache ABI described by ``NVFP4CacheLayout`` and the same physical page table,
so they move together across PD separation. In CP Prefill, the full suffix is
gathered for projection, but writes remain rank-sharded and Q stays
rank-local; the FP4 IndexScore and sparse-attention operators consume the
packed pages directly.

The non-CP physical slot for ``(request b, token position p)`` is::

    slot = block_table[b, p // page_size] * page_size + (p % page_size)

The index branch (``index_q_proj`` / ``index_k_proj`` + per-head Gemma RMSNorm
+ partial RoPE) selects top-k blocks. With ``disable_index_value=True`` it
does not contribute to the attention value.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_CP_PACKED_KV_OVERLAP = os.environ.get("RTP_LLM_CP_PACKED_KV_OVERLAP", "0") == "1"
_CP_PREFIX_PREFETCH = os.environ.get("RTP_LLM_CP_PREFIX_PREFETCH", "0") == "1"
_CP_COMPACT_PREFILL = os.environ.get("M3_MSA_CP_COMPACT_PREFILL", "0") == "1"
_MAX_LIVE_PREFETCH = 2
_BF16_BYTES = 2
_FP8_SCALE_BYTES = 4
_FP8_E4M3_MAX = tl.constexpr(448.0)


def _should_use_cp_compact_prefill(compact_enabled: bool, nvfp4_kv_cache: bool) -> bool:
    """Keep the legacy BF16 compact path out of M3.1's packed-KV4 route."""
    return bool(compact_enabled and not nvfp4_kv_cache)


import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    build_cp_sharded_prefix_gather_plan,
    gather_cp_sharded_prefix_pool,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import (
    CudaMxfp8Linear,
)
from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    NVFP4_GROUP_SIZE,
    build_decode_physical_slots,
)
from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    cache_layout as nvfp4_cache_layout,
)
from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    clear_packed_working_tail_scales,
)
from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    quantize_cp_main_index_rows_to_planes as nvfp4_quantize_cp_main_index_rows_to_planes,
)
from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    quantize_main_index_rows as nvfp4_quantize_main_index_rows,
)
from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    quantize_query_rows_mma as nvfp4_quantize_query_rows_mma,
)
from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    round_to_e4m3_compute_grid_ as nvfp4_round_to_e4m3_compute_grid_,
)
from rtp_llm.models_py.triton_kernels.common.nvfp4_prefix_restore import (
    restore_prefix_planes,
)
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

try:
    from rtp_llm.ops.compute_ops import (
        cuda_graph_capture_forward_enabled,
        cuda_graph_warmup_forward_enabled,
    )
except ImportError:

    def cuda_graph_capture_forward_enabled() -> bool:
        return False

    def cuda_graph_warmup_forward_enabled() -> bool:
        return False


from rtp_llm.utils.model_weight import W

device_type = get_device_type()
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.norm import FusedQKRMSNorm
else:
    from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm


def _repeat_request_block_table_for_verify_tokens(
    block_table: torch.Tensor, batch_size: int, total_tokens: int
) -> torch.Tensor:
    if batch_size <= 0 or total_tokens % batch_size != 0:
        raise RuntimeError(
            "MSA target verify expects flat [batch * verify_tokens, hidden] input; "
            f"got tokens={total_tokens}, batch={batch_size}"
        )
    if int(block_table.shape[0]) != batch_size:
        raise RuntimeError(
            "MSA target verify block table batch mismatch: "
            f"block_table={tuple(block_table.shape)}, batch={batch_size}"
        )
    verify_tokens = total_tokens // batch_size
    return block_table.repeat_interleave(verify_tokens, dim=0)


def _build_target_verify_token_metadata(
    prefix_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    total_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand request-row target-verify metadata into token-row MSA metadata."""
    batch_size = int(prefix_lengths.numel())
    if batch_size <= 0 or total_tokens % batch_size != 0:
        raise RuntimeError(
            "MSA target verify expects flat [batch * verify_tokens, hidden] input; "
            f"got tokens={total_tokens}, batch={batch_size}"
        )
    if int(input_lengths.numel()) != batch_size:
        raise RuntimeError(
            "MSA target verify input length batch mismatch: "
            f"input_lengths={input_lengths.numel()}, batch={batch_size}"
        )

    verify_tokens = total_tokens // batch_size
    prefix = prefix_lengths.to(device=device, dtype=torch.int64)
    relative_positions = torch.arange(verify_tokens, device=device, dtype=torch.int64)
    positions_i64 = (prefix[:, None] + relative_positions[None, :]).reshape(-1)

    # Decode CUDA Graph may replay a larger captured batch bucket. The shared
    # runner marks padded request rows with input_lengths == 0.
    valid_requests = input_lengths.to(device=device) > 0
    valid_tokens = valid_requests[:, None].expand(batch_size, verify_tokens).reshape(-1)
    sequence_lengths = torch.where(
        valid_tokens, positions_i64 + 1, torch.zeros_like(positions_i64)
    )
    return positions_i64.to(torch.int32), sequence_lengths.to(torch.int32), valid_tokens


def _prepare_target_verify_addressing(
    request_block_table: torch.Tensor,
    prefix_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    total_tokens: int,
    device: torch.device,
    use_fused_cuda: bool = False,
    is_ragged: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build token-row MSA addressing, optionally using one CUDA launch."""
    batch_size = int(prefix_lengths.numel())
    if is_ragged:
        if batch_size <= 0 or int(input_lengths.numel()) != batch_size:
            raise RuntimeError(
                "ragged MSA target verify requires one input length per request: "
                f"input_lengths={input_lengths.numel()}, batch={batch_size}"
            )
        if cu_seqlens is None or int(cu_seqlens.numel()) != batch_size + 1:
            raise RuntimeError(
                "ragged MSA target verify requires CUDA cu_seqlens [batch + 1]"
            )
        if not request_block_table.is_cuda:
            raise RuntimeError("ragged MSA target verify requires CUDA addressing")
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        return tuple(
            rtp_llm_ops.mtp_msa_target_verify_ragged_addressing_prepare(
                request_block_table,
                prefix_lengths,
                cu_seqlens,
                total_tokens,
            )
        )
    if batch_size <= 0 or total_tokens % batch_size != 0:
        raise RuntimeError(
            "MSA target verify expects flat [batch * verify_tokens, hidden] input; "
            f"got tokens={total_tokens}, batch={batch_size}"
        )
    verify_tokens = total_tokens // batch_size
    if use_fused_cuda and request_block_table.is_cuda:
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        return tuple(
            rtp_llm_ops.mtp_msa_target_verify_addressing_prepare(
                request_block_table,
                prefix_lengths,
                input_lengths,
                verify_tokens,
            )
        )

    physical_block_table = _repeat_request_block_table_for_verify_tokens(
        request_block_table, batch_size, total_tokens
    )
    positions, sequence_lengths, valid_token_mask = _build_target_verify_token_metadata(
        prefix_lengths,
        input_lengths,
        total_tokens,
        device,
    )
    return physical_block_table, positions, sequence_lengths, valid_token_mask


# ----------------------------------------------------------------------------
# Fused QKV split + RoPE(K) + pack for CP prefill.
#
# Replaces:
#   k = qkv[:, q_size:q_size+kv_size].reshape(T, kv_head, hd).contiguous()  # DtoD
#   v = qkv[:, q_size+kv_size:].reshape(T, kv_head, hd).contiguous()        # DtoD
#   self._apply_rope(k, dummy, positions)                                    # launch
#   packed = torch.cat([k.reshape(T,nk), v.reshape(T,nk), idx_k], dim=-1)  # DtoD
#
# with a single Triton kernel that reads K/V directly from the strided qkv
# tensor, applies NeoX RoPE to K in-register, and writes the packed output.
# Persistent cache writes still go through the scheduler-provided paged KV cache;
# no side-cache fallback is introduced.
# ----------------------------------------------------------------------------


@triton.jit
def _fused_split_rope_pack_kernel(
    qkv_ptr,  # [T, QKV_DIM] bf16, contiguous
    idx_k_ptr,  # [T, NI] bf16, contiguous (already RoPE'd)
    cos_sin_ptr,  # [max_pos, rotary_dim] float32 (cos[:HALF_ROT], sin[HALF_ROT:])
    pos_ids_ptr,  # [T] int32
    packed_ptr,  # [T, PACKED_DIM] bf16, output
    Q_OFFSET,  # element offset of K within each qkv row
    qkv_row_stride,
    idx_k_row_stride,
    cos_sin_row_stride,
    packed_row_stride,
    NK: tl.constexpr,  # kv_head_num * head_dim
    NI: tl.constexpr,  # idx_head_dim
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,  # partial RoPE dimension (≤ HEAD_DIM)
    HALF_ROT: tl.constexpr,  # ROTARY_DIM // 2
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_HALF: tl.constexpr,  # next_pow2(HALF_ROT)
    BLOCK_NK: tl.constexpr,  # next_pow2(NK)
    BLOCK_NI: tl.constexpr,  # next_pow2(NI)
    REM: tl.constexpr,  # HEAD_DIM - ROTARY_DIM
    BLOCK_REM: tl.constexpr,  # next_pow2(REM) or 1
):
    pid = tl.program_id(0).to(tl.int64)

    # Load position and cos/sin for this token (based on rotary_dim)
    pos = tl.load(pos_ids_ptr + pid).to(tl.int64)
    rot_off = tl.arange(0, BLOCK_HALF)
    rot_mask = rot_off < HALF_ROT
    cos = tl.load(
        cos_sin_ptr + pos * cos_sin_row_stride + rot_off,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + pos * cos_sin_row_stride + HALF_ROT + rot_off,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)

    qkv_row = pid * qkv_row_stride
    packed_row = pid * packed_row_stride

    # K: read from qkv (strided), apply NeoX RoPE on first rotary_dim,
    # pass through remaining elements, write to packed
    for h in tl.static_range(NUM_KV_HEADS):
        h_off = Q_OFFSET + h * HEAD_DIM
        out_off = h * HEAD_DIM

        # --- RoPE on first rotary_dim elements (NeoX non-interleaved) ---
        k_first = tl.load(
            qkv_ptr + qkv_row + h_off + rot_off,
            mask=rot_mask,
            other=0.0,
        ).to(tl.float32)
        k_second = tl.load(
            qkv_ptr + qkv_row + h_off + HALF_ROT + rot_off,
            mask=rot_mask,
            other=0.0,
        ).to(tl.float32)
        # NeoX (non-interleaved) RoPE:
        #   k_rot[:half] = k[:half] * cos - k[half:] * sin
        #   k_rot[half:] = k[half:] * cos + k[:half] * sin
        k_rot_first = k_first * cos - k_second * sin
        k_rot_second = k_second * cos + k_first * sin
        tl.store(
            packed_ptr + packed_row + out_off + rot_off,
            k_rot_first.to(packed_ptr.dtype.element_ty),
            mask=rot_mask,
        )
        tl.store(
            packed_ptr + packed_row + out_off + HALF_ROT + rot_off,
            k_rot_second.to(packed_ptr.dtype.element_ty),
            mask=rot_mask,
        )

        # --- Pass-through: rotary_dim to HEAD_DIM (no RoPE) ---
        rem_off = tl.arange(0, BLOCK_REM)
        rem_mask = rem_off < REM
        if REM > 0:
            k_rem = tl.load(
                qkv_ptr + qkv_row + h_off + ROTARY_DIM + rem_off,
                mask=rem_mask,
                other=0.0,
            )
            tl.store(
                packed_ptr + packed_row + out_off + ROTARY_DIM + rem_off,
                k_rem,
                mask=rem_mask,
            )

    # V: copy from qkv to packed (no RoPE)
    v_off = tl.arange(0, BLOCK_NK)
    v_mask = v_off < NK
    v = tl.load(
        qkv_ptr + qkv_row + Q_OFFSET + NK + v_off,
        mask=v_mask,
        other=0.0,
    )
    tl.store(packed_ptr + packed_row + NK + v_off, v, mask=v_mask)

    # idx_k: copy (already RoPE'd) to packed
    idx_off = tl.arange(0, BLOCK_NI)
    idx_mask = idx_off < NI
    idx_k = tl.load(
        idx_k_ptr + pid * idx_k_row_stride + idx_off,
        mask=idx_mask,
        other=0.0,
    )
    tl.store(packed_ptr + packed_row + 2 * NK + idx_off, idx_k, mask=idx_mask)


def _fused_split_rope_pack(
    qkv: torch.Tensor,  # [T, q_size + 2*kv_size] contiguous
    idx_k: torch.Tensor,  # [T, 1, idx_head_dim] or [T, idx_head_dim]
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim] float32
    pos_ids: torch.Tensor,  # [T] int32/int64
    packed_kv: torch.Tensor,  # [T, 2*nk + ni] output
    q_offset: int,  # = q_size
    nk: int,  # = kv_head_num * head_dim
    ni: int,  # = idx_head_dim
    head_dim: int,
    num_kv_heads: int,
    rotary_dim: int,  # partial RoPE dimension (≤ head_dim)
) -> None:
    """Fused QKV split + NeoX RoPE on K + pack [K_rope|V|idx_k].

    Reads K and V directly from the strided ``qkv`` GEMM output, applies
    RoPE to K in-register using ``cos_sin_cache``, and writes the packed
    layout to ``packed_kv``. ``idx_k`` must already be RoPE'd.

    Supports **partial RoPE** (``rotary_dim < head_dim``): only the first
    ``rotary_dim`` elements of each head are rotated; the remaining
    ``head_dim - rotary_dim`` elements pass through unchanged.
    """
    T = qkv.shape[0]
    if T == 0:
        return
    half_rot = rotary_dim // 2
    rem = head_dim - rotary_dim
    BLOCK_HALF = triton.next_power_of_2(half_rot)
    BLOCK_REM = max(triton.next_power_of_2(rem), 1) if rem > 0 else 1
    BLOCK_NK = triton.next_power_of_2(nk)
    BLOCK_NI = triton.next_power_of_2(ni)

    # Ensure idx_k is 2-D [T, ni] for simple pointer arithmetic
    if idx_k.dim() == 3:
        idx_k = idx_k.reshape(T, ni)

    # Ensure pos_ids is int32 for the kernel
    if pos_ids.dtype != torch.int32:
        pos_ids = pos_ids.to(torch.int32)

    _fused_split_rope_pack_kernel[(T,)](
        qkv,
        idx_k,
        cos_sin_cache,
        pos_ids,
        packed_kv,
        q_offset,
        qkv.stride(0),
        idx_k.stride(0),
        cos_sin_cache.stride(0),
        packed_kv.stride(0),
        NK=nk,
        NI=ni,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        HALF_ROT=half_rot,
        NUM_KV_HEADS=num_kv_heads,
        BLOCK_HALF=BLOCK_HALF,
        BLOCK_NK=BLOCK_NK,
        BLOCK_NI=BLOCK_NI,
        REM=rem,
        BLOCK_REM=BLOCK_REM,
        num_warps=1,
    )


@triton.jit
def _fused_qk_idx_norm_rope_write_paged_decode_kernel(
    fused_ptr,  # [T, Q|K|V|idx_Q|idx_K] bf16, contiguous
    q_out_ptr,  # [T, num_q_heads, head_dim] bf16, contiguous
    idx_q_out_ptr,  # [T, num_idx_q_heads, head_dim] bf16, contiguous
    q_weight_ptr,  # [HEAD_DIM]
    k_weight_ptr,  # [HEAD_DIM]
    idx_q_weight_ptr,  # [HEAD_DIM]
    idx_k_weight_ptr,  # [HEAD_DIM]
    cos_sin_ptr,  # [max_pos, rotary_dim]
    pos_ids_ptr,  # [T]
    seq_lens_ptr,  # [T] int32, current kv length after writing decode token
    block_table_ptr,  # [T, max_blocks]
    paged_kv_ptr,  # [block,2,kv_head,page,head_dim]
    paged_idx_k_ptr,  # [block, page, idx_dim]
    paged_idx_scale_ptr,  # [block, page] fp32, only read when SCALED_IDX
    FUSED_ROW_STRIDE: tl.constexpr,
    Q_STRIDE_T: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    Q_STRIDE_D: tl.constexpr,
    IDX_Q_STRIDE_T: tl.constexpr,
    IDX_Q_STRIDE_H: tl.constexpr,
    IDX_Q_STRIDE_D: tl.constexpr,
    COS_SIN_ROW_STRIDE: tl.constexpr,
    BT_STRIDE_B: tl.constexpr,
    BT_STRIDE_BLK: tl.constexpr,
    KV_STRIDE_BLOCK: tl.constexpr,
    KV_STRIDE_KV: tl.constexpr,
    KV_STRIDE_HEAD: tl.constexpr,
    KV_STRIDE_PAGE: tl.constexpr,
    KV_STRIDE_DIM: tl.constexpr,
    IDX_VALUE_S0: tl.constexpr,
    IDX_VALUE_S1: tl.constexpr,
    IDX_VALUE_S2: tl.constexpr,
    IDX_SCALE_S0: tl.constexpr,
    IDX_SCALE_S1: tl.constexpr,
    MAX_PHYSICAL_BLOCKS: tl.constexpr,
    MAX_BLOCKS_PER_ROW: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROT: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    NUM_IDX_Q_HEADS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    REM: tl.constexpr,
    BLOCK_REM: tl.constexpr,
    SCALED_IDX: tl.constexpr,
):
    token_id = tl.program_id(0).to(tl.int64)
    output_group = tl.program_id(1)

    q_group_end = NUM_Q_HEADS
    kv_k_group_end = q_group_end + NUM_KV_HEADS
    idx_q_group_end = kv_k_group_end + NUM_IDX_Q_HEADS
    idx_k_output_head = NUM_Q_HEADS + 2 * NUM_KV_HEADS + NUM_IDX_Q_HEADS

    is_q = output_group < q_group_end
    is_k = (output_group >= q_group_end) & (output_group < kv_k_group_end)
    is_idx_q = (output_group >= kv_k_group_end) & (output_group < idx_q_group_end)
    fused_output_head = tl.where(
        is_idx_q,
        output_group + NUM_KV_HEADS,  # skip V heads between K and idx_Q
        tl.where(output_group >= idx_q_group_end, idx_k_output_head, output_group),
    )

    fused_row = token_id * FUSED_ROW_STRIDE
    fused_head_base = fused_row + fused_output_head * HEAD_DIM

    decode_kv_len = tl.load(seq_lens_ptr + token_id).to(tl.int64)
    token_pos = decode_kv_len - 1
    page_index = token_pos // PAGE_SIZE
    page_offset = token_pos - page_index * PAGE_SIZE
    valid_page_index = (
        (decode_kv_len > 0) & (page_index >= 0) & (page_index < MAX_BLOCKS_PER_ROW)
    )
    kv_block_id = tl.load(
        block_table_ptr + token_id * BT_STRIDE_B + page_index * BT_STRIDE_BLK,
        mask=valid_page_index,
        other=-1,
    ).to(tl.int64)
    head_off = tl.arange(0, BLOCK_HEAD)
    head_mask = head_off < HEAD_DIM
    x = tl.load(fused_ptr + fused_head_base + head_off, mask=head_mask, other=0.0).to(
        tl.float32
    )

    weight_ptr = tl.where(
        is_q,
        q_weight_ptr,
        tl.where(
            is_k,
            k_weight_ptr,
            tl.where(is_idx_q, idx_q_weight_ptr, idx_k_weight_ptr),
        ),
    )
    w = tl.load(weight_ptr + head_off, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x) / HEAD_DIM
    rrms = tl.rsqrt(var + EPS)

    pos = tl.load(pos_ids_ptr + token_id).to(tl.int64)
    rot_off = tl.arange(0, BLOCK_HALF)
    rot_mask = rot_off < HALF_ROT
    cos = tl.load(
        cos_sin_ptr + pos * COS_SIN_ROW_STRIDE + rot_off,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + pos * COS_SIN_ROW_STRIDE + HALF_ROT + rot_off,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)

    first = tl.load(fused_ptr + fused_head_base + rot_off, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    second = tl.load(
        fused_ptr + fused_head_base + HALF_ROT + rot_off,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)
    w_first = tl.load(weight_ptr + rot_off, mask=rot_mask, other=0.0).to(tl.float32)
    w_second = tl.load(weight_ptr + HALF_ROT + rot_off, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    # Match the original path: RMSNorm materializes BF16 Q/K/idx_Q/idx_K
    # before RoPE reads them. Keeping FP32 normalized values here is a real
    # numerical behavior change from the unfused decode path.
    first = (first * rrms * w_first).to(tl.bfloat16).to(tl.float32)
    second = (second * rrms * w_second).to(tl.bfloat16).to(tl.float32)
    rot_first = first * cos - second * sin
    rot_second = second * cos + first * sin
    q_out_head = tl.where(is_q, output_group, 0)
    idx_q_head = tl.where(is_idx_q, output_group - kv_k_group_end, 0)
    q_out_base = token_id * Q_STRIDE_T + q_out_head * Q_STRIDE_H
    idx_q_out_base = token_id * IDX_Q_STRIDE_T + idx_q_head * IDX_Q_STRIDE_H
    tl.store(
        q_out_ptr + q_out_base + rot_off * Q_STRIDE_D,
        rot_first.to(q_out_ptr.dtype.element_ty),
        mask=rot_mask & is_q,
    )
    tl.store(
        q_out_ptr + q_out_base + (HALF_ROT + rot_off) * Q_STRIDE_D,
        rot_second.to(q_out_ptr.dtype.element_ty),
        mask=rot_mask & is_q,
    )
    tl.store(
        idx_q_out_ptr + idx_q_out_base + rot_off * IDX_Q_STRIDE_D,
        rot_first.to(idx_q_out_ptr.dtype.element_ty),
        mask=rot_mask & is_idx_q,
    )
    tl.store(
        idx_q_out_ptr + idx_q_out_base + (HALF_ROT + rot_off) * IDX_Q_STRIDE_D,
        rot_second.to(idx_q_out_ptr.dtype.element_ty),
        mask=rot_mask & is_idx_q,
    )

    rem_off = tl.arange(0, BLOCK_REM)
    rem_mask = rem_off < REM
    if REM > 0:
        w_rem = tl.load(
            weight_ptr + ROTARY_DIM + rem_off,
            mask=rem_mask,
            other=0.0,
        ).to(tl.float32)
        rem = (
            tl.load(
                fused_ptr + fused_head_base + ROTARY_DIM + rem_off,
                mask=rem_mask,
                other=0.0,
            ).to(tl.float32)
            * rrms
            * w_rem
        )
        tl.store(
            q_out_ptr + q_out_base + (ROTARY_DIM + rem_off) * Q_STRIDE_D,
            rem.to(q_out_ptr.dtype.element_ty),
            mask=rem_mask & is_q,
        )
        tl.store(
            idx_q_out_ptr + idx_q_out_base + (ROTARY_DIM + rem_off) * IDX_Q_STRIDE_D,
            rem.to(idx_q_out_ptr.dtype.element_ty),
            mask=rem_mask & is_idx_q,
        )

    valid_paged_slot = (
        valid_page_index & (kv_block_id >= 0) & (kv_block_id < MAX_PHYSICAL_BLOCKS)
    )
    store_block = tl.where(valid_paged_slot, kv_block_id, 0)
    store_page_offset = tl.where(valid_paged_slot, page_offset, 0)

    kv_head = output_group - q_group_end
    store_kv_head = tl.where(is_k, kv_head, 0)
    paged_k_offset = (
        store_block * KV_STRIDE_BLOCK
        + store_kv_head * KV_STRIDE_HEAD
        + store_page_offset * KV_STRIDE_PAGE
        + head_off * KV_STRIDE_DIM
    )
    kv_store_mask = head_mask & valid_paged_slot & is_k
    v_output_head = NUM_Q_HEADS + NUM_KV_HEADS + store_kv_head
    v_output_base = fused_row + v_output_head * HEAD_DIM
    tl.store(
        paged_kv_ptr + paged_k_offset + KV_STRIDE_KV,
        tl.load(fused_ptr + v_output_base + head_off, mask=head_mask, other=0.0),
        mask=kv_store_mask,
    )

    is_idx_k = output_group >= idx_q_group_end

    # K/idx_K are consumed only by paged caches, so write them directly. Reloading
    # from fused_ptr after an in-kernel store is not ordered and can corrupt K.
    k_store_base = (
        store_block * KV_STRIDE_BLOCK
        + store_kv_head * KV_STRIDE_HEAD
        + store_page_offset * KV_STRIDE_PAGE
    )
    tl.store(
        paged_kv_ptr + k_store_base + rot_off * KV_STRIDE_DIM,
        rot_first.to(tl.bfloat16).to(paged_kv_ptr.dtype.element_ty),
        mask=rot_mask & valid_paged_slot & is_k,
    )
    tl.store(
        paged_kv_ptr + k_store_base + (HALF_ROT + rot_off) * KV_STRIDE_DIM,
        rot_second.to(tl.bfloat16).to(paged_kv_ptr.dtype.element_ty),
        mask=rot_mask & valid_paged_slot & is_k,
    )
    if SCALED_IDX:
        # Match _write_decode_kv_idx_to_paged: the unfused path materializes
        # BF16 idx_K after RoPE and quantizes the whole head with one absmax
        # scale. Round to BF16 first so both paths share bit-identical
        # quantization inputs.
        idx_first = rot_first.to(tl.bfloat16).to(tl.float32)
        idx_second = rot_second.to(tl.bfloat16).to(tl.float32)
        idx_absmax = tl.maximum(
            tl.max(tl.where(rot_mask, tl.abs(idx_first), 0.0), axis=0),
            tl.max(tl.where(rot_mask, tl.abs(idx_second), 0.0), axis=0),
        )
        if REM > 0:
            idx_rem = rem.to(tl.bfloat16).to(tl.float32)
            idx_absmax = tl.maximum(
                idx_absmax,
                tl.max(tl.where(rem_mask, tl.abs(idx_rem), 0.0), axis=0),
            )
        idx_scale = tl.maximum(idx_absmax / _FP8_E4M3_MAX, 1.0e-12)
        idx_first = idx_first / idx_scale
        idx_second = idx_second / idx_scale
        tl.store(
            paged_idx_scale_ptr
            + store_block * IDX_SCALE_S0
            + store_page_offset * IDX_SCALE_S1,
            idx_scale,
            mask=valid_paged_slot & is_idx_k,
        )
    else:
        idx_first = rot_first
        idx_second = rot_second
    idx_value_base = store_block * IDX_VALUE_S0 + store_page_offset * IDX_VALUE_S1
    tl.store(
        paged_idx_k_ptr + idx_value_base + rot_off * IDX_VALUE_S2,
        idx_first.to(paged_idx_k_ptr.dtype.element_ty),
        mask=rot_mask & valid_paged_slot & is_idx_k,
    )
    tl.store(
        paged_idx_k_ptr + idx_value_base + (HALF_ROT + rot_off) * IDX_VALUE_S2,
        idx_second.to(paged_idx_k_ptr.dtype.element_ty),
        mask=rot_mask & valid_paged_slot & is_idx_k,
    )
    if REM > 0:
        tl.store(
            paged_kv_ptr + k_store_base + (ROTARY_DIM + rem_off) * KV_STRIDE_DIM,
            rem.to(tl.bfloat16).to(paged_kv_ptr.dtype.element_ty),
            mask=rem_mask & valid_paged_slot & is_k,
        )
        if SCALED_IDX:
            idx_rem_store = idx_rem / idx_scale
        else:
            idx_rem_store = rem
        tl.store(
            paged_idx_k_ptr + idx_value_base + (ROTARY_DIM + rem_off) * IDX_VALUE_S2,
            idx_rem_store.to(paged_idx_k_ptr.dtype.element_ty),
            mask=rem_mask & valid_paged_slot & is_idx_k,
        )


def _fused_qk_idx_norm_rope_write_paged_decode(
    fused_qkv_idx_out: torch.Tensor,
    q_out: torch.Tensor,
    idx_q_out: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    idx_q_weight: torch.Tensor,
    idx_k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    phys_block_table: torch.Tensor,
    paged_kv_base: torch.Tensor,
    paged_idx_k: torch.Tensor,
    paged_idx_k_scale: Optional[torch.Tensor],
    page_size: int,
    head_dim: int,
    rotary_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_idx_q_heads: int,
    eps: float,
) -> None:
    """SGLang-style 4-group Gemma RMSNorm + NeoX RoPE for paged decode.

    Q and idx_Q are written directly to their downstream contiguous outputs.
    K/V and idx_K are persisted into the paged cache. idx_K is stored as BF16
    (``paged_idx_k_scale is None``) or as scaled E4M3 with one fp32 scale per
    token, matching the ``_write_decode_kv_idx_to_paged`` layout. The original
    fallback path materializes bf16 after RMSNorm before RoPE, so keep that as
    the numerical reference when needed.
    """
    T = fused_qkv_idx_out.shape[0]
    if T == 0:
        return
    if paged_idx_k.dim() != 3:
        raise RuntimeError(
            f"fused paged decode needs a [block,page,idx_dim] idx_K view, got "
            f"{tuple(paged_idx_k.shape)}"
        )
    if paged_idx_k_scale is not None and tuple(paged_idx_k_scale.shape) != tuple(
        paged_idx_k.shape[:2]
    ):
        raise RuntimeError(
            f"idx_K scale shape {tuple(paged_idx_k_scale.shape)} does not match "
            f"value pages {tuple(paged_idx_k.shape[:2])}"
        )
    half_rot = rotary_dim // 2
    rem = head_dim - rotary_dim
    block_head = triton.next_power_of_2(head_dim)
    block_half = triton.next_power_of_2(half_rot)
    block_rem = max(triton.next_power_of_2(rem), 1) if rem > 0 else 1
    if pos_ids.dtype != torch.int32:
        pos_ids = pos_ids.to(torch.int32)

    total_norm_heads = num_q_heads + num_kv_heads + num_idx_q_heads + 1
    _fused_qk_idx_norm_rope_write_paged_decode_kernel[(T, total_norm_heads)](
        fused_qkv_idx_out,
        q_out,
        idx_q_out,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        cos_sin_cache,
        pos_ids,
        seq_lens,
        phys_block_table,
        paged_kv_base,
        paged_idx_k,
        paged_idx_k_scale if paged_idx_k_scale is not None else paged_idx_k,
        FUSED_ROW_STRIDE=int(fused_qkv_idx_out.stride(0)),
        Q_STRIDE_T=int(q_out.stride(0)),
        Q_STRIDE_H=int(q_out.stride(1)),
        Q_STRIDE_D=int(q_out.stride(2)),
        IDX_Q_STRIDE_T=int(idx_q_out.stride(0)),
        IDX_Q_STRIDE_H=int(idx_q_out.stride(1)),
        IDX_Q_STRIDE_D=int(idx_q_out.stride(2)),
        COS_SIN_ROW_STRIDE=int(cos_sin_cache.stride(0)),
        BT_STRIDE_B=int(phys_block_table.stride(0)),
        BT_STRIDE_BLK=int(phys_block_table.stride(1)),
        KV_STRIDE_BLOCK=int(paged_kv_base.stride(0)),
        KV_STRIDE_KV=int(paged_kv_base.stride(1)),
        KV_STRIDE_HEAD=int(paged_kv_base.stride(2)),
        KV_STRIDE_PAGE=int(paged_kv_base.stride(3)),
        KV_STRIDE_DIM=int(paged_kv_base.stride(4)),
        IDX_VALUE_S0=int(paged_idx_k.stride(0)),
        IDX_VALUE_S1=int(paged_idx_k.stride(1)),
        IDX_VALUE_S2=int(paged_idx_k.stride(2)),
        IDX_SCALE_S0=(
            0 if paged_idx_k_scale is None else int(paged_idx_k_scale.stride(0))
        ),
        IDX_SCALE_S1=(
            0 if paged_idx_k_scale is None else int(paged_idx_k_scale.stride(1))
        ),
        MAX_PHYSICAL_BLOCKS=int(paged_kv_base.shape[0]),
        MAX_BLOCKS_PER_ROW=int(phys_block_table.shape[1]),
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        HALF_ROT=half_rot,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        NUM_IDX_Q_HEADS=num_idx_q_heads,
        EPS=eps,
        BLOCK_HEAD=block_head,
        BLOCK_HALF=block_half,
        REM=rem,
        BLOCK_REM=block_rem,
        SCALED_IDX=paged_idx_k_scale is not None,
    )


@triton.jit
def _rows_to_contig_kernel(
    src, out, T, row_stride, ROW: tl.constexpr, BLK: tl.constexpr
):
    # CP keeps only the local query rows, but their source stride is the full
    # fused-QKV width.  At 1M context / CP4, t * row_stride exceeds INT32 even
    # though both operands fit individually.  Promote the row id before either
    # source or destination address arithmetic.
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    s = t * row_stride
    d = t * ROW
    for o in range(0, ROW, BLK):
        off = o + tl.arange(0, BLK)
        m = off < ROW
        tl.store(out + d + off, tl.load(src + s + off, mask=m, other=0), mask=m)


def _rows_to_contig(x: torch.Tensor) -> torch.Tensor:
    """Contiguous copy of a [T, H, hd] tensor whose dim-0 is strided but whose last two
    dims are contiguous (e.g. q = qkv[:, :q_size].reshape(T,H,hd), a column-slice of the
    fused QKV). A token-parallel coalesced copy hits ~75% HBM BW vs aten .contiguous()'s
    ~20% on this strided slice (~3.4x, ~150->44us at T=8192). Falls back to .contiguous()
    if the layout is not the expected row-contiguous form."""
    if x.is_contiguous():
        return x
    T, H, hd = x.shape
    if x.stride(2) != 1 or x.stride(1) != hd:
        return x.contiguous()
    out = torch.empty((T, H, hd), dtype=x.dtype, device=x.device)
    _rows_to_contig_kernel[(T,)](
        x, out, T, x.stride(0), ROW=H * hd, BLK=2048, num_warps=8
    )
    return out


@triton.jit
def _write_decode_kv_idx_kernel(
    k_ptr,
    v_ptr,
    idx_ptr,
    seq_lens_ptr,
    block_table_ptr,
    base_ptr,
    idx_value_ptr,
    idx_scale_ptr,
    TOKEN_COUNT: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IDX_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BT_STRIDE_B: tl.constexpr,
    BT_STRIDE_BLK: tl.constexpr,
    BASE_S0: tl.constexpr,
    BASE_S1: tl.constexpr,
    BASE_S2: tl.constexpr,
    BASE_S3: tl.constexpr,
    BASE_S4: tl.constexpr,
    IDX_VALUE_S0: tl.constexpr,
    IDX_VALUE_S1: tl.constexpr,
    IDX_VALUE_S2: tl.constexpr,
    IDX_SCALE_S0: tl.constexpr,
    IDX_SCALE_S1: tl.constexpr,
    MAX_PHYSICAL_BLOCKS: tl.constexpr,
    MAX_BLOCKS_PER_ROW: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_IDX: tl.constexpr,
    SCALED_IDX: tl.constexpr,
):
    token = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + token, mask=token < TOKEN_COUNT, other=0).to(
        tl.int64
    )
    prefix = seq_len - 1
    block_idx = prefix // PAGE_SIZE
    block_off = prefix - block_idx * PAGE_SIZE
    valid_block_idx = (
        (token < TOKEN_COUNT)
        & (seq_len > 0)
        & (block_idx >= 0)
        & (block_idx < MAX_BLOCKS_PER_ROW)
    )
    physical_block = tl.load(
        block_table_ptr + token * BT_STRIDE_B + block_idx * BT_STRIDE_BLK,
        mask=valid_block_idx,
        other=-1,
    ).to(tl.int64)
    valid_physical_block = (
        valid_block_idx & (physical_block >= 0) & (physical_block < MAX_PHYSICAL_BLOCKS)
    )
    offs = tl.arange(0, BLOCK_KV)
    head = offs // HEAD_DIM
    dim = offs - head * HEAD_DIM
    kv_mask = valid_physical_block & (offs < NUM_KV_HEADS * HEAD_DIM)
    k_vals = tl.load(
        k_ptr + token * NUM_KV_HEADS * HEAD_DIM + offs,
        mask=kv_mask,
        other=0.0,
    )
    v_vals = tl.load(
        v_ptr + token * NUM_KV_HEADS * HEAD_DIM + offs,
        mask=kv_mask,
        other=0.0,
    )
    base_k = (
        physical_block * BASE_S0 + head * BASE_S2 + block_off * BASE_S3 + dim * BASE_S4
    )
    tl.store(base_ptr + base_k, k_vals, mask=kv_mask)
    tl.store(base_ptr + base_k + BASE_S1, v_vals, mask=kv_mask)

    idx_offs = tl.arange(0, BLOCK_IDX)
    idx_mask = valid_physical_block & (idx_offs < IDX_DIM)
    idx_vals = tl.load(
        idx_ptr + token * IDX_DIM + idx_offs,
        mask=idx_mask,
        other=0.0,
    )
    if SCALED_IDX:
        absmax = tl.max(tl.where(idx_offs < IDX_DIM, tl.abs(idx_vals), 0.0), axis=0)
        idx_scale = tl.maximum(absmax / _FP8_E4M3_MAX, 1.0e-12)
        idx_vals = idx_vals / idx_scale
        tl.store(
            idx_scale_ptr + physical_block * IDX_SCALE_S0 + block_off * IDX_SCALE_S1,
            idx_scale,
            mask=valid_physical_block,
        )
    tl.store(
        idx_value_ptr
        + physical_block * IDX_VALUE_S0
        + block_off * IDX_VALUE_S1
        + idx_offs * IDX_VALUE_S2,
        idx_vals,
        mask=idx_mask,
    )


def _write_decode_kv_idx_to_paged(
    k: torch.Tensor,
    v: torch.Tensor,
    idx_k: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    base: torch.Tensor,
    idx_values: torch.Tensor,
    idx_scales: Optional[torch.Tensor],
    page_size: int,
    idx_dim: int,
) -> None:
    token_count = int(k.shape[0])
    if token_count == 0:
        return
    _write_decode_kv_idx_kernel[(token_count,)](
        k.reshape(token_count, -1),
        v.reshape(token_count, -1),
        idx_k.reshape(token_count, idx_dim),
        seq_lens,
        block_table,
        base,
        idx_values,
        idx_scales if idx_scales is not None else idx_values,
        TOKEN_COUNT=token_count,
        NUM_KV_HEADS=int(k.shape[1]),
        HEAD_DIM=int(k.shape[2]),
        IDX_DIM=idx_dim,
        PAGE_SIZE=page_size,
        BT_STRIDE_B=int(block_table.stride(0)),
        BT_STRIDE_BLK=int(block_table.stride(1)),
        BASE_S0=int(base.stride(0)),
        BASE_S1=int(base.stride(1)),
        BASE_S2=int(base.stride(2)),
        BASE_S3=int(base.stride(3)),
        BASE_S4=int(base.stride(4)),
        IDX_VALUE_S0=int(idx_values.stride(0)),
        IDX_VALUE_S1=int(idx_values.stride(1)),
        IDX_VALUE_S2=int(idx_values.stride(2)),
        IDX_SCALE_S0=0 if idx_scales is None else int(idx_scales.stride(0)),
        IDX_SCALE_S1=0 if idx_scales is None else int(idx_scales.stride(1)),
        MAX_PHYSICAL_BLOCKS=int(base.shape[0]),
        MAX_BLOCKS_PER_ROW=int(block_table.shape[1]),
        BLOCK_KV=triton.next_power_of_2(int(k.shape[1]) * int(k.shape[2])),
        BLOCK_IDX=triton.next_power_of_2(idx_dim),
        SCALED_IDX=idx_scales is not None,
    )


@triton.jit
def _write_main_kv_to_paged_kernel(
    k_ptr,
    v_ptr,
    slot_ptr,
    base_ptr,
    TOKEN_COUNT,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    K_S2: tl.constexpr,
    V_S0: tl.constexpr,
    V_S1: tl.constexpr,
    V_S2: tl.constexpr,
    BASE_S0: tl.constexpr,
    BASE_S1: tl.constexpr,
    BASE_S2: tl.constexpr,
    BASE_S3: tl.constexpr,
    BASE_S4: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slot_ptr + row).to(tl.int64)
    block = slot // PAGE_SIZE
    page_off = slot - block * PAGE_SIZE
    offs = tl.arange(0, BLOCK_KV)
    head = offs // HEAD_DIM
    dim = offs - head * HEAD_DIM
    valid = (slot >= 0) & (block < NUM_BLOCKS) & (offs < NUM_KV_HEADS * HEAD_DIM)
    k = tl.load(
        k_ptr + row * K_S0 + head * K_S1 + dim * K_S2,
        mask=valid,
        other=0.0,
    )
    v = tl.load(
        v_ptr + row * V_S0 + head * V_S1 + dim * V_S2,
        mask=valid,
        other=0.0,
    )
    dst = (
        base_ptr + block * BASE_S0 + head * BASE_S2 + page_off * BASE_S3 + dim * BASE_S4
    )
    tl.store(dst, k, mask=valid)
    tl.store(dst + BASE_S1, v, mask=valid)


def _write_main_kv_to_paged(
    k: torch.Tensor,
    v: torch.Tensor,
    base: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Persist K/V into the 5-D paged pool, including padded block strides."""
    if k.shape != v.shape or k.ndim != 3:
        raise ValueError("K/V must have matching [token, head, dim] shapes")
    if base.ndim != 5 or tuple(base.shape[1:3]) != (2, k.shape[1]):
        raise ValueError("paged K/V base must be [block, 2, head, page, dim]")
    if base.shape[4] != k.shape[2] or slot_mapping.numel() != k.shape[0]:
        raise ValueError("paged K/V base or slots do not match K/V rows")
    rows = int(k.shape[0])
    if rows == 0:
        return
    _write_main_kv_to_paged_kernel[(rows,)](
        k,
        v,
        slot_mapping,
        base,
        rows,
        NUM_KV_HEADS=int(k.shape[1]),
        HEAD_DIM=int(k.shape[2]),
        PAGE_SIZE=int(base.shape[3]),
        NUM_BLOCKS=int(base.shape[0]),
        K_S0=int(k.stride(0)),
        K_S1=int(k.stride(1)),
        K_S2=int(k.stride(2)),
        V_S0=int(v.stride(0)),
        V_S1=int(v.stride(1)),
        V_S2=int(v.stride(2)),
        BASE_S0=int(base.stride(0)),
        BASE_S1=int(base.stride(1)),
        BASE_S2=int(base.stride(2)),
        BASE_S3=int(base.stride(3)),
        BASE_S4=int(base.stride(4)),
        BLOCK_KV=triton.next_power_of_2(int(k.shape[1] * k.shape[2])),
        num_warps=1,
    )


@triton.jit
def _write_idx_rows_kernel(
    src_ptr,
    slot_ptr,
    dst_ptr,
    dst_scale_ptr,
    N,
    ROW_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DST_BLOCKS: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    DST_S2: tl.constexpr,
    SCALE_S0: tl.constexpr,
    SCALE_S1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALED_FP8: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    valid_row = row < N
    slot = tl.load(slot_ptr + row, mask=valid_row, other=-1).to(tl.int64)
    valid = valid_row & (slot >= 0) & (slot < DST_BLOCKS * PAGE_SIZE)
    block = slot // PAGE_SIZE
    offset = slot - block * PAGE_SIZE
    values = tl.load(
        src_ptr + row * ROW_DIM + d,
        mask=valid & (d < ROW_DIM),
        other=0.0,
    )
    dim_mask = d < ROW_DIM
    if SCALED_FP8:
        absmax = tl.max(tl.where(dim_mask, tl.abs(values), 0.0), axis=0)
        scale = tl.maximum(absmax / _FP8_E4M3_MAX, 1.0e-12)
        values = values / scale
        tl.store(
            dst_scale_ptr + block * SCALE_S0 + offset * SCALE_S1,
            scale,
            mask=valid,
        )
    tl.store(
        dst_ptr + block * DST_S0 + offset * DST_S1 + d * DST_S2,
        values,
        mask=valid & dim_mask,
    )


def _write_idx_rows(
    src: torch.Tensor,
    slots: torch.Tensor,
    dst: torch.Tensor,
    dst_scale: Optional[torch.Tensor],
) -> None:
    rows, dim = src.shape
    if int(slots.numel()) != int(rows):
        raise ValueError(f"idx row/slot mismatch: rows={rows} slots={slots.numel()}")
    if dst.dim() != 3 or int(dst.shape[2]) != int(dim):
        raise ValueError(
            f"idx destination must be [block,page,{dim}], got {tuple(dst.shape)}"
        )
    if dst_scale is not None and tuple(dst_scale.shape) != tuple(dst.shape[:2]):
        raise ValueError(
            f"idx scale shape {tuple(dst_scale.shape)} does not match "
            f"destination pages {tuple(dst.shape[:2])}"
        )
    _write_idx_rows_kernel[(rows,)](
        src,
        slots,
        dst,
        dst_scale if dst_scale is not None else dst,
        rows,
        ROW_DIM=dim,
        PAGE_SIZE=int(dst.shape[1]),
        DST_BLOCKS=int(dst.shape[0]),
        DST_S0=int(dst.stride(0)),
        DST_S1=int(dst.stride(1)),
        DST_S2=int(dst.stride(2)),
        SCALE_S0=0 if dst_scale is None else int(dst_scale.stride(0)),
        SCALE_S1=0 if dst_scale is None else int(dst_scale.stride(1)),
        BLOCK_D=triton.next_power_of_2(dim),
        SCALED_FP8=dst_scale is not None,
        num_warps=1,
    )


@triton.jit
def _fused_cp_paged_write_kernel(
    packed_ptr,
    unpad_ptr,
    write_slots_ptr,
    slot_mapping_ptr,
    working_k_ptr,
    working_v_ptr,
    scratch_idx_ptr,
    base_ptr,
    idx_value_ptr,
    idx_scale_ptr,
    kv_lens_ptr,
    TOKEN_COUNT,
    BATCH_SIZE,
    NK: tl.constexpr,
    NI: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BASE_S0: tl.constexpr,
    BASE_S1: tl.constexpr,
    BASE_S2: tl.constexpr,
    BASE_S3: tl.constexpr,
    BASE_S4: tl.constexpr,
    SCRATCH_SEQ_LEN: tl.constexpr,
    WRITE_MAIN_PAGES: tl.constexpr,
    IDX_VALUE_S0: tl.constexpr,
    IDX_VALUE_S1: tl.constexpr,
    IDX_VALUE_S2: tl.constexpr,
    IDX_SCALE_S0: tl.constexpr,
    IDX_SCALE_S1: tl.constexpr,
    SCALED_IDX: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_DIM
    di = tl.arange(0, BLOCK_I)
    imask = di < NI

    if t < TOKEN_COUNT:
        per_token = 2 * NK + NI
        src_row = tl.load(unpad_ptr + t).to(tl.int64) * per_token
        dst_slot = tl.load(write_slots_ptr + t).to(tl.int64)
        physical_slot = tl.load(slot_mapping_ptr + t).to(tl.int64)

        valid = physical_slot >= 0
        safe_slot = tl.where(valid, physical_slot, 0)
        block_id = safe_slot // PAGE_SIZE
        page_off = safe_slot - block_id * PAGE_SIZE
        paged_k = block_id * BASE_S0 + page_off * BASE_S3

        for h in tl.range(0, NUM_KV_HEADS):
            k = tl.load(packed_ptr + src_row + h * HEAD_DIM + d, mask=dmask, other=0.0)
            v = tl.load(
                packed_ptr + src_row + NK + h * HEAD_DIM + d,
                mask=dmask,
                other=0.0,
            )
            if WRITE_MAIN_PAGES:
                scratch_page = dst_slot // PAGE_SIZE
                scratch_off = dst_slot - scratch_page * PAGE_SIZE
                scratch_addr = (
                    scratch_page * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
                    + h * PAGE_SIZE * HEAD_DIM
                    + scratch_off * HEAD_DIM
                )
                tl.store(working_k_ptr + scratch_addr + d, k, mask=dmask)
                tl.store(working_v_ptr + scratch_addr + d, v, mask=dmask)
            tl.store(
                base_ptr + paged_k + h * BASE_S2 + d * BASE_S4,
                k,
                mask=dmask & valid,
            )
            tl.store(
                base_ptr + paged_k + BASE_S1 + h * BASE_S2 + d * BASE_S4,
                v,
                mask=dmask & valid,
            )

        idx = tl.load(packed_ptr + src_row + 2 * NK + di, mask=imask, other=0.0)
        tl.store(scratch_idx_ptr + dst_slot * NI + di, idx, mask=imask)
        if SCALED_IDX:
            absmax = tl.max(tl.where(imask, tl.abs(idx), 0.0), axis=0)
            idx_scale = tl.maximum(absmax / _FP8_E4M3_MAX, 1.0e-12)
            stored_idx = idx / idx_scale
            tl.store(
                idx_scale_ptr + block_id * IDX_SCALE_S0 + page_off * IDX_SCALE_S1,
                idx_scale,
                mask=valid,
            )
        else:
            stored_idx = idx
        tl.store(
            idx_value_ptr
            + block_id * IDX_VALUE_S0
            + page_off * IDX_VALUE_S1
            + di * IDX_VALUE_S2,
            stored_idx,
            mask=imask & valid,
        )

    # The scratch pool is reused across requests. Clear the short tail between
    # each real KV length and its page boundary in the same launch so padded CP
    # queries cannot observe stale K/V/index values.
    if t < BATCH_SIZE * PAGE_SIZE:
        batch_idx = t // PAGE_SIZE
        page_offset = t - batch_idx * PAGE_SIZE
        kv_len = tl.load(kv_lens_ptr + batch_idx).to(tl.int64)
        tail = (PAGE_SIZE - kv_len % PAGE_SIZE) % PAGE_SIZE
        scratch_row = batch_idx * SCRATCH_SEQ_LEN + kv_len + page_offset
        clear = (page_offset < tail) & (scratch_row < (batch_idx + 1) * SCRATCH_SEQ_LEN)
        if WRITE_MAIN_PAGES:
            for h in tl.range(0, NUM_KV_HEADS):
                scratch_page = scratch_row // PAGE_SIZE
                scratch_off = scratch_row - scratch_page * PAGE_SIZE
                scratch_addr = (
                    scratch_page * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
                    + h * PAGE_SIZE * HEAD_DIM
                    + scratch_off * HEAD_DIM
                )
                tl.store(
                    working_k_ptr + scratch_addr + d,
                    0.0,
                    mask=dmask & clear,
                )
                tl.store(
                    working_v_ptr + scratch_addr + d,
                    0.0,
                    mask=dmask & clear,
                )
        tl.store(
            scratch_idx_ptr + scratch_row * NI + di,
            0.0,
            mask=imask & clear,
        )


def _fused_cp_paged_write(
    packed: torch.Tensor,
    unpad_indices: torch.Tensor,
    write_slots: torch.Tensor,
    slot_mapping: torch.Tensor,
    working_k: torch.Tensor,
    working_v: torch.Tensor,
    idx_scratch: torch.Tensor,
    base: torch.Tensor,
    idx_values: torch.Tensor,
    idx_scales: Optional[torch.Tensor],
    kv_lens: torch.Tensor,
    scratch_seq_len: int,
    nk: int,
    ni: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    token_count: Optional[int] = None,
    write_main_pages: bool = True,
) -> None:
    """Unpad CP output into paged working K/V and persistent cache pages."""
    if token_count is None:
        token_count = int(write_slots.numel())
    batch_size = int(kv_lens.numel())
    if token_count == 0 and batch_size == 0:
        return
    if nk != num_kv_heads * head_dim:
        raise ValueError(
            f"_fused_cp_paged_write expects nk == num_kv_heads * head_dim, got "
            f"nk={nk}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
        )
    if tuple(idx_values.shape[1:]) != (page_size, ni):
        raise ValueError(
            f"paged idx values must be [block,{page_size},{ni}], got "
            f"{tuple(idx_values.shape)}"
        )
    if idx_scales is not None and tuple(idx_scales.shape) != tuple(
        idx_values.shape[:2]
    ):
        raise ValueError(
            f"paged idx scales {tuple(idx_scales.shape)} do not match value pages "
            f"{tuple(idx_values.shape[:2])}"
        )
    grid_size = max(token_count, batch_size * page_size)
    _fused_cp_paged_write_kernel[(grid_size,)](
        packed,
        unpad_indices,
        write_slots,
        slot_mapping,
        working_k.reshape(-1, nk) if write_main_pages else packed,
        working_v.reshape(-1, nk) if write_main_pages else packed,
        idx_scratch.reshape(-1, ni),
        base,
        idx_values,
        idx_scales if idx_scales is not None else idx_values,
        kv_lens,
        token_count,
        batch_size,
        NK=nk,
        NI=ni,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BASE_S0=int(base.stride(0)),
        BASE_S1=int(base.stride(1)),
        BASE_S2=int(base.stride(2)),
        BASE_S3=int(base.stride(3)),
        BASE_S4=int(base.stride(4)),
        SCRATCH_SEQ_LEN=scratch_seq_len,
        WRITE_MAIN_PAGES=write_main_pages,
        IDX_VALUE_S0=int(idx_values.stride(0)),
        IDX_VALUE_S1=int(idx_values.stride(1)),
        IDX_VALUE_S2=int(idx_values.stride(2)),
        IDX_SCALE_S0=0 if idx_scales is None else int(idx_scales.stride(0)),
        IDX_SCALE_S1=0 if idx_scales is None else int(idx_scales.stride(1)),
        SCALED_IDX=idx_scales is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_I=triton.next_power_of_2(ni),
        num_warps=1,
    )


@triton.jit
def _scatter_cp_prefix_pages_kernel(
    main_pages_ptr,
    idx_pages_ptr,
    idx_scales_ptr,
    dst_pages_ptr,
    src_pages_ptr,
    k_pages_ptr,
    v_pages_ptr,
    idx_scratch_ptr,
    PAGE_COUNT,
    MAIN_PAGE_ELEMS: tl.constexpr,
    IDX_PAGE_ELEMS: tl.constexpr,
    IDX_DIM: tl.constexpr,
    MAIN_SRC_PAGE_STRIDE: tl.constexpr,
    MAIN_SRC_V_OFFSET: tl.constexpr,
    IDX_SRC_PAGE_STRIDE: tl.constexpr,
    IDX_SCALE_PAGE_STRIDE: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    USE_SRC_PAGES: tl.constexpr,
    HAS_IDX_SCALE: tl.constexpr,
):
    """Scatter gathered logical prefix pages into request-local page slots.

    E4M3 source values are converted to BF16 while scattering; BF16 source
    values are copied directly. The idx-K pages remain BF16. One launch
    restores K, V and idx-K together.
    """
    logical_page = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    offsets = chunk * COPY_BLOCK + tl.arange(0, COPY_BLOCK).to(tl.int64)
    page_valid = logical_page < PAGE_COUNT
    dst_page = tl.load(dst_pages_ptr + logical_page).to(tl.int64)
    if USE_SRC_PAGES:
        src_page = tl.load(src_pages_ptr + logical_page).to(tl.int64)
    else:
        src_page = logical_page

    src_main = src_page * MAIN_SRC_PAGE_STRIDE + offsets
    dst_main = dst_page * MAIN_PAGE_ELEMS + offsets
    k = tl.load(main_pages_ptr + src_main)
    v = tl.load(main_pages_ptr + src_main + MAIN_SRC_V_OFFSET)
    tl.store(k_pages_ptr + dst_main, k)
    tl.store(v_pages_ptr + dst_main, v)

    idx_valid = page_valid & (offsets < IDX_PAGE_ELEMS)
    src_idx = src_page * IDX_SRC_PAGE_STRIDE + offsets
    dst_idx = dst_page * IDX_PAGE_ELEMS + offsets
    idx = tl.load(idx_pages_ptr + src_idx, mask=idx_valid, other=0.0)
    if HAS_IDX_SCALE:
        token_in_page = offsets // IDX_DIM
        idx_scale = tl.load(
            idx_scales_ptr + src_page * IDX_SCALE_PAGE_STRIDE + token_in_page,
            mask=idx_valid,
            other=0.0,
        )
        idx = idx * idx_scale
    tl.store(idx_scratch_ptr + dst_idx, idx, mask=idx_valid)


def _scatter_cp_prefix_pages(
    main_pages: torch.Tensor,
    idx_pages: torch.Tensor,
    idx_scales: Optional[torch.Tensor],
    dst_pages: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    idx_scratch: torch.Tensor,
    src_pages: Optional[torch.Tensor] = None,
) -> None:
    """Gather BF16/E4M3 pool pages into contiguous HND working pages.

    Source pools may have a larger stride between pages, as in hybrid KV
    storage. ``src_pages`` maps compact destination pages to physical pages.
    """
    page_count = int(dst_pages.numel())
    if page_count == 0:
        return
    if main_pages.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(
            f"prefix main pages must be BF16 or E4M3, got {main_pages.dtype}"
        )
    if k_paged.dtype != torch.bfloat16 or v_paged.dtype != torch.bfloat16:
        raise ValueError(
            "prefix main destination must be BF16 working pages: "
            f"src={main_pages.dtype} K={k_paged.dtype} V={v_paged.dtype}"
        )
    if idx_pages.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ) or idx_scratch.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise ValueError(
            "prefix idx pages and scratch must use BF16 or E4M3, got "
            f"src={idx_pages.dtype} dst={idx_scratch.dtype}"
        )
    if src_pages is None:
        src_pages = dst_pages
        use_src_pages = False
    else:
        if int(src_pages.numel()) != page_count:
            raise ValueError(
                f"prefix source page count {src_pages.numel()} != {page_count}"
            )
        use_src_pages = True
    tensors = (
        main_pages,
        idx_pages,
        idx_scales if idx_scales is not None else idx_pages,
        dst_pages,
        src_pages,
        k_paged,
        v_paged,
        idx_scratch,
    )
    if not all(t.is_cuda for t in tensors):
        raise ValueError("fused prefix-page scatter requires CUDA tensors")
    if not all(
        t.is_contiguous() for t in (dst_pages, src_pages, k_paged, v_paged, idx_scratch)
    ):
        raise ValueError(
            "fused prefix-page scatter requires contiguous destinations and page ids"
        )
    if (
        not main_pages[0, 0].is_contiguous()
        or int(main_pages.stride(1)) != int(main_pages[0, 0].numel())
        or not idx_pages[0].is_contiguous()
        or (idx_scales is not None and not idx_scales[0].is_contiguous())
    ):
        raise ValueError(
            "fused prefix-page scatter requires contiguous data within each page"
        )
    if int(main_pages.shape[0]) < page_count or int(idx_pages.shape[0]) < page_count:
        raise ValueError(
            "prefix source has fewer pages than the logical restore: "
            f"main={main_pages.shape[0]} idx={idx_pages.shape[0]} dst={page_count}"
        )

    main_page_elems = int(main_pages[0, 0].numel())
    idx_page_elems = int(idx_pages[0].numel())
    if (
        int(k_paged[0].numel()) != main_page_elems
        or int(v_paged[0].numel()) != main_page_elems
    ):
        raise ValueError("prefix main source and destination page shapes differ")
    if int(idx_scratch.numel()) % int(k_paged.shape[0]) != 0:
        raise ValueError("idx scratch cannot be addressed by the main page namespace")
    if int(idx_scratch.numel()) // int(k_paged.shape[0]) != idx_page_elems:
        raise ValueError("prefix idx source and destination page shapes differ")

    copy_block = min(4096, triton.next_power_of_2(main_page_elems))
    if main_page_elems < idx_page_elems or main_page_elems % copy_block != 0:
        raise ValueError(
            "fused prefix-page scatter requires the main page to cover idx and "
            f"align to {copy_block} elements: main={main_page_elems} "
            f"idx={idx_page_elems}"
        )
    grid = (
        page_count,
        triton.cdiv(max(main_page_elems, idx_page_elems), copy_block),
    )
    _scatter_cp_prefix_pages_kernel[grid](
        main_pages,
        idx_pages,
        idx_scales if idx_scales is not None else idx_pages,
        dst_pages,
        src_pages,
        k_paged,
        v_paged,
        idx_scratch,
        page_count,
        MAIN_PAGE_ELEMS=main_page_elems,
        IDX_PAGE_ELEMS=idx_page_elems,
        IDX_DIM=int(idx_pages.shape[-1]),
        MAIN_SRC_PAGE_STRIDE=int(main_pages.stride(0)),
        MAIN_SRC_V_OFFSET=int(main_pages.stride(1)),
        IDX_SRC_PAGE_STRIDE=int(idx_pages.stride(0)),
        IDX_SCALE_PAGE_STRIDE=(0 if idx_scales is None else int(idx_scales.stride(0))),
        COPY_BLOCK=copy_block,
        USE_SRC_PAGES=use_src_pages,
        HAS_IDX_SCALE=idx_scales is not None,
        num_warps=8,
    )


def _gemma_rmsnorm_per_head(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Per-head RMSNorm over the last dim using the loaded gamma.

    MiniMax-M3 weight loading already bakes Gemma's ``+1`` offset into norm
    weights, matching the dense Q/K norm path — so this is plain RMSNorm and
    we route it through flashinfer's fused kernel instead of a Python op
    chain (cast/pow/mean/rsqrt/mul/cast). Last-dim reduction means the (T,H,D)
    input can be reshaped to (T*H, D) where each row is normalized
    independently against the shared D-dim weight.
    """
    import flashinfer.norm

    orig_shape = x.shape
    return flashinfer.norm.rmsnorm(
        x.reshape(-1, orig_shape[-1]).contiguous(), weight, eps=eps
    ).view(orig_shape)


class _IdxKScratch:
    """Shared compact idx_K working tensor consumed by FMHA index scoring."""

    def __init__(self) -> None:
        self._t: Optional[torch.Tensor] = None
        self._graph_tensors: Dict[tuple, torch.Tensor] = {}

    def acquire(
        self,
        slots: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
        graph_stable: bool = False,
    ):
        if graph_stable:
            key = (str(device), slots, heads, dim, dtype)
            tensor = self._graph_tensors.get(key)
            if tensor is None:
                tensor = torch.zeros(slots, heads, dim, dtype=dtype, device=device)
                self._graph_tensors[key] = tensor
            return tensor
        if (
            self._t is None
            or self._t.shape[0] < slots
            or self._t.shape[1] != heads
            or self._t.shape[2] != dim
            or self._t.dtype != dtype
            or self._t.device != device
        ):
            self._t = torch.zeros(slots, heads, dim, dtype=dtype, device=device)
        return self._t[:slots]


_IDX_K_SCRATCH = _IdxKScratch()


class _Bf16WorkingPages:
    """Process-wide shared BF16 HND working pages for CP paged prefill.

    Each MSA sparse layer used to ``torch.empty`` a fresh
    ``[2, page_count, kv_heads, page, dim]`` buffer (~2.3 GiB at bs16/75k).
    With CP side/prefetch streams those frees are deferred across stream
    events, so N layer-local copies piled up in the caching allocator
    (~12 gens -> 30GB+). Sparse layers run strictly sequentially, so one
    grown-on-demand buffer is enough — footprint stays 1x.
    """

    def __init__(self) -> None:
        self._kv: Optional[torch.Tensor] = None
        self._graph_kv: Dict[tuple, torch.Tensor] = {}

    def acquire(
        self,
        page_count: int,
        heads: int,
        page_size: int,
        dim: int,
        device: torch.device,
        graph_stable: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if graph_stable:
            key = (str(device), page_count, heads, page_size, dim)
            kv = self._graph_kv.get(key)
            if kv is None:
                kv = torch.empty(
                    (2, page_count, heads, page_size, dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                self._graph_kv[key] = kv
            return kv[0], kv[1]
        kv = self._kv
        if (
            kv is None
            or kv.shape[1] < page_count
            or kv.shape[2] != heads
            or kv.shape[3] != page_size
            or kv.shape[4] != dim
            or kv.dtype != torch.bfloat16
            or kv.device != device
        ):
            # Leading pair dim keeps K/V individually contiguous (base[:, 0]
            # would not be). Match the previous per-layer empty() contract.
            kv = torch.empty(
                (2, page_count, heads, page_size, dim),
                dtype=torch.bfloat16,
                device=device,
            )
            self._kv = kv
        return kv[0, :page_count], kv[1, :page_count]


_BF16_WORKING_PAGES = _Bf16WorkingPages()


class _Nvfp4WorkingPages:
    """One process-wide packed working set for native CP sparse prefill.

    Prefix pages are copied here in their opaque packed representation after
    the CP page-RR gather.  The already all-gathered suffix is quantized
    directly into the same logical page namespace.  Sparse layers execute
    serially, so the largest observed shape can be reused across layers and
    requests without retaining one working set per layer.
    """

    def __init__(self) -> None:
        self._main: Optional[torch.Tensor] = None
        self._main_scales: Optional[torch.Tensor] = None
        self._idx: Optional[torch.Tensor] = None
        self._idx_scales: Optional[torch.Tensor] = None

    def acquire(
        self,
        page_count: int,
        heads: int,
        page_size: int,
        dim: int,
        index_dim: int,
        device: torch.device,
    ):
        groups = dim // NVFP4_GROUP_SIZE
        index_groups = index_dim // NVFP4_GROUP_SIZE
        main = self._main
        main_scales = self._main_scales
        idx = self._idx
        idx_scales = self._idx_scales
        if (
            main is None
            or main_scales is None
            or idx is None
            or idx_scales is None
            or main.shape[1] < page_count
            or tuple(main.shape[2:]) != (heads, page_size, dim // 2)
            or tuple(idx.shape[1:]) != (1, page_size, index_dim // 2)
            or main.device != device
        ):
            main = torch.empty(
                2,
                page_count,
                heads,
                page_size,
                dim // 2,
                dtype=torch.uint8,
                device=device,
            )
            main_scales = torch.empty(
                2,
                page_count,
                heads * page_size * groups,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            idx = torch.empty(
                page_count,
                1,
                page_size,
                index_dim // 2,
                dtype=torch.uint8,
                device=device,
            )
            idx_scales = torch.empty(
                page_count,
                page_size * index_groups,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            self._main = main
            self._main_scales = main_scales
            self._idx = idx
            self._idx_scales = idx_scales
        return (
            main[:, :page_count],
            main_scales[:, :page_count],
            idx[:page_count],
            idx_scales[:page_count],
        )


_NVFP4_WORKING_PAGES = _Nvfp4WorkingPages()


class _RopeDummyScratch:

    def __init__(self) -> None:
        self._t: Optional[torch.Tensor] = None

    def acquire(self, rows: int, heads: int, dim: int, dtype, device):
        t = self._t
        if (
            t is None
            or t.shape[0] < rows
            or t.shape[1] != heads
            or t.shape[2] != dim
            or t.dtype != dtype
            or t.device != device
        ):
            t = torch.zeros(rows, heads, dim, dtype=dtype, device=device)
            self._t = t
        return t[:rows]


_ROPE_DUMMY_SCRATCH = _RopeDummyScratch()


class _Mxfp8FusedQKVIndexProj(nn.Module):
    """MXFP8 output-dim concat projection for QKV + idx_Q + idx_K.

    Prefill and decode can share this projection when QKV is also MXFP8.
    Otherwise the index branch uses its own MXFP8 projection.
    """

    def __init__(self, fused_linear: CudaMxfp8Linear) -> None:
        super().__init__()
        self.fused_linear = fused_linear

    @staticmethod
    def _valid_weight_shapes(
        qkv_w: Optional[torch.Tensor],
        idx_q_w: torch.Tensor,
        idx_k_w: torch.Tensor,
        expected_qkv_dim: int,
    ) -> bool:
        return (
            qkv_w is not None
            and qkv_w.dim() == 2
            and idx_q_w.dim() == 2
            and idx_k_w.dim() == 2
            and int(qkv_w.shape[0]) == int(expected_qkv_dim)
            and int(qkv_w.shape[1]) == int(idx_q_w.shape[1])
            and int(qkv_w.shape[1]) == int(idx_k_w.shape[1])
        )

    @staticmethod
    def _mxfp8_scale_inv_to_weight_scale(
        weight: torch.Tensor, scale_inv: torch.Tensor
    ) -> torch.Tensor:
        if weight.dtype != torch.float8_e4m3fn or scale_inv.dtype != torch.uint8:
            raise ValueError(
                "MSA idx weight must be float8_e4m3fn with uint8 scale_inv"
            )
        if weight.dim() != 2 or scale_inv.dim() != 2:
            raise ValueError("MSA idx weight and scale_inv must be 2D")
        n, k = weight.shape
        expected = (int(k) + 31) // 32
        if int(scale_inv.shape[0]) != int(n) or int(scale_inv.shape[1]) != expected:
            raise ValueError(
                f"MSA idx scale_inv shape mismatch: weight={tuple(weight.shape)}, "
                f"scale={tuple(scale_inv.shape)}, expected second dim {expected}"
            )
        return torch.exp2(scale_inv.to(torch.float32) - 127.0).contiguous()

    @classmethod
    @torch.inference_mode()
    def build(
        cls,
        qkv_proj: nn.Module,
        expected_qkv_dim: int,
        idx_q_w: Optional[torch.Tensor],
        idx_q_s: Optional[torch.Tensor],
        idx_k_w: Optional[torch.Tensor],
        idx_k_s: Optional[torch.Tensor],
    ) -> Optional["_Mxfp8FusedQKVIndexProj"]:
        if not isinstance(qkv_proj, CudaMxfp8Linear):
            return None
        qkv_w = getattr(qkv_proj, "weight", None)
        qkv_s = getattr(qkv_proj, "weight_scale", None)
        qkv_b = getattr(qkv_proj, "bias", None)
        if (
            qkv_b is not None
            or qkv_w is None
            or qkv_w.dtype != torch.float8_e4m3fn
            or qkv_s is None
            or qkv_s.dtype != torch.float32
            or idx_q_w is None
            or idx_q_s is None
            or idx_k_w is None
            or idx_k_s is None
        ):
            return None
        if not cls._valid_weight_shapes(qkv_w, idx_q_w, idx_k_w, expected_qkv_dim):
            return None
        scale_cols = (int(qkv_w.shape[1]) + 31) // 32
        if (
            int(qkv_s.shape[0]) != int(qkv_w.shape[0])
            or int(qkv_s.shape[1]) != scale_cols
        ):
            return None

        idx_q_weight_scale = cls._mxfp8_scale_inv_to_weight_scale(idx_q_w, idx_q_s)
        idx_k_weight_scale = cls._mxfp8_scale_inv_to_weight_scale(idx_k_w, idx_k_s)
        if (
            int(idx_q_weight_scale.shape[1]) != scale_cols
            or int(idx_k_weight_scale.shape[1]) != scale_cols
        ):
            return None
        fused_w = torch.cat(
            [qkv_w.contiguous(), idx_q_w.contiguous(), idx_k_w.contiguous()],
            dim=0,
        ).contiguous()
        fused_s = torch.cat(
            [qkv_s.contiguous(), idx_q_weight_scale, idx_k_weight_scale], dim=0
        ).contiguous()
        fused_linear = CudaMxfp8Linear(
            weight=fused_w,
            weight_scales=fused_s,
            input_scales=None,
            bias=None,
            quant_config=None,
        )
        # Build-time packing keeps the first decode/capture step free of this
        # one-time MXFP8 scale transform.
        fused_linear._packed_weight_scale()
        return cls(fused_linear=fused_linear)

    def forward(
        self,
        x: torch.Tensor,
        input_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.fused_linear(x, input_scales=input_scales)


class MSAAttention(nn.Module):
    """MiniMax-M3 sparse attention for a single sparse layer."""

    # Keep the feature flag defined even for lightweight test doubles that
    # construct the module with ``__new__`` and therefore skip ``__init__``.
    # Real instances overwrite this with the per-layer AttentionConfig value.
    nvfp4_kv_cache: bool = False

    # Shared prefill metadata and working pages are reused across sparse layers.
    _cp_shared_meta: Optional[Dict[str, Any]] = None
    _prefill_shared_meta: Optional[Dict[str, Any]] = None
    _paged_decode_shared_meta: Optional[Dict[str, Any]] = None
    _target_verify_shared_meta: Optional[Dict[str, Any]] = None
    _cp_side_stream: Dict[torch.device, torch.cuda.Stream] = {}
    _cp_side_event: Dict[torch.device, torch.cuda.Event] = {}
    _cp_prefetch_stream: Dict[torch.device, torch.cuda.Stream] = {}
    _cp_prefetch_entries: Dict[int, Dict[str, Any]] = {}
    _cp_native_prefetch_buffers: Dict[tuple, Dict[str, Any]] = {}
    _cp_prefetch_disabled: bool = False

    def _maybe_build_mxfp8_fused_qkv_idx_proj(self) -> None:
        self._mxfp8_idx_proj = None
        if not self._has_raw_mxfp8_idx_weights:
            self._mxfp8_fused_qkv_idx_proj = None
            self._can_use_mxfp8_fused_qkv_idx_decode = False
            return

        expected_qkv_dim = self.q_size + 2 * self.kv_size
        self._mxfp8_fused_qkv_idx_proj = _Mxfp8FusedQKVIndexProj.build(
            self.qkv_proj,
            expected_qkv_dim,
            self.idx_q_raw_w,
            self.idx_q_raw_s,
            self.idx_k_raw_w,
            self.idx_k_raw_s,
        )
        if self._mxfp8_fused_qkv_idx_proj is None:
            idx_q_scale = _Mxfp8FusedQKVIndexProj._mxfp8_scale_inv_to_weight_scale(
                self.idx_q_raw_w, self.idx_q_raw_s
            )
            idx_k_scale = _Mxfp8FusedQKVIndexProj._mxfp8_scale_inv_to_weight_scale(
                self.idx_k_raw_w, self.idx_k_raw_s
            )
            self._mxfp8_idx_proj = CudaMxfp8Linear(
                weight=torch.cat((self.idx_q_raw_w, self.idx_k_raw_w), dim=0),
                weight_scales=torch.cat((idx_q_scale, idx_k_scale), dim=0),
                input_scales=None,
                bias=None,
                quant_config=None,
            )
            self._mxfp8_idx_proj._packed_weight_scale()
        fused_decode_ready = (
            self._mxfp8_fused_qkv_idx_proj is not None
            and self.qk_fuse_norm is not None
            and self.cos_sin_cache is not None
            and not self._rope_interleave
            and int(self.head_dim) == int(self.idx_head_dim)
            and int(self.rotary_dim) <= int(self.head_dim)
        )
        self._can_use_mxfp8_fused_qkv_idx_decode = fused_decode_ready

    def _project_qkv_idx(
        self,
        hidden_states: torch.Tensor,
        x_fp8: Optional[torch.Tensor],
        x_scale: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project QKV and index Q/K using the checkpoint's MXFP8 idx weights."""
        use_quantized_input = x_fp8 is not None and x_scale is not None
        projection_input = x_fp8 if use_quantized_input else hidden_states
        projection_scale = x_scale if use_quantized_input else None
        fused = getattr(self, "_mxfp8_fused_qkv_idx_proj", None)
        if fused is not None:
            projected = fused(projection_input, input_scales=projection_scale)
            return torch.split(
                projected,
                [
                    self.q_size + 2 * self.kv_size,
                    self.num_idx_heads * self.idx_head_dim,
                    self.idx_head_dim,
                ],
                dim=-1,
            )
        if use_quantized_input and isinstance(self.qkv_proj, CudaMxfp8Linear):
            qkv = self.qkv_proj(x_fp8, input_scales=x_scale)
        else:
            qkv = self.qkv_proj(hidden_states)
        idx_proj = getattr(self, "_mxfp8_idx_proj", None)
        if idx_proj is not None:
            idx = idx_proj(projection_input, input_scales=projection_scale)
            idx_q, idx_k = torch.split(
                idx,
                [self.num_idx_heads * self.idx_head_dim, self.idx_head_dim],
                dim=-1,
            )
        else:
            idx_q = F.linear(hidden_states, self.idx_q_w)
            idx_k = F.linear(hidden_states, self.idx_k_w)
        return qkv, idx_q, idx_k

    def _should_use_mxfp8_fused_qkv_idx_decode(
        self,
        x_fp8: Optional[torch.Tensor],
        x_scale: Optional[torch.Tensor],
    ) -> bool:
        return (
            not self.nvfp4_kv_cache
            and self._m31_raw_attention_norms is None
            and self._can_use_mxfp8_fused_qkv_idx_decode
            and x_fp8 is not None
            and x_scale is not None
        )

    def _decode_project_fused_qkv_idx(
        self,
        total_tokens: int,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        phys_block_table: torch.Tensor,
        paged_kv_base: torch.Tensor,
        paged_idx_k: torch.Tensor,
        paged_idx_k_scale: Optional[torch.Tensor],
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
    ):
        fused_qkv_idx = self._mxfp8_fused_qkv_idx_proj(x_fp8, input_scales=x_scale)
        if (
            paged_kv_base.dtype not in (fused_qkv_idx.dtype, torch.float8_e4m3fn)
            or paged_idx_k.dtype != self._idx_k_persistent_dtype
            or (paged_idx_k_scale is not None) != (self.idx_k_fp8_mode > 0)
        ):
            raise RuntimeError(
                "MXFP8 fused paged decode requires the idx_K "
                "side region to match the configured cache mode: BF16 values "
                "for mode 0, scaled-E4M3 values plus per-token scales for "
                "mode 1/2."
            )
        q = torch.empty(
            total_tokens,
            self.head_num,
            self.head_dim,
            dtype=fused_qkv_idx.dtype,
            device=fused_qkv_idx.device,
        )
        idx_q = torch.empty(
            total_tokens,
            self.num_idx_heads,
            self.idx_head_dim,
            dtype=fused_qkv_idx.dtype,
            device=fused_qkv_idx.device,
        )
        _fused_qk_idx_norm_rope_write_paged_decode(
            fused_qkv_idx,
            q,
            idx_q,
            self.qk_fuse_norm.q_weight,
            self.qk_fuse_norm.k_weight,
            self.idx_q_norm_w,
            self.idx_k_norm_w,
            self.cos_sin_cache,
            positions,
            seq_lens,
            phys_block_table,
            paged_kv_base,
            paged_idx_k,
            paged_idx_k_scale,
            int(self.page_size),
            self.head_dim,
            self.rotary_dim,
            self.head_num,
            self.kv_head_num,
            self.num_idx_heads,
            self.layernorm_eps,
        )
        return q, idx_q

    def __init__(
        self,
        attn_config: AttentionConfigs,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        sparse_config: Dict[str, Any],
        layer_idx: int,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.parallelism_config = parallelism_config
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.tp_rank = parallelism_config.get_attn_tp_rank()
        self.layernorm_eps = layernorm_eps

        # CP (context parallelism) uses the raw TP dimension for sequence
        # splitting. get_attn_tp_size() returns 1 when CP is active so weights
        # are NOT sharded, but tp_size/tp_rank still identify the CP group.
        cp_cfg = parallelism_config.prefill_cp_config
        self.cp_enabled = cp_cfg.method.value != 0  # NONE = 0
        # CP page-RR KV sharding geometry. Mirrors C++ DeviceData::props:
        #   sharded = prefill_cp enabled AND kv_cache_sharded AND raw tp_size>1.
        # The CP group is the raw TP dimension (get_attn_tp_size()==1 under CP).
        # When not sharded, cp_size=1 makes the slot mapping a plain global-slot
        # passthrough (bit-equal to the pre-sharding global-slot behaviour).
        raw_tp_size = int(parallelism_config.tp_size)
        raw_tp_rank = int(parallelism_config.tp_rank)
        self._kv_sharded = bool(
            self.cp_enabled
            and getattr(cp_cfg, "kv_cache_sharded", False)
            and raw_tp_size > 1
        )
        self._cp_size = raw_tp_size if self._kv_sharded else 1
        self._cp_rank = raw_tp_rank if self._kv_sharded else 0
        self.head_num = attn_config.head_num
        self.kv_head_num = attn_config.kv_head_num
        self.head_dim = attn_config.size_per_head
        self.q_size = self.head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.page_size = attn_config.kernel_tokens_per_block
        self.physical_page_size = attn_config.tokens_per_block
        self.nvfp4_kv_cache = bool(getattr(attn_config, "nvfp4_kv_cache", False))
        if self.nvfp4_kv_cache and self.layer_idx == 0:
            logger.info("MiniMax-M3.1 NVFP4 uses native packed-FP4 prefill/decode")

        # --- main GQA branch (identical construction to CausalAttention) ---
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_qkv_w,
            W.attn_qkv_s,
            W.attn_qkv_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_qkv_s2,
            input_scale_key=W.attn_qkv_i_s,
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_o_w,
            W.attn_o_s,
            W.attn_o_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_o_s2,
            input_scale_key=W.attn_o_i_s,
        )
        self.o_proj.maybe_cache_quant_scale(1024)

        raw_norm_keys = (
            "minimax_m31.raw_q_norm",
            "minimax_m31.raw_k_norm",
            "minimax_m31.raw_index_q_norm",
            "minimax_m31.raw_index_k_norm",
        )
        raw_norm_count = sum(key in weights for key in raw_norm_keys)
        if raw_norm_count not in (0, 4):
            raise ValueError("M3.1 fused norm/RoPE requires all four raw norm weights")
        self._m31_raw_attention_norms = (
            tuple(weights[key] for key in raw_norm_keys) if raw_norm_count else None
        )
        self.qk_fuse_norm = None
        if W.q_ln_gamma in weights and W.k_ln_gamma in weights:
            self.qk_fuse_norm = FusedQKRMSNorm(
                weights[W.q_ln_gamma],
                weights[W.k_ln_gamma],
                self.head_num,
                self.kv_head_num,
                self.head_dim,
                layernorm_eps,
            )

        # --- index branch ---
        # Native checkpoints keep MXFP8 idx weights and scales. BF16 idx
        # weights are accepted only for checkpoints that store BF16 directly.
        self.idx_head_dim = int(sparse_config["idx_head_dim"])
        self.idx_k_fp8_mode = int(sparse_config.get("idx_k_fp8_mode", 0))
        if self.idx_k_fp8_mode not in (0, 1, 2, 3):
            raise ValueError(
                f"invalid idx_k_fp8_mode={self.idx_k_fp8_mode}; "
                "expected 0, 1, 2, or internal NVFP4 mode 3"
            )
        if self.nvfp4_kv_cache != (self.idx_k_fp8_mode == 3):
            raise ValueError(
                "MiniMax-M3 NVFP4 main K/V and indexer-K cache modes must be "
                f"enabled together (nvfp4={self.nvfp4_kv_cache}, "
                f"idx_mode={self.idx_k_fp8_mode})"
            )
        self._idx_k_persistent_dtype = (
            torch.uint8
            if self.idx_k_fp8_mode == 3
            else (torch.float8_e4m3fn if self.idx_k_fp8_mode > 0 else torch.bfloat16)
        )
        self._idx_k_working_dtype = (
            torch.float8_e4m3fn if self.idx_k_fp8_mode == 2 else torch.bfloat16
        )
        self.idx_q_norm_w = weights[W.msa_idx_q_norm]  # [idx_dim]
        self.idx_k_norm_w = weights[W.msa_idx_k_norm]  # [idx_dim]
        has_bf16_idx_w = W.msa_idx_q_w in weights and W.msa_idx_k_w in weights
        raw_idx_key_count = sum(
            key in weights
            for key in (
                W.msa_idx_q_raw_w,
                W.msa_idx_q_raw_s,
                W.msa_idx_k_raw_w,
                W.msa_idx_k_raw_s,
            )
        )
        if raw_idx_key_count not in (0, 4) or (
            not has_bf16_idx_w and raw_idx_key_count == 0
        ):
            raise RuntimeError(
                "MSA idx weights require BF16 q/k weights or all four raw "
                "MXFP8 q/k weights and scales."
            )

        self.idx_q_w: Optional[torch.Tensor] = None
        self.idx_k_w: Optional[torch.Tensor] = None
        self.idx_q_raw_w: Optional[torch.Tensor] = None
        self.idx_q_raw_s: Optional[torch.Tensor] = None
        self.idx_k_raw_w: Optional[torch.Tensor] = None
        self.idx_k_raw_s: Optional[torch.Tensor] = None
        self._has_raw_mxfp8_idx_weights = raw_idx_key_count == 4

        full_idx_q_w_for_heads = weights[
            W.msa_idx_q_raw_w if self._has_raw_mxfp8_idx_weights else W.msa_idx_q_w
        ]
        self.total_idx_heads = int(
            sparse_config.get(
                "num_idx_heads", full_idx_q_w_for_heads.shape[0] // self.idx_head_dim
            )
        )
        self.num_idx_heads = self._local_idx_heads()
        loaded_idx_heads = full_idx_q_w_for_heads.shape[0] // self.idx_head_dim
        if loaded_idx_heads == self.total_idx_heads:
            start_head = self.idx_head_rank * self.num_idx_heads
            start = start_head * self.idx_head_dim
            end = start + self.num_idx_heads * self.idx_head_dim
        elif loaded_idx_heads == self.num_idx_heads:
            start = 0
            end = full_idx_q_w_for_heads.shape[0]
        else:
            raise RuntimeError(
                "unexpected MSA index_q weight shape: "
                f"loaded_idx_heads={loaded_idx_heads}, "
                f"total_idx_heads={self.total_idx_heads}, "
                f"local_idx_heads={self.num_idx_heads}"
            )

        if has_bf16_idx_w:
            self.idx_q_w = weights[W.msa_idx_q_w][start:end].contiguous()
            self.idx_k_w = weights[W.msa_idx_k_w].contiguous()
        if self._has_raw_mxfp8_idx_weights:
            self.idx_q_raw_w = weights[W.msa_idx_q_raw_w][start:end].contiguous()
            self.idx_q_raw_s = weights[W.msa_idx_q_raw_s][start:end].contiguous()
            self.idx_k_raw_w = weights[W.msa_idx_k_raw_w].contiguous()
            self.idx_k_raw_s = weights[W.msa_idx_k_raw_s].contiguous()

        self._mxfp8_fused_qkv_idx_proj: Optional[_Mxfp8FusedQKVIndexProj] = None
        self._can_use_mxfp8_fused_qkv_idx_decode = False

        # --- sparse params ---
        self.topk_blocks = int(sparse_config["topk_blocks"])
        self.block_size = int(sparse_config["block_size"])
        self.init_blocks = int(sparse_config["init_blocks"])
        self.local_blocks = int(sparse_config["local_blocks"])
        self.score_type = str(sparse_config.get("score_type", "max"))
        self.disable_index_value = layer_idx in set(
            sparse_config.get("disable_value_layer_ids", [])
        )

        # --- partial RoPE cos/sin cache.  Match the dense C++ fused RoPE
        # path for M3: rope_style=1 uses the non-interleaved LLaMA layout.
        from rtp_llm.ops import get_rope_cache_once

        self._rope_theta = attn_config.rope_config.base
        self._rope_interleave = False
        try:
            self._cuda_graph_max_seq_len = int(attn_config.max_seq_len)
            rope_cache_len = int(
                attn_config.max_seq_len + attn_config.gen_num_per_cycle + 1
            )
            rope_cache = get_rope_cache_once(
                attn_config.rope_config,
                rope_cache_len,
                is_cuda=True,
                interleave=self._rope_interleave,
            )
            self.cos_sin_cache = rope_cache.data
            self.rotary_dim = self.cos_sin_cache.shape[1]
        except Exception:
            self.cos_sin_cache = None
            self.rotary_dim = 0

        self._maybe_build_mxfp8_fused_qkv_idx_proj()

        # Persistent main K/V and idx_K share the cache-manager page table.
        # CP prefill uses a request-local paged working set.
        self._scratch_batch_size = 0
        self._scratch_seq_len = 0

        # The index-score path still uses a compact per-request idx_K tensor.
        self._scratch_idx_k: Optional[torch.Tensor] = None
        # CP request-local page namespace capacity.
        self._scratch_slots = 0
        self._paged_decode_static_ok: Optional[bool] = None

    def _paged_kv_base_view(self, kv_cache: LayerKVCache) -> Optional[torch.Tensor]:
        base = None if kv_cache is None else kv_cache.kv_cache_base
        if self.nvfp4_kv_cache:
            return base
        if base is None or base.dim() != 2:
            return base
        from rtp_llm.models_py.modules.factory.attention.common import (
            reshape_paged_kv_cache,
        )

        return reshape_paged_kv_cache(
            base, self.kv_head_num, self.physical_page_size, self.head_dim
        )

    def _check_paged_decode_static(self, kv_cache: LayerKVCache) -> bool:
        if self.nvfp4_kv_cache:
            if (
                kv_cache is None
                or self._kv_sharded
                or int(self.page_size) != int(self.block_size)
                or int(self.page_size) != int(self.physical_page_size)
                or (not self.disable_index_value)
            ):
                return False
            try:
                layout = nvfp4_cache_layout(
                    kv_cache.kv_cache_base,
                    kv_cache.kv_scale_base,
                    self.kv_head_num,
                    self.physical_page_size,
                    self.head_dim,
                )
                layout.indexer(self.idx_head_dim)
            except (RuntimeError, ValueError):
                return False
            return True
        if (
            kv_cache is None
            or self._kv_sharded
            or int(self.page_size) != int(self.block_size)
            or int(self.page_size) != int(self.physical_page_size)
            or (not self.disable_index_value)
        ):
            return False

        base = self._paged_kv_base_view(kv_cache)
        scale = kv_cache.kv_scale_base
        if (
            base is None
            or base.dim() != 5
            or base.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
            or int(base.shape[2]) != int(self.kv_head_num)
            or int(base.shape[3]) != int(self.page_size)
            or int(base.shape[4]) != int(self.head_dim)
            or scale is None
            or scale.dim() != 2
            or scale.stride(-1) != 1
        ):
            return False

        actual_bytes = int(scale.shape[1]) * int(scale.element_size())
        expected_bytes = int(self.page_size) * (
            int(self.idx_head_dim) * 2
            if self.idx_k_fp8_mode == 0
            else int(self.idx_head_dim) + 4
        )
        return actual_bytes == expected_bytes

    def _use_paged_decode_path(
        self, attn_inputs: PyAttentionInputs, kv_cache: LayerKVCache
    ) -> bool:
        if attn_inputs.is_prefill:
            return False
        if self._paged_decode_static_ok is None:
            self._paged_decode_static_ok = self._check_paged_decode_static(kv_cache)
        return self._paged_decode_static_ok

    def _paged_decode_addressing(
        self, attn_inputs: PyAttentionInputs, device: torch.device
    ):
        # Sparse layers execute in increasing layer order within one decode step.
        cache = MSAAttention._paged_decode_shared_meta
        if (
            cache is not None
            and cache.get("owner") is attn_inputs
            and cache["layer_idx"] < self.layer_idx
        ):
            cache["layer_idx"] = self.layer_idx
            return cache["addressing"]

        seq = attn_inputs.sequence_lengths
        phys_block_table = self._physical_block_table(attn_inputs)
        prefix_i64 = seq.to(device=device, dtype=torch.int64)
        kv_lens = prefix_i64 + 1
        seq_lens = kv_lens.to(torch.int32)
        positions = prefix_i64.to(torch.int32)
        addressing = (kv_lens, seq_lens, positions, phys_block_table)
        MSAAttention._paged_decode_shared_meta = {
            "owner": attn_inputs,
            "layer_idx": self.layer_idx,
            "addressing": addressing,
        }
        return addressing

    def _target_verify_addressing(
        self,
        attn_inputs: PyAttentionInputs,
        total_tokens: int,
        device: torch.device,
        use_fused_cuda: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sparse layers execute in increasing layer order within one target
        # forward. The MSA request block table, positions, and validity metadata
        # are shared across those layers, so expand them once in the first sparse
        # layer. The cache is process-wide because the layers are separate module
        # instances, so its owner must also match: target verify and MTP draft
        # prefill/refresh reuse this path in one process with overlapping indices.
        cache = MSAAttention._target_verify_shared_meta
        if (
            cache is not None
            and cache.get("owner") is attn_inputs
            and cache["layer_idx"] < self.layer_idx
        ):
            cache["layer_idx"] = self.layer_idx
            return cache["addressing"]

        prefix_lengths = attn_inputs.prefix_lengths
        input_lengths = attn_inputs.input_lengths
        request_block_table = self._physical_block_table(attn_inputs)
        phys_block_table, positions, seq_lens, valid_token_mask = (
            _prepare_target_verify_addressing(
                request_block_table,
                prefix_lengths,
                input_lengths,
                total_tokens,
                device,
                use_fused_cuda=use_fused_cuda,
                is_ragged=bool(getattr(attn_inputs, "is_ragged_target_verify", False)),
                cu_seqlens=attn_inputs.cu_seqlens,
            )
        )
        addressing = (
            request_block_table,
            phys_block_table,
            positions,
            seq_lens,
            valid_token_mask,
        )
        MSAAttention._target_verify_shared_meta = {
            "owner": attn_inputs,
            "layer_idx": self.layer_idx,
            "addressing": addressing,
        }
        return addressing

    @staticmethod
    def _cuda_graph_forward_active() -> bool:
        return (
            cuda_graph_capture_forward_enabled() or cuda_graph_warmup_forward_enabled()
        )

    def _cuda_graph_max_kv(
        self,
        attn_inputs: PyAttentionInputs,
        physical_block_table: Optional[torch.Tensor] = None,
    ) -> int:
        max_kv = int(self._cuda_graph_max_seq_len)
        bt = (
            physical_block_table
            if physical_block_table is not None
            else self._physical_block_table(attn_inputs)
        )
        if isinstance(bt, torch.Tensor) and bt.dim() >= 2:
            max_kv = min(max_kv, int(bt.shape[1]) * int(self.page_size))
        return max(max_kv, 1)

    def _paged_decode_max_kv(
        self,
        attn_inputs: PyAttentionInputs,
        kv_lens: torch.Tensor,
        physical_block_table: torch.Tensor,
    ) -> int:
        if self._cuda_graph_forward_active():
            return self._cuda_graph_max_kv(attn_inputs, physical_block_table)
        return int(kv_lens.max().item())

    def _local_idx_heads(self) -> int:
        """Match SGLang's GQA-style sharding for sparse index-Q heads."""
        if self.total_idx_heads >= self.tp_size:
            if self.total_idx_heads % self.tp_size != 0:
                raise RuntimeError(
                    "MSA index heads must be divisible by TP size: "
                    f"idx_heads={self.total_idx_heads}, tp_size={self.tp_size}"
                )
            self.idx_head_tp_size = self.tp_size
            self.idx_replica_size = 1
        else:
            if self.tp_size % self.total_idx_heads != 0:
                raise RuntimeError(
                    "TP size must be divisible by MSA index heads when "
                    f"tp_size > idx_heads: tp_size={self.tp_size}, "
                    f"idx_heads={self.total_idx_heads}"
                )
            self.idx_head_tp_size = self.total_idx_heads
            self.idx_replica_size = self.tp_size // self.idx_head_tp_size
        self.idx_head_rank = self.tp_rank // self.idx_replica_size
        return self.total_idx_heads // self.idx_head_tp_size

    def _fuse_m31_projected_norm_rope(
        self,
        qkv,
        idx_q,
        idx_k,
        positions,
        *,
        contiguous_outputs=None,
        query_fp8_outputs=None,
    ):
        """M3.1 raw-weight producer; callers must skip legacy norm and RoPE."""
        raw_weights = getattr(self, "_m31_raw_attention_norms", None)
        if raw_weights is None:
            return False
        if self._rope_interleave or self.head_dim != 128 or self.rotary_dim != 64:
            raise ValueError("M3.1 fused norm/RoPE requires partial NeoX 64/128")
        from rtp_llm.models_py.triton_kernels.minimax_m31_gemma_rope import (
            minimax_m31_gemma_norm_rope_,
        )

        rows = qkv.shape[0]
        minimax_m31_gemma_norm_rope_(
            qkv,
            idx_q.reshape(rows, self.num_idx_heads * self.idx_head_dim),
            idx_k.reshape(rows, self.idx_head_dim),
            raw_weights,
            positions,
            self.cos_sin_cache,
            num_q_heads=self.head_num,
            num_kv_heads=self.kv_head_num,
            num_index_heads=self.num_idx_heads,
            eps=self.layernorm_eps,
            contiguous_outputs=contiguous_outputs,
            query_fp8_outputs=query_fp8_outputs,
        )
        return True

    def _legacy_index_norm(self, rows, weight, eps):
        if getattr(self, "_m31_raw_attention_norms", None) is not None:
            return rows
        return _gemma_rmsnorm_per_head(rows, weight, eps)

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """In-place partial RoPE on q/k ([T, H, head_dim])."""
        import flashinfer.rope as fi_rope

        if self.cos_sin_cache is not None:
            fi_rope._apply_rope_pos_ids_cos_sin_cache(
                q=q,
                k=k,
                q_rope=q,
                k_rope=k,
                cos_sin_cache=self.cos_sin_cache,
                pos_ids=positions,
                interleave=self._rope_interleave,
            )
        else:
            import flashinfer

            flashinfer.apply_rope_pos_ids_inplace(
                q, k, positions, rope_theta=self._rope_theta
            )

    def _apply_rope_contiguous(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack projection views while applying the existing cached RoPE."""
        if self.cos_sin_cache is None or (q.is_contiguous() and k.is_contiguous()):
            q, k = q.contiguous(), k.contiguous()
            self._apply_rope(q, k, positions)
            return q, k

        import flashinfer.rope as fi_rope

        # Cached RoPE accepts separate input/output strides and writes the
        # non-rotary tail too. Keep its arithmetic, but eliminate the two
        # projection-to-contiguous copies preceding the old in-place call.
        q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        k_out = torch.empty(k.shape, device=k.device, dtype=k.dtype)
        fi_rope._apply_rope_pos_ids_cos_sin_cache(
            q=q,
            k=k,
            q_rope=q_out,
            k_rope=k_out,
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=positions,
            interleave=self._rope_interleave,
        )
        return q_out, k_out

    def _ensure_scratch_addressing_capacity(
        self,
        bsz: Optional[int] = None,
        max_kv: Optional[int] = None,
        exact_cp_shape: bool = False,
    ) -> None:
        if bsz is None or max_kv is None:
            raise RuntimeError("CP MSA working pages require batch size and KV length")
        if (
            not exact_cp_shape
            and self._scratch_slots > 0
            and self._scratch_batch_size >= int(bsz)
            and self._scratch_seq_len >= int(max_kv)
        ):
            return
        if exact_cp_shape:
            target_bsz = max(int(bsz), 1)
            requested_seq_len = max(int(max_kv), 1)
        else:
            target_bsz = max(int(bsz), self._scratch_batch_size, 1)
            requested_seq_len = max(int(max_kv), self._scratch_seq_len, 1)
        grow_granularity = max(int(self.page_size), 256)
        target_seq_len = (
            (requested_seq_len + grow_granularity - 1) // grow_granularity
        ) * grow_granularity
        self._scratch_batch_size = target_bsz
        self._scratch_seq_len = target_seq_len
        self._scratch_slots = target_bsz * target_seq_len

    def _get_lengths(self, attn_inputs: PyAttentionInputs):
        if attn_inputs.is_prefill:
            prefix = attn_inputs.prefix_lengths.to(torch.int64)
            inlen = attn_inputs.input_lengths.to(torch.int64)
            kv_lens = prefix + inlen
        else:
            seqlen = attn_inputs.sequence_lengths.to(torch.int64)
            kv_lens = seqlen + 1
            prefix = kv_lens - 1
            inlen = torch.ones_like(kv_lens)
        return kv_lens, prefix, inlen

    def _physical_block_table(self, attn_inputs: PyAttentionInputs) -> torch.Tensor:
        """Resolve this MSA layer's page table from shared request metadata."""
        gid = 0
        layer_to_group = getattr(attn_inputs, "kv_cache_layer_to_group", None)
        if (
            isinstance(layer_to_group, torch.Tensor)
            and layer_to_group.numel() > self.layer_idx
        ):
            gid = int(layer_to_group[self.layer_idx].item())

        # MSA uses 128-token physical and kernel pages, so the framework's
        # existing per-group kernel block table is also the physical page table.
        grouped_tables = getattr(
            attn_inputs, "kv_cache_kernel_block_id_device_by_group", None
        )
        if grouped_tables is not None and len(grouped_tables) > gid:
            if self.page_size != self.physical_page_size:
                raise RuntimeError(
                    "MSA cannot use a kernel block table as a physical page table when "
                    f"kernel_page_size={self.page_size} differs from "
                    f"physical_page_size={self.physical_page_size}"
                )
            group_table = grouped_tables[gid]
            if isinstance(group_table, torch.Tensor) and group_table.numel() > 0:
                return group_table

        phys = getattr(attn_inputs, "kv_cache_block_id_device", None)
        if isinstance(phys, torch.Tensor) and phys.numel() > 0:
            return phys
        return attn_inputs.kv_cache_kernel_block_id_device

    def _kernel_slots_to_paged(
        self, kernel_slots: torch.Tensor, attn_inputs: PyAttentionInputs
    ) -> torch.Tensor:
        """Map kernel-space slots to physical paged-pool slots.

        Three regimes (all addressed through the *physical* block table, the
        same table GLM5/DSV4 use for paged cache I/O):

        * non-CP: kernel slots are already global ``block*page+off`` → identity.
        * CP, full-replicated pool (``_cp_size == 1``): compact kernel slots
          ``b*scratch_seq_len + pos`` → resolve ``(b, pos)`` through the block
          table to the plain global slot.
        * CP page-RR sharded (``_cp_size > 1``): reuse GLM5/DSV4's
          ``cp_kv_slot_mapping`` (ratio=1, uncompressed MHA K/V). Non-owned
          tokens (and block-0 sentinels) become ``-1`` so the writer skips them.
        """
        ks = kernel_slots.to(torch.int64)
        if not self.cp_enabled:
            return ks
        seq_len = int(self._scratch_seq_len)
        b_idx = ks // seq_len
        positions = ks % seq_len
        bt = self._physical_block_table(attn_inputs).to(torch.int64)
        if not self._kv_sharded:
            blk = positions // self.page_size
            return bt[b_idx, blk] * self.page_size + (positions % self.page_size)
        from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
            cp_kv_slot_mapping,
        )

        return cp_kv_slot_mapping(
            positions,
            bt,
            b_idx,
            self.page_size,  # tokens_per_block
            self.page_size,  # kv_eb (entries per block, ratio=1)
            1,  # ratio (uncompressed)
            self._cp_size,
            self._cp_rank,
            owner_tokens_per_block=self.page_size,
        )

    def _write_cp_suffix_to_bf16_working_pages(
        self,
        kv_cache: LayerKVCache,
        packed: torch.Tensor,
        unpad_indices: torch.Tensor,
        write_slots: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_lens: torch.Tensor,
        nk: int,
        ni: int,
        token_count: int,
        *,
        write_main_pages: bool = True,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Persist suffix/idx and optionally materialize full working pages.

        Compact prefill skips BF16 main working pages while retaining persistent
        writes and the idx_K working tensor.
        The same launch persists rank-owned suffix pages and fills the configured
        BF16/FP8 idx_K working tensor. Main working pages come from ``_BF16_WORKING_PAGES``
        so all sparse layers of one forward share a single buffer.
        """
        if self.nvfp4_kv_cache:
            raise RuntimeError(
                "MiniMax-M3.1 NVFP4 must use the native fmha_fp4 prefill "
                "path; BF16 working-page materialization has been removed"
            )

        base = self._paged_kv_base_view(kv_cache)
        if base is None or base.dim() != 5:
            raise RuntimeError(
                "MSA paged main K/V requires a 5-D paged cache "
                "[block,2,head,page,dim], got "
                f"{None if base is None else tuple(base.shape)}"
            )
        if base.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise RuntimeError(
                "MSA CP prefill persistent KV pool must be BF16 or E4M3, "
                f"got {base.dtype}"
            )
        if packed.dtype != torch.bfloat16:
            raise RuntimeError(
                "MSA CP paged prefill requires BF16 packed K/V/idx, "
                f"got {packed.dtype}"
            )
        idx_view, idx_scale = self._idx_k_paged_storage(kv_cache)
        if idx_view.dtype != self._idx_k_persistent_dtype:
            raise RuntimeError(
                f"MSA paged idx_K dtype mismatch: paged={idx_view.dtype} vs "
                f"configured={self._idx_k_persistent_dtype}"
            )
        scratch_slots = int(self._scratch_slots)
        if scratch_slots % int(self.page_size) != 0:
            raise RuntimeError(
                f"MSA CP scratch slots {scratch_slots} are not page-aligned "
                f"to page_size={self.page_size}"
            )
        page_count = scratch_slots // int(self.page_size)
        device = packed.device
        # Process-wide pool: one BF16 HND working set shared across all sparse
        # layers of a forward. Avoids N x ~2GiB empties that pile up under CP
        # side/prefetch stream deferred frees.
        k_paged, v_paged = None, None
        if write_main_pages:
            k_paged, v_paged = _BF16_WORKING_PAGES.acquire(
                page_count,
                self.kv_head_num,
                int(self.page_size),
                self.head_dim,
                device,
            )
        idx_scratch = _IDX_K_SCRATCH.acquire(
            scratch_slots, 1, self.idx_head_dim, self._idx_k_working_dtype, device
        )
        _fused_cp_paged_write(
            packed,
            unpad_indices,
            write_slots,
            slot_mapping,
            k_paged,
            v_paged,
            idx_scratch,
            base,
            idx_view,
            idx_scale,
            kv_lens,
            int(self._scratch_seq_len),
            nk,
            ni,
            self.kv_head_num,
            self.head_dim,
            self.page_size,
            token_count=token_count,
            write_main_pages=write_main_pages,
        )
        self._scratch_idx_k = idx_scratch
        return k_paged, v_paged

    def _write_cp_suffix_to_nvfp4_working_pages(
        self,
        kv_cache: LayerKVCache,
        packed: torch.Tensor,
        unpad_indices: torch.Tensor,
        write_slots: torch.Tensor,
        slot_mapping: torch.Tensor,
        nk: int,
        ni: int,
        token_count: int,
        prefix_cpu_list,
        prefix_dst_pages: torch.Tensor,
        prefix_gather_plan,
        attn_inputs: PyAttentionInputs,
        kv_lens: torch.Tensor,
    ):
        """Persist rank-owned rows and build a packed FP4 CP working set.

        Prefix pages stay opaque during the CP gather (packed values plus the
        side/scale block).  The full BF16 suffix already produced by CP
        all-gather is quantized directly into the request-local page namespace.
        No historical row is materialized as BF16.
        """
        persistent_layout = nvfp4_cache_layout(
            kv_cache.kv_cache_base,
            kv_cache.kv_scale_base,
            self.kv_head_num,
            self.physical_page_size,
            self.head_dim,
        )
        if (
            nk != 512
            or ni != 128
            or self.kv_head_num != 4
            or self.head_dim != 128
            or self.page_size != 128
        ):
            raise ValueError(
                "M3.1 CP mapped writer requires head4/dim128/index128/page128"
            )
        unpad_rows = unpad_indices[:token_count]

        # The persistent scale bytes use the same 128x4 swizzle consumed by
        # both fmha_sm100 prefill readers and the native decode readers.
        views = persistent_layout.logical_views(ni)

        scratch_slots = int(self._scratch_slots)
        if scratch_slots % int(self.page_size) != 0:
            raise RuntimeError(
                f"MSA CP FP4 scratch slots {scratch_slots} are not page-aligned "
                f"to page_size={self.page_size}"
            )
        page_count = scratch_slots // int(self.page_size)
        main, main_scales, idx_packed, idx_scales = _NVFP4_WORKING_PAGES.acquire(
            page_count,
            self.kv_head_num,
            int(self.page_size),
            self.head_dim,
            ni,
            packed.device,
        )

        if prefix_cpu_list and any(prefix_cpu_list):
            prefetched = (
                self._take_prefetched_cp_prefix(
                    kv_cache,
                    attn_inputs,
                    self._physical_block_table(attn_inputs),
                    prefix_gather_plan,
                )
                if _CP_PREFIX_PREFETCH and self._kv_sharded
                else None
            )
            if prefetched is not None:
                prefix_values, prefix_side, _ = prefetched
            else:
                prefix_values = self._gather_cp_compact_prefix_pool(
                    kv_cache.kv_cache_base,
                    attn_inputs,
                    prefix_cpu_list,
                    prefix_gather_plan,
                )
                prefix_side = self._gather_cp_compact_prefix_pool(
                    kv_cache.kv_scale_base,
                    attn_inputs,
                    prefix_cpu_list,
                    prefix_gather_plan,
                )
            restore = (
                prefix_gather_plan.restore_indices
                if prefix_gather_plan is not None and self._kv_sharded
                else None
            )
            logical_prefix_pages = (
                int(restore.numel())
                if restore is not None
                else int(prefix_values.shape[0])
            )
            if logical_prefix_pages != int(prefix_dst_pages.numel()):
                raise RuntimeError(
                    "MSA CP FP4 prefix page count mismatch: "
                    f"logical={logical_prefix_pages} gathered={prefix_values.shape[0]} "
                    f"destination={prefix_dst_pages.numel()}"
                )
            # Restore directly from rank-major opaque blocks. Six independent
            # views preserve larger-capacity working-pool plane offsets.
            restore_prefix_planes(
                prefix_values,
                prefix_side,
                restore,
                prefix_dst_pages,
                (
                    main[0],
                    main[1],
                    main_scales[0],
                    main_scales[1],
                    idx_packed,
                    idx_scales,
                ),
            )

        # Quantize once into both independent destinations. The full persistent
        # map contains -1 for non-owned rows; never substitute the compressed
        # owned-row map. Prefix restoration above cannot overlap suffix pages.
        nvfp4_quantize_cp_main_index_rows_to_planes(
            packed,
            unpad_rows,
            write_slots[:token_count].contiguous(),
            main[0],
            main_scales[0],
            main[1],
            main_scales[1],
            idx_packed,
            idx_scales,
            persistent_slots=slot_mapping[:token_count],
            persistent_planes=(
                views.main_k_fp4,
                views.main_k_scale,
                views.main_v_fp4,
                views.main_v_scale,
                views.idx_k_fp4,
                views.idx_k_scale,
            ),
        )
        clear_packed_working_tail_scales(
            main_scales[0],
            main_scales[1],
            idx_scales,
            kv_lens,
            int(self._scratch_seq_len),
            self.kv_head_num,
            self.head_dim,
            ni,
        )
        return main, main_scales, idx_packed, idx_scales

    # ------------------------------------------------------------------
    # Task-2: source idx_K from the main paged pool's scale region.
    # The C++ cache manager sizes the MHA scale region (kv_scale_base) to hold
    # one BF16 or E4M3 idx_K per token (indexer_head_dim). It is exposed to
    # Python as FP32; reinterpret it using the configured storage dtype and view
    # it as [block, page, idx_head_dim]
    # so idx_K is addressed by the same block table as the main K/V and travels
    # with it under PD separation.
    # ------------------------------------------------------------------
    def _idx_k_paged_storage(
        self, kv_cache: LayerKVCache
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return idx_K values and optional per-token scales from the side region."""
        if self.nvfp4_kv_cache:
            layout = nvfp4_cache_layout(
                kv_cache.kv_cache_base,
                kv_cache.kv_scale_base,
                self.kv_head_num,
                self.physical_page_size,
                self.head_dim,
            )
            values, scales = layout.indexer(self.idx_head_dim)
            return values.view(
                layout.num_blocks,
                self.page_size,
                self.idx_head_dim // 2,
            ), scales.view(
                layout.num_blocks,
                self.page_size,
                self.idx_head_dim // NVFP4_GROUP_SIZE,
            )

        scale = kv_cache.kv_scale_base
        if scale is None or scale.dim() != 2:
            raise RuntimeError(
                "MSA paged idx_K requires a 2-D kv_scale_base "
                "[block, scale_elems]; got "
                f"{None if scale is None else tuple(scale.shape)}. Launch with "
                "the M3 MHA indexer scale sizing (indexer_head_dim set)."
            )
        blk = int(scale.shape[0])
        block_bytes = int(scale.shape[1]) * int(scale.element_size())
        if self.idx_k_fp8_mode == 0:
            expect_bytes = self.page_size * self.idx_head_dim * _BF16_BYTES
        else:
            expect_bytes = self.page_size * (self.idx_head_dim + _FP8_SCALE_BYTES)
        if block_bytes != expect_bytes:
            raise RuntimeError(
                f"MSA idx_K side-region mismatch: bytes/block={block_bytes} "
                f"!= expected={expect_bytes} (mode={self.idx_k_fp8_mode}, "
                f"page={self.page_size}, "
                f"idx_head_dim={self.idx_head_dim}); check C++ kv_scale_stride_bytes"
            )
        raw = scale.view(torch.uint8)
        if self.idx_k_fp8_mode == 0:
            values = raw.view(torch.bfloat16).view(
                blk, self.page_size, self.idx_head_dim
            )
            return values, None

        value_bytes = self.page_size * self.idx_head_dim
        values = raw.as_strided(
            (blk, self.page_size, self.idx_head_dim),
            (int(raw.stride(0)), self.idx_head_dim, 1),
        ).view(torch.float8_e4m3fn)
        scales = raw[:, value_bytes:].view(torch.float32).view(blk, self.page_size)
        return values, scales

    def _restore_cp_prefix_working_pages(
        self,
        kv_cache: LayerKVCache,
        prefix_lengths: Any,
        req_to_token: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        k_paged: torch.Tensor,
        v_paged: torch.Tensor,
        dst_pages: Optional[torch.Tensor] = None,
        gather_plan=None,
    ) -> None:
        """Restore cached prefixes into transient HND working pages."""
        if isinstance(prefix_lengths, torch.Tensor):
            prefix_cpu = prefix_lengths.detach().cpu().to(torch.int64)
        else:
            prefix_cpu = torch.tensor(list(prefix_lengths), dtype=torch.int64)
        if not bool((prefix_cpu > 0).any().item()):
            return
        if bool((prefix_cpu % int(self.page_size) != 0).any().item()):
            raise RuntimeError(
                "MSA CP prefix restore requires page-aligned prefix lengths; got "
                f"{prefix_cpu.tolist()} with page_size={self.page_size}"
            )
        if self._scratch_idx_k is None:
            raise RuntimeError("MSA CP prefix restore requires idx-K scratch")

        block_table = self._physical_block_table(attn_inputs)
        idx_pool, idx_scale_pool = self._idx_k_paged_storage(kv_cache)
        if self._kv_sharded:
            prefetched = self._take_prefetched_cp_prefix(
                kv_cache, attn_inputs, block_table, gather_plan
            )
            if prefetched is not None:
                main_pages, idx_pages, idx_scales = prefetched
            else:
                main_pages = gather_cp_sharded_prefix_pool(
                    self._paged_kv_base_view(kv_cache),
                    block_table,
                    prefix_cpu,
                    page_size=self.page_size,
                    cp_size=self._cp_size,
                    cp_rank=self._cp_rank,
                    gather_plan=gather_plan,
                    restore_logical_order=gather_plan is None,
                )
                idx_pages = gather_cp_sharded_prefix_pool(
                    idx_pool,
                    block_table,
                    prefix_cpu,
                    page_size=self.page_size,
                    cp_size=self._cp_size,
                    cp_rank=self._cp_rank,
                    gather_plan=gather_plan,
                    restore_logical_order=gather_plan is None,
                )
                idx_scales = (
                    None
                    if idx_scale_pool is None
                    else gather_cp_sharded_prefix_pool(
                        idx_scale_pool,
                        block_table,
                        prefix_cpu,
                        page_size=self.page_size,
                        cp_size=self._cp_size,
                        cp_rank=self._cp_rank,
                        gather_plan=gather_plan,
                        restore_logical_order=gather_plan is None,
                    )
                )
        else:
            physical_page_parts = [
                block_table[batch_idx, : int(prefix_len) // int(self.page_size)]
                for batch_idx, prefix_len in enumerate(prefix_cpu.tolist())
                if int(prefix_len) > 0
            ]
            physical_pages = torch.cat(physical_page_parts).to(torch.long)
            main_pages = self._paged_kv_base_view(kv_cache).index_select(
                0, physical_pages
            )
            idx_pages = idx_pool.index_select(0, physical_pages)
            idx_scales = (
                None
                if idx_scale_pool is None
                else idx_scale_pool.index_select(0, physical_pages)
            )

        if dst_pages is None:
            dst_page_parts = []
            for batch_idx, prefix_len in enumerate(prefix_cpu.tolist()):
                prefix_pages = int(prefix_len) // int(self.page_size)
                if prefix_pages == 0:
                    continue
                dst_page_parts.append(
                    req_to_token[batch_idx, : int(prefix_len) : self.page_size]
                    .to(torch.long)
                    .div(self.page_size, rounding_mode="floor")
                )
            dst_pages = torch.cat(dst_page_parts)
        logical_page_count = int(dst_pages.numel())
        expected_pages = (
            int(main_pages.shape[0])
            if gather_plan is None
            else int(gather_plan.total_logical_blocks)
        )
        if logical_page_count != expected_pages:
            raise RuntimeError(
                f"MSA CP prefix page count mismatch: dst={dst_pages.numel()} "
                f"logical={expected_pages} gathered={main_pages.shape[0]}"
            )
        # One fused scatter restores K, V and idx-K. Persistent E4M3 K/V are
        # converted to BF16 working pages; persistent BF16 K/V are copied.
        if not k_paged.is_cuda:
            raise RuntimeError("MSA CP prefix restore requires CUDA working pages")
        _scatter_cp_prefix_pages(
            main_pages,
            idx_pages,
            idx_scales,
            dst_pages,
            k_paged,
            v_paged,
            self._scratch_idx_k,
            src_pages=(
                None
                if gather_plan is None or not self._kv_sharded
                else gather_plan.restore_indices
            ),
        )

    def _gather_cp_compact_prefix_pool(
        self, pool, attn_inputs, prefix_lengths, gather_plan
    ):
        """Keep complete gathered prefixes in pool dtype, without BF16 expansion."""
        if not any(prefix_lengths):
            return pool[:0]
        if any(length % self.page_size for length in prefix_lengths):
            raise ValueError("compact CP prefixes must be page aligned")
        block_table = self._physical_block_table(attn_inputs)
        if self._kv_sharded:
            return gather_cp_sharded_prefix_pool(
                pool,
                block_table,
                torch.tensor(prefix_lengths, dtype=torch.int64),
                page_size=self.page_size,
                cp_size=self._cp_size,
                cp_rank=self._cp_rank,
                gather_plan=gather_plan,
                restore_logical_order=False,
            )
        pages = torch.cat(
            [
                block_table[b, : length // self.page_size]
                for b, length in enumerate(prefix_lengths)
                if length
            ]
        ).long()
        return pool.index_select(0, pages)

    def _write_kv_cache_and_idx_k_for_decode(
        self,
        kv_cache: LayerKVCache,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_k: torch.Tensor,
        seq_lens: torch.Tensor,
        phys_block_table: torch.Tensor,
    ):
        """Persist the current decode token and return paged decode views."""
        # Caller contract: this helper is only entered after
        # _check_paged_decode_static() has accepted the paged cache layout. Keep
        # the hot path to dynamic dtype checks; layout mismatches should fall back
        # before _forward_paged_decode() is selected.
        base = self._paged_kv_base_view(kv_cache)
        scale = kv_cache.kv_scale_base
        if self.nvfp4_kv_cache:
            layout = nvfp4_cache_layout(
                base,
                scale,
                self.kv_head_num,
                self.physical_page_size,
                self.head_dim,
            )
            physical_slots = build_decode_physical_slots(
                seq_lens, phys_block_table, page_size=self.page_size
            )
            # Decode owns K, V and the shared indexer-K row at the same token
            # slot. Emit all three persistent NVFP4 planes in one launch; the
            # fused primitive is bitwise-equivalent to the three independent
            # writers and keeps the phase-1 cache ABI unchanged.
            nvfp4_quantize_main_index_rows(
                k.contiguous(),
                # V is an unrotated column view of the fused projection. The
                # writer consumes explicit strides; materializing it adds a
                # copy per layer without changing the stored cache bytes.
                v,
                idx_k.contiguous(),
                physical_slots.contiguous(),
                layout,
                mma_scale_layout=True,
            )
            # Callers materialize their bounded request working set below.
            return ()
        # base may be an FP8 (e4m3) pool; _write_decode_kv_idx_to_paged casts the
        # bf16 K/V to e4m3 on store, and the paged decode kernel upconverts on read.
        if (
            base is None
            or scale is None
            or (base.dtype != k.dtype and base.dtype != torch.float8_e4m3fn)
        ):
            return None

        idx_view, idx_scale = self._idx_k_paged_storage(kv_cache)
        if idx_view.dtype != self._idx_k_persistent_dtype:
            return None

        _write_decode_kv_idx_to_paged(
            k.contiguous(),
            v.contiguous(),
            idx_k.reshape(-1, self.idx_head_dim).contiguous(),
            seq_lens,
            phys_block_table,
            base,
            idx_view,
            idx_scale,
            int(self.page_size),
            int(self.idx_head_dim),
        )

        return base[:, 0], base[:, 1], phys_block_table, idx_view, idx_scale

    @classmethod
    def cp_prefix_prefetch_enabled(cls) -> bool:
        return _CP_PREFIX_PREFETCH

    @classmethod
    def _drop_all_prefetch(cls) -> None:
        while cls._cp_prefetch_entries:
            entry = cls._cp_prefetch_entries.popitem()[1]
            entry["event"].synchronize()

    @classmethod
    def _drop_stale_prefetch(cls, owner: PyAttentionInputs) -> None:
        stale = [
            layer_idx
            for layer_idx, entry in cls._cp_prefetch_entries.items()
            if entry["owner"] is not owner
        ]
        for layer_idx in stale:
            entry = cls._cp_prefetch_entries.pop(layer_idx)
            entry["event"].synchronize()
        while len(cls._cp_prefetch_entries) > _MAX_LIVE_PREFETCH:
            oldest = next(iter(cls._cp_prefetch_entries))
            entry = cls._cp_prefetch_entries.pop(oldest)
            entry["event"].synchronize()

    @classmethod
    def join_cp_side_comms(cls) -> None:
        for stream in cls._cp_prefetch_stream.values():
            torch.cuda.current_stream(stream.device).wait_stream(stream)
        for stream in cls._cp_side_stream.values():
            torch.cuda.current_stream(stream.device).wait_stream(stream)
        for entry in cls._cp_prefetch_entries.values():
            torch.cuda.current_stream(entry["device"]).wait_event(entry["event"])

    join_cp_prefix_prefetch = join_cp_side_comms

    def maybe_prefetch_cp_prefix(
        self, kv_cache: Optional[LayerKVCache], attn_inputs: PyAttentionInputs
    ) -> None:
        if not _CP_PREFIX_PREFETCH or kv_cache is None:
            return
        if MSAAttention._cp_prefetch_disabled:
            if self.nvfp4_kv_cache:
                raise RuntimeError(
                    "MSA CP FP4 prefetch cannot inherit a disabled schedule"
                )
            return
        try:
            self._issue_cp_prefix_prefetch(kv_cache, attn_inputs)
        except Exception:
            # A rank-local fallback can leave peers inside TP_PREFETCH while
            # this rank enters an inline TP collective. Native CP must fail
            # closed rather than silently change its collective schedule.
            if self.nvfp4_kv_cache:
                raise
            MSAAttention._cp_prefetch_disabled = True
            MSAAttention._drop_all_prefetch()
            logging.warning(
                "[msa] disabling CP prefix prefetch after issue failure; "
                "falling back to inline prefix gather",
                exc_info=True,
            )

    def _issue_cp_prefix_prefetch(
        self, kv_cache: LayerKVCache, attn_inputs: PyAttentionInputs
    ) -> None:
        self._drop_stale_prefetch(attn_inputs)
        cp_size = int(self._cp_size)
        if not self._kv_sharded or cp_size <= 1:
            return
        if self.layer_idx in MSAAttention._cp_prefetch_entries:
            return

        meta = MSAAttention._cp_shared_meta
        if meta is None or meta.get("owner") is not attn_inputs:
            return
        if int(meta.get("prefix_sum", 0)) <= 0:
            return
        addr = meta.get("addr")
        if addr is None:
            if self.nvfp4_kv_cache:
                raise RuntimeError(
                    "MSA CP FP4 prefetch metadata has no addressing plan"
                )
            return

        block_table = self._physical_block_table(attn_inputs)
        plan_key = (
            block_table.device,
            int(block_table.data_ptr()),
            tuple(block_table.shape),
        )
        plans = addr["prefix_gather_plans"]
        plan = plans.get(plan_key)
        if plan is None and self.nvfp4_kv_cache:
            # A new cache-group table may appear on only one rank. A local
            # cache miss must rebuild addressing, not change collective order.
            plan = build_cp_sharded_prefix_gather_plan(
                block_table,
                torch.tensor(meta["prefix_cpu_list"], dtype=torch.int64),
                page_size=self.page_size,
                cp_size=cp_size,
                cp_rank=self._cp_rank,
            )
            plans[plan_key] = plan
        if plan is None or plan.total_logical_blocks == 0:
            if self.nvfp4_kv_cache:
                raise RuntimeError(
                    "MSA CP FP4 prefetch has an empty nonzero-prefix plan"
                )
            return

        main_pool = self._paged_kv_base_view(kv_cache)
        if main_pool is None or not main_pool.is_cuda:
            if self.nvfp4_kv_cache:
                raise RuntimeError("MSA CP FP4 prefetch requires a CUDA cache pool")
            return
        if self.nvfp4_kv_cache:
            # Native KV4 has two opaque page blocks. The second contains main
            # scales and packed index-K/scales; do not reinterpret it as BF16.
            idx_pool, idx_scale_pool = kv_cache.kv_scale_base, None
            if (
                main_pool.dim() != 2
                or main_pool.dtype != torch.uint8
                or idx_pool is None
                or idx_pool.dim() != 2
                or idx_pool.dtype != torch.uint8
                or idx_pool.shape[0] != main_pool.shape[0]
            ):
                raise RuntimeError("MSA CP FP4 prefetch requires two uint8 page pools")
        else:
            if main_pool.dim() != 5:
                return
            idx_pool, idx_scale_pool = self._idx_k_paged_storage(kv_cache)
        device = main_pool.device
        if idx_pool.dim() < 2 or idx_pool.device != device:
            if self.nvfp4_kv_cache:
                raise RuntimeError("MSA CP FP4 prefetch side pool device mismatch")
            return
        if plan.packed_block_ids.device != device:
            if self.nvfp4_kv_cache:
                raise RuntimeError("MSA CP FP4 prefetch plan device mismatch")
            return

        stream = MSAAttention._cp_prefetch_stream.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            MSAAttention._cp_prefetch_stream[device] = stream

        main_rows = int(plan.packed_block_ids.numel()) * cp_size
        main_stream = torch.cuda.current_stream(device)
        workspace = None
        if self.nvfp4_kv_cache:
            # Layer L is issued before L-1, after the L-2 consumer has been
            # queued. The existing side.wait_stream(main) below orders a
            # two-slot reuse after that consumer. Keep storage alive instead
            # of accumulating per-layer cross-stream allocator reservations.
            slot_key = (device, self.layer_idx % 2)
            if any(
                entry.get("workspace_key") == slot_key
                for entry in MSAAttention._cp_prefetch_entries.values()
            ):
                raise RuntimeError("MSA CP FP4 prefetch slot is still unconsumed")
            layout = (cp_size, main_pool.shape[1], idx_pool.shape[1])
            workspace = MSAAttention._cp_native_prefetch_buffers.get(slot_key)
            if workspace is not None and (
                workspace["stream"] != main_stream.cuda_stream
                or workspace["layout"] != layout
            ):
                raise RuntimeError(
                    "MSA CP FP4 prefetch workspace stream/layout changed"
                )
            if workspace is None:
                workspace = dict(
                    stream=main_stream.cuda_stream,
                    layout=layout,
                    rows=0,
                    generation=0,
                )
                MSAAttention._cp_native_prefetch_buffers[slot_key] = workspace
            if workspace["rows"] < main_rows:
                # Replaced buffers retain the record_stream protection below;
                # never drop an in-flight buffer without allocator tracking.
                workspace["main"] = torch.empty(
                    (main_rows, main_pool.shape[1]), dtype=torch.uint8, device=device
                )
                workspace["idx"] = torch.empty(
                    (main_rows, idx_pool.shape[1]), dtype=torch.uint8, device=device
                )
                workspace["rows"] = main_rows
            workspace["generation"] += 1
            main_gathered = workspace["main"][:main_rows]
            idx_gathered = workspace["idx"][:main_rows]
        else:
            main_gathered = torch.empty(
                (main_rows, *main_pool.shape[1:]), dtype=main_pool.dtype, device=device
            )
            idx_gathered = torch.empty(
                (main_rows, *idx_pool.shape[1:]), dtype=idx_pool.dtype, device=device
            )
        idx_scale_gathered = (
            None
            if idx_scale_pool is None
            else torch.empty(
                (main_rows, *idx_scale_pool.shape[1:]),
                dtype=idx_scale_pool.dtype,
                device=device,
            )
        )
        stream.wait_stream(main_stream)
        plan.packed_block_ids.record_stream(stream)
        main_pool.record_stream(stream)
        idx_pool.record_stream(stream)
        if idx_scale_pool is not None:
            idx_scale_pool.record_stream(stream)
        main_gathered.record_stream(stream)
        idx_gathered.record_stream(stream)
        if idx_scale_gathered is not None:
            idx_scale_gathered.record_stream(stream)
        with torch.cuda.stream(stream):
            all_gather(
                main_pool.index_select(0, plan.packed_block_ids),
                group=Group.TP_PREFETCH,
                out=main_gathered,
            )
            all_gather(
                idx_pool.index_select(0, plan.packed_block_ids),
                group=Group.TP_PREFETCH,
                out=idx_gathered,
            )
            if idx_scale_gathered is not None:
                all_gather(
                    idx_scale_pool.index_select(0, plan.packed_block_ids),
                    group=Group.TP_PREFETCH,
                    out=idx_scale_gathered,
                )
        event = torch.cuda.Event()
        event.record(stream)
        MSAAttention._cp_prefetch_entries[self.layer_idx] = {
            "workspace_key": slot_key if workspace is not None else None,
            "workspace_generation": (
                workspace["generation"] if workspace is not None else None
            ),
            "owner": attn_inputs,
            "plan": plan,
            "block_table_ptr": int(block_table.data_ptr()),
            "main_pool_ptr": int(main_pool.data_ptr()),
            "source_pools": (main_pool, idx_pool, idx_scale_pool),
            "nvfp4": self.nvfp4_kv_cache,
            "side_pool_ptr": (
                int(idx_pool.data_ptr()) if self.nvfp4_kv_cache else None
            ),
            "main": main_gathered,
            "idx": idx_gathered,
            "idx_scale": idx_scale_gathered,
            "event": event,
            "device": device,
        }

    def _take_prefetched_cp_prefix(
        self,
        kv_cache: LayerKVCache,
        attn_inputs: PyAttentionInputs,
        block_table: torch.Tensor,
        gather_plan,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        layer_idx = getattr(self, "layer_idx", None)
        if layer_idx is None:
            return None
        entry = MSAAttention._cp_prefetch_entries.pop(layer_idx, None)
        if entry is None:
            if self.nvfp4_kv_cache and _CP_PREFIX_PREFETCH and self._kv_sharded:
                meta = MSAAttention._cp_shared_meta
                if meta is None or meta.get("owner") is not attn_inputs:
                    raise RuntimeError(
                        "MSA CP FP4 prefetch has no current owner metadata"
                    )
                # Layer zero establishes CP metadata after its next-layer hook.
                # It and the immediately following layer bootstrap inline on
                # every rank. All subsequent layers must consume a prefetch;
                # a missing entry cannot silently choose another collective.
                first_layer = int(meta["layer_idx"])
                if int(meta.get("prefix_sum", 0)) > 0 and layer_idx > first_layer + 1:
                    raise RuntimeError("MSA CP FP4 prefix prefetch entry is missing")
            return None
        main_pool = self._paged_kv_base_view(kv_cache)
        workspace_key = entry.get("workspace_key")
        if workspace_key is not None:
            workspace = MSAAttention._cp_native_prefetch_buffers[workspace_key]
            if (
                workspace["generation"] != entry["workspace_generation"]
                or workspace["stream"]
                != torch.cuda.current_stream(entry["device"]).cuda_stream
            ):
                raise RuntimeError("MSA CP FP4 prefetch workspace consumer changed")
        if (
            entry["owner"] is not attn_inputs
            or entry["plan"] is not gather_plan
            or entry["block_table_ptr"] != int(block_table.data_ptr())
            or main_pool is None
            or entry["main_pool_ptr"] != int(main_pool.data_ptr())
            or entry["nvfp4"] != self.nvfp4_kv_cache
            or (
                self.nvfp4_kv_cache
                and (
                    kv_cache.kv_scale_base is None
                    or entry["side_pool_ptr"] != int(kv_cache.kv_scale_base.data_ptr())
                )
            )
        ):
            entry["event"].synchronize()
            if self.nvfp4_kv_cache:
                raise RuntimeError(
                    "MSA CP FP4 prefix prefetch identity changed; "
                    "cannot safely switch a single rank to inline gather"
                )
            return None
        current_stream = torch.cuda.current_stream(entry["device"])
        current_stream.wait_event(entry["event"])
        # Allocation/production happened on another stream. Keep the allocator
        # from recycling these pages before the consuming restore completes.
        for pages in (entry["main"], entry["idx"], entry["idx_scale"]):
            if pages is not None:
                pages.record_stream(current_stream)
        return entry["main"], entry["idx"], entry["idx_scale"]

    def _cp_all_gather_packed_kv(
        self, packed_kv: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.cuda.Event]]:
        cp_size = int(self.parallelism_config.tp_size)
        if not _CP_PACKED_KV_OVERLAP or not packed_kv.is_cuda or cp_size <= 1:
            return all_gather(packed_kv, group=Group.TP), None

        device = packed_kv.device
        stream = MSAAttention._cp_side_stream.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            MSAAttention._cp_side_stream[device] = stream
            MSAAttention._cp_side_event[device] = torch.cuda.Event()
        event = MSAAttention._cp_side_event[device]

        main_stream = torch.cuda.current_stream(device)
        all_packed = torch.empty(
            (packed_kv.shape[0] * cp_size, *packed_kv.shape[1:]),
            dtype=packed_kv.dtype,
            device=device,
        )
        stream.wait_stream(main_stream)
        packed_kv.record_stream(stream)
        all_packed.record_stream(stream)
        with torch.cuda.stream(stream):
            all_gather(packed_kv, group=Group.TP_SIDE, out=all_packed)
        event.record(stream)
        return all_packed, event

    # ------------------------------------------------------------------
    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: LayerKVCache,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ordinary MSA prefill over compact HND pages, including prefixes."""
        from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
            minimax_sparse_prefill,
        )
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.score_chunk import (
            m3_index_score_chunk_enabled,
        )
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
            build_index_score_plan,
            build_kv_page_indices,
            build_sparse_attn_plan,
        )

        if kv_cache is None or self.nvfp4_kv_cache:
            raise RuntimeError("MSA non-CP prefill requires a BF16/FP8 paged cache")
        if (
            self.page_size != self.block_size
            or self.page_size != self.physical_page_size
            or not self.disable_index_value
            or self.num_idx_heads != self.kv_head_num
        ):
            raise RuntimeError(
                "MSA paged prefill requires aligned pages, matching index/KV "
                "heads and disabled index values"
            )
        device = hidden_states.device
        total_tokens = int(hidden_states.shape[0])
        kv_lens, prefix_lens, inlens = self._get_lengths(attn_inputs)
        kv_cpu = [int(x) for x in kv_lens.cpu().tolist()]
        prefix_cpu = [int(x) for x in prefix_lens.cpu().tolist()]
        inlen_cpu = [int(x) for x in inlens.cpu().tolist()]
        if not kv_cpu or sum(inlen_cpu) != total_tokens:
            raise RuntimeError(
                "MSA prefill input token count does not match request lengths"
            )
        table = self._physical_block_table(attn_inputs)
        counts = [(n + self.page_size - 1) // self.page_size for n in kv_cpu]
        if not sum(counts) or max(counts) > int(table.shape[1]):
            raise RuntimeError("MSA prefill physical page table is too short")
        pages = torch.cat([table[b, :n] for b, n in enumerate(counts)]).long()
        offsets, page_count = [], 0
        for count in counts:
            offsets.append(page_count)
            page_count += count
        req_to_token = (
            torch.tensor(offsets, device=device, dtype=torch.int32)[:, None]
            * self.page_size
            + torch.arange(max(kv_cpu), device=device, dtype=torch.int32)[None, :]
        ).contiguous()
        pos_parts, slot_parts = [], []
        for b, (prefix, length) in enumerate(zip(prefix_cpu, kv_cpu)):
            pos = torch.arange(prefix, length, device=device, dtype=torch.int64)
            pos_parts.append(pos)
            slot_parts.append(
                table[b, pos // self.page_size].long() * self.page_size
                + pos % self.page_size
            )
        positions = torch.cat(pos_parts).to(torch.int32)
        physical_slots = torch.cat(slot_parts).contiguous()

        qkv, idx_q, idx_k = self._project_qkv_idx(hidden_states, x_fp8, x_scale)
        m31_fused = self._fuse_m31_projected_norm_rope(qkv, idx_q, idx_k, positions)
        if self.qk_fuse_norm is not None and not m31_fused:
            qkv = self.qk_fuse_norm(qkv)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(total_tokens, self.head_num, self.head_dim).contiguous()
        k = k.reshape(total_tokens, self.kv_head_num, self.head_dim).contiguous()
        v = v.reshape(total_tokens, self.kv_head_num, self.head_dim).contiguous()
        idx_q = idx_q.reshape(total_tokens, self.num_idx_heads, self.idx_head_dim)
        idx_k = idx_k.reshape(total_tokens, 1, self.idx_head_dim)
        idx_q = self._legacy_index_norm(
            idx_q, self.idx_q_norm_w, self.layernorm_eps
        ).contiguous()
        idx_k = self._legacy_index_norm(
            idx_k, self.idx_k_norm_w, self.layernorm_eps
        ).contiguous()
        if not m31_fused:
            self._apply_rope(q, k, positions)
            self._apply_rope(idx_q, idx_k, positions)
        base = self._paged_kv_base_view(kv_cache)
        if (
            base is None
            or base.dim() != 5
            or base.dtype
            not in (
                torch.bfloat16,
                torch.float8_e4m3fn,
            )
        ):
            raise RuntimeError("MSA prefill requires BF16 or E4M3 paged K/V")
        idx_view, idx_scale = self._idx_k_paged_storage(kv_cache)
        _write_main_kv_to_paged(k, v, base, physical_slots)
        _write_idx_rows(
            idx_k.reshape(-1, self.idx_head_dim),
            physical_slots,
            idx_view,
            idx_scale,
        )
        if attn_inputs.cache_store_inputs:
            from rtp_llm.models_py.modules.factory.attention import common

            write_impl = common.create_write_cache_store_impl(attn_inputs)
            common.apply_write_cache_store(write_impl, attn_inputs, kv_cache)
        working_k, working_v = _BF16_WORKING_PAGES.acquire(
            page_count, self.kv_head_num, self.page_size, self.head_dim, device
        )
        idx_scratch = _IDX_K_SCRATCH.acquire(
            page_count * self.page_size,
            1,
            self.idx_head_dim,
            self._idx_k_working_dtype,
            device,
        )
        _scatter_cp_prefix_pages(
            base,
            idx_view,
            idx_scale,
            torch.arange(page_count, device=device, dtype=torch.long),
            working_k,
            working_v,
            idx_scratch,
            src_pages=pages.contiguous(),
        )
        cu_seqlens = attn_inputs.cu_seqlens[: len(kv_cpu) + 1].to(torch.int32)
        seq_lens, prefix_i32 = kv_lens.to(torch.int32), prefix_lens.to(torch.int32)
        geometry = (
            tuple(kv_cpu),
            tuple(prefix_cpu),
            tuple(inlen_cpu),
            self.page_size,
            self.head_num,
            self.kv_head_num,
            self.num_idx_heads,
            self.topk_blocks,
            self.idx_k_fp8_mode,
        )
        shared = MSAAttention._prefill_shared_meta
        if (
            shared is not None
            and shared["owner"] is attn_inputs
            and shared["layer_idx"] < self.layer_idx
            and shared["geometry"] == geometry
        ):
            plan = shared["sparse_attn_plan"]
            index_score_plan = shared["index_score_plan"]
            page_indices = shared["page_indices"]
            shared["layer_idx"] = self.layer_idx
        else:
            plan = build_sparse_attn_plan(
                cu_seqlens,
                seq_lens,
                prefix_i32,
                self.head_num,
                self.kv_head_num,
                self.block_size,
                self.topk_blocks,
                use_fp8_kvcache=False,
            )
            if self.nvfp4_kv_cache or m3_index_score_chunk_enabled(total_tokens):
                index_score_plan = {}
            else:
                index_score_plan = build_index_score_plan(
                    cu_seqlens,
                    seq_lens,
                    prefix_i32,
                    self.num_idx_heads,
                    1,
                    self.block_size,
                    use_fp8_kvcache=self.idx_k_fp8_mode == 2,
                )
            page_indices = build_kv_page_indices(
                req_to_token, seq_lens, self.block_size
            )
            MSAAttention._prefill_shared_meta = {
                "owner": attn_inputs,
                "layer_idx": self.layer_idx,
                "geometry": geometry,
                "sparse_attn_plan": plan,
                "index_score_plan": index_score_plan,
                "page_indices": page_indices,
            }
        _, output = minimax_sparse_prefill(
            q=q,
            idx_q=idx_q,
            idx_k_cache=idx_scratch,
            req_to_token=req_to_token,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_i32,
            max_seqlen_q=max(inlen_cpu),
            max_seqlen_k=max(kv_cpu),
            block_size_k=self.block_size,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            disable_index_value=self.disable_index_value,
            index_score_plan=index_score_plan,
            sparse_attn_plan=plan,
            kv_indices=page_indices,
            k_paged_cache=working_k,
            v_paged_cache=working_v,
        )
        projected = self.o_proj(output.reshape(total_tokens, -1).contiguous())
        if self.tp_size > 1:
            projected = all_reduce(projected, group=Group.TP)
        return projected

    def _forward_nvfp4_prefill(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: LayerKVCache,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ordinary TP1 prefill with packed persistent and working FP4 pages.

        Unlike CP, each request is one contiguous query segment, including odd
        and single-token suffixes. Working slots and physical persistent slots
        are separate namespaces; neither CP metadata nor CP collectives apply.
        """
        if self.tp_size != 1 or int(self.parallelism_config.tp_size) != 1:
            raise RuntimeError("MSA ordinary NVFP4 prefill currently requires TP1")
        if (
            self.cp_enabled
            or self._kv_sharded
            or getattr(attn_inputs, "context_parallel_info", None) is not None
        ):
            raise RuntimeError(
                "MSA ordinary NVFP4 prefill does not accept CP metadata or sharded KV"
            )
        if (
            kv_cache is None
            or not self.nvfp4_kv_cache
            or self.page_size != 128
            or self.block_size != self.page_size
            or self.physical_page_size != self.page_size
            or not self.disable_index_value
            or self.num_idx_heads != self.kv_head_num
        ):
            raise RuntimeError(
                "MSA ordinary NVFP4 prefill requires aligned native FP4 cache geometry"
            )
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.score_chunk import (
            PrefillScoreHostMetadata,
        )
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
            build_kv_page_indices,
            build_sparse_attn_plan,
            flash_prefill_topk_to_block_tables_fp4,
            sparse_prefill_from_topk_fp4,
        )

        device = hidden_states.device
        total_tokens = int(hidden_states.shape[0])
        kv_lens, prefix_lens, inlens = self._get_lengths(attn_inputs)
        prefix_cpu = [int(x) for x in prefix_lens.cpu().tolist()]
        inlen_cpu = [int(x) for x in inlens.cpu().tolist()]
        kv_cpu = [p + n for p, n in zip(prefix_cpu, inlen_cpu)]
        if (
            not kv_cpu
            or len(prefix_cpu) != len(inlen_cpu)
            or sum(inlen_cpu) != total_tokens
            or any(n <= 0 for n in inlen_cpu)
        ):
            raise RuntimeError(
                "MSA ordinary NVFP4 prefill token/request lengths mismatch"
            )
        if any(p < 0 or p % self.page_size for p in prefix_cpu):
            raise RuntimeError(
                "MSA ordinary NVFP4 prefill requires page-aligned prefixes"
            )
        bsz, max_kv = len(kv_cpu), max(kv_cpu)
        table = self._physical_block_table(attn_inputs)
        if table.shape[0] < bsz or table.shape[1] < triton.cdiv(max_kv, self.page_size):
            raise RuntimeError(
                "MSA ordinary NVFP4 prefill physical page table is too short"
            )
        self._ensure_scratch_addressing_capacity(bsz, max_kv, exact_cp_shape=True)
        stride = int(self._scratch_seq_len)
        req_to_token = (
            torch.arange(bsz, device=device, dtype=torch.int32)[:, None] * stride
            + torch.arange(max_kv, device=device, dtype=torch.int32)[None, :]
        ).contiguous()
        positions, working_slots, physical_slots, prefix_pages = [], [], [], []
        for b, (prefix, length) in enumerate(zip(prefix_cpu, kv_cpu)):
            pos = torch.arange(prefix, length, device=device, dtype=torch.int64)
            positions.append(pos)
            working_slots.append(pos + b * stride)
            # _kernel_slots_to_paged is identity outside CP. Resolve physical
            # slots explicitly instead of passing compact working addresses.
            physical_slots.append(
                table[b, pos // self.page_size].long() * self.page_size
                + pos % self.page_size
            )
            prefix_pages.append(
                torch.arange(prefix // self.page_size, device=device, dtype=torch.int64)
                + b * (stride // self.page_size)
            )
        positions = torch.cat(positions)
        working_slots = torch.cat(working_slots)
        physical_slots = torch.cat(physical_slots)
        prefix_pages = torch.cat(prefix_pages)
        identity_rows = torch.arange(total_tokens, device=device, dtype=torch.int64)
        seq_lens = kv_lens.to(device=device, dtype=torch.int32)
        prefix_i32 = prefix_lens.to(device=device, dtype=torch.int32)
        cu_seqlens = torch.zeros(bsz + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(
            inlens.to(device=device, dtype=torch.int32), dim=0
        )
        kv_indices = build_kv_page_indices(req_to_token, seq_lens, self.block_size)
        sparse_plan = build_sparse_attn_plan(
            cu_seqlens,
            seq_lens,
            prefix_i32,
            self.head_num,
            self.kv_head_num,
            self.block_size,
            self.topk_blocks,
            use_fp8_kvcache=False,
        )
        host = PrefillScoreHostMetadata(
            tuple(inlen_cpu), tuple(kv_cpu), tuple(prefix_cpu), tuple(range(bsz))
        )
        # Packed-KV4 IndexScore uses the RTP Q8K4 reader; the fmha-sm100
        # Q4K4 plan is neither used nor allocated on this path.
        index_plan = {"_fp4_host_metadata": host}

        qkv, idx_q, idx_k = self._project_qkv_idx(hidden_states, x_fp8, x_scale)
        query_fp8_outputs = None
        if self._m31_raw_attention_norms is not None:
            query_fp8_outputs = tuple(
                torch.empty(
                    (total_tokens, heads, self.head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                for heads in (self.head_num, self.num_idx_heads)
            )
        m31_fused = self._fuse_m31_projected_norm_rope(
            qkv, idx_q, idx_k, positions, query_fp8_outputs=query_fp8_outputs
        )
        if self.qk_fuse_norm is not None and not m31_fused:
            qkv = self.qk_fuse_norm(qkv)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = (
            query_fp8_outputs[0]
            if query_fp8_outputs is not None
            else q.reshape(total_tokens, self.head_num, self.head_dim).contiguous()
        )
        k = k.reshape(total_tokens, self.kv_head_num, self.head_dim).contiguous()
        v = v.reshape(total_tokens, self.kv_head_num, self.head_dim).contiguous()
        idx_q = (
            query_fp8_outputs[1]
            if query_fp8_outputs is not None
            else self._legacy_index_norm(
                idx_q.reshape(total_tokens, self.num_idx_heads, self.idx_head_dim),
                self.idx_q_norm_w,
                self.layernorm_eps,
            ).contiguous()
        )
        del query_fp8_outputs
        idx_k = self._legacy_index_norm(
            idx_k.reshape(total_tokens, 1, self.idx_head_dim),
            self.idx_k_norm_w,
            self.layernorm_eps,
        ).contiguous()
        if not m31_fused:
            self._apply_rope(q, k, positions)
            self._apply_rope(idx_q, idx_k, positions)
        nk, ni = self.kv_size, self.idx_head_dim
        packed = torch.cat(
            (
                k.reshape(total_tokens, nk),
                v.reshape(total_tokens, nk),
                idx_k.reshape(total_tokens, ni),
            ),
            dim=-1,
        )
        # Existing writer preserves both opaque persistent planes and builds
        # FP4 working pages without ever expanding historical KV to BF16.
        main, scales, idx_fp4, idx_scales = (
            self._write_cp_suffix_to_nvfp4_working_pages(
                kv_cache,
                packed,
                identity_rows,
                working_slots,
                physical_slots,
                nk,
                ni,
                total_tokens,
                prefix_cpu,
                prefix_pages,
                None,
                attn_inputs,
                seq_lens,
            )
        )
        del packed, qkv, k, v, idx_k
        if attn_inputs.cache_store_inputs:
            from rtp_llm.models_py.modules.factory.attention import common

            write_impl = common.create_write_cache_store_impl(attn_inputs)
            common.apply_write_cache_store(write_impl, attn_inputs, kv_cache)
        pages = int(main.shape[1])
        groups = self.head_dim // NVFP4_GROUP_SIZE
        k_scale = (
            scales[0]
            .view(torch.uint8)
            .view(pages * self.kv_head_num * self.page_size, groups)
        )
        v_scale = scales[1].view(torch.uint8).view_as(k_scale)
        idx_scale_mma = idx_scales.view(
            pages, 1, (ni // NVFP4_GROUP_SIZE) // 4, 32, 4, 4
        )
        _, _, topk = flash_prefill_topk_to_block_tables_fp4(
            idx_q=idx_q,
            idx_k_fp4=idx_fp4,
            idx_k_scale_mma=idx_scale_mma,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_i32,
            max_seqlen_q=max(inlen_cpu),
            max_seqlen_k=max_kv,
            block_size_k=self.block_size,
            topk=self.topk_blocks,
            num_pages=triton.cdiv(max_kv, self.block_size),
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            index_score_plan=index_plan,
            kv_indices=kv_indices,
            emit_block_table=False,
        )
        output = sparse_prefill_from_topk_fp4(
            q,
            main[0],
            main[1],
            k_scale,
            v_scale,
            topk,
            kv_indices,
            sparse_plan,
            self.topk_blocks,
            self.block_size,
            self.head_dim**-0.5,
        )
        return self.o_proj(output.reshape(total_tokens, -1).contiguous())

    def _forward_cp_prefill(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: LayerKVCache,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CP prefill with native NVFP4 or the legacy non-NVFP4 MSA path."""
        from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
            minimax_sparse_prefill,
        )

        # Compact BF16 prefill remains an old-M3 memory option. M3.1 NVFP4
        # always takes the native packed-FP4 path below.
        if _should_use_cp_compact_prefill(_CP_COMPACT_PREFILL, self.nvfp4_kv_cache):
            from .msa_cp_compact import validate_compact_mode

            validate_compact_mode(_CP_PACKED_KV_OVERLAP, _CP_PREFIX_PREFETCH)
            if self.block_size != self.page_size:
                raise ValueError(
                    "compact CP prefill requires sparse block size == cache page size"
                )

        cp_info = attn_inputs.context_parallel_info
        device = hidden_states.device
        local_tokens = hidden_states.shape[0]
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.score_chunk import (
            PrefillScoreHostMetadata,
            m3_index_score_chunk_enabled,
            m3_index_score_chunk_rows,
        )

        index_score_chunk_enabled = m3_index_score_chunk_enabled(local_tokens)
        index_score_host_metadata = None
        cache = MSAAttention._cp_shared_meta
        if (
            cache is not None
            and cache.get("owner") is attn_inputs
            and cache.get("layer_idx", -1) < self.layer_idx
        ):
            local_positions = cache["local_positions"]
            unpad_indices = cache["unpad_indices"]
            segment_req_ids_t = cache["segment_req_ids_t"]
            segment_lengths_t = cache["segment_lengths_t"]
            prefix_i32 = cache["prefix_i32"]
            cu_seqlens = cache["cu_seqlens"]
            seq_lens_i32 = cache["seq_lens_i32"]
            max_seqlen_q = cache["max_seqlen_q"]
            max_seqlen_k = cache["max_seqlen_k"]
            n_seg = cache["n_seg"]
            kv_lens_i32 = cache["kv_lens_i32"]
            kv_lens_cpu_list = cache["kv_lens_cpu_list"]
            prefix_cpu_list = cache["prefix_cpu_list"]
            prefix_sum = cache["prefix_sum"]
            token_count_py = cache["token_count"]
            bsz = cache["bsz"]
            max_kv = cache["max_kv"]
            nk = cache["nk"]
            ni = cache["ni"]
            index_score_plan = cache["index_score_plan"]
            index_score_host_metadata = cache["index_score_host_metadata"]
            sparse_attn_plan = cache["sparse_attn_plan"]
            need_build_new_meta = False
        else:
            from rtp_llm.models_py.modules.hybrid.cp_host_metadata import (
                cp_planning_host_mirrors,
            )

            host_mirrors = cp_planning_host_mirrors(cp_info)
            chunk_source = (
                (
                    cp_info.prefill_cp_chunk_lengths
                    if host_mirrors is None
                    else host_mirrors[0]
                )
                .detach()
                .to(dtype=torch.int64)
            )
            prefix_dev = attn_inputs.prefix_lengths.detach().to(
                device=device, dtype=torch.int64
            )
            n_chunks = chunk_source.numel()
            packed_pinned = torch.empty(
                n_chunks + prefix_dev.numel(), dtype=torch.int64, pin_memory=True
            )
            packed_pinned[:n_chunks].copy_(chunk_source, non_blocking=True)
            packed_pinned[n_chunks:].copy_(prefix_dev, non_blocking=True)
            if host_mirrors is not None:
                shuffle_pinned = host_mirrors[1].to(torch.int64)
            else:
                shuffle_pinned = torch.empty(
                    cp_info.prefill_shuffle_indices.numel(),
                    dtype=torch.int64,
                    pin_memory=True,
                )
                shuffle_pinned.copy_(
                    cp_info.prefill_shuffle_indices.detach().to(torch.int64),
                    non_blocking=True,
                )
            need_build_new_meta = True

        qkv, idx_q, idx_k = self._project_qkv_idx(hidden_states, x_fp8, x_scale)
        if self.qk_fuse_norm is not None and self._m31_raw_attention_norms is None:
            qkv = self.qk_fuse_norm(qkv)
        q = qkv[:, : self.q_size].reshape(local_tokens, self.head_num, self.head_dim)

        idx_q = idx_q.reshape(local_tokens, self.num_idx_heads, self.idx_head_dim)
        idx_k = idx_k.reshape(local_tokens, 1, self.idx_head_dim)
        if self._m31_raw_attention_norms is None:
            idx_q = _gemma_rmsnorm_per_head(
                idx_q, self.idx_q_norm_w, self.layernorm_eps
            )
            idx_k = _gemma_rmsnorm_per_head(
                idx_k, self.idx_k_norm_w, self.layernorm_eps
            )

        if need_build_new_meta:
            torch.cuda.current_stream().synchronize()
            chunk_lengths_cpu = packed_pinned[:n_chunks].tolist()
            prefix_cpu = packed_pinned[n_chunks:]
            prefix_cpu_list = prefix_cpu.tolist()
            if any(
                int(prefix_len) % int(self.page_size) != 0
                for prefix_len in prefix_cpu_list
            ):
                raise RuntimeError(
                    "MSA CP prefix restore requires page-aligned prefix lengths; "
                    f"got {prefix_cpu_list} with page_size={self.page_size}"
                )
            if sum(int(x) for x in chunk_lengths_cpu) != local_tokens:
                raise RuntimeError(
                    "MSA CP prefill expects rank-local token count to match "
                    "prefill_cp_chunk_lengths; got "
                    f"local_tokens={local_tokens}, chunks={chunk_lengths_cpu}"
                )
            # Zigzag splits each chunk into two equal halves; an odd chunk would
            # make ``chunk // 2 * 2`` drop a token, leaving its output row
            # unwritten.
            if any(int(x) % 2 != 0 for x in chunk_lengths_cpu):
                raise RuntimeError(
                    "MSA CP prefill requires even per-request chunk lengths for "
                    f"zigzag CP; got chunks={chunk_lengths_cpu}"
                )
            shuffle_cpu = shuffle_pinned.tolist()

            chunk_arr = np.asarray(chunk_lengths_cpu, dtype=np.int64)
            prefix_arr = np.asarray(prefix_cpu_list, dtype=np.int64)
            shuffle_arr = np.asarray(shuffle_cpu, dtype=np.int64)
            bsz_py = chunk_arr.shape[0]
            batch_id = np.repeat(np.arange(bsz_py, dtype=np.int64), chunk_arr)
            positions_np = (np.maximum(shuffle_arr, 0) + prefix_arr[batch_id]).astype(
                np.int32, copy=False
            )
            local_positions = torch.from_numpy(np.ascontiguousarray(positions_np)).to(
                device=device, non_blocking=True
            )

            pair_arr = chunk_arr // 2
            segment_req_ids_np = np.repeat(np.arange(bsz_py, dtype=np.int64), 2)
            segment_lengths_np = np.repeat(pair_arr, 2)
            cursor_arr = np.concatenate([[0], np.cumsum(chunk_arr[:-1])])
            seg0_starts = prefix_arr + np.maximum(shuffle_arr[cursor_arr], 0)
            seg1_starts = prefix_arr + np.maximum(shuffle_arr[cursor_arr + pair_arr], 0)
            segment_starts_np = np.empty(2 * bsz_py, dtype=np.int64)
            segment_starts_np[0::2] = seg0_starts
            segment_starts_np[1::2] = seg1_starts
            empty_mask = pair_arr == 0
            if empty_mask.any():
                segment_starts_np[0::2][empty_mask] = prefix_arr[empty_mask]
                segment_starts_np[1::2][empty_mask] = prefix_arr[empty_mask]

            unpad_indices = cp_info.prefill_qkv_restore_indice[
                cp_info.prefill_qkv_padding_mask == 1
            ].to(torch.long)
            full_input_lengths_cpu = cp_info.prefill_actual_input_lengths_cpu.to(
                torch.int64
            )
            full_input_lengths_cpu_list = [
                int(x) for x in full_input_lengths_cpu.tolist()
            ]
            kv_lens_cpu = prefix_cpu + full_input_lengths_cpu
            kv_lens_cpu_list = [
                int(prefix_cpu_list[i]) + full_input_lengths_cpu_list[i]
                for i in range(len(prefix_cpu_list))
            ]
            prefix_sum = int(prefix_arr.sum())
            token_count_py = int(sum(full_input_lengths_cpu_list))
            bsz = len(kv_lens_cpu_list)
            max_kv = max(kv_lens_cpu_list) if kv_lens_cpu_list else 0
            nk = self.kv_head_num * self.head_dim
            ni = self.idx_head_dim

            n_seg = len(segment_lengths_np)
            packed_seg_np = np.concatenate(
                [segment_req_ids_np, segment_lengths_np, segment_starts_np]
            )
            packed_seg_dev = torch.from_numpy(packed_seg_np).to(
                device=device, non_blocking=True
            )
            segment_req_ids_t = packed_seg_dev[:n_seg].contiguous()
            segment_lengths_t = packed_seg_dev[n_seg : 2 * n_seg].to(torch.int32)
            prefix_i32 = packed_seg_dev[2 * n_seg :].to(torch.int32)
            cu_seqlens = torch.zeros(n_seg + 1, device=device, dtype=torch.int32)
            cu_seqlens[1:] = torch.cumsum(segment_lengths_t, dim=0)
            kv_lens_i32 = kv_lens_cpu.to(device=device, dtype=torch.int32)
            seq_lens_i32 = kv_lens_i32.index_select(0, segment_req_ids_t)
            max_seqlen_q = int(pair_arr.max()) if len(pair_arr) > 0 else 0
            max_seqlen_k = max_kv

            # Build or initialize the fmha index-score plan cache once per forward
            # here, then reuse it across sparse layers via _cp_shared_meta.
            from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
                build_index_score_plan,
                build_sparse_attn_plan,
            )

            if self.nvfp4_kv_cache or index_score_chunk_enabled:
                # Avoid the unusable full-Q OnlyScore plan: its int32 maxscore
                # geometry can overflow on long contexts. The chunk plans are
                # prepared below once the shared physical page table is ready.
                index_score_plan = {}
                segment_req_ids_host = [int(value) for value in segment_req_ids_np]
                index_score_host_metadata = PrefillScoreHostMetadata(
                    query_lens=tuple(int(value) for value in segment_lengths_np),
                    seq_lens=tuple(
                        kv_lens_cpu_list[req_id] for req_id in segment_req_ids_host
                    ),
                    prefix_lens=tuple(int(value) for value in segment_starts_np),
                    slot_ids=tuple(range(n_seg)),
                )
            else:
                index_score_plan = build_index_score_plan(
                    cu_seqlens,
                    seq_lens_i32,
                    prefix_i32,
                    self.num_idx_heads,
                    1,
                    self.block_size,
                    use_fp8_kvcache=self.idx_k_fp8_mode == 2,
                )
            # step3 sparse-attention plan (fmha): GQA num_q_heads/num_kv_heads,
            # kv_block_num=topk. Same per-forward reuse as index_score_plan.
            sparse_attn_plan = build_sparse_attn_plan(
                cu_seqlens,
                seq_lens_i32,
                prefix_i32,
                self.head_num,
                self.kv_head_num,
                self.block_size,
                self.topk_blocks,
                use_fp8_kvcache=False,
            )

            MSAAttention._cp_shared_meta = {
                "owner": attn_inputs,
                "layer_idx": self.layer_idx,
                "local_positions": local_positions,
                "unpad_indices": unpad_indices,
                "segment_req_ids_t": segment_req_ids_t,
                "segment_lengths_t": segment_lengths_t,
                "prefix_i32": prefix_i32,
                "cu_seqlens": cu_seqlens,
                "seq_lens_i32": seq_lens_i32,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "n_seg": n_seg,
                "kv_lens_i32": kv_lens_i32,
                "kv_lens_cpu_list": kv_lens_cpu_list,
                "prefix_cpu_list": prefix_cpu_list,
                "prefix_sum": prefix_sum,
                "token_count": token_count_py,
                "bsz": bsz,
                "max_kv": max_kv,
                "nk": nk,
                "ni": ni,
                "index_score_plan": index_score_plan,
                "index_score_host_metadata": index_score_host_metadata,
                "sparse_attn_plan": sparse_attn_plan,
            }

        self._ensure_scratch_addressing_capacity(
            bsz=bsz,
            max_kv=max_kv,
            exact_cp_shape=True,
        )
        # Only reuse addressing when this layer also reused CP metadata. When a
        # new request rebuilds metadata, the entry-local ``cache`` still points
        # at the previous request, so its addr must be ignored.
        addr_cache = None if need_build_new_meta else cache.get("addr")
        # The old forward's metadata is no longer consumed below. In particular,
        # do not retain its native index workspace during this forward's score.
        del cache
        if (
            addr_cache is not None
            and addr_cache.get("scratch_seq_len") == int(self._scratch_seq_len)
            and addr_cache.get("token_count") == token_count_py
        ):
            req_to_token = addr_cache["req_to_token"]
            write_slots = addr_cache["write_slots"]
            req_to_token_segments = addr_cache["req_to_token_segments"]
            slot_ids = addr_cache["slot_ids"]
            slot_mapping = addr_cache["slot_mapping"]
            kv_page_indices = addr_cache["kv_page_indices"]
            prefix_dst_pages = addr_cache["prefix_dst_pages"]
            prefix_gather_plans = addr_cache["prefix_gather_plans"]
        else:
            pos_range = torch.arange(max_kv, device=device, dtype=torch.int32)
            cache_row_offsets = torch.arange(bsz, device=device, dtype=torch.int32)[
                :, None
            ] * int(self._scratch_seq_len)
            req_to_token = cache_row_offsets + pos_range[None, :]
            slot_parts = []
            for b in range(bsz):
                p0 = int(prefix_cpu_list[b])
                p1 = int(kv_lens_cpu_list[b])
                slot_parts.append(req_to_token[b, p0:p1])
            write_slots = torch.cat(slot_parts).to(torch.int64)
            req_to_token_segments = req_to_token.index_select(
                0, segment_req_ids_t
            ).contiguous()
            slot_ids = torch.arange(n_seg, device=device, dtype=torch.int64)
            slot_mapping = self._kernel_slots_to_paged(write_slots, attn_inputs)
            # fmha physical page table: built once here (per forward), shared by the
            # index-score and step3 fmha kernels across all sparse layers.
            from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
                build_kv_page_indices,
            )

            kv_page_indices = build_kv_page_indices(
                req_to_token_segments, seq_lens_i32, self.block_size
            )
            if self.nvfp4_kv_cache:
                from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.score_chunk import (
                    publish_fp4_prefill_metadata_table,
                )

                # This table is read-only across layers. Every rebuild publishes
                # a fresh epoch; a new forward already owns a new score plan.
                publish_fp4_prefill_metadata_table(index_score_plan, kv_page_indices)
            if prefix_sum > 0:
                prefix_dst_pages = torch.cat(
                    [
                        req_to_token[b, : prefix_len : self.page_size]
                        .to(torch.long)
                        .div(self.page_size, rounding_mode="floor")
                        for b, prefix_len in enumerate(prefix_cpu_list)
                        if prefix_len > 0
                    ]
                )
            else:
                prefix_dst_pages = torch.empty(0, dtype=torch.long, device=device)
            prefix_gather_plans = {}
            if MSAAttention._cp_shared_meta is not None:
                MSAAttention._cp_shared_meta["addr"] = {
                    "scratch_seq_len": int(self._scratch_seq_len),
                    "token_count": token_count_py,
                    "req_to_token": req_to_token,
                    "write_slots": write_slots,
                    "req_to_token_segments": req_to_token_segments,
                    "slot_ids": slot_ids,
                    "slot_mapping": slot_mapping,
                    "kv_page_indices": kv_page_indices,
                    "prefix_dst_pages": prefix_dst_pages,
                    "prefix_gather_plans": prefix_gather_plans,
                }

        if self._kv_sharded and prefix_sum > 0:
            prefix_block_table = self._physical_block_table(attn_inputs)
            prefix_plan_key = (
                prefix_block_table.device,
                int(prefix_block_table.data_ptr()),
                tuple(prefix_block_table.shape),
            )
            prefix_gather_plan = prefix_gather_plans.get(prefix_plan_key)
            if prefix_gather_plan is None:
                prefix_gather_plan = build_cp_sharded_prefix_gather_plan(
                    prefix_block_table,
                    torch.tensor(prefix_cpu_list, dtype=torch.int64),
                    page_size=self.page_size,
                    cp_size=self._cp_size,
                    cp_rank=self._cp_rank,
                )
                prefix_gather_plans[prefix_plan_key] = prefix_gather_plan
        else:
            prefix_gather_plan = None

        if _should_use_cp_compact_prefill(_CP_COMPACT_PREFILL, self.nvfp4_kv_cache):
            from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
                m3_fmha_prefill_enabled,
            )

            if not m3_fmha_prefill_enabled(
                sparse_attn_plan=sparse_attn_plan,
                num_idx_heads=self.num_idx_heads,
                num_kv_heads=self.kv_head_num,
                disable_index_value=self.disable_index_value,
                has_idx_sink=False,
                has_sink=False,
                max_seqlen_k=max_seqlen_k,
                total_q=local_tokens,
            ):
                raise ValueError("compact CP prefill requires the native FMHA path")
        # Native Q8K4 prepares its own score chunks in the FP4 reader below;
        # FMHA OnlyScore plans are consumed only by the legacy cache path.
        if index_score_chunk_enabled and not self.nvfp4_kv_cache:
            assert index_score_host_metadata is not None
            from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
                prepare_fmha_index_score_chunks,
            )

            prepare_fmha_index_score_chunks(
                index_score_plan=index_score_plan,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens_i32,
                prefix_lens=prefix_i32,
                kv_indices=kv_page_indices,
                chunk_rows=m3_index_score_chunk_rows(),
                block_size_k=self.block_size,
                num_heads=self.num_idx_heads,
                idx_kv_heads=1,
                total_q=local_tokens,
                max_seqlen_k=max_seqlen_k,
                host_metadata=index_score_host_metadata,
                use_fp8_kvcache=self.idx_k_fp8_mode == 2,
            )

        query_fp8_outputs = None
        if self.nvfp4_kv_cache and self._m31_raw_attention_norms is not None:
            query_fp8_outputs = tuple(
                torch.empty(
                    (local_tokens, heads, self.head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                for heads in (self.head_num, self.num_idx_heads)
            )
        m31_fused = self._fuse_m31_projected_norm_rope(
            qkv, idx_q, idx_k, local_positions, query_fp8_outputs=query_fp8_outputs
        )
        idx_k = idx_k.contiguous()
        if not m31_fused:
            dummy_idx = _ROPE_DUMMY_SCRATCH.acquire(
                idx_k.shape[0],
                idx_k.shape[1],
                idx_k.shape[2],
                idx_k.dtype,
                idx_k.device,
            )
            self._apply_rope(idx_k, dummy_idx, local_positions)

        can_fuse = self.cos_sin_cache is not None and not self._rope_interleave
        if m31_fused:
            packed_kv = torch.cat(
                (
                    qkv[:, self.q_size : self.q_size + nk],
                    qkv[:, self.q_size + nk : self.q_size + 2 * nk],
                    idx_k.reshape(local_tokens, ni),
                ),
                dim=-1,
            )
        elif can_fuse:
            packed_kv = torch.empty(
                local_tokens, 2 * nk + ni, dtype=qkv.dtype, device=device
            )
            _fused_split_rope_pack(
                qkv,
                idx_k,
                self.cos_sin_cache,
                local_positions,
                packed_kv,
                q_offset=self.q_size,
                nk=nk,
                ni=ni,
                head_dim=self.head_dim,
                num_kv_heads=self.kv_head_num,
                rotary_dim=self.rotary_dim,
            )
        else:
            _, k_fb, v_fb = torch.split(
                qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            k_fb = k_fb.reshape(
                local_tokens, self.kv_head_num, self.head_dim
            ).contiguous()
            v_fb = v_fb.reshape(
                local_tokens, self.kv_head_num, self.head_dim
            ).contiguous()
            dummy_k = torch.zeros_like(k_fb[:, :1, :])
            self._apply_rope(k_fb, dummy_k, local_positions)
            packed_kv = torch.cat(
                [
                    k_fb.reshape(local_tokens, nk),
                    v_fb.reshape(local_tokens, nk),
                    idx_k.reshape(local_tokens, ni),
                ],
                dim=-1,
            )

        all_packed, packed_kv_event = self._cp_all_gather_packed_kv(packed_kv)

        if query_fp8_outputs is not None:
            q, idx_q = query_fp8_outputs
            del query_fp8_outputs
        else:
            q = _rows_to_contig(q)
            idx_q = idx_q.contiguous()
        if m31_fused:
            pass  # Q and index Q were rotated together before the KV pack.
        elif self.head_dim == self.idx_head_dim:
            self._apply_rope(q, idx_q, local_positions)
        else:
            dummy_q = torch.zeros_like(q[:, :1, :])
            self._apply_rope(q, dummy_q, local_positions)
            dummy_iq = torch.zeros_like(idx_q[:, :1, :])
            self._apply_rope(idx_q, dummy_iq, local_positions)
        # q is now an independent contiguous tensor and the packed all-gather
        # has already been enqueued. Drop the large fused QKV allocation before
        # index scoring and sparse attention instead of retaining it through
        # output projection. CUDA stream ordering keeps the queued pack safe.
        del qkv
        if not can_fuse and not m31_fused:
            del k_fb, v_fb

        if packed_kv_event is not None:
            torch.cuda.current_stream(all_packed.device).wait_event(packed_kv_event)

        if _should_use_cp_compact_prefill(_CP_COMPACT_PREFILL, self.nvfp4_kv_cache):
            from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
                flash_prefill_topk_to_block_tables,
            )

            from .msa_cp_compact import (
                build_source_metadata,
                restore_idx_pages,
                run_compact_attention,
            )

            self._write_cp_suffix_to_bf16_working_pages(
                kv_cache,
                all_packed,
                unpad_indices,
                write_slots,
                slot_mapping,
                kv_lens_i32,
                nk,
                ni,
                token_count_py,
                write_main_pages=False,
            )
            prefix_rows = (
                prefix_gather_plan.restore_indices
                if prefix_gather_plan is not None
                else None
            )
            idx_pool, idx_scale_pool = self._idx_k_paged_storage(kv_cache)
            idx_pages = self._gather_cp_compact_prefix_pool(
                idx_pool,
                attn_inputs,
                prefix_cpu_list,
                prefix_gather_plan,
            )
            idx_scales = (
                None
                if idx_scale_pool is None
                else self._gather_cp_compact_prefix_pool(
                    idx_scale_pool,
                    attn_inputs,
                    prefix_cpu_list,
                    prefix_gather_plan,
                )
            )
            restore_idx_pages(
                idx_pages,
                prefix_dst_pages,
                self._scratch_idx_k,
                prefix_rows,
                idx_scales=idx_scales,
            )
            del idx_pages, idx_scales
            if attn_inputs.cache_store_inputs:
                from rtp_llm.models_py.modules.factory.attention import (
                    common as _attn_common,
                )

                write_impl = _attn_common.create_write_cache_store_impl(attn_inputs)
                _attn_common.apply_write_cache_store(write_impl, attn_inputs, kv_cache)
            _, _, topk_idx = flash_prefill_topk_to_block_tables(
                idx_q=idx_q,
                idx_k_cache=self._scratch_idx_k,
                req_to_token=req_to_token_segments,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens_i32,
                prefix_lens=prefix_i32,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                block_size_k=self.block_size,
                topk=self.topk_blocks,
                num_pages=triton.cdiv(max_seqlen_k, self.block_size),
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                index_score_plan=index_score_plan,
                kv_indices=kv_page_indices,
                emit_block_table=False,
            )
            main_pages = self._gather_cp_compact_prefix_pool(
                self._paged_kv_base_view(kv_cache),
                attn_inputs,
                prefix_cpu_list,
                prefix_gather_plan,
            )
            source_meta = build_source_metadata(
                prefix_cpu_list,
                kv_lens_i32,
                int(self._scratch_seq_len),
                self.page_size,
                prefix_dst_pages,
                prefix_rows,
            )
            o = run_compact_attention(
                q,
                topk_idx,
                kv_page_indices,
                sparse_attn_plan,
                main_pages,
                all_packed,
                unpad_indices,
                source_meta,
                int(self._scratch_seq_len),
                self.page_size,
                self.kv_head_num,
                self.head_dim,
                ni,
                self.topk_blocks,
            )
            del all_packed, packed_kv, main_pages, q, idx_q, topk_idx, source_meta
            return self.o_proj(o.reshape(local_tokens, -1).contiguous())

        if self.nvfp4_kv_cache:
            from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
                flash_prefill_topk_to_block_tables_fp4,
                sparse_prefill_from_topk_fp4,
            )

            main, main_scales, idx_k_fp4, idx_scales = (
                self._write_cp_suffix_to_nvfp4_working_pages(
                    kv_cache,
                    all_packed,
                    unpad_indices,
                    write_slots,
                    slot_mapping,
                    nk,
                    ni,
                    token_count_py,
                    prefix_cpu_list,
                    prefix_dst_pages,
                    prefix_gather_plan,
                    attn_inputs,
                    kv_lens_i32,
                )
            )
            del all_packed, packed_kv
            if attn_inputs.cache_store_inputs:
                from rtp_llm.models_py.modules.factory.attention import (
                    common as _attn_common,
                )

                write_impl = _attn_common.create_write_cache_store_impl(attn_inputs)
                _attn_common.apply_write_cache_store(write_impl, attn_inputs, kv_cache)

            page_count = int(main.shape[1])
            groups = self.head_dim // NVFP4_GROUP_SIZE
            idx_groups = self.idx_head_dim // NVFP4_GROUP_SIZE
            k_fp4, v_fp4 = main[0], main[1]
            # Main sparse attention accepts scale bytes as uint8. IndexScore
            # receives the packed idxK and MMA-ordered scales directly through
            # RTP's Q8K4 page reader.
            k_scale = (
                main_scales[0]
                .view(torch.uint8)
                .view(page_count * self.kv_head_num * self.page_size, groups)
            )
            v_scale = main_scales[1].view(torch.uint8).view_as(k_scale)
            idx_k_scale_mma = idx_scales.view(
                page_count,
                1,
                idx_groups // 4,
                32,
                4,
                4,
            )
            if isinstance(index_score_plan, dict):
                index_score_plan["_fp4_host_metadata"] = index_score_host_metadata
            _, _, topk_idx = flash_prefill_topk_to_block_tables_fp4(
                idx_q=idx_q,
                idx_k_fp4=idx_k_fp4,
                idx_k_scale_mma=idx_k_scale_mma,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens_i32,
                prefix_lens=prefix_i32,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                block_size_k=self.block_size,
                topk=self.topk_blocks,
                num_pages=triton.cdiv(max_seqlen_k, self.block_size),
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                index_score_plan=index_score_plan,
                kv_indices=kv_page_indices,
                emit_block_table=False,
            )
            o = sparse_prefill_from_topk_fp4(
                q,
                k_fp4,
                v_fp4,
                k_scale,
                v_scale,
                topk_idx,
                kv_page_indices,
                sparse_attn_plan,
                self.topk_blocks,
                self.block_size,
                self.head_dim**-0.5,
            )
            del q, idx_q, topk_idx
            return self.o_proj(o.reshape(local_tokens, -1).contiguous())

        # The fused writer persists rank-owned suffix pages, fills idx-K scratch,
        # and builds the BF16 HND working pages consumed by native sparse FMHA.
        working_k_pages, working_v_pages = self._write_cp_suffix_to_bf16_working_pages(
            kv_cache,
            all_packed,
            unpad_indices,
            write_slots,
            slot_mapping,
            kv_lens_i32,
            nk,
            ni,
            token_count_py,
        )
        self._restore_cp_prefix_working_pages(
            kv_cache,
            prefix_cpu_list,
            req_to_token,
            attn_inputs,
            working_k_pages,
            working_v_pages,
            prefix_dst_pages,
            prefix_gather_plan,
        )
        del all_packed, packed_kv
        if attn_inputs.cache_store_inputs:
            from rtp_llm.models_py.modules.factory.attention import (
                common as _attn_common,
            )

            write_impl = _attn_common.create_write_cache_store_impl(attn_inputs)
            _attn_common.apply_write_cache_store(write_impl, attn_inputs, kv_cache)

        _, o = minimax_sparse_prefill(
            q=q,
            idx_q=idx_q,
            idx_k_cache=self._scratch_idx_k,
            req_to_token=req_to_token_segments,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens_i32,
            prefix_lens=prefix_i32,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size_k=self.block_size,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            disable_index_value=self.disable_index_value,
            index_score_plan=index_score_plan,
            sparse_attn_plan=sparse_attn_plan,
            kv_indices=kv_page_indices,
            k_paged_cache=working_k_pages,
            v_paged_cache=working_v_pages,
        )

        # The sparse kernels are enqueued on the current stream, so allocator
        # stream ordering makes it safe to release their read-only inputs here.
        # Do this before o_proj: for long CP prefill, otherwise the logical
        # paged working set remains live while the projection allocates its
        # output/workspace.
        del q, idx_q, working_k_pages, working_v_pages

        return self.o_proj(o.reshape(local_tokens, -1).contiguous())

    # ------------------------------------------------------------------
    def _forward_paged_decode(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: LayerKVCache,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
            minimax_paged_sparse_decode,
        )

        input_shape = hidden_states.shape[:-1]
        total_tokens = hidden_states.shape[0]
        device = hidden_states.device

        kv_lens, seq_lens, positions, phys_block_table = self._paged_decode_addressing(
            attn_inputs, device
        )
        # Ordinary BF16/FP8 index scoring addresses the same physical table as
        # main attention. NVFP4 returns directly through Q8KV4 below, while
        # feature-off paths still pass an explicit None.
        score_block_table = None
        # The fused write kernel casts K/V to the paged-pool dtype, so this path
        # is valid for both BF16 and FP8 KV cache. Keep draft decode aligned with
        # target verify instead of silently disabling the fused decode path for
        # the production FP8 configuration.
        if (
            self._should_use_mxfp8_fused_qkv_idx_decode(x_fp8, x_scale)
            and not self.nvfp4_kv_cache
        ):
            paged_kv_base = self._paged_kv_base_view(kv_cache)
            paged_idx_k, paged_idx_scale = self._idx_k_paged_storage(kv_cache)
            q, idx_q = self._decode_project_fused_qkv_idx(
                total_tokens,
                positions,
                seq_lens,
                phys_block_table,
                paged_kv_base,
                paged_idx_k,
                paged_idx_scale,
                x_fp8=x_fp8,
                x_scale=x_scale,
            )
            paged_decode_views = (
                paged_kv_base[:, 0],
                paged_kv_base[:, 1],
                phys_block_table,
                paged_idx_k,
                paged_idx_scale,
            )
        else:
            qkv, idx_q, idx_k = self._project_qkv_idx(hidden_states, x_fp8, x_scale)
            m31_fused = self._fuse_m31_projected_norm_rope(qkv, idx_q, idx_k, positions)
            if self.qk_fuse_norm is not None and not m31_fused:
                qkv = self.qk_fuse_norm(qkv)
            q, k, v = torch.split(
                qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q = q.reshape(total_tokens, self.head_num, self.head_dim)
            k = k.reshape(total_tokens, self.kv_head_num, self.head_dim)
            v = v.reshape(total_tokens, self.kv_head_num, self.head_dim)

            idx_q = idx_q.reshape(total_tokens, self.num_idx_heads, self.idx_head_dim)
            idx_k = idx_k.reshape(total_tokens, 1, self.idx_head_dim)
            idx_q = self._legacy_index_norm(
                idx_q, self.idx_q_norm_w, self.layernorm_eps
            )
            idx_k = self._legacy_index_norm(
                idx_k, self.idx_k_norm_w, self.layernorm_eps
            )

            q = q.contiguous()
            k = k.contiguous()
            if not m31_fused:
                self._apply_rope(q, k, positions)
            idx_q = idx_q.contiguous()
            idx_k = idx_k.contiguous()
            if not m31_fused:
                self._apply_rope(idx_q, idx_k, positions)
            if self.nvfp4_kv_cache:
                fuse_bf16_query_rounding = (
                    q.dtype == torch.bfloat16 and idx_q.dtype == torch.bfloat16
                )
                if not fuse_bf16_query_rounding:
                    nvfp4_round_to_e4m3_compute_grid_(q)
                    nvfp4_round_to_e4m3_compute_grid_(idx_q)

            paged_decode_views = self._write_kv_cache_and_idx_k_for_decode(
                kv_cache, k, v, idx_k, seq_lens, phys_block_table
            )
            if self.nvfp4_kv_cache:
                from rtp_llm.models_py.triton_kernels.sparse_msa.decode.q8kv4_decode import (
                    q8kv4_paged_sparse_decode,
                )

                layout = nvfp4_cache_layout(
                    kv_cache.kv_cache_base,
                    kv_cache.kv_scale_base,
                    self.kv_head_num,
                    self.physical_page_size,
                    self.head_dim,
                )
                q8kv4 = q8kv4_paged_sparse_decode(
                    q,
                    idx_q,
                    layout,
                    phys_block_table,
                    seq_lens,
                    indexer_dim=self.idx_head_dim,
                    block_size=self.block_size,
                    topk=self.topk_blocks,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    score_type=self.score_type,
                    mma_scale_layout=True,
                    fuse_bf16_query_rounding=fuse_bf16_query_rounding,
                    max_seq_len=self._cuda_graph_max_seq_len,
                )
                attn_output = q8kv4.output.reshape(*input_shape, -1).contiguous()
                output = self.o_proj(attn_output)
                if self.tp_size > 1:
                    output = all_reduce(output, group=Group.TP)
                return output
        if paged_decode_views is None:
            raise RuntimeError(
                "MSA paged decode requires a BF16 or FP8 5-D paged KV cache and "
                "an idx_K side region matching the configured cache mode"
            )
        paged_main_k, paged_main_v, phys_block_table, paged_idx_k, paged_idx_scale = (
            paged_decode_views
        )
        max_seqlen_k = self._paged_decode_max_kv(attn_inputs, kv_lens, phys_block_table)
        _idx_o, o = minimax_paged_sparse_decode(
            q=q,
            sink=None,
            idx_q=idx_q,
            seq_lens=seq_lens,
            max_seqlen=max_seqlen_k,
            block_size_k=self.block_size,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            disable_index_value=self.disable_index_value,
            paged_main_k=paged_main_k,
            paged_main_v=paged_main_v,
            phys_block_table=phys_block_table,
            paged_idx_k=paged_idx_k,
            paged_idx_scale=paged_idx_scale,
            score_block_table=score_block_table,
        )
        attn_output = o.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output

    def _forward_target_verify(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: LayerKVCache,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        use_fused_addressing: bool = False,
        use_paged_capacity_bound: bool = False,
    ) -> torch.Tensor:
        from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
            minimax_paged_sparse_decode,
        )

        if self._paged_decode_static_ok is None:
            self._paged_decode_static_ok = self._check_paged_decode_static(kv_cache)
        if not self._paged_decode_static_ok:
            raise RuntimeError(
                "MSA target verify requires the paged decode cache layout"
            )

        input_shape = hidden_states.shape[:-1]
        total_tokens = int(hidden_states.shape[0])
        device = hidden_states.device

        # The shared target-verify contract remains request-row based. Expand it
        # only inside MiniMax-M3 MSA, immediately before the sparse operator.
        (
            request_block_table,
            phys_block_table,
            positions,
            seq_lens,
            valid_token_mask,
        ) = self._target_verify_addressing(
            attn_inputs,
            total_tokens,
            device,
            use_fused_cuda=use_fused_addressing,
        )
        request_batch_size = int(request_block_table.shape[0])
        is_ragged = bool(getattr(attn_inputs, "is_ragged_target_verify", False))
        write_seq_lens = seq_lens
        score_block_table = phys_block_table if is_ragged else request_block_table
        norm_rope_outputs = None
        if (
            self.nvfp4_kv_cache
            and getattr(self, "_m31_raw_attention_norms", None) is not None
            and hidden_states.dtype == torch.bfloat16
            and (self.head_num, self.kv_head_num, self.num_idx_heads) == (64, 4, 4)
            and self.head_dim == self.idx_head_dim == 128
            and 1 < total_tokens <= 128
        ):
            # Replace the four existing contiguous copies, without introducing
            # history workspace or per-layer persistent storage. M1 previously
            # aliases its projection and must keep the allocation-free path.
            norm_rope_outputs = tuple(
                torch.empty(
                    (total_tokens, heads, self.head_dim),
                    dtype=hidden_states.dtype,
                    device=device,
                )
                for heads in (self.head_num, self.kv_head_num, self.num_idx_heads, 1)
            )

        if (
            self._should_use_mxfp8_fused_qkv_idx_decode(x_fp8, x_scale)
            and not self.nvfp4_kv_cache
        ):
            paged_kv_base = self._paged_kv_base_view(kv_cache)
            paged_idx_k, paged_idx_scale = self._idx_k_paged_storage(kv_cache)
            q, idx_q = self._decode_project_fused_qkv_idx(
                total_tokens,
                positions,
                write_seq_lens,
                phys_block_table,
                paged_kv_base,
                paged_idx_k,
                paged_idx_scale,
                x_fp8=x_fp8,
                x_scale=x_scale,
            )
            paged_decode_views = (
                paged_kv_base[:, 0],
                paged_kv_base[:, 1],
                phys_block_table,
                paged_idx_k,
                paged_idx_scale,
            )
        else:
            qkv, idx_q, idx_k = self._project_qkv_idx(hidden_states, x_fp8, x_scale)
            m31_fused = self._fuse_m31_projected_norm_rope(
                qkv, idx_q, idx_k, positions, contiguous_outputs=norm_rope_outputs
            )
            if self.qk_fuse_norm is not None and not m31_fused:
                qkv = self.qk_fuse_norm(qkv)
            q, k, v = torch.split(
                qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q = q.reshape(total_tokens, self.head_num, self.head_dim)
            k = k.reshape(total_tokens, self.kv_head_num, self.head_dim)
            v = v.reshape(total_tokens, self.kv_head_num, self.head_dim)

            idx_q = idx_q.reshape(total_tokens, self.num_idx_heads, self.idx_head_dim)
            idx_k = idx_k.reshape(total_tokens, 1, self.idx_head_dim)
            idx_q = self._legacy_index_norm(
                idx_q, self.idx_q_norm_w, self.layernorm_eps
            )
            idx_k = self._legacy_index_norm(
                idx_k, self.idx_k_norm_w, self.layernorm_eps
            )

            if norm_rope_outputs is not None:
                q, k, idx_q, idx_k = norm_rope_outputs
            elif m31_fused:
                q = q.contiguous()
                k = k.contiguous()
            elif self.nvfp4_kv_cache:
                q, k = self._apply_rope_contiguous(q, k, positions)
            else:
                q = q.contiguous()
                k = k.contiguous()
                self._apply_rope(q, k, positions)
            idx_q = idx_q.contiguous()
            idx_k = idx_k.contiguous()
            if not m31_fused:
                self._apply_rope(idx_q, idx_k, positions)
            if self.nvfp4_kv_cache:
                fuse_bf16_query_rounding = (
                    q.dtype == torch.bfloat16 and idx_q.dtype == torch.bfloat16
                )
                if not fuse_bf16_query_rounding:
                    nvfp4_round_to_e4m3_compute_grid_(q)
                    nvfp4_round_to_e4m3_compute_grid_(idx_q)

            paged_decode_views = self._write_kv_cache_and_idx_k_for_decode(
                kv_cache, k, v, idx_k, write_seq_lens, phys_block_table
            )
            if self.nvfp4_kv_cache:
                from rtp_llm.models_py.triton_kernels.sparse_msa.decode.q8kv4_decode import (
                    q8kv4_paged_sparse_decode,
                )

                layout = nvfp4_cache_layout(
                    kv_cache.kv_cache_base,
                    kv_cache.kv_scale_base,
                    self.kv_head_num,
                    self.physical_page_size,
                    self.head_dim,
                )
                q8kv4 = q8kv4_paged_sparse_decode(
                    q,
                    idx_q,
                    layout,
                    phys_block_table,
                    seq_lens,
                    indexer_dim=self.idx_head_dim,
                    block_size=self.block_size,
                    topk=self.topk_blocks,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    score_type=self.score_type,
                    mma_scale_layout=True,
                    fuse_bf16_query_rounding=fuse_bf16_query_rounding,
                    # Addressing expands one shared page table per request
                    # into contiguous query rows with individual causal lengths.
                    query_width=(
                        1 if is_ragged else total_tokens // request_batch_size
                    ),
                    query_cu_seqlens=(attn_inputs.cu_seqlens if is_ragged else None),
                    # Released M3.1 DSpARK has gamma=7, hence at most anchor +
                    # seven proposal rows in one compact request group.
                    max_query_width=(8 if is_ragged else 1),
                    valid_token_mask=valid_token_mask,
                    max_seq_len=self._cuda_graph_max_seq_len,
                )
                attn_output = q8kv4.output.reshape(*input_shape, -1).contiguous()
                output = self.o_proj(attn_output)
                output = torch.where(
                    valid_token_mask[:, None], output, torch.zeros_like(output)
                )
                if self.tp_size > 1:
                    output = all_reduce(output, group=Group.TP)
                return output
        if paged_decode_views is None:
            raise RuntimeError(
                "MSA target verify requires BF16 or FP8 paged K/V and idx_K scale storage"
            )
        paged_main_k, paged_main_v, phys_block_table, paged_idx_k, paged_idx_scale = (
            paged_decode_views
        )
        if self._cuda_graph_forward_active() or use_paged_capacity_bound:
            max_seqlen_k = self._cuda_graph_max_kv(attn_inputs, request_block_table)
        else:
            max_seqlen_k = int(seq_lens.max().item())
        score_seq_lens = (
            seq_lens
            if is_ragged
            else seq_lens.view(request_batch_size, -1)[:, -1].contiguous()
        )
        _idx_o, o = minimax_paged_sparse_decode(
            q=q,
            sink=None,
            idx_q=idx_q,
            seq_lens=seq_lens,
            max_seqlen=max_seqlen_k,
            block_size_k=self.block_size,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            disable_index_value=self.disable_index_value,
            paged_main_k=paged_main_k,
            paged_main_v=paged_main_v,
            phys_block_table=phys_block_table,
            paged_idx_k=paged_idx_k,
            paged_idx_scale=paged_idx_scale,
            score_block_table=score_block_table,
            score_seq_lens=score_seq_lens,
            decode_query_len=(1 if is_ragged else total_tokens // request_batch_size),
        )
        o = torch.where(valid_token_mask[:, None, None], o, torch.zeros_like(o))

        attn_output = o.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        output = torch.where(
            valid_token_mask[:, None], output, torch.zeros_like(output)
        )
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output

    def forward_paged_continuation(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: LayerKVCache,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a fixed-width continuation directly from paged MSA state.

        MiniMax-M3 MTP uses this after target verification.  Unlike prompt
        prefill, the complete history is already in paged K/V and index-K
        storage, so rebuilding full-history prefill scratch is unnecessary.
        """
        if not attn_inputs.is_prefill:
            raise RuntimeError("paged MTP continuation must be represented as prefill")
        request_rows = int(attn_inputs.input_lengths.numel())
        total_tokens = int(hidden_states.shape[0])
        if (
            request_rows <= 0
            or total_tokens <= 0
            or total_tokens % request_rows != 0
            or total_tokens // request_rows > 8
        ):
            raise RuntimeError(
                "invalid recurrent MTP draft-prefill shape: "
                f"tokens={total_tokens}, requests={request_rows}"
            )
        if getattr(attn_inputs, "context_parallel_info", None) is not None:
            # CP prefill owns a different sequence-sharding contract. Preserve
            # its existing correct fallback instead of interpreting CP metadata
            # as fixed request rows.
            return self.forward(
                hidden_states,
                attn_inputs,
                kv_cache,
                x_fp8=x_fp8,
                x_scale=x_scale,
            )
        return self._forward_target_verify(
            hidden_states,
            attn_inputs,
            kv_cache,
            x_fp8=x_fp8,
            x_scale=x_scale,
            use_fused_addressing=True,
            use_paged_capacity_bound=True,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert kv_cache is not None, "MSAAttention requires a KV cache"
        assert (
            attn_inputs.kv_cache_kernel_block_id_device is not None
        ), "MSAAttention requires a block table"

        if bool(getattr(attn_inputs, "is_target_verify", False)):
            return self._forward_target_verify(
                hidden_states,
                attn_inputs,
                kv_cache,
                x_fp8=x_fp8,
                x_scale=x_scale,
            )

        if not attn_inputs.is_prefill:
            if not self._use_paged_decode_path(attn_inputs, kv_cache):
                raise RuntimeError(
                    "MSA decode requires paged KV/index cache; "
                    "flat-scratch decode is unsupported"
                )
            return self._forward_paged_decode(
                hidden_states, attn_inputs, kv_cache, x_fp8=x_fp8, x_scale=x_scale
            )
        if self.cp_enabled:
            if attn_inputs.context_parallel_info is None:
                raise RuntimeError("MSA CP prefill requires context-parallel metadata")
            return self._forward_cp_prefill(
                hidden_states, attn_inputs, kv_cache, x_fp8=x_fp8, x_scale=x_scale
            )
        if self.nvfp4_kv_cache:
            return self._forward_nvfp4_prefill(
                hidden_states, attn_inputs, kv_cache, x_fp8=x_fp8, x_scale=x_scale
            )
        return self._forward_prefill(
            hidden_states, attn_inputs, kv_cache, x_fp8=x_fp8, x_scale=x_scale
        )
