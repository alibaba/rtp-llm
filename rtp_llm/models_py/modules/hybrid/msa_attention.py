"""MiniMax-M3 sparse attention (MSA) module.

Wires the ported Triton MSA kernels (``rtp_llm/models_py/triton_kernels/
sparse_msa``) into rtp-llm's GenericMoe decoder for MiniMax-M3 *sparse* layers
(e.g. layers 3,4 in the 5-layer mini model). Dense layers keep using the
shared FlashInfer FMHA impl; only sparse layers are routed here.

Design (paged-only store):

* The persistent store for both the main K/V and the index-K is the standard
  cache-manager paged pool — there is NO self-built per-layer side cache.
  Main K/V live in ``kv_cache.kv_cache_base`` (HND paged pool) and idx_K lives
  in that pool's scale region ``kv_cache.kv_scale_base`` (reinterpreted as
  BF16). Both are addressed by the same block table and therefore travel
  together under PD separation.

* Prefill MSA kernels consume flat *token-slot* tensors
  ``[max_slots, num_kv_heads, head_dim]`` addressed by a
  ``req_to_token [max_reqs, max_kv_len]`` map plus ``slot_ids [batch]``. Since
  that layout differs from the paged pool, prefill gathers the active sequence
  out of the paged pool into a process-wide *transient* scratch
  (``_MainKVScratch`` / ``_IdxKScratch``) that the prefill kernel reads.
  The opt-in paged decode path writes only the current K/V/idx_K token into the
  persistent paged pool and reads history directly via the physical block table.

* In the normal non-CP path the physical slot for ``(request b, token
  position p)`` is the paged block table::

      slot = block_table[b, p // page_size] * page_size + (p % page_size)

* In CP prefill, K/V are all-gathered into full sequence order while Q stays
  rank-local, then written into this rank's paged shard; the gather scratch is
  indexed by a compact ``b*seq_len + pos`` grid for the kernel.

The index branch (``index_q_proj`` / ``index_k_proj`` + per-head Gemma RMSNorm
+ partial RoPE) only selects top-k blocks; with ``disable_index_value=True``
(M3 default) it does not contribute to the attention value, so ``idx_v`` is
``None`` and the index output ``idx_o`` is discarded.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import triton
import triton.language as tl

# Optimized CP prefill path for the paged-cache implementation. Default-on;
# set M3_MSA_USE_V2_CP_PREFILL=0 to fall back to the simpler reference path.
_USE_V2_CP_PREFILL = os.environ.get("M3_MSA_USE_V2_CP_PREFILL", "1") != "0"
# Fused CP paged write removes the unpack tensors plus mha_kv_write_cache
# for cold/sharded v2 prefill. Set to 0 to fall back to the two-kernel path.
_USE_FUSED_CP_PAGED_WRITE = os.environ.get("M3_MSA_FUSED_CP_PAGED_WRITE", "1") != "0"
# Fused paged->scratch main-K/V gather (one pass instead of torch's
# index -> cast -> index_put). Set M3_MSA_FUSED_KV_GATHER=0 for the torch path.
_FUSED_KV_GATHER = os.environ.get("M3_MSA_FUSED_KV_GATHER", "1") != "0"

import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    gather_cp_sharded_prefix_pool,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import (
    CudaMxfp8Linear,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build token-row MSA addressing, optionally using one CUDA launch."""
    batch_size = int(prefix_lengths.numel())
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
    paged_idx_k_ptr,  # [block*page, idx_dim]
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
    paged_token_slot = store_block * PAGE_SIZE + store_page_offset

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
    tl.store(
        paged_idx_k_ptr + paged_token_slot * HEAD_DIM + rot_off,
        rot_first.to(paged_idx_k_ptr.dtype.element_ty),
        mask=rot_mask & valid_paged_slot & is_idx_k,
    )
    tl.store(
        paged_idx_k_ptr + paged_token_slot * HEAD_DIM + HALF_ROT + rot_off,
        rot_second.to(paged_idx_k_ptr.dtype.element_ty),
        mask=rot_mask & valid_paged_slot & is_idx_k,
    )
    if REM > 0:
        tl.store(
            paged_kv_ptr + k_store_base + (ROTARY_DIM + rem_off) * KV_STRIDE_DIM,
            rem.to(tl.bfloat16).to(paged_kv_ptr.dtype.element_ty),
            mask=rem_mask & valid_paged_slot & is_k,
        )
        tl.store(
            paged_idx_k_ptr + paged_token_slot * HEAD_DIM + ROTARY_DIM + rem_off,
            rem.to(paged_idx_k_ptr.dtype.element_ty),
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
    paged_idx_k_flat: torch.Tensor,
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
    K/V and idx_K are persisted into the paged cache. The original fallback path
    materializes bf16 after RMSNorm before RoPE, so keep that as the numerical
    reference when needed.
    """
    T = fused_qkv_idx_out.shape[0]
    if T == 0:
        return
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
        paged_idx_k_flat,
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
    )


@triton.jit
def _fused_unpack_packed_cp_kernel(
    packed_ptr,
    unpad_ptr,
    k_ptr,
    v_ptr,
    idx_ptr,
    T,
    PACKED_DIM: tl.constexpr,
    NK: tl.constexpr,
    NI: tl.constexpr,
    BLK: tl.constexpr,
):
    # One program per output token: read packed[unpad[t]] (contiguous row) and split the
    # three column ranges into contiguous k/v/idx rows. Token-parallel + contiguous
    # read/write is ~2-3x faster than the element-parallel split (better coalescing).
    t = tl.program_id(0)
    if t >= T:
        return
    base = tl.load(unpad_ptr + t).to(tl.int64) * PACKED_DIM
    for o in range(0, NK, BLK):
        off = o + tl.arange(0, BLK)
        m = off < NK
        tl.store(
            k_ptr + t * NK + off,
            tl.load(packed_ptr + base + off, mask=m, other=0),
            mask=m,
        )
        tl.store(
            v_ptr + t * NK + off,
            tl.load(packed_ptr + base + NK + off, mask=m, other=0),
            mask=m,
        )
    for o in range(0, NI, BLK):
        off = o + tl.arange(0, BLK)
        m = off < NI
        tl.store(
            idx_ptr + t * NI + off,
            tl.load(packed_ptr + base + 2 * NK + off, mask=m, other=0),
            mask=m,
        )


def _fused_unpack_packed_cp(
    packed: torch.Tensor,
    unpad_indices: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    full_idx_k: torch.Tensor,
    nk: int,
    ni: int,
    token_count: Optional[int] = None,
) -> None:
    """Unpad packed [K|V|idx_K] all-gather output in one Triton launch."""
    if token_count is None:
        token_count = int(unpad_indices.numel())
    if token_count == 0:
        return
    packed_dim = 2 * nk + ni
    _fused_unpack_packed_cp_kernel[(token_count,)](
        packed,
        unpad_indices,
        full_k.reshape(token_count, nk),
        full_v.reshape(token_count, nk),
        full_idx_k.reshape(token_count, ni),
        token_count,
        PACKED_DIM=packed_dim,
        NK=nk,
        NI=ni,
        BLK=512,
        num_warps=2,
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
def _fused_scatter_cp_gathered_kernel(
    k_ptr,
    v_ptr,
    idx_ptr,
    write_slots_ptr,
    slot_mapping_ptr,
    scratch_k_ptr,
    scratch_v_ptr,
    scratch_idx_ptr,
    scale_flat_ptr,
    TOTAL: tl.constexpr,
    NK: tl.constexpr,
    NI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    per_token = 2 * NK + NI
    token = offs // per_token
    field = offs - token * per_token
    dst_slot = tl.load(write_slots_ptr + token, mask=mask, other=0).to(tl.int64)

    is_k = field < NK
    is_v = (field >= NK) & (field < 2 * NK)
    is_idx = field >= 2 * NK

    k_field = field
    v_field = field - NK
    idx_field = field - 2 * NK
    k_vals = tl.load(k_ptr + token * NK + k_field, mask=mask & is_k, other=0.0)
    v_vals = tl.load(v_ptr + token * NK + v_field, mask=mask & is_v, other=0.0)
    idx_vals = tl.load(idx_ptr + token * NI + idx_field, mask=mask & is_idx, other=0.0)

    tl.store(scratch_k_ptr + dst_slot * NK + k_field, k_vals, mask=mask & is_k)
    tl.store(scratch_v_ptr + dst_slot * NK + v_field, v_vals, mask=mask & is_v)
    tl.store(
        scratch_idx_ptr + dst_slot * NI + idx_field,
        idx_vals,
        mask=mask & is_idx,
    )

    physical_slot = tl.load(slot_mapping_ptr + token, mask=mask & is_idx, other=-1).to(
        tl.int64
    )
    tl.store(
        scale_flat_ptr + physical_slot * NI + idx_field,
        idx_vals,
        mask=mask & is_idx & (physical_slot >= 0),
    )


def _fused_scatter_cp_gathered(
    k: torch.Tensor,
    v: torch.Tensor,
    idx_k: torch.Tensor,
    write_slots: torch.Tensor,
    slot_mapping: torch.Tensor,
    scratch_k: torch.Tensor,
    scratch_v: torch.Tensor,
    idx_scratch: torch.Tensor,
    scale_flat: torch.Tensor,
    nk: int,
    ni: int,
    token_count: Optional[int] = None,
) -> None:
    """Scatter CP-gathered K/V/idx_K to scratch and persist idx_K in one launch."""
    if token_count is None:
        token_count = int(write_slots.numel())
    if token_count == 0:
        return
    total = token_count * (2 * nk + ni)
    _fused_scatter_cp_gathered_kernel[(triton.cdiv(total, 256),)](
        k.reshape(token_count, nk),
        v.reshape(token_count, nk),
        idx_k.reshape(token_count, ni),
        write_slots,
        slot_mapping,
        scratch_k.reshape(-1, nk),
        scratch_v.reshape(-1, nk),
        idx_scratch.reshape(-1, ni),
        scale_flat,
        TOTAL=total,
        NK=nk,
        NI=ni,
        BLOCK=256,
    )


@triton.jit
def _write_decode_kv_idx_kernel(
    k_ptr,
    v_ptr,
    idx_ptr,
    seq_lens_ptr,
    block_table_ptr,
    base_ptr,
    scale_flat_ptr,
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
    MAX_PHYSICAL_BLOCKS: tl.constexpr,
    MAX_BLOCKS_PER_ROW: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_IDX: tl.constexpr,
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
    physical_slot = physical_block * PAGE_SIZE + block_off

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
    tl.store(
        scale_flat_ptr + physical_slot * IDX_DIM + idx_offs,
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
    scale_flat: torch.Tensor,
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
        scale_flat,
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
        MAX_PHYSICAL_BLOCKS=int(base.shape[0]),
        MAX_BLOCKS_PER_ROW=int(block_table.shape[1]),
        BLOCK_KV=triton.next_power_of_2(int(k.shape[1]) * int(k.shape[2])),
        BLOCK_IDX=triton.next_power_of_2(idx_dim),
    )


def _write_main_kv_to_paged(
    k: torch.Tensor,
    v: torch.Tensor,
    base: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Persist K/V into the 5-D paged pool via the tuned C++ writer.

    When ``base`` is float8_e4m3fn the C++ op casts bf16/half/float activations
    to e4m3 on store (no-scale cast, matching the sparse decode kernels)."""
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    rtp_llm_ops.mha_kv_write_cache(k.contiguous(), v.contiguous(), base, slot_mapping)


@triton.jit
def _gather_paged_kv_to_scratch_kernel(
    base_ptr,
    gf_ptr,
    dst_ptr,
    out_k_ptr,
    out_v_ptr,
    N,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BASE_S0: tl.constexpr,
    BASE_S1: tl.constexpr,
    BASE_S2: tl.constexpr,
    BASE_S3: tl.constexpr,
    BASE_S4: tl.constexpr,
    OUT_S0: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """One pass: read paged K/V at physical slot gf[t], cast, write scratch[dst[t]].

    Replaces ``scratch[dst] = pool[gf // p, gf % p].to(bf16)``, which torch
    lowers to three full passes over the whole active history (index -> cast ->
    index_put) per K and per V per layer.
    """
    # Every term that scales with the scratch or the pool is promoted to int64
    # before it is multiplied. The scratch is sized from the *historical* maxima
    # of batch size and sequence length (see _ensure_gather_scratch), so dst can
    # be far larger than the current request implies, and int32 offsets would
    # wrap into unrelated tensors -- cf. the _kv_flat_to_paged_kernel INT32 fix.
    pid = tl.program_id(0).to(tl.int64)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    t_ok = t < N
    gf = tl.load(gf_ptr + t, mask=t_ok, other=-1).to(tl.int64)
    dst = tl.load(dst_ptr + t, mask=t_ok, other=0).to(tl.int64)
    # gf < 0 marks a non-owned token under CP page-RR sharding; skip it rather
    # than letting a negative index wrap into unrelated pool rows.
    row_ok = t_ok & (gf >= 0)
    blk = gf // PAGE_SIZE
    off = gf % PAGE_SIZE

    hd = tl.arange(0, BLOCK_HD)
    head = hd // HEAD_DIM
    dim = hd % HEAD_DIM
    m = row_ok[:, None] & (hd < NUM_KV_HEADS * HEAD_DIM)[None, :]

    src = (
        blk[:, None] * BASE_S0
        + head[None, :] * BASE_S2
        + off[:, None] * BASE_S3
        + dim[None, :] * BASE_S4
    )
    k_vals = tl.load(base_ptr + src, mask=m, other=0.0)
    v_vals = tl.load(base_ptr + src + BASE_S1, mask=m, other=0.0)
    # out is [slots, head, dim] contiguous, so hd == head * HEAD_DIM + dim.
    dof = dst[:, None] * OUT_S0 + hd[None, :]
    tl.store(out_k_ptr + dof, k_vals.to(out_k_ptr.dtype.element_ty), mask=m)
    tl.store(out_v_ptr + dof, v_vals.to(out_v_ptr.dtype.element_ty), mask=m)


def _gather_paged_main_kv_to_scratch(
    base: torch.Tensor,
    gf: torch.Tensor,
    dst_full: torch.Tensor,
    scratch_k: torch.Tensor,
    scratch_v: torch.Tensor,
    page_size: int,
) -> None:
    """Fused paged->scratch gather+cast+scatter for the full active history."""
    n = int(dst_full.numel())
    if n == 0:
        return
    if base.dim() != 5:
        raise RuntimeError(
            f"fused KV gather needs a 5-D paged base, got {tuple(base.shape)}"
        )
    heads = int(scratch_k.shape[1])
    head_dim = int(scratch_k.shape[2])
    for name, t in (("scratch_k", scratch_k), ("scratch_v", scratch_v)):
        if t.stride(2) != 1 or t.stride(1) != head_dim:
            raise RuntimeError(
                f"fused KV gather needs a contiguous {name} [slots, head, dim], "
                f"got strides {tuple(t.stride())}"
            )
    block_t = int(os.environ.get("M3_MSA_KV_GATHER_BLOCK_T", "8"))
    _gather_paged_kv_to_scratch_kernel[(triton.cdiv(n, block_t),)](
        base,
        gf,
        dst_full,
        scratch_k,
        scratch_v,
        n,
        PAGE_SIZE=int(page_size),
        NUM_KV_HEADS=heads,
        HEAD_DIM=head_dim,
        BASE_S0=int(base.stride(0)),
        BASE_S1=int(base.stride(1)),
        BASE_S2=int(base.stride(2)),
        BASE_S3=int(base.stride(3)),
        BASE_S4=int(base.stride(4)),
        OUT_S0=int(scratch_k.stride(0)),
        BLOCK_T=block_t,
        BLOCK_HD=triton.next_power_of_2(heads * head_dim),
    )


@triton.jit
def _gather_flat_rows_kernel(
    src_ptr,
    gf_ptr,
    dst_ptr,
    out_ptr,
    N,
    ROW_DIM: tl.constexpr,
    SRC_S0: tl.constexpr,
    OUT_S0: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One pass: out[dst[t], :] = src[gf[t], :] over flat rows.

    The idx_K counterpart of _gather_paged_kv_to_scratch_kernel: the paged scale
    region and the idx scratch are both flat [row, idx_dim] and share a dtype,
    so this is a pure gather+scatter with no cast. Replaces
    ``idx_scratch[dst_full, 0] = scale_flat[gf]``, which torch lowers to a
    full-history ``index`` plus a full-history ``index_put`` per layer.
    """
    # int64 promotion before every scratch/pool multiply -- same reasoning as
    # _gather_paged_kv_to_scratch_kernel.
    pid = tl.program_id(0).to(tl.int64)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    t_ok = t < N
    gf = tl.load(gf_ptr + t, mask=t_ok, other=-1).to(tl.int64)
    dst = tl.load(dst_ptr + t, mask=t_ok, other=0).to(tl.int64)
    row_ok = t_ok & (gf >= 0)
    d = tl.arange(0, BLOCK_D)
    m = row_ok[:, None] & (d < ROW_DIM)[None, :]
    vals = tl.load(src_ptr + gf[:, None] * SRC_S0 + d[None, :], mask=m, other=0.0)
    tl.store(out_ptr + dst[:, None] * OUT_S0 + d[None, :], vals, mask=m)


def _gather_flat_rows(
    src: torch.Tensor,
    gf: torch.Tensor,
    dst_full: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Fused row gather+scatter: out[dst_full] = src[gf], for 2-D row tensors."""
    n = int(dst_full.numel())
    if n == 0:
        return
    row_dim = int(src.shape[1])
    if src.dim() != 2 or out.dim() != 2 or int(out.shape[1]) != row_dim:
        raise RuntimeError(
            f"fused row gather needs 2-D [row, dim] src/out with equal dim, got "
            f"{tuple(src.shape)} -> {tuple(out.shape)}"
        )
    if src.stride(1) != 1 or out.stride(1) != 1:
        raise RuntimeError(
            f"fused row gather needs row-contiguous src/out, got strides "
            f"{tuple(src.stride())} -> {tuple(out.stride())}"
        )
    block_t = int(os.environ.get("M3_MSA_IDX_GATHER_BLOCK_T", "8"))
    _gather_flat_rows_kernel[(triton.cdiv(n, block_t),)](
        src,
        gf,
        dst_full,
        out,
        n,
        ROW_DIM=row_dim,
        SRC_S0=int(src.stride(0)),
        OUT_S0=int(out.stride(0)),
        BLOCK_T=block_t,
        BLOCK_D=triton.next_power_of_2(row_dim),
    )


@triton.jit
def _fused_cp_paged_write_kernel(
    packed_ptr,
    unpad_ptr,
    write_slots_ptr,
    slot_mapping_ptr,
    scratch_k_ptr,
    scratch_v_ptr,
    scratch_idx_ptr,
    base_flat_ptr,
    scale_flat_ptr,
    kv_lens_ptr,
    TOKEN_COUNT,
    BATCH_SIZE,
    NK: tl.constexpr,
    NI: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCRATCH_SEQ_LEN: tl.constexpr,
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
        head_stride = PAGE_SIZE * HEAD_DIM
        kv_stride = NUM_KV_HEADS * head_stride
        paged_k = block_id * 2 * kv_stride + page_off * HEAD_DIM

        for h in tl.range(0, NUM_KV_HEADS):
            k = tl.load(packed_ptr + src_row + h * HEAD_DIM + d, mask=dmask, other=0.0)
            v = tl.load(
                packed_ptr + src_row + NK + h * HEAD_DIM + d,
                mask=dmask,
                other=0.0,
            )
            tl.store(scratch_k_ptr + dst_slot * NK + h * HEAD_DIM + d, k, mask=dmask)
            tl.store(scratch_v_ptr + dst_slot * NK + h * HEAD_DIM + d, v, mask=dmask)
            tl.store(
                base_flat_ptr + paged_k + h * head_stride + d,
                k,
                mask=dmask & valid,
            )
            tl.store(
                base_flat_ptr + paged_k + kv_stride + h * head_stride + d,
                v,
                mask=dmask & valid,
            )

        idx = tl.load(packed_ptr + src_row + 2 * NK + di, mask=imask, other=0.0)
        tl.store(scratch_idx_ptr + dst_slot * NI + di, idx, mask=imask)
        tl.store(scale_flat_ptr + safe_slot * NI + di, idx, mask=imask & valid)

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
        for h in tl.range(0, NUM_KV_HEADS):
            tl.store(
                scratch_k_ptr + scratch_row * NK + h * HEAD_DIM + d,
                0.0,
                mask=dmask & clear,
            )
            tl.store(
                scratch_v_ptr + scratch_row * NK + h * HEAD_DIM + d,
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
    scratch_k: torch.Tensor,
    scratch_v: torch.Tensor,
    idx_scratch: torch.Tensor,
    base: torch.Tensor,
    scale_flat: torch.Tensor,
    kv_lens: torch.Tensor,
    scratch_seq_len: int,
    nk: int,
    ni: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    token_count: Optional[int] = None,
) -> None:
    """Unpad CP all-gather output and write scratch plus scheduler paged caches."""
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
    grid_size = max(token_count, batch_size * page_size)
    _fused_cp_paged_write_kernel[(grid_size,)](
        packed,
        unpad_indices,
        write_slots,
        slot_mapping,
        scratch_k.reshape(-1, nk),
        scratch_v.reshape(-1, nk),
        idx_scratch.reshape(-1, ni),
        base.reshape(-1),
        scale_flat,
        kv_lens,
        token_count,
        batch_size,
        NK=nk,
        NI=ni,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        SCRATCH_SEQ_LEN=scratch_seq_len,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_I=triton.next_power_of_2(ni),
        num_warps=1,
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


class _MainKVScratch:
    """Process-wide shared, transient gather scratch for MSA main K/V.

    The persistent store is the standard cache-manager paged pool; the MSA
    Triton kernels still need the active K/V in flat
    token-slot layout, so each forward we gather the full active sequence out
    of the paged pool into this scratch. Sparse layers run strictly
    sequentially within one model forward (layer i finishes before layer i+1),
    so a single buffer grown on demand serves all sparse layers — the scratch
    footprint is 1x, not num_sparse_layers x.
    """

    def __init__(self) -> None:
        self._k: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None

    def acquire(
        self,
        slots: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if (
            self._k is None
            or self._k.shape[0] < slots
            or self._k.shape[1] != heads
            or self._k.shape[2] != dim
            or self._k.dtype != dtype
            or self._k.device != device
        ):
            self._k = torch.zeros(slots, heads, dim, dtype=dtype, device=device)
            self._v = torch.zeros_like(self._k)
        return self._k[:slots], self._v[:slots]


_MAIN_KV_SCRATCH = _MainKVScratch()


class _IdxKScratch:
    """Process-wide shared, transient gather scratch for MSA idx_K.

    Counterpart to ``_MainKVScratch`` for the index branch: the persistent
    store is the main paged pool's scale region (PD-transferable); the MSA
    Triton kernels still want idx_K in flat ``[slot, 1, idx_head_dim]`` layout,
    so each forward we gather the active sequence out of the scale region into
    this single buffer (one grown-on-demand buffer for all sparse layers).
    """

    def __init__(self) -> None:
        self._t: Optional[torch.Tensor] = None

    def acquire(
        self,
        slots: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
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
    """MXFP8 output-dim concat projection for MSA decode QKV + idx_Q + idx_K.

    This is intentionally narrow: it only supports the current MiniMax-M3
    decode business path where qkv_proj is CudaMxfp8Linear and idx_Q/idx_K are
    loaded in MXFP8 form. Normal BF16-dequantized idx weights use the
    original unfused F.linear fallback.
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
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
    ) -> torch.Tensor:
        return self.fused_linear(x_fp8, input_scales=x_scale)


class MSAAttention(nn.Module):
    """MiniMax-M3 sparse attention for a single sparse layer."""

    # Class-level workspace shared across all sparse layers (trtllm-gen needs
    # a 256 MB scratch buffer; one per device is enough). Lazily allocated on
    # the first prefill that takes the trtllm-gen fast path.
    _trtllm_workspace: Dict[torch.device, torch.Tensor] = {}
    _cp_shared_meta: Optional[Dict[str, Any]] = None
    _paged_decode_shared_meta: Optional[Dict[str, Any]] = None
    _target_verify_shared_meta: Optional[Dict[str, Any]] = None

    @classmethod
    def _get_trtllm_workspace(cls, device: torch.device):
        ws = cls._trtllm_workspace.get(device)
        if ws is None:
            ws = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            cls._trtllm_workspace[device] = ws
        return ws

    def _maybe_build_mxfp8_fused_qkv_idx_proj(self) -> None:
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
        fused_decode_ready = (
            self._mxfp8_fused_qkv_idx_proj is not None
            and self.qk_fuse_norm is not None
            and self.cos_sin_cache is not None
            and not self._rope_interleave
            and int(self.head_dim) == int(self.idx_head_dim)
            and int(self.rotary_dim) <= int(self.head_dim)
        )
        self._can_use_mxfp8_fused_qkv_idx_decode = fused_decode_ready

    def _should_use_mxfp8_fused_qkv_idx_decode(
        self,
        x_fp8: Optional[torch.Tensor],
        x_scale: Optional[torch.Tensor],
    ) -> bool:
        return (
            self._can_use_mxfp8_fused_qkv_idx_decode
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
        paged_idx_k_flat: torch.Tensor,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
    ):
        fused_qkv_idx = self._mxfp8_fused_qkv_idx_proj(x_fp8=x_fp8, x_scale=x_scale)
        if (
            paged_kv_base.dtype
            not in (
                fused_qkv_idx.dtype,
                torch.float8_e4m3fn,
            )
            or paged_idx_k_flat.dtype != fused_qkv_idx.dtype
        ):
            raise RuntimeError(
                "M3_MSA_RAW_IDX_MXFP8 fused paged decode requires BF16 or FP8 "
                "paged KV cache and a BF16 idx_K cache view. Disable "
                "M3_MSA_RAW_IDX_MXFP8 to use the BF16 idx fallback."
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
            paged_idx_k_flat,
            int(self.page_size),
            self.head_dim,
            self.rotary_dim,
            self.head_num,
            self.kv_head_num,
            self.num_idx_heads,
            self.layernorm_eps,
        )
        return q, idx_q

    def _maybe_trtllm_workspace(self, device: torch.device):
        """Workspace for the trtllm-gen MSA fast path, or None to force the Triton path.

        The trtllm-gen mega-kernel emits page ids as ``pid_h * num_pages + block_idx``
        with no per-block offset, i.e. it assumes the MSA side cache is a single
        physically-contiguous slice. Only the CP path (_build_compact_addressing)
        produces that layout; the non-CP path addresses the side cache through the
        scattered paged block table, which the kernel would misread -> corruption.
        So the trtllm fast path is only valid when CP is enabled. The
        M3_DISABLE_TRTLLM_GEN=1 escape hatch additionally forces the Triton path
        (e.g. on boxes whose flashinfer build lacks the M3 trtllm-gen cubin).
        """
        if not self.cp_enabled:
            return None
        if __import__("os").environ.get("M3_DISABLE_TRTLLM_GEN", "0") == "1":
            return None
        return MSAAttention._get_trtllm_workspace(device)

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
        # BF16 idx projections are always present for the original/F.linear
        # fallback. Optional raw MXFP8 copies feed only the fused decode matmul.
        self.idx_head_dim = int(sparse_config["idx_head_dim"])
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
        if not has_bf16_idx_w or raw_idx_key_count not in (0, 4):
            raise RuntimeError(
                "MSA idx weights must contain BF16 q/k weights and either all "
                "raw MXFP8 q/k weights+scales or none. Check M3_MSA_RAW_IDX_MXFP8."
            )

        self.idx_q_w: Optional[torch.Tensor] = None
        self.idx_k_w: Optional[torch.Tensor] = None
        self.idx_q_raw_w: Optional[torch.Tensor] = None
        self.idx_q_raw_s: Optional[torch.Tensor] = None
        self.idx_k_raw_w: Optional[torch.Tensor] = None
        self.idx_k_raw_s: Optional[torch.Tensor] = None
        self._has_raw_mxfp8_idx_weights = raw_idx_key_count == 4

        full_idx_q_w_for_heads = weights[W.msa_idx_q_w]
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

        # Paged-only store: the main K/V live in the standard cache-manager
        # paged pool (kv_cache_base) and idx_K in its scale region
        # (kv_scale_base, reinterpreted as BF16). Both are PD-transferable and
        # addressed by the same block table. The self-built persistent side
        # cache was removed; only a process-wide gather scratch is kept for the
        # MSA kernel.
        self._scratch_batch_size = 0
        self._scratch_seq_len = 0

        # Views into the process-wide shared gather scratch, refreshed by
        # the original scratch-backed prefill/decode helpers.
        self._scratch_k: Optional[torch.Tensor] = None
        self._scratch_v: Optional[torch.Tensor] = None
        self._scratch_idx_k: Optional[torch.Tensor] = None
        # Allocated kernel slot span (anchors scratch sizing).
        self._scratch_slots = 0
        self._paged_decode_static_ok: Optional[bool] = None

    def _paged_kv_base_view(self, kv_cache: LayerKVCache) -> Optional[torch.Tensor]:
        base = None if kv_cache is None else kv_cache.kv_cache_base
        if base is None or base.dim() != 2:
            return base
        from rtp_llm.models_py.modules.factory.attention.common import (
            reshape_paged_kv_cache,
        )

        return reshape_paged_kv_cache(
            base, self.kv_head_num, self.physical_page_size, self.head_dim
        )

    def _check_paged_decode_static(self, kv_cache: LayerKVCache) -> bool:
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

        bf16_elems_per_block = (
            int(scale.shape[1])
            * scale.element_size()
            // torch.empty((), dtype=torch.bfloat16).element_size()
        )
        return bf16_elems_per_block == int(self.page_size) * int(self.idx_head_dim)

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

    def _ensure_gather_scratch(
        self,
        kv_cache: LayerKVCache,
        device: torch.device,
        dtype: torch.dtype,
        bsz: Optional[int] = None,
        max_kv: Optional[int] = None,
        max_slot: Optional[int] = None,
    ) -> None:
        # Paged-only: main K/V and idx_K live in the cache-manager pool, so no
        # persistent side tensors are allocated here. This only tracks the
        # kernel slot span (and, under CP, the [bsz, seq_len] addressing grid)
        # that sizes the process-wide gather scratch acquired per forward.
        if self.cp_enabled:
            if bsz is None or max_kv is None:
                raise RuntimeError(
                    "CP MSA gather scratch requires batch size and kv length"
                )
            if (
                self._scratch_slots > 0
                and self._scratch_batch_size >= int(bsz)
                and self._scratch_seq_len >= int(max_kv)
            ):
                return
            target_bsz = max(int(bsz), self._scratch_batch_size, 1)
            requested_seq_len = max(int(max_kv), self._scratch_seq_len, 1)
            grow_granularity = max(int(self.page_size), 256)
            target_seq_len = (
                (requested_seq_len + grow_granularity - 1)
                // grow_granularity
                * grow_granularity
            )
            self._scratch_batch_size = target_bsz
            self._scratch_seq_len = target_seq_len
            self._scratch_slots = target_bsz * target_seq_len
            return

        if max_slot is None:
            raise RuntimeError("non-CP MSA gather scratch requires max active slot")
        requested_slots = max(int(max_slot) + 1, 1)
        if self._scratch_slots >= requested_slots:
            return
        grow_granularity = max(int(self.page_size), 256)
        self._scratch_slots = (
            (requested_slots + grow_granularity - 1) // grow_granularity
        ) * grow_granularity

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

    def _build_compact_addressing(
        self, attn_inputs: PyAttentionInputs, device: torch.device
    ):
        """CP path addressing over the compact per-request gather scratch."""
        if self._scratch_seq_len <= 0:
            raise RuntimeError("compact MSA gather scratch is not initialized")
        kv_lens, prefix, inlen = self._get_lengths(attn_inputs)
        bsz = int(kv_lens.numel())
        if (not attn_inputs.is_prefill) and self._cuda_graph_forward_active():
            max_kv = self._cuda_graph_max_kv(attn_inputs)
            pos = torch.arange(max_kv, device=device, dtype=torch.int32)
            row_offsets = (
                torch.arange(bsz, device=device, dtype=torch.int32)[:, None] * max_kv
            )
            req_to_token = row_offsets + pos[None, :]
            slot_ids = torch.arange(bsz, device=device, dtype=torch.int64)
            positions = prefix.to(device=device, dtype=torch.int32)
            batch_ids = torch.arange(bsz, device=device, dtype=torch.long)
            write_slots = req_to_token[
                batch_ids, prefix.to(device=device, dtype=torch.long)
            ].to(torch.int64)
            return (
                req_to_token,
                slot_ids,
                kv_lens,
                positions,
                write_slots,
                prefix,
                inlen,
            )

        max_kv = int(kv_lens.max().item())
        pos = torch.arange(max_kv, device=device, dtype=torch.int32)
        row_offsets = torch.arange(bsz, device=device, dtype=torch.int32)[
            :, None
        ] * int(self._scratch_seq_len)
        req_to_token = row_offsets + pos[None, :]
        slot_ids = torch.arange(bsz, device=device, dtype=torch.int64)

        prefix_cpu = prefix.detach().cpu().tolist()
        kv_cpu = kv_lens.detach().cpu().tolist()
        pos_parts = []
        slot_parts = []
        for b in range(bsz):
            p0, p1 = int(prefix_cpu[b]), int(kv_cpu[b])
            pos_parts.append(torch.arange(p0, p1, device=device, dtype=torch.int64))
            slot_parts.append(req_to_token[b, p0:p1])
        positions = torch.cat(pos_parts).to(torch.int32)
        write_slots = torch.cat(slot_parts).to(torch.int64)
        return req_to_token, slot_ids, kv_lens, positions, write_slots, prefix, inlen

    def _build_addressing(self, attn_inputs: PyAttentionInputs, device: torch.device):
        """Return (req_to_token [B, max_kv], slot_ids [B], kv_lens [B],
        positions [T], write_slots [T]) from rtp-llm block table + lengths."""
        block_table = attn_inputs.kv_cache_kernel_block_id_device  # [B, max_blocks]
        bsz = block_table.size(0)
        max_blocks = block_table.size(1)
        kv_lens, prefix, inlen = self._get_lengths(attn_inputs)

        max_kv = int(kv_lens.max().item())
        pos = torch.arange(max_kv, device=device, dtype=torch.int64)
        blk_idx = (pos // self.page_size).clamp(max=max_blocks - 1)
        blk_off = pos % self.page_size
        bt = block_table.index_select(1, blk_idx).to(torch.int64)  # [B, max_kv]
        req_to_token = (bt * self.page_size + blk_off[None, :]).to(torch.int32)
        slot_ids = torch.arange(bsz, device=device, dtype=torch.int64)

        # token order: per-request concat of new tokens [prefix[b], kv_len[b])
        prefix_cpu = prefix.tolist()
        kv_cpu = kv_lens.tolist()
        pos_parts = []
        slot_parts = []
        for b in range(bsz):
            p0, p1 = prefix_cpu[b], kv_cpu[b]
            pos_parts.append(torch.arange(p0, p1, device=device, dtype=torch.int64))
            slot_parts.append(req_to_token[b, p0:p1])
        positions = torch.cat(pos_parts).to(torch.int32)
        write_slots = torch.cat(slot_parts).to(torch.int64)
        return req_to_token, slot_ids, kv_lens, positions, write_slots, prefix, inlen

    @staticmethod
    def _max_active_slot(req_to_token: torch.Tensor, kv_lens: torch.Tensor) -> int:
        """Return the largest physical slot read by the sparse kernels."""
        max_slot = 0
        kv_lens_cpu = kv_lens.detach().cpu().to(torch.int64).tolist()
        for b, kv_len in enumerate(kv_lens_cpu):
            kv_len = int(kv_len)
            if kv_len <= 0:
                continue
            row_max = int(req_to_token[b, :kv_len].max().item())
            max_slot = max(max_slot, row_max)
        return max_slot

    # ------------------------------------------------------------------
    # Source main K/V from the standard cache-manager paged pool.
    # The paged pool (kv_cache.kv_cache_base) is the persistent, PD-transferable
    # store; the per-step gather scratch (_scratch_k / _scratch_v) is filled
    # from it and read by the MSA kernel (req_to_token unchanged).
    # ------------------------------------------------------------------
    def _paged_main_views(self, kv_cache: LayerKVCache):
        """Token-major [block, page, head, dim] views of the standard HND paged
        pool [block, 2, head, page, head_dim] for K and V (non-contiguous views;
        fine for advanced-index read/write)."""
        base = self._paged_kv_base_view(kv_cache)
        if base is None or base.dim() != 5:
            raise RuntimeError(
                "MSA paged main K/V requires a 5-D paged cache "
                "[block,2,head,page,dim], got "
                f"{None if base is None else tuple(base.shape)}"
            )
        kpv = base[:, 0].permute(0, 2, 1, 3)
        vpv = base[:, 1].permute(0, 2, 1, 3)
        return kpv, vpv

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

    def _zero_scratch_padding_tail(self, kv_lens: Any, bsz: int) -> None:
        """Zero each request's scratch slots between kv_len and its page end.

        Zigzag CP pads a request's prefill tokens up to a multiple of
        ``2 * cp_size`` and places the padding at the tail of the padded
        sequence, which is exactly the range rank 0's second segment covers.
        That segment reaches fmha as ``qo_offset + segment_len == padded_len``
        while ``kv_segment_lens`` carries the real ``prefix + input_len``, so the
        padded queries get a causal limit past the real KV length. Scratch is
        only sourced over ``[0, kv_len)`` and the scratch pools are reused across
        requests without clearing, so those slots would otherwise return the
        previous request's residual K/V and idx_K -- and a residual idx_K can win
        the top-k block selection outright. Padding stays below ``2 * cp_size``
        and ``page_size`` is a multiple of it, so the overflow never leaves the
        request's last page.
        """
        page = int(self.page_size)
        seq_len = int(self._scratch_seq_len)
        if page <= 0 or seq_len <= 0:
            return
        for b in range(int(bsz)):
            kv_len = int(kv_lens[b])
            tail = (-kv_len) % page
            if tail == 0:
                continue
            lo = b * seq_len + kv_len
            hi = min(lo + tail, (b + 1) * seq_len)
            if hi <= lo:
                continue
            for scratch in (self._scratch_k, self._scratch_v, self._scratch_idx_k):
                if scratch is not None and hi <= scratch.shape[0]:
                    scratch[lo:hi] = 0

    def _source_cp_from_packed(
        self,
        kv_cache: LayerKVCache,
        packed: torch.Tensor,
        unpad_indices: torch.Tensor,
        write_slots: torch.Tensor,
        slot_mapping: torch.Tensor,
        device: torch.device,
        kv_lens: torch.Tensor,
        nk: int,
        ni: int,
        token_count: int,
    ) -> None:
        """Persist CP packed all-gather output and fill MSA scratch in one kernel.

        This is the cold/sharded v2 fast path: gathered K/V/idx_K already cover
        the active sequence, so the kernel can unpad directly into both the
        transient MSA scratch and scheduler-provided paged cache. It keeps the
        paged store contract and avoids the side-cache fallback.
        """
        base = self._paged_kv_base_view(kv_cache)
        if base is None or base.dim() != 5:
            raise RuntimeError(
                "MSA paged main K/V requires a 5-D paged cache "
                "[block,2,head,page,dim], got "
                f"{None if base is None else tuple(base.shape)}"
            )
        if base.dtype != packed.dtype and base.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"MSA paged main K/V dtype mismatch: paged={base.dtype} vs "
                f"act={packed.dtype}; expected a match or float8_e4m3fn (fp8 KV)"
            )
        idx_view = self._idx_k_paged_view(kv_cache)
        if idx_view.dtype != packed.dtype:
            raise RuntimeError(
                f"MSA paged idx_K dtype mismatch: paged={idx_view.dtype} vs "
                f"act={packed.dtype} (scale region is reinterpreted as bf16)"
            )

        scratch_slots = int(self._scratch_slots)
        scratch_k, scratch_v = _MAIN_KV_SCRATCH.acquire(
            scratch_slots, self.kv_head_num, self.head_dim, packed.dtype, device
        )
        idx_scratch = _IDX_K_SCRATCH.acquire(
            scratch_slots, 1, self.idx_head_dim, packed.dtype, device
        )
        _fused_cp_paged_write(
            packed,
            unpad_indices,
            write_slots,
            slot_mapping,
            scratch_k,
            scratch_v,
            idx_scratch,
            base,
            idx_view.reshape(-1, self.idx_head_dim),
            kv_lens,
            int(self._scratch_seq_len),
            nk,
            ni,
            self.kv_head_num,
            self.head_dim,
            self.page_size,
            token_count=token_count,
        )
        self._scratch_k = scratch_k
        self._scratch_v = scratch_v
        self._scratch_idx_k = idx_scratch

    def _source_cp_from_gathered(
        self,
        kv_cache: LayerKVCache,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_k: torch.Tensor,
        write_slots: torch.Tensor,
        slot_mapping: torch.Tensor,
        device: torch.device,
        token_count: Optional[int] = None,
    ) -> None:
        """Persist CP gathered tensors and fill MSA scratch.

        Cold CP prefill already has full-sequence K/V/idx_K after all_gather,
        so scratch can be filled directly without reading back from paged cache.
        The fused Triton scatter also replaces PyTorch advanced-index writes
        and the slot_mapping >= 0 mask/nonzero path for idx_K persistence.
        """
        base = self._paged_kv_base_view(kv_cache)
        if base is None or base.dim() != 5:
            raise RuntimeError(
                "MSA paged main K/V requires a 5-D paged cache "
                "[block,2,head,page,dim], got "
                f"{None if base is None else tuple(base.shape)}"
            )
        if base.dtype != k.dtype and base.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"MSA paged main K/V dtype mismatch: paged={base.dtype} vs "
                f"act={k.dtype}; expected a match or float8_e4m3fn (fp8 KV)"
            )
        idx_view = self._idx_k_paged_view(kv_cache)
        if idx_view.dtype != idx_k.dtype:
            raise RuntimeError(
                f"MSA paged idx_K dtype mismatch: paged={idx_view.dtype} vs "
                f"act={idx_k.dtype} (scale region is reinterpreted as bf16)"
            )

        # FP8-aware paged write (Triton scatter casts bf16 -> e4m3 for an fp8 pool;
        # the tuned C++ writer is used only when dtypes match).
        _write_main_kv_to_paged(k, v, base, slot_mapping)

        scratch_slots = int(self._scratch_slots)
        scratch_k, scratch_v = _MAIN_KV_SCRATCH.acquire(
            scratch_slots, self.kv_head_num, self.head_dim, k.dtype, device
        )
        idx_scratch = _IDX_K_SCRATCH.acquire(
            scratch_slots, 1, self.idx_head_dim, idx_k.dtype, device
        )
        _fused_scatter_cp_gathered(
            k,
            v,
            idx_k,
            write_slots,
            slot_mapping,
            scratch_k,
            scratch_v,
            idx_scratch,
            idx_view.reshape(-1, self.idx_head_dim),
            self.kv_head_num * self.head_dim,
            self.idx_head_dim,
            token_count=token_count,
        )
        self._scratch_k = scratch_k
        self._scratch_v = scratch_v
        self._scratch_idx_k = idx_scratch

    def _full_history_slots(
        self,
        req_to_token: torch.Tensor,
        kv_lens: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        device: torch.device,
        graph_decode: bool,
    ):
        """Scratch/paged slot addressing for every token of the active history.

        Returns ``(dst_full, gf)``: the flat scratch rows and their physical
        paged slots.

        These depend only on per-batch metadata (``req_to_token``, ``kv_lens``,
        the physical block table), not on the layer, but both the main-K/V and
        the idx_K sourcing path needed them — so they were rebuilt twice per
        sparse layer, i.e. ~2x num_sparse_layers times per forward. Each rebuild
        is a chain of bs*max_kv-sized int64 ops plus a ``_kernel_slots_to_paged``
        call, and that call resolves the block table through
        ``layer_to_group[...].item()`` — a device sync. So memoize.

        The memo lives inside the shared CP ``addr`` dict, which is replaced
        wholesale whenever the first sparse layer rebuilds metadata, so a cached
        entry can never outlive its batch. Reuse additionally requires the very
        same ``req_to_token``/``kv_lens`` objects, which covers the call sites
        that run without CP metadata. Like the ``slot_mapping`` entry already in
        that dict, this assumes all sparse layers resolve the same physical block
        table (they share one KV cache group).
        """
        meta = MSAAttention._cp_shared_meta
        addr = meta.get("addr") if isinstance(meta, dict) else None
        key = (int(self._scratch_seq_len), int(self.page_size), bool(graph_decode))
        if (
            addr is not None
            and addr.get("hist_key") == key
            and addr.get("hist_rtt") is req_to_token
            and addr.get("hist_lens") is kv_lens
        ):
            return addr["hist_dst_full"], addr["hist_gf"]

        if graph_decode:
            dst_full = req_to_token.reshape(-1).to(torch.int64)
        else:
            max_kv = req_to_token.shape[1]
            ar = torch.arange(max_kv, device=device, dtype=torch.int64)
            mask = ar[None, :] < kv_lens.to(device=device, dtype=torch.int64)[:, None]
            dst_full = req_to_token.to(torch.int64)[mask]
        gf = self._kernel_slots_to_paged(dst_full, attn_inputs)

        if addr is not None:
            addr["hist_key"] = key
            addr["hist_rtt"] = req_to_token
            addr["hist_lens"] = kv_lens
            addr["hist_dst_full"] = dst_full
            addr["hist_gf"] = gf
        return dst_full, gf

    def _source_main_kv_from_paged(
        self,
        kv_cache: LayerKVCache,
        k: torch.Tensor,
        v: torch.Tensor,
        write_slots: torch.Tensor,
        req_to_token: torch.Tensor,
        kv_lens: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        device: torch.device,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> None:
        """Persist K/V into the standard paged pool (PD-store) and build the
        transient gather scratch the MSA kernel reads.

        The paged write goes through the sharding-aware C++ writer
        ``mha_kv_write_cache`` with a physical-block-table slot mapping, so it is
        correct under CP page-RR sharding (each rank stores only its 1/cp_size of
        the tokens; non-owned tokens get a -1 slot and are skipped).

        Scratch sourcing depends on sharding:
        * sharded CP prefill fills scratch directly from the already-all-gathered
          full sequence (``k``/``v`` are the full sequence in CP prefill).
        * not sharded (non-CP prefill, or the original decode path): the full
          active history is read back from the persistent paged pool."""
        base = self._paged_kv_base_view(kv_cache)
        if base is None or base.dim() != 5:
            raise RuntimeError(
                "MSA paged main K/V requires a 5-D paged cache "
                "[block,2,head,page,dim], got "
                f"{None if base is None else tuple(base.shape)}"
            )
        if base.dtype != k.dtype and base.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"MSA paged main K/V dtype mismatch: paged={base.dtype} vs "
                f"act={k.dtype}; expected a match or float8_e4m3fn (fp8 KV)"
            )

        # 1) persist into the paged pool via the sharding-aware writer (C++ casts
        # bf16 activations to e4m3 when the pool is float8_e4m3fn).
        if slot_mapping is None:
            slot_mapping = self._kernel_slots_to_paged(write_slots, attn_inputs)
        _write_main_kv_to_paged(k, v, base, slot_mapping)

        # 2) build the transient gather scratch the prefill MSA kernel reads.
        scratch_slots = int(self._scratch_slots)
        scratch_k, scratch_v = _MAIN_KV_SCRATCH.acquire(
            scratch_slots, self.kv_head_num, self.head_dim, k.dtype, device
        )
        if self._kv_sharded:
            scratch_k[write_slots] = k
            scratch_v[write_slots] = v
        else:
            graph_decode = (
                not attn_inputs.is_prefill
            ) and self._cuda_graph_forward_active()
            dst_full, gf = self._full_history_slots(
                req_to_token, kv_lens, attn_inputs, device, graph_decode
            )
            # The paged pool holds e4m3 for fp8 KV; upconvert to the bf16 gather
            # scratch the MSA step-3 kernel reads. One fused kernel does
            # gather+cast+scatter in a single pass; the torch form below needs
            # three full passes over the whole history per K and per V per layer.
            if _FUSED_KV_GATHER:
                _gather_paged_main_kv_to_scratch(
                    base, gf, dst_full, scratch_k, scratch_v, self.page_size
                )
            else:
                kpv, vpv = self._paged_main_views(kv_cache)
                p = self.page_size
                gf_blk, gf_off = gf // p, gf % p
                scratch_k[dst_full] = kpv[gf_blk, gf_off].to(scratch_k.dtype)
                scratch_v[dst_full] = vpv[gf_blk, gf_off].to(scratch_v.dtype)
        self._scratch_k = scratch_k
        self._scratch_v = scratch_v

    # ------------------------------------------------------------------
    # Task-2: source idx_K from the main paged pool's scale region.
    # The C++ cache manager sizes the MHA scale region (kv_scale_base) to hold
    # one BF16 idx_K per token (indexer_head_dim). It is exposed to Python as
    # FP32; we reinterpret it as BF16 and view it as [block, page, idx_head_dim]
    # so idx_K is addressed by the same block table as the main K/V and travels
    # with it under PD separation.
    # ------------------------------------------------------------------
    def _idx_k_paged_view(self, kv_cache: LayerKVCache) -> torch.Tensor:
        """[block, page, idx_head_dim] BF16 view of the FP32 scale region."""
        scale = kv_cache.kv_scale_base
        if scale is None or scale.dim() != 2:
            raise RuntimeError(
                "MSA paged idx_K requires a 2-D kv_scale_base "
                "[block, scale_elems]; got "
                f"{None if scale is None else tuple(scale.shape)}. Launch with "
                "M3_IDX_PAGED=1 and a th_transformer built with the M3 MHA "
                "indexer scale sizing (indexer_head_dim set)."
            )
        blk = int(scale.shape[0])
        # FP32 storage reinterpreted as BF16: scale_elems fp32 -> 2*scale_elems
        # bf16 == page_size * idx_head_dim.
        sb = scale.view(torch.bfloat16)
        expect = self.page_size * self.idx_head_dim
        if int(sb.shape[1]) != expect:
            raise RuntimeError(
                f"MSA idx_K scale region mismatch: bf16 elems/block={int(sb.shape[1])} "
                f"!= page_size*idx_head_dim={expect} (page={self.page_size}, "
                f"idx_head_dim={self.idx_head_dim}); check C++ kv_scale_stride_bytes"
            )
        return sb.view(blk, self.page_size, self.idx_head_dim)

    def _source_idx_k_from_paged(
        self,
        kv_cache: LayerKVCache,
        idx_k: torch.Tensor,
        write_slots: torch.Tensor,
        req_to_token: torch.Tensor,
        kv_lens: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        device: torch.device,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> None:
        """Persist idx_K into the paged scale region (PD-store) and build the
        shared idx scratch the MSA kernel reads.

        Uses the same sharding-aware physical slot mapping as the main K/V: the
        scale region is a ``[block, page, idx_head_dim]`` view, so a token's
        flat row is exactly its physical slot. Non-owned tokens (slot == -1)
        are skipped. The kernel scratch is filled directly from the
        all-gathered ``idx_k`` (no paged read-back)."""
        idx_view = self._idx_k_paged_view(kv_cache)  # [block, page, idx_head_dim]
        if idx_view.dtype != idx_k.dtype:
            raise RuntimeError(
                f"MSA paged idx_K dtype mismatch: paged={idx_view.dtype} vs "
                f"act={idx_k.dtype} (scale region is reinterpreted as bf16)"
            )
        p = self.page_size
        idx_flat = idx_k.reshape(-1, self.idx_head_dim)
        scale_flat = idx_view.reshape(-1, self.idx_head_dim)  # [block*page, idx_dim]

        # 1) persist into the scale region at physical slots (skip -1 non-owned).
        #    The scale region is [block, page, idx_dim]; a token's flat row is
        #    exactly its physical slot (block*page + off), same mapping as K/V.
        if slot_mapping is None:
            slot_mapping = self._kernel_slots_to_paged(write_slots, attn_inputs)
        graph_decode = (
            not attn_inputs.is_prefill
        ) and self._cuda_graph_forward_active()
        if graph_decode and not self._kv_sharded:
            scale_flat[slot_mapping] = idx_flat
        else:
            valid = slot_mapping >= 0
            scale_flat[slot_mapping[valid]] = idx_flat[valid]

        # 2) build the transient scratch the MSA kernel reads.
        scratch_slots = int(self._scratch_slots)
        idx_scratch = _IDX_K_SCRATCH.acquire(
            scratch_slots, 1, self.idx_head_dim, idx_k.dtype, device
        )
        if self._kv_sharded:
            idx_scratch[write_slots, 0] = idx_flat
        else:
            dst_full, gf = self._full_history_slots(
                req_to_token, kv_lens, attn_inputs, device, graph_decode
            )
            if _FUSED_KV_GATHER:
                _gather_flat_rows(
                    scale_flat,
                    gf,
                    dst_full,
                    idx_scratch.view(-1, self.idx_head_dim),
                )
            else:
                idx_scratch[dst_full, 0] = scale_flat[gf]
        self._scratch_idx_k = idx_scratch

    def _restore_cp_sharded_prefix_scratch(
        self,
        kv_cache: LayerKVCache,
        prefix_lengths: Any,
        req_to_token: torch.Tensor,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        """Restore cached prefix K/V/idx_K from page-RR owner ranks.

        The current-step CP all-gather contains only the suffix, while the MSA
        kernels consume ``prefix + suffix`` from flat scratch. Under physical
        page-RR sharding no rank can reconstruct that prefix from its local
        pool alone, so gather owned pages and interleave them into logical page
        order before filling each request's prefix scratch rows.
        """
        if not self._kv_sharded:
            return
        if isinstance(prefix_lengths, torch.Tensor):
            prefix_cpu = prefix_lengths.detach().cpu().to(torch.int64)
        else:
            prefix_cpu = torch.tensor(list(prefix_lengths), dtype=torch.int64)
        if not bool((prefix_cpu > 0).any().item()):
            return
        if self._scratch_k is None or self._scratch_idx_k is None:
            raise RuntimeError("MSA sharded prefix restore requires allocated scratch")

        block_table = self._physical_block_table(attn_inputs)
        main_pages = gather_cp_sharded_prefix_pool(
            self._paged_kv_base_view(kv_cache),
            block_table,
            prefix_cpu,
            page_size=self.page_size,
            cp_size=self._cp_size,
            cp_rank=self._cp_rank,
            debug_label="msa-main-kv",
        )
        idx_pages = gather_cp_sharded_prefix_pool(
            self._idx_k_paged_view(kv_cache),
            block_table,
            prefix_cpu,
            page_size=self.page_size,
            cp_size=self._cp_size,
            cp_rank=self._cp_rank,
            debug_label="msa-idx-k",
        )

        # Main pool is HND: [page,2,head,token,dim]. Convert to token-major
        # logical history; idx_K is already [page,token,dim].
        prefix_k = (
            main_pages[:, 0]
            .permute(0, 2, 1, 3)
            .reshape(-1, self.kv_head_num, self.head_dim)
        )
        prefix_v = (
            main_pages[:, 1]
            .permute(0, 2, 1, 3)
            .reshape(-1, self.kv_head_num, self.head_dim)
        )
        prefix_idx = idx_pages.reshape(-1, self.idx_head_dim)

        token_offset = 0
        for batch_idx, prefix_len in enumerate(prefix_cpu.tolist()):
            prefix_len = int(prefix_len)
            if prefix_len == 0:
                continue
            dst = req_to_token[batch_idx, :prefix_len].to(torch.long)
            src = slice(token_offset, token_offset + prefix_len)
            self._scratch_k[dst] = prefix_k[src].to(self._scratch_k.dtype)
            self._scratch_v[dst] = prefix_v[src].to(self._scratch_v.dtype)
            self._scratch_idx_k[dst, 0] = prefix_idx[src].to(self._scratch_idx_k.dtype)
            token_offset += prefix_len

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
        # base may be an FP8 (e4m3) pool; _write_decode_kv_idx_to_paged casts the
        # bf16 K/V to e4m3 on store, and the paged decode kernel upconverts on read.
        if (
            base is None
            or scale is None
            or (base.dtype != k.dtype and base.dtype != torch.float8_e4m3fn)
        ):
            return None

        idx_view = scale.view(torch.bfloat16).view(
            int(scale.shape[0]), int(self.page_size), int(self.idx_head_dim)
        )
        if idx_view.dtype != idx_k.dtype:
            return None

        _write_decode_kv_idx_to_paged(
            k.contiguous(),
            v.contiguous(),
            idx_k.reshape(-1, self.idx_head_dim).contiguous(),
            seq_lens,
            phys_block_table,
            base,
            idx_view.reshape(-1, self.idx_head_dim),
            int(self.page_size),
            int(self.idx_head_dim),
        )

        return base[:, 0], base[:, 1], phys_block_table, idx_view

    # ------------------------------------------------------------------
    def _forward_cp_prefill(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: LayerKVCache,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CP-aware prefill: local Q attends to all-gathered full-sequence KV.

        When ``x_fp8`` / ``x_scale`` are supplied (the upstream fused
        norm+quant path in GenericMoeDecoderLayer), feed them straight into
        ``qkv_proj`` to skip the per-token-group quant that the projection
        would otherwise run on its bf16 input. ``hidden_states`` still drives
        the index-branch F.linear paths (which are bf16 GEMMs).
        """
        from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
            minimax_sparse_prefill,
        )

        cp_info = attn_inputs.context_parallel_info
        device = hidden_states.device
        local_tokens = hidden_states.shape[0]

        if x_fp8 is not None and x_scale is not None:
            qkv = self.qkv_proj(x_fp8, input_scales=x_scale)
        else:
            qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(local_tokens, self.head_num, self.head_dim)
        k = k.reshape(local_tokens, self.kv_head_num, self.head_dim)
        v = v.reshape(local_tokens, self.kv_head_num, self.head_dim)

        idx_q = F.linear(hidden_states, self.idx_q_w)
        idx_k = F.linear(hidden_states, self.idx_k_w)
        idx_q = idx_q.reshape(local_tokens, self.num_idx_heads, self.idx_head_dim)
        idx_k = idx_k.reshape(local_tokens, 1, self.idx_head_dim)
        idx_q = _gemma_rmsnorm_per_head(idx_q, self.idx_q_norm_w, self.layernorm_eps)
        idx_k = _gemma_rmsnorm_per_head(idx_k, self.idx_k_norm_w, self.layernorm_eps)

        # Coalesce the two small per-layer D2H transfers (chunk_lengths +
        # prefix_lengths) into one packed copy so we pay a single
        # cudaStreamSynchronize instead of two. The tiny H2D for chunk_lengths
        # when it originates on CPU is negligible (<1us pinned->device).
        _chunk_dev = cp_info.prefill_cp_chunk_lengths.detach().to(
            device=device, dtype=torch.int64
        )
        _prefix_dev = attn_inputs.prefix_lengths.detach().to(
            device=device, dtype=torch.int64
        )
        _packed_cpu = torch.cat([_chunk_dev, _prefix_dev]).cpu()
        _n_chunks = _chunk_dev.numel()
        chunk_lengths_cpu = _packed_cpu[:_n_chunks].tolist()
        prefix_cpu = _packed_cpu[_n_chunks:]
        prefix_cpu_list = prefix_cpu.tolist()
        if sum(int(x) for x in chunk_lengths_cpu) != local_tokens:
            raise RuntimeError(
                "MSA CP prefill expects rank-local token count to match "
                "prefill_cp_chunk_lengths; got "
                f"local_tokens={local_tokens}, chunks={chunk_lengths_cpu}"
            )
        # Zigzag splits each chunk into two equal halves; an odd chunk would make
        # ``chunk // 2 * 2`` drop a token, leaving its output row unwritten.
        if any(int(x) % 2 != 0 for x in chunk_lengths_cpu):
            raise RuntimeError(
                "MSA CP prefill requires even per-request chunk lengths for "
                f"zigzag CP; got chunks={chunk_lengths_cpu}"
            )

        # Bring shuffle_indices back to CPU once; we compute local_positions
        # entirely on CPU with numpy (cheap, all inputs are small ints) and
        # do a single H2D, eliminating the per-batch GPU op chain
        # (clamp + add + cat + cast = 2*bsz + 2 launches → 1 H2D).
        shuffle_cpu = (
            cp_info.prefill_shuffle_indices.detach().cpu().to(torch.int64).tolist()
        )

        # Vectorized CPU build of local_positions.
        # positions[i] = max(shuffle[i], 0) + prefix[batch_id[i]]
        # where batch_id[i] = which chunk the i-th token belongs to.
        chunk_arr = np.asarray(chunk_lengths_cpu, dtype=np.int64)
        prefix_arr = np.asarray(prefix_cpu_list, dtype=np.int64)
        shuffle_arr = np.asarray(shuffle_cpu, dtype=np.int64)
        bsz_py = chunk_arr.shape[0]
        batch_id = np.repeat(np.arange(bsz_py, dtype=np.int64), chunk_arr)
        positions_np = (np.maximum(shuffle_arr, 0) + prefix_arr[batch_id]).astype(
            np.int32, copy=False
        )
        # Force a fresh contiguous device tensor (single async H2D).
        local_positions = torch.from_numpy(np.ascontiguousarray(positions_np)).to(
            device=device, non_blocking=True
        )

        # Segment metadata: pure-Python list construction (no GPU op here).
        segment_lengths = []
        segment_starts = []
        segment_req_ids = []
        cursor = 0
        for b in range(bsz_py):
            chunk_len = int(chunk_arr[b])
            pair_len = chunk_len // 2
            req_prefix = int(prefix_arr[b])
            for rel_start in (0, pair_len):
                segment_lengths.append(pair_len)
                segment_req_ids.append(b)
                if pair_len > 0:
                    start_pos = int(shuffle_cpu[cursor + rel_start])
                    segment_starts.append(req_prefix + max(start_pos, 0))
                else:
                    segment_starts.append(req_prefix)
            cursor += chunk_len

        q = _rows_to_contig(q)
        k = k.contiguous()
        self._apply_rope(q, k, local_positions)
        idx_q = idx_q.contiguous()
        idx_k = idx_k.contiguous()
        self._apply_rope(idx_q, idx_k, local_positions)

        # Pack K, V, idx_k along the last dim and issue ONE all_gather
        # instead of three. NCCL launch + small-message latency dominates
        # over per-byte bandwidth here, so 3→1 saves ~75–130us / layer.
        # k / idx_k were already made contiguous above by the RoPE block;
        # v is fresh out of torch.split and needs realising once.
        v = v.contiguous()
        nk = self.kv_head_num * self.head_dim
        ni = self.idx_head_dim
        packed_kv = torch.cat(
            [
                k.reshape(local_tokens, nk),
                v.reshape(local_tokens, nk),
                idx_k.reshape(local_tokens, ni),
            ],
            dim=-1,
        )  # [local_tokens, 2*nk + ni], contiguous
        all_packed = all_gather(packed_kv, group=Group.TP)
        restore_indices = cp_info.prefill_qkv_restore_indice
        padding_mask = cp_info.prefill_qkv_padding_mask
        unpad_indices = restore_indices[padding_mask == 1].to(torch.long)
        # Fused unpad+split: one token-parallel kernel gathers all_packed[unpad_indices]
        # and splits the K/V/idx_K column ranges into contiguous outputs -- replaces the
        # reshape(x3) + fancy-index(x3) copy chain (~2x faster, bit-identical).
        full_T = int(unpad_indices.numel())
        full_k = torch.empty(
            full_T,
            self.kv_head_num,
            self.head_dim,
            dtype=all_packed.dtype,
            device=device,
        )
        full_v = torch.empty(
            full_T,
            self.kv_head_num,
            self.head_dim,
            dtype=all_packed.dtype,
            device=device,
        )
        full_idx_k = torch.empty(
            full_T, 1, self.idx_head_dim, dtype=all_packed.dtype, device=device
        )
        _fused_unpack_packed_cp(
            all_packed, unpad_indices, full_k, full_v, full_idx_k, nk, ni, full_T
        )

        full_input_lengths_cpu = cp_info.prefill_actual_input_lengths_cpu.to(
            torch.int64
        )
        kv_lens_cpu = prefix_cpu + full_input_lengths_cpu

        bsz = int(kv_lens_cpu.numel())
        max_kv = int(kv_lens_cpu.max().item())
        self._ensure_gather_scratch(
            kv_cache, device, full_k.dtype, bsz=bsz, max_kv=max_kv
        )

        pos_range = torch.arange(max_kv, device=device, dtype=torch.int32)
        cache_row_offsets = torch.arange(bsz, device=device, dtype=torch.int32)[
            :, None
        ] * int(self._scratch_seq_len)
        req_to_token = cache_row_offsets + pos_range[None, :]

        slot_parts = []
        for b in range(bsz):
            p0, p1 = int(prefix_cpu_list[b]), int(kv_lens_cpu[b].item())
            slot_parts.append(req_to_token[b, p0:p1])
        write_slots = torch.cat(slot_parts).to(torch.int64)

        # Persist into the cache-manager pool: idx_K -> scale region,
        # main K/V -> paged pool. Both are PD-transferable.
        self._source_idx_k_from_paged(
            kv_cache,
            full_idx_k,
            write_slots,
            req_to_token,
            kv_lens_cpu,
            attn_inputs,
            device,
        )
        self._source_main_kv_from_paged(
            kv_cache,
            full_k,
            full_v,
            write_slots,
            req_to_token,
            kv_lens_cpu,
            attn_inputs,
            device,
        )
        self._restore_cp_sharded_prefix_scratch(
            kv_cache, prefix_cpu, req_to_token, attn_inputs
        )
        self._zero_scratch_padding_tail(kv_lens_cpu, bsz)

        # PD separation: register this MSA layer's paged K/V (and idx_K on the
        # scale region) with the cache store, exactly like the non-CP prefill
        # path. cache_store_inputs already carries the prefill_cp_size / tp_rank
        # so the C++ writer stores only this rank's 1/cp_size page-RR shard.
        # Without this the decode side waits forever for the missing MSA blocks.
        if (
            kv_cache is not None
            and attn_inputs.is_prefill
            and attn_inputs.cache_store_inputs
        ):
            from rtp_llm.models_py.modules.factory.attention import (
                common as _attn_common,
            )

            _write_impl = _attn_common.create_write_cache_store_impl(attn_inputs)
            _attn_common.apply_write_cache_store(_write_impl, attn_inputs, kv_cache)

        # Pack three small CPU lists (segment_req_ids/segment_lengths/
        # segment_starts) into ONE contiguous int64 buffer and do a single
        # pinned H2D, then split + cast on-device. Replaces 3 separate
        # torch.tensor(list, device=cuda) calls (each a pageable H2D).
        n_seg = len(segment_lengths)
        _packed_np = np.concatenate(
            [
                np.asarray(segment_req_ids, dtype=np.int64),
                np.asarray(segment_lengths, dtype=np.int64),
                np.asarray(segment_starts, dtype=np.int64),
            ]
        )
        _packed_dev = torch.from_numpy(_packed_np).to(device=device, non_blocking=True)
        # dim-0 slices of a 1-D tensor are always contiguous; .contiguous()
        # afterwards is a no-op but defensive in case allocator returns a view.
        segment_req_ids_t = _packed_dev[:n_seg].contiguous()  # int64
        segment_lengths_t = _packed_dev[n_seg : 2 * n_seg].to(
            torch.int32
        )  # fresh contiguous
        prefix_i32 = _packed_dev[2 * n_seg :].to(torch.int32)  # fresh contiguous

        req_to_token_segments = req_to_token.index_select(
            0, segment_req_ids_t
        ).contiguous()
        slot_ids = torch.arange(n_seg, device=device, dtype=torch.int64)
        cu_seqlens = torch.zeros(n_seg + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(segment_lengths_t, dim=0)
        kv_lens_device = kv_lens_cpu.to(device=device, dtype=torch.int32)
        seq_lens_i32 = kv_lens_device.index_select(0, segment_req_ids_t)
        max_seqlen_q = max(int(x) for x in segment_lengths)
        max_seqlen_k = int(kv_lens_cpu.max().item())

        # Q is already in rank-local zigzag order. The Triton/trtllm-gen
        # kernel stores O by cu_seqlens offsets, so no output restore /
        # all-gather is needed here.
        main_k = self._scratch_k
        main_v = self._scratch_v
        idx_kc = self._scratch_idx_k
        _idx_o, o = minimax_sparse_prefill(
            q=q,
            k_cache=main_k,
            v_cache=main_v,
            sink=None,
            idx_q=idx_q,
            idx_k_cache=idx_kc,
            idx_v_cache=None,
            idx_sink=None,
            req_to_token=req_to_token_segments,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens_i32,
            prefix_lens=prefix_i32,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size_q=1,
            block_size_k=self.block_size,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            disable_index_value=self.disable_index_value,
            workspace=self._maybe_trtllm_workspace(device),
        )

        output = self.o_proj(o.reshape(local_tokens, -1).contiguous())
        return output

    def _forward_cp_prefill_v2(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: LayerKVCache,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optimized CP prefill for the paged-cache MSA path.

        Ports the safe parts of the old side-cache v3 path while keeping the
        new storage contract: K/V and idx_K are persisted through the
        scheduler-provided paged cache. The optimizations are cross-layer CP
        metadata reuse, early D2H for CP metadata, fused K-RoPE+pack, and
        Q/idx_q RoPE overlap with packed all_gather.
        """
        from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
            m3_fmha_prefill_enabled,
            minimax_sparse_prefill,
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
            kv_lens_cpu = cache["kv_lens_cpu"]
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
            need_meta = False
        else:
            chunk_dev = cp_info.prefill_cp_chunk_lengths.detach().to(
                device=device, dtype=torch.int64
            )
            prefix_dev = attn_inputs.prefix_lengths.detach().to(
                device=device, dtype=torch.int64
            )
            n_chunks = chunk_dev.numel()
            packed_pinned = torch.empty(
                n_chunks + prefix_dev.numel(), dtype=torch.int64, pin_memory=True
            )
            packed_pinned[:n_chunks].copy_(chunk_dev, non_blocking=True)
            packed_pinned[n_chunks:].copy_(prefix_dev, non_blocking=True)
            shuffle_pinned = torch.empty(
                cp_info.prefill_shuffle_indices.numel(),
                dtype=torch.int64,
                pin_memory=True,
            )
            shuffle_pinned.copy_(
                cp_info.prefill_shuffle_indices.detach().to(torch.int64),
                non_blocking=True,
            )
            need_meta = True

        if x_fp8 is not None and x_scale is not None:
            qkv = self.qkv_proj(x_fp8, input_scales=x_scale)
        else:
            qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        q = qkv[:, : self.q_size].reshape(local_tokens, self.head_num, self.head_dim)

        idx_q = F.linear(hidden_states, self.idx_q_w)
        idx_k = F.linear(hidden_states, self.idx_k_w)
        idx_q = idx_q.reshape(local_tokens, self.num_idx_heads, self.idx_head_dim)
        idx_k = idx_k.reshape(local_tokens, 1, self.idx_head_dim)
        idx_q = _gemma_rmsnorm_per_head(idx_q, self.idx_q_norm_w, self.layernorm_eps)
        idx_k = _gemma_rmsnorm_per_head(idx_k, self.idx_k_norm_w, self.layernorm_eps)

        if need_meta:
            torch.cuda.current_stream().synchronize()
            chunk_lengths_cpu = packed_pinned[:n_chunks].tolist()
            prefix_cpu = packed_pinned[n_chunks:]
            prefix_cpu_list = prefix_cpu.tolist()
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

            if index_score_chunk_enabled:
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
                "kv_lens_cpu": kv_lens_cpu,
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

        self._ensure_gather_scratch(kv_cache, device, qkv.dtype, bsz=bsz, max_kv=max_kv)
        # Only reuse addressing when this layer also reused CP metadata. When a
        # new request rebuilds metadata, the entry-local ``cache`` still points
        # at the previous request, so its addr must be ignored.
        addr_cache = None if need_meta or cache is None else cache.get("addr")
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
                }

        trtllm_workspace = self._maybe_trtllm_workspace(device)
        if index_score_chunk_enabled:
            assert index_score_host_metadata is not None
            if m3_fmha_prefill_enabled(
                workspace=trtllm_workspace,
                sparse_attn_plan=sparse_attn_plan,
                num_idx_heads=self.num_idx_heads,
                num_kv_heads=self.kv_head_num,
                disable_index_value=self.disable_index_value,
                has_idx_sink=False,
                has_sink=False,
                max_seqlen_k=max_seqlen_k,
                total_q=local_tokens,
            ):
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
                )

        idx_k = idx_k.contiguous()
        dummy_idx = _ROPE_DUMMY_SCRATCH.acquire(
            idx_k.shape[0], idx_k.shape[1], idx_k.shape[2], idx_k.dtype, idx_k.device
        )
        self._apply_rope(idx_k, dummy_idx, local_positions)

        can_fuse = self.cos_sin_cache is not None and not self._rope_interleave
        if can_fuse:
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

        all_packed = all_gather(packed_kv, group=Group.TP)

        q = _rows_to_contig(q)
        idx_q = idx_q.contiguous()
        if self.head_dim == self.idx_head_dim:
            self._apply_rope(q, idx_q, local_positions)
        else:
            dummy_q = torch.zeros_like(q[:, :1, :])
            self._apply_rope(q, dummy_q, local_positions)
            dummy_iq = torch.zeros_like(idx_q[:, :1, :])
            self._apply_rope(idx_q, dummy_iq, local_positions)

        token_count = token_count_py

        # Cold/sharded CP prefill has no missing prefix tokens in the gathered
        # payload. The fused paged writer unpads directly into scratch and the
        # scheduler-provided paged caches, avoiding full_* temporaries,
        # mha_kv_write_cache, and the separate unpack+scatter launches.
        used_fused_cp_paged_write = _USE_FUSED_CP_PAGED_WRITE and (
            self._kv_sharded or prefix_sum == 0
        )
        if used_fused_cp_paged_write:
            self._source_cp_from_packed(
                kv_cache,
                all_packed,
                unpad_indices,
                write_slots,
                slot_mapping,
                device,
                kv_lens_i32,
                nk,
                ni,
                token_count,
            )
        else:
            full_k = torch.empty(
                token_count,
                self.kv_head_num,
                self.head_dim,
                dtype=all_packed.dtype,
                device=device,
            )
            full_v = torch.empty_like(full_k)
            full_idx_k = torch.empty(
                token_count, 1, self.idx_head_dim, dtype=all_packed.dtype, device=device
            )
            _fused_unpack_packed_cp(
                all_packed,
                unpad_indices,
                full_k,
                full_v,
                full_idx_k,
                nk,
                ni,
                token_count=token_count,
            )
            if self._kv_sharded or prefix_sum == 0:
                self._source_cp_from_gathered(
                    kv_cache,
                    full_k,
                    full_v,
                    full_idx_k,
                    write_slots,
                    slot_mapping,
                    device,
                    token_count=token_count,
                )
            else:
                self._source_idx_k_from_paged(
                    kv_cache,
                    full_idx_k,
                    write_slots,
                    req_to_token,
                    kv_lens_cpu,
                    attn_inputs,
                    device,
                    slot_mapping=slot_mapping,
                )
                self._source_main_kv_from_paged(
                    kv_cache,
                    full_k,
                    full_v,
                    write_slots,
                    req_to_token,
                    kv_lens_cpu,
                    attn_inputs,
                    device,
                    slot_mapping=slot_mapping,
                )
        self._restore_cp_sharded_prefix_scratch(
            kv_cache, prefix_cpu_list, req_to_token, attn_inputs
        )
        if not used_fused_cp_paged_write:
            self._zero_scratch_padding_tail(kv_lens_cpu_list, bsz)

        if (
            kv_cache is not None
            and attn_inputs.is_prefill
            and attn_inputs.cache_store_inputs
        ):
            from rtp_llm.models_py.modules.factory.attention import (
                common as _attn_common,
            )

            write_impl = _attn_common.create_write_cache_store_impl(attn_inputs)
            _attn_common.apply_write_cache_store(write_impl, attn_inputs, kv_cache)

        _idx_o, o = minimax_sparse_prefill(
            q=q,
            k_cache=self._scratch_k,
            v_cache=self._scratch_v,
            sink=None,
            idx_q=idx_q,
            idx_k_cache=self._scratch_idx_k,
            idx_v_cache=None,
            idx_sink=None,
            req_to_token=req_to_token_segments,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens_i32,
            prefix_lens=prefix_i32,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size_q=1,
            block_size_k=self.block_size,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            disable_index_value=self.disable_index_value,
            workspace=trtllm_workspace,
            index_score_plan=index_score_plan,
            sparse_attn_plan=sparse_attn_plan,
            kv_indices=kv_page_indices,
        )

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

        # The fused write kernel casts K/V to the paged-pool dtype, so this path
        # is valid for both BF16 and FP8 KV cache. Keep draft decode aligned with
        # target verify instead of silently disabling M3_MSA_RAW_IDX_MXFP8 for
        # the production FP8 configuration.
        if self._should_use_mxfp8_fused_qkv_idx_decode(x_fp8, x_scale):
            paged_kv_base = self._paged_kv_base_view(kv_cache)
            scale = kv_cache.kv_scale_base
            paged_idx_k = scale.view(torch.bfloat16).view(
                int(scale.shape[0]), int(self.page_size), int(self.idx_head_dim)
            )
            q, idx_q = self._decode_project_fused_qkv_idx(
                total_tokens,
                positions,
                seq_lens,
                phys_block_table,
                paged_kv_base,
                paged_idx_k.reshape(-1, self.idx_head_dim),
                x_fp8=x_fp8,
                x_scale=x_scale,
            )
            paged_decode_views = (
                paged_kv_base[:, 0],
                paged_kv_base[:, 1],
                phys_block_table,
                paged_idx_k,
            )
        else:
            if x_fp8 is not None and x_scale is not None:
                qkv = self.qkv_proj(x_fp8, input_scales=x_scale)
            else:
                qkv = self.qkv_proj(hidden_states)
            if self.qk_fuse_norm is not None:
                qkv = self.qk_fuse_norm(qkv)
            q, k, v = torch.split(
                qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q = q.reshape(total_tokens, self.head_num, self.head_dim)
            k = k.reshape(total_tokens, self.kv_head_num, self.head_dim)
            v = v.reshape(total_tokens, self.kv_head_num, self.head_dim)

            idx_q = F.linear(hidden_states, self.idx_q_w)
            idx_k = F.linear(hidden_states, self.idx_k_w)
            idx_q = idx_q.reshape(total_tokens, self.num_idx_heads, self.idx_head_dim)
            idx_k = idx_k.reshape(total_tokens, 1, self.idx_head_dim)
            idx_q = _gemma_rmsnorm_per_head(
                idx_q, self.idx_q_norm_w, self.layernorm_eps
            )
            idx_k = _gemma_rmsnorm_per_head(
                idx_k, self.idx_k_norm_w, self.layernorm_eps
            )

            q = q.contiguous()
            k = k.contiguous()
            self._apply_rope(q, k, positions)
            idx_q = idx_q.contiguous()
            idx_k = idx_k.contiguous()
            self._apply_rope(idx_q, idx_k, positions)

            paged_decode_views = self._write_kv_cache_and_idx_k_for_decode(
                kv_cache, k, v, idx_k, seq_lens, phys_block_table
            )
        if paged_decode_views is None:
            raise RuntimeError(
                "MSA paged decode requires a BF16 or FP8 5-D paged KV cache and "
                "a BF16-compatible idx_K scale region. The original forward "
                "decode path is selected automatically when static paged KV "
                "conditions are not satisfied."
            )
        paged_main_k, paged_main_v, phys_block_table, paged_idx_k = paged_decode_views
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

        if self._should_use_mxfp8_fused_qkv_idx_decode(x_fp8, x_scale):
            paged_kv_base = self._paged_kv_base_view(kv_cache)
            scale = kv_cache.kv_scale_base
            paged_idx_k = scale.view(torch.bfloat16).view(
                int(scale.shape[0]), int(self.page_size), int(self.idx_head_dim)
            )
            q, idx_q = self._decode_project_fused_qkv_idx(
                total_tokens,
                positions,
                seq_lens,
                phys_block_table,
                paged_kv_base,
                paged_idx_k.reshape(-1, self.idx_head_dim),
                x_fp8=x_fp8,
                x_scale=x_scale,
            )
            paged_decode_views = (
                paged_kv_base[:, 0],
                paged_kv_base[:, 1],
                phys_block_table,
                paged_idx_k,
            )
        else:
            if x_fp8 is not None and x_scale is not None:
                qkv = self.qkv_proj(x_fp8, input_scales=x_scale)
            else:
                qkv = self.qkv_proj(hidden_states)
            if self.qk_fuse_norm is not None:
                qkv = self.qk_fuse_norm(qkv)
            q, k, v = torch.split(
                qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q = q.reshape(total_tokens, self.head_num, self.head_dim)
            k = k.reshape(total_tokens, self.kv_head_num, self.head_dim)
            v = v.reshape(total_tokens, self.kv_head_num, self.head_dim)

            idx_q = F.linear(hidden_states, self.idx_q_w)
            idx_k = F.linear(hidden_states, self.idx_k_w)
            idx_q = idx_q.reshape(total_tokens, self.num_idx_heads, self.idx_head_dim)
            idx_k = idx_k.reshape(total_tokens, 1, self.idx_head_dim)
            idx_q = _gemma_rmsnorm_per_head(
                idx_q, self.idx_q_norm_w, self.layernorm_eps
            )
            idx_k = _gemma_rmsnorm_per_head(
                idx_k, self.idx_k_norm_w, self.layernorm_eps
            )

            q = q.contiguous()
            k = k.contiguous()
            self._apply_rope(q, k, positions)
            idx_q = idx_q.contiguous()
            idx_k = idx_k.contiguous()
            self._apply_rope(idx_q, idx_k, positions)

            paged_decode_views = self._write_kv_cache_and_idx_k_for_decode(
                kv_cache, k, v, idx_k, seq_lens, phys_block_table
            )
        if paged_decode_views is None:
            raise RuntimeError(
                "MSA target verify requires BF16 or FP8 paged K/V and idx_K scale storage"
            )
        paged_main_k, paged_main_v, phys_block_table, paged_idx_k = paged_decode_views

        if self._cuda_graph_forward_active() or use_paged_capacity_bound:
            max_seqlen_k = self._cuda_graph_max_kv(attn_inputs, request_block_table)
        else:
            max_seqlen_k = int(seq_lens.max().item())
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
            score_block_table=request_block_table,
            score_seq_lens=seq_lens.view(request_batch_size, -1)[:, -1],
            decode_query_len=total_tokens // request_batch_size,
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
        from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
            minimax_sparse_decode,
            minimax_sparse_prefill,
        )

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

        if self._use_paged_decode_path(attn_inputs, kv_cache):
            return self._forward_paged_decode(
                hidden_states,
                attn_inputs,
                kv_cache,
                x_fp8=x_fp8,
                x_scale=x_scale,
            )

        if (
            self.cp_enabled
            and attn_inputs.is_prefill
            and attn_inputs.context_parallel_info is not None
        ):
            if _USE_V2_CP_PREFILL:
                return self._forward_cp_prefill_v2(
                    hidden_states,
                    attn_inputs,
                    kv_cache,
                    x_fp8=x_fp8,
                    x_scale=x_scale,
                )
            return self._forward_cp_prefill(
                hidden_states,
                attn_inputs,
                kv_cache,
                x_fp8=x_fp8,
                x_scale=x_scale,
            )

        input_shape = hidden_states.shape[:-1]
        total_tokens = hidden_states.shape[0]
        device = hidden_states.device

        # --- main QKV + per-head Gemma QK norm ---
        if x_fp8 is not None and x_scale is not None:
            qkv = self.qkv_proj(x_fp8, input_scales=x_scale)
        else:
            qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(total_tokens, self.head_num, self.head_dim)
        k = k.reshape(total_tokens, self.kv_head_num, self.head_dim)
        v = v.reshape(total_tokens, self.kv_head_num, self.head_dim)

        # --- index branch: proj -> per-head Gemma norm ---
        idx_q = F.linear(hidden_states, self.idx_q_w)
        idx_k = F.linear(hidden_states, self.idx_k_w)
        idx_q = idx_q.reshape(total_tokens, self.num_idx_heads, self.idx_head_dim)
        idx_k = idx_k.reshape(total_tokens, 1, self.idx_head_dim)
        idx_q = _gemma_rmsnorm_per_head(idx_q, self.idx_q_norm_w, self.layernorm_eps)
        idx_k = _gemma_rmsnorm_per_head(idx_k, self.idx_k_norm_w, self.layernorm_eps)

        # --- addressing (req_to_token / slot_ids / positions / write slots) ---
        if self.cp_enabled:
            alloc_kv_lens, _, _ = self._get_lengths(attn_inputs)
            if (not attn_inputs.is_prefill) and self._cuda_graph_forward_active():
                max_kv = self._cuda_graph_max_kv(attn_inputs)
            else:
                max_kv = int(alloc_kv_lens.max().item())
            self._ensure_gather_scratch(
                kv_cache,
                device,
                k.dtype,
                bsz=int(alloc_kv_lens.numel()),
                max_kv=max_kv,
            )
            (
                req_to_token,
                slot_ids,
                kv_lens,
                positions,
                write_slots,
                prefix_lens,
                inlens,
            ) = self._build_compact_addressing(attn_inputs, device)
        else:
            (
                req_to_token,
                slot_ids,
                kv_lens,
                positions,
                write_slots,
                prefix_lens,
                inlens,
            ) = self._build_addressing(attn_inputs, device)
            self._ensure_gather_scratch(
                kv_cache,
                device,
                k.dtype,
                max_slot=self._max_active_slot(req_to_token, kv_lens),
            )

        # --- partial RoPE on main q/k and index q/k ---
        q = _rows_to_contig(q)
        k = k.contiguous()
        self._apply_rope(q, k, positions)
        idx_q = idx_q.contiguous()
        idx_k = idx_k.contiguous()
        self._apply_rope(idx_q, idx_k, positions)

        # --- write current tokens into the cache-manager pool ---
        # idx_K -> paged scale region; main K/V -> paged pool. Both are
        # PD-transferable and the scratch is filled from paged so the kernel
        # reads paged-sourced data.
        if kv_cache is not None:
            self._source_idx_k_from_paged(
                kv_cache, idx_k, write_slots, req_to_token, kv_lens, attn_inputs, device
            )
            self._source_main_kv_from_paged(
                kv_cache, k, v, write_slots, req_to_token, kv_lens, attn_inputs, device
            )

        # PD separation: register this MSA layer's paged K/V (and the idx_K
        # piggybacked on the scale region) with the cache store, exactly like
        # dense CausalAttention does through its fmha_impl. Without this the
        # decode side waits forever for the missing MSA-layer blocks.
        if (
            kv_cache is not None
            and attn_inputs.is_prefill
            and attn_inputs.cache_store_inputs
        ):
            from rtp_llm.models_py.modules.factory.attention import (
                common as _attn_common,
            )

            _write_impl = _attn_common.create_write_cache_store_impl(attn_inputs)
            _attn_common.apply_write_cache_store(_write_impl, attn_inputs, kv_cache)

        # --- sparse attention via Triton MSA kernels ---
        main_k = self._scratch_k
        main_v = self._scratch_v
        idx_kc = self._scratch_idx_k
        if (not attn_inputs.is_prefill) and self._cuda_graph_forward_active():
            max_seqlen_k = int(req_to_token.shape[1])
        else:
            max_seqlen_k = int(kv_lens.max().item())
        if attn_inputs.is_prefill:
            cu_seqlens = attn_inputs.cu_seqlens[: slot_ids.numel() + 1].to(torch.int32)
            seq_lens = kv_lens.to(torch.int32)
            prefix_i32 = prefix_lens.to(torch.int32)
            max_seqlen_q = int(inlens.max().item())
            _idx_o, o = minimax_sparse_prefill(
                q=q,
                k_cache=main_k,
                v_cache=main_v,
                sink=None,
                idx_q=idx_q,
                idx_k_cache=idx_kc,
                idx_v_cache=None,
                idx_sink=None,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens,
                prefix_lens=prefix_i32,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                block_size_q=1,
                block_size_k=self.block_size,
                topk=self.topk_blocks,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                score_type=self.score_type,
                disable_index_value=self.disable_index_value,
                workspace=self._maybe_trtllm_workspace(device),
            )
        else:
            seq_lens = kv_lens.to(torch.int32)
            # Decode = 1 query token per request: cu_seqlens is [0,1,...,batch].
            # Pass it + prefix_lens + the trtllm workspace so minimax_sparse_decode
            # can take the trtllm-gen sparse-decode fast path (same op as prefill)
            # instead of the legacy triton step3; falls back to triton when the
            # trtllm gate is not satisfied (e.g. multi-request batch).
            decode_bsz = int(slot_ids.numel())
            decode_cu_seqlens = torch.arange(
                decode_bsz + 1, device=device, dtype=torch.int32
            )
            _idx_o, o = minimax_sparse_decode(
                q=q,
                sink=None,
                k_cache=main_k,
                v_cache=main_v,
                idx_q=idx_q,
                idx_sink=None,
                idx_k_cache=idx_kc,
                idx_v_cache=None,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                seq_lens=seq_lens,
                max_seqlen=max_seqlen_k,
                block_size_k=self.block_size,
                topk=self.topk_blocks,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                score_type=self.score_type,
                disable_index_value=self.disable_index_value,
                workspace=self._maybe_trtllm_workspace(device),
                cu_seqlens=decode_cu_seqlens,
                prefix_lens=prefix_lens.to(torch.int32),
            )

        attn_output = o.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
