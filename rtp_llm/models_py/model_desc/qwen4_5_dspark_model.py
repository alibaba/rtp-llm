"""Qwen4.5-Flash DFlash2 (DFlashQwenDraftModel / SP_TYPE_DSPARK) proposer model."""

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import inspect

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.qwen4_5.fast_comm import (
    get_qwen4_5_fast_all_reduce,
)

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.qwen4_5.qwen4_5_dspark import (
    DFLASH2_W,
    DFlash2ArchSpec,
)
from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import FULL_ATTN_KV_TAG
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.base import Embedding
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase

try:
    from rtp_llm.models_py.modules.factory.attention.common import (
        create_write_cache_store_impl,
    )
except Exception:  # pragma: no cover - pure-Python fallback
    create_write_cache_store_impl = None  # type: ignore[assignment]
from rtp_llm.models_py.modules.qwen4_5.attention import (
    GemmaRMSNorm,
    apply_partial_neox_rope,
    gemma_rmsnorm_and_partial_rope,
)
from rtp_llm.models_py.modules.qwen4_5.hyper_connection import Qwen4_5HyperConnection
from rtp_llm.models_py.modules.qwen4_5.moe import Qwen4_5LatentMoE
from rtp_llm.models_py.speculative.dspark_proposer_mixin import (
    DSparkProposerMixin,
    device_metadata_tensor,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    rtp_llm_ops,
)
from rtp_llm.utils.model_weight import W

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

logger = logging.getLogger(__name__)


QWEN4_5_OPAQUE_STATE_TAGS: Tuple[str, ...] = (
    "gdn_ssm_state",
    "gdn_conv_state",
    "gdn_consumer_q_conv",
    "full_attn_kv_sconv",
    "full_attn_output_sconv",
    "linear_attn_output_sconv",
    "ffn_output_sconv",
    "ple_conv_state",
    "ple_ngram_ctx",
    "linear",
    "indexer_state",
    "indexer_kv",
)


def _apply_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Zero-centered Gemma RMSNorm: ``(1.0 + weight) * rmsnorm(x)``."""
    orig_dtype = x.dtype
    xf = x.float()
    normed = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (normed * (1.0 + weight.float())).to(orig_dtype)


def _apply_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.float()
    normed = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (normed * weight.float()).to(orig_dtype)


def _resolve_host_or_device(
    primary: Optional[torch.Tensor], fallback: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if primary is not None and primary.numel() > 0:
        return primary
    if fallback is not None and fallback.numel() > 0:
        return fallback
    return primary if primary is not None else fallback


if _HAS_TRITON:

    @triton.jit
    def _scatter_kv_pages_kernel(
        k_ptr,
        v_ptr,
        req_ids_ptr,
        pos_ptr,
        bt_ptr,
        kv_cache_ptr,
        num_rows,
        num_kv_heads,
        page_size,
        head_dim,
        max_blocks,
        num_blocks,
        bt_stride_req: tl.int64,
        bt_stride_col: tl.int64,
        kv_stride_page: tl.int64,
        kv_stride_kv: tl.int64,
        kv_stride_head: tl.int64,
        kv_stride_pos: tl.int64,
        kv_stride_dim: tl.int64,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        if row >= num_rows or head >= num_kv_heads:
            return
        req_id = tl.load(req_ids_ptr + row).to(tl.int32)
        pos = tl.load(pos_ptr + row).to(tl.int32)
        if req_id < 0 or pos < 0:
            return
        block_idx = pos // page_size
        if block_idx < 0 or block_idx >= max_blocks:
            return
        slot_idx = pos - block_idx * page_size
        page_id = tl.load(
            bt_ptr + req_id.to(tl.int64) * bt_stride_req + block_idx.to(tl.int64) * bt_stride_col
        ).to(tl.int32)
        if page_id < 0 or page_id >= num_blocks:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        src_off = (row.to(tl.int64) * num_kv_heads + head.to(tl.int64)) * head_dim + offs_d
        dst_k = (
            page_id.to(tl.int64) * kv_stride_page
            + head.to(tl.int64) * kv_stride_head
            + slot_idx.to(tl.int64) * kv_stride_pos
            + offs_d * kv_stride_dim
        )
        dst_v = dst_k + kv_stride_kv
        k_val = tl.load(k_ptr + src_off, mask=mask_d)
        v_val = tl.load(v_ptr + src_off, mask=mask_d)
        tl.store(kv_cache_ptr + dst_k, k_val, mask=mask_d)
        tl.store(kv_cache_ptr + dst_v, v_val, mask=mask_d)

    @triton.jit
    def _dflash2_scatter_block_kv_triton(
        k_ptr,
        v_ptr,
        active_ptr,
        pos_ptr,
        bt_ptr,
        kv_cache_ptr,
        batch_size: tl.int32,
        query_len: tl.int32,
        num_kv_heads: tl.int32,
        page_size: tl.int32,
        head_dim: tl.int32,
        max_blocks: tl.int32,
        num_blocks: tl.int32,
        k_stride_tok: tl.int64,
        k_stride_h: tl.int64,
        k_stride_d: tl.int64,
        v_stride_tok: tl.int64,
        v_stride_h: tl.int64,
        v_stride_d: tl.int64,
        bt_stride_req: tl.int64,
        bt_stride_col: tl.int64,
        kv_stride_page: tl.int64,
        kv_stride_kv: tl.int64,
        kv_stride_head: tl.int64,
        kv_stride_pos: tl.int64,
        kv_stride_dim: tl.int64,
        BLOCK_D: tl.constexpr,
    ):
        tok = tl.program_id(0)
        head = tl.program_id(1)
        b = tok // query_len
        if b >= batch_size or head >= num_kv_heads:
            return
        if tl.load(active_ptr + b).to(tl.int32) == 0:
            return
        pos = tl.load(pos_ptr + tok).to(tl.int32)
        if pos < 0:
            return
        page = pos // page_size
        if page < 0 or page >= max_blocks:
            return
        slot = pos - page * page_size
        phys = tl.load(
            bt_ptr + b.to(tl.int64) * bt_stride_req + page.to(tl.int64) * bt_stride_col
        ).to(tl.int32)
        if phys < 0 or phys >= num_blocks:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        k_off = tok.to(tl.int64) * k_stride_tok + head.to(tl.int64) * k_stride_h + offs_d * k_stride_d
        v_off = tok.to(tl.int64) * v_stride_tok + head.to(tl.int64) * v_stride_h + offs_d * v_stride_d
        k_val = tl.load(k_ptr + k_off, mask=mask_d)
        v_val = tl.load(v_ptr + v_off, mask=mask_d)
        dst_k = (
            phys.to(tl.int64) * kv_stride_page
            + head.to(tl.int64) * kv_stride_head
            + slot.to(tl.int64) * kv_stride_pos
            + offs_d * kv_stride_dim
        )
        tl.store(kv_cache_ptr + dst_k, k_val, mask=mask_d)
        tl.store(kv_cache_ptr + dst_k + kv_stride_kv, v_val, mask=mask_d)

    @triton.jit
    def _dflash2_paged_noncausal_mha_stage1_triton(
        q_ptr,
        kv_cache_ptr,
        bt_ptr,
        prefix_lens_ptr,
        pos_ptr,
        active_ptr,
        ws_m_ptr,
        ws_l_ptr,
        ws_acc_ptr,
        sm_scale,
        batch_size: tl.int32,
        query_len: tl.int32,
        num_kv_heads: tl.int32,
        gqa_groups: tl.int32,
        num_g_tiles: tl.int32,
        num_splits: tl.int32,
        page_size: tl.int32,
        head_dim: tl.int32,
        max_blocks: tl.int32,
        num_blocks: tl.int32,
        window_left: tl.int32,
        q_stride_tok: tl.int64,
        q_stride_h: tl.int64,
        q_stride_d: tl.int64,
        bt_stride_req: tl.int64,
        bt_stride_col: tl.int64,
        kv_stride_page: tl.int64,
        kv_stride_kv: tl.int64,
        kv_stride_head: tl.int64,
        kv_stride_pos: tl.int64,
        kv_stride_dim: tl.int64,
        ws_stride_b: tl.int64,
        ws_stride_h: tl.int64,
        ws_stride_q: tl.int64,
        ws_stride_s: tl.int64,
        acc_stride_b: tl.int64,
        acc_stride_h: tl.int64,
        acc_stride_q: tl.int64,
        acc_stride_s: tl.int64,
        acc_stride_d: tl.int64,
        HAS_SWA: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        qg_tile_id = tl.program_id(0)
        pid_bh = tl.program_id(1)
        split_id = tl.program_id(2)

        q_tile_id = qg_tile_id // num_g_tiles
        g_tile_id = qg_tile_id % num_g_tiles
        b = pid_bh // num_kv_heads
        h_kv = pid_bh % num_kv_heads
        if b >= batch_size or h_kv >= num_kv_heads or split_id >= num_splits:
            return

        offs_m = tl.arange(0, BLOCK_M)
        q_local = offs_m // BLOCK_G
        g_local = offs_m % BLOCK_G
        q_idx = q_tile_id * BLOCK_Q + q_local
        g_idx = g_tile_id * BLOCK_G + g_local
        valid_m = (q_idx < query_len) & (g_idx < gqa_groups)
        safe_q = tl.where(q_idx < query_len, q_idx, 0)
        safe_g = tl.where(g_idx < gqa_groups, g_idx, 0)
        tok_m = b * query_len + safe_q
        hq_m = h_kv * gqa_groups + safe_g

        ml_off = (
            b.to(tl.int64) * ws_stride_b
            + hq_m.to(tl.int64) * ws_stride_h
            + safe_q.to(tl.int64) * ws_stride_q
            + split_id.to(tl.int64) * ws_stride_s
        )

        if tl.load(active_ptr + b).to(tl.int32) == 0:
            tl.store(ws_m_ptr + ml_off, tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32), mask=valid_m)
            tl.store(ws_l_ptr + ml_off, tl.zeros((BLOCK_M,), dtype=tl.float32), mask=valid_m)
            return

        pref_len = tl.load(prefix_lens_ptr + b).to(tl.int32)
        total_kv = pref_len + query_len
        max_kv_slots = max_blocks * page_size
        if total_kv > max_kv_slots:
            total_kv = max_kv_slots
        if total_kv <= 0:
            tl.store(ws_m_ptr + ml_off, tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32), mask=valid_m)
            tl.store(ws_l_ptr + ml_off, tl.zeros((BLOCK_M,), dtype=tl.float32), mask=valid_m)
            return

        if HAS_SWA:
            q_pos_m = tl.load(pos_ptr + tok_m, mask=(q_idx < query_len), other=0).to(tl.int32)
            kv_start_m = tl.maximum(0, q_pos_m - window_left)
            tile_kv_start = tl.min(tl.where(valid_m, kv_start_m, total_kv), axis=0)
        else:
            kv_start_m = tl.zeros((BLOCK_M,), dtype=tl.int32)
            tile_kv_start = 0

        kv_span = total_kv - tile_kv_start
        if kv_span <= 0:
            tl.store(ws_m_ptr + ml_off, tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32), mask=valid_m)
            tl.store(ws_l_ptr + ml_off, tl.zeros((BLOCK_M,), dtype=tl.float32), mask=valid_m)
            return

        total_n_tiles = tl.cdiv(kv_span, BLOCK_N)
        tiles_per_split = tl.cdiv(total_n_tiles, num_splits)
        start_n = tile_kv_start + split_id * tiles_per_split * BLOCK_N
        end_n = tl.minimum(total_kv, start_n + tiles_per_split * BLOCK_N)
        if start_n >= end_n:
            tl.store(ws_m_ptr + ml_off, tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32), mask=valid_m)
            tl.store(ws_l_ptr + ml_off, tl.zeros((BLOCK_M,), dtype=tl.float32), mask=valid_m)
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        mask_md = valid_m[:, None] & mask_d[None, :]
        q_ptrs = (
            q_ptr
            + tok_m[:, None].to(tl.int64) * q_stride_tok
            + hq_m[:, None].to(tl.int64) * q_stride_h
            + offs_d[None, :] * q_stride_d
        )
        q_tile = tl.load(q_ptrs, mask=mask_md, other=0.0)
        if q_tile.dtype == tl.float32:
            q_tile = q_tile.to(tl.bfloat16)

        m_i = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        offs_n = tl.arange(0, BLOCK_N)
        cur_n = start_n
        while cur_n < end_n:
            kv_pos = cur_n + offs_n
            valid_n = (kv_pos >= tile_kv_start) & (kv_pos < end_n)
            safe_pos = tl.where(valid_n, kv_pos, 0)
            pages = safe_pos // page_size
            slots = safe_pos - pages * page_size
            phys = tl.load(
                bt_ptr + b.to(tl.int64) * bt_stride_req + pages.to(tl.int64) * bt_stride_col,
                mask=valid_n,
                other=0,
            ).to(tl.int32)
            valid_k = valid_n & (phys >= 0) & (phys < num_blocks)
            safe_phys = tl.where(valid_k, phys, 0)
            k_base = (
                safe_phys[:, None].to(tl.int64) * kv_stride_page
                + h_kv.to(tl.int64) * kv_stride_head
                + slots[:, None].to(tl.int64) * kv_stride_pos
                + offs_d[None, :] * kv_stride_dim
            )
            mask_kd = valid_k[:, None] & mask_d[None, :]
            k_tile = tl.load(kv_cache_ptr + k_base, mask=mask_kd, other=0.0)
            v_tile = tl.load(
                kv_cache_ptr + k_base + kv_stride_kv, mask=mask_kd, other=0.0
            )
            if k_tile.dtype != q_tile.dtype:
                k_tile = k_tile.to(q_tile.dtype)
            if v_tile.dtype != q_tile.dtype:
                v_tile = v_tile.to(q_tile.dtype)

            qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            if HAS_SWA:
                valid_mn = (
                    valid_m[:, None]
                    & valid_k[None, :]
                    & (kv_pos[None, :] >= kv_start_m[:, None])
                )
            else:
                valid_mn = valid_m[:, None] & valid_k[None, :]
            qk = tl.where(valid_mn, qk, -1.0e20)

            row_max = tl.max(qk, axis=1)
            m_ij = tl.maximum(m_i, row_max)
            p = tl.where(valid_mn, tl.exp(qk - m_ij[:, None]), 0.0)
            alpha = tl.where(m_i > -1.0e19, tl.exp(m_i - m_ij), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            m_i = m_ij
            cur_n += BLOCK_N

        tl.store(ws_m_ptr + ml_off, m_i, mask=valid_m)
        tl.store(ws_l_ptr + ml_off, l_i, mask=valid_m)
        acc_off = (
            b.to(tl.int64) * acc_stride_b
            + hq_m[:, None].to(tl.int64) * acc_stride_h
            + safe_q[:, None].to(tl.int64) * acc_stride_q
            + split_id.to(tl.int64) * acc_stride_s
            + offs_d[None, :] * acc_stride_d
        )
        mask_acc = (valid_m & (l_i > 0.0))[:, None] & mask_d[None, :]
        tl.store(ws_acc_ptr + acc_off, acc, mask=mask_acc)

    @triton.jit
    def _dflash2_paged_noncausal_mha_stage2_triton(
        ws_m_ptr,
        ws_l_ptr,
        ws_acc_ptr,
        active_ptr,
        gate_ptr,
        out_ptr,
        batch_size: tl.int32,
        query_len: tl.int32,
        num_heads: tl.int32,
        num_splits: tl.int32,
        head_dim: tl.int32,
        ws_stride_b: tl.int64,
        ws_stride_h: tl.int64,
        ws_stride_q: tl.int64,
        ws_stride_s: tl.int64,
        acc_stride_b: tl.int64,
        acc_stride_h: tl.int64,
        acc_stride_q: tl.int64,
        acc_stride_s: tl.int64,
        acc_stride_d: tl.int64,
        g_stride_tok: tl.int64,
        g_stride_h: tl.int64,
        g_stride_d: tl.int64,
        o_stride_tok: tl.int64,
        o_stride_h: tl.int64,
        o_stride_d: tl.int64,
        HAS_GATE: tl.constexpr,
        SPLIT_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        tok = tl.program_id(0)
        h = tl.program_id(1)
        b = tok // query_len
        q_idx = tok - b * query_len
        if b >= batch_size or h >= num_heads:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        o_ptrs = (
            out_ptr
            + tok.to(tl.int64) * o_stride_tok
            + h.to(tl.int64) * o_stride_h
            + offs_d * o_stride_d
        )

        if tl.load(active_ptr + b).to(tl.int32) == 0:
            tl.store(
                o_ptrs,
                tl.zeros((BLOCK_D,), dtype=tl.float32).to(out_ptr.dtype.element_ty),
                mask=mask_d,
            )
            return

        ml_base = (
            b.to(tl.int64) * ws_stride_b
            + h.to(tl.int64) * ws_stride_h
            + q_idx.to(tl.int64) * ws_stride_q
        )
        offs_s = tl.arange(0, SPLIT_BLOCK)
        mask_s = offs_s < num_splits
        m_vec = tl.load(
            ws_m_ptr + ml_base + offs_s.to(tl.int64) * ws_stride_s,
            mask=mask_s,
            other=-1.0e20,
        )
        l_vec = tl.load(
            ws_l_ptr + ml_base + offs_s.to(tl.int64) * ws_stride_s,
            mask=mask_s,
            other=0.0,
        )
        valid_s = mask_s & (l_vec > 0.0) & (m_vec > -1.0e19)
        m_max = tl.max(tl.where(valid_s, m_vec, -1.0e20), axis=0)
        w_vec = tl.where(valid_s, tl.exp(m_vec - m_max), 0.0)
        l_total = tl.sum(w_vec * l_vec, axis=0)
        if l_total <= 0.0:
            tl.store(
                o_ptrs,
                tl.zeros((BLOCK_D,), dtype=tl.float32).to(out_ptr.dtype.element_ty),
                mask=mask_d,
            )
            return

        acc_base = (
            b.to(tl.int64) * acc_stride_b
            + h.to(tl.int64) * acc_stride_h
            + q_idx.to(tl.int64) * acc_stride_q
        )
        a_mat = tl.load(
            ws_acc_ptr
            + acc_base
            + offs_s[:, None].to(tl.int64) * acc_stride_s
            + offs_d[None, :] * acc_stride_d,
            mask=valid_s[:, None] & mask_d[None, :],
            other=0.0,
        )
        out_acc = tl.sum(w_vec[:, None] * a_mat, axis=0)

        out_val = out_acc / tl.maximum(l_total, 1.0e-12)
        if HAS_GATE:
            g_ptrs = (
                gate_ptr
                + tok.to(tl.int64) * g_stride_tok
                + h.to(tl.int64) * g_stride_h
                + offs_d * g_stride_d
            )
            g_val = tl.load(g_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            out_val = out_val * tl.sigmoid(g_val)
        tl.store(o_ptrs, out_val.to(out_ptr.dtype.element_ty), mask=mask_d)

    @triton.jit
    def _dflash2_grouped_conv_kernel(
        x_ptr,
        delta_ptr,
        base_ptr,
        out_ptr,
        x_stride_row,
        delta_stride_row,
        delta_stride_tap,
        base_stride_tap,
        out_stride_row,
        num_rows,
        NUM_CHANNELS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        TAPS: tl.constexpr,
        ELEMENT_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blocks_per_row = tl.cdiv(NUM_CHANNELS, ELEMENT_BLOCK)
        row = pid // blocks_per_row
        if row >= num_rows:
            return
        col_blk = pid - row * blocks_per_row
        offs = col_blk * ELEMENT_BLOCK + tl.arange(0, ELEMENT_BLOCK)
        mask = offs < NUM_CHANNELS
        groups = offs // GROUP_SIZE
        pos_in_blk = row % BLOCK_SIZE

        acc = tl.zeros((ELEMENT_BLOCK,), dtype=tl.float32)
        for tap in tl.static_range(0, TAPS):
            src_row = row - tap
            valid = pos_in_blk >= tap
            safe_row = tl.where(valid, src_row, 0)
            x_val = tl.load(
                x_ptr + safe_row * x_stride_row + offs,
                mask=mask & valid,
                other=0.0,
            ).to(tl.float32)
            base_val = tl.load(
                base_ptr + tap * base_stride_tap + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            delta_val = tl.load(
                delta_ptr + row * delta_stride_row + tap * delta_stride_tap + groups,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += (base_val + delta_val) * x_val

        tl.store(out_ptr + row * out_stride_row + offs, acc, mask=mask)

    @triton.jit
    def _selector_greedy_walk_kernel(
        scores_ptr,
        candidate_ptr,
        tokens_ptr,
        num_steps: tl.constexpr,
        top_k: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < top_k
        previous = 0

        for step in range(num_steps):
            flat = row * num_steps + step
            score_base = (flat * top_k + previous) * top_k
            scores = tl.load(
                scores_ptr + score_base + offsets,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            index = tl.argmax(scores, axis=0)
            candidate_base = flat * top_k
            token = tl.load(candidate_ptr + candidate_base + index)
            tl.store(tokens_ptr + flat, token)
            previous = index


def _as_hnd_kv_pool(
    kv_base: torch.Tensor, kv_head_num: int, page_size: int, head_dim: int
) -> torch.Tensor:
    """Normalize either raw 2D hybrid packed cache or 5D cache to HND ``[B, 2, H, P, D]``."""
    if kv_base.dim() == 2:
        expected = 2 * kv_head_num * page_size * head_dim
        if kv_base.shape[1] < expected:
            raise ValueError(
                f"packed kv_cache_base has stride {kv_base.shape[1]}, need {expected}"
            )
        return kv_base[:, :expected].reshape(-1, 2, kv_head_num, page_size, head_dim)
    if (
        kv_base.dim() == 5
        and int(kv_base.shape[2]) != kv_head_num
        and int(kv_base.shape[3]) == kv_head_num
    ):
        return kv_base.transpose(2, 3)
    return kv_base


def scatter_kv_to_paged_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    req_ids: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    kv_cache_base: torch.Tensor,
    page_size: int,
) -> None:
    """Scatter ``[rows, kv_heads, head_dim]`` K/V into HND paged KV cache."""
    num_rows, num_kv_heads, head_dim = key.shape
    if num_rows == 0 or block_table.numel() == 0:
        return
    kv_hnd = _as_hnd_kv_pool(kv_cache_base, num_kv_heads, page_size, head_dim)
    max_blocks = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    num_blocks = int(kv_hnd.shape[0])
    if max_blocks == 0 or num_blocks == 0:
        return
    if key.is_cuda and kv_hnd.is_cuda and _HAS_TRITON:
        key_c = key.contiguous()
        val_c = value.contiguous()
        req_c = req_ids if req_ids.dtype == torch.int32 and req_ids.is_contiguous() else req_ids.to(torch.int32).contiguous()
        pos_c = positions if positions.dtype == torch.int32 and positions.is_contiguous() else positions.to(torch.int32).contiguous()
        bt_c = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
        block_d = max(16, int(triton.next_power_of_2(head_dim)))
        grid = (num_rows, num_kv_heads)
        _scatter_kv_pages_kernel[grid](
            key_c,
            val_c,
            req_c,
            pos_c,
            bt_c,
            kv_hnd,
            num_rows,
            num_kv_heads,
            page_size,
            head_dim,
            max_blocks,
            num_blocks,
            int(bt_c.stride(0)),
            int(bt_c.stride(1)),
            int(kv_hnd.stride(0)),
            int(kv_hnd.stride(1)),
            int(kv_hnd.stride(2)),
            int(kv_hnd.stride(3)),
            int(kv_hnd.stride(4)),
            BLOCK_D=block_d,
        )
        return

    valid = (req_ids >= 0) & (positions >= 0) & ((positions // page_size) < max_blocks)
    if not bool(valid.any().item()):
        return
    req_v = req_ids[valid].long()
    pos_v = positions[valid].long()
    k_v = key[valid].to(kv_hnd.dtype)
    v_v = value[valid].to(kv_hnd.dtype)
    pages = block_table[req_v, pos_v // page_size].long()
    slots = pos_v % page_size
    page_ok = (pages >= 0) & (pages < num_blocks)
    if not bool(page_ok.all().item()):
        pages = pages[page_ok]
        slots = slots[page_ok]
        k_v = k_v[page_ok]
        v_v = v_v[page_ok]
    kv_hnd[pages, 0, :, slots, :] = k_v
    kv_hnd[pages, 1, :, slots, :] = v_v


def grouped_conv_ref(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Exact PyTorch reference for request-local dynamic grouped conv."""
    if hidden_states.size(0) == 0:
        return torch.empty_like(hidden_states)
    blocks = hidden_states.unflatten(-1, (num_groups, group_size)).float()
    coefficients = (
        base.view(1, taps, num_groups, group_size).float()
        + delta.unsqueeze(-1).float()
    )
    output = coefficients[:, 0] * blocks

    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size

    for tap in range(1, taps):
        if hidden_states.shape[0] <= tap:
            break
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        within_request = (position >= tap).view(-1, 1, 1)
        output = output + coefficients[:, tap] * shifted * within_request
    return output.flatten(-2).to(hidden_states.dtype)


def grouped_conv_forward(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    num_rows, num_channels = hidden_states.shape
    if num_rows == 0:
        return torch.empty_like(hidden_states)
    if hidden_states.is_cuda and _HAS_TRITON:
        x_c = (
            hidden_states
            if (hidden_states.ndim == 2 and hidden_states.stride(-1) == 1)
            else hidden_states.contiguous()
        )
        delta_c = (
            delta
            if (delta.ndim == 3 and delta.stride(-1) == 1)
            else delta.contiguous()
        )
        base_c = (
            base
            if (base.ndim == 2 and base.stride(-1) == 1)
            else base.contiguous()
        )
        out = torch.empty_like(x_c)
        element_block = 512 if num_channels % 512 == 0 else 256
        grid = (num_rows * triton.cdiv(num_channels, element_block),)
        _dflash2_grouped_conv_kernel[grid](
            x_c,
            delta_c,
            base_c,
            out,
            x_c.stride(0),
            delta_c.stride(0),
            delta_c.stride(1),
            base_c.stride(0),
            out.stride(0),
            num_rows,
            NUM_CHANNELS=num_channels,
            BLOCK_SIZE=block_size,
            GROUP_SIZE=group_size,
            TAPS=taps,
            ELEMENT_BLOCK=element_block,
            num_warps=4,
        )
        return out
    return grouped_conv_ref(
        hidden_states, delta, base, block_size, num_groups, group_size, taps
    )


class DFlashGroupedConv(nn.Module):
    """DFlash2 per-layer dynamic grouped convolution (side 0 prepare / side 1 finish)."""

    def __init__(
        self,
        hidden_size: int,
        taps: int = 2,
        group_size: int = 16,
        block_size: int = 8,
        identity_folded: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if group_size <= 0 or hidden_size % group_size != 0:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}"
            )
        if taps <= 0 or taps > block_size:
            raise ValueError(
                f"conv_kernel_size={taps} must be in [1, block_size={block_size}]"
            )
        self.hidden_size = int(hidden_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = self.hidden_size // self.group_size
        self.block_size = int(block_size)

        base_init = torch.zeros(2, self.taps, self.hidden_size, dtype=dtype, device=device)
        if identity_folded:
            base_init[:, 0, :] = 1.0
        self.base_kernel = nn.Parameter(base_init, requires_grad=False)
        self.kernel_projection = nn.Linear(
            self.hidden_size,
            2 * self.taps * self.num_groups,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return grouped_conv_forward(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


def greedy_select_path_ref(
    candidate_ids: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    """PyTorch reference for CandidateSelector sequential greedy path walk."""
    batch_size, num_steps, _ = candidate_ids.shape
    batch_indices = torch.arange(batch_size, device=candidate_ids.device)
    previous = torch.zeros(batch_size, dtype=torch.int64, device=candidate_ids.device)
    tokens = torch.empty(
        batch_size, num_steps, dtype=candidate_ids.dtype, device=candidate_ids.device
    )
    for step in range(num_steps):
        choice = scores[batch_indices, step, previous].argmax(dim=-1)
        tokens[:, step] = candidate_ids[batch_indices, step, choice]
        previous = choice
    return tokens


def greedy_select_path(
    candidate_ids: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    """Select coherent token chain from ``[B, L, K, K]`` transition scores."""
    batch_size, num_steps, top_k = candidate_ids.shape
    if batch_size == 0 or num_steps == 0:
        return torch.empty(
            (batch_size, num_steps),
            dtype=candidate_ids.dtype,
            device=candidate_ids.device,
        )
    if candidate_ids.is_cuda and _HAS_TRITON:
        tokens = torch.empty(
            (batch_size, num_steps),
            dtype=candidate_ids.dtype,
            device=candidate_ids.device,
        )
        _selector_greedy_walk_kernel[(batch_size,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            tokens,
            num_steps=num_steps,
            top_k=top_k,
            BLOCK_K=triton.next_power_of_2(top_k),
            num_warps=4,
        )
        return tokens
    return greedy_select_path_ref(candidate_ids, scores)


class CandidateSelector(nn.Module):
    """Low-rank bigram CandidateSelector mapped onto W.dspark_markov_w1/w2."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int = 256,
        top_k: int = 16,
        predecessor_codebook: Optional[torch.Tensor] = None,
        successor_codebook: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"selector_rank must be positive, got {rank}")
        if top_k <= 0 or top_k > vocab_size:
            raise ValueError(f"selector_top_k={top_k} out of range [1, {vocab_size}]")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.top_k = int(top_k)

        self.hidden_projection = nn.Linear(
            self.hidden_size, self.rank, bias=False, dtype=dtype, device=device
        )
        if predecessor_codebook is not None:
            self.predecessor_codebook = nn.Parameter(
                predecessor_codebook, requires_grad=False
            )
        else:
            self.predecessor_codebook = nn.Parameter(
                torch.zeros(self.vocab_size, self.rank, dtype=dtype, device=device),
                requires_grad=False,
            )
        if successor_codebook is not None:
            self.successor_codebook = nn.Parameter(
                successor_codebook, requires_grad=False
            )
        else:
            self.successor_codebook = nn.Parameter(
                torch.zeros(self.vocab_size, self.rank, dtype=dtype, device=device),
                requires_grad=False,
            )

    def score_edges(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``[B, L, K, K]`` transition scores from hidden + codebooks."""
        hidden = self.hidden_projection(hidden_states)
        cand_l = candidate_ids.long()
        successors = self.successor_codebook[cand_l]
        predecessor_ids = torch.cat(
            (
                anchor_token_ids.long()[:, None, None].expand(-1, 1, self.top_k),
                cand_l[:, :-1],
            ),
            dim=1,
        )
        predecessors = self.predecessor_codebook[predecessor_ids]
        return unary_logits[:, :, None].float() + torch.einsum(
            "blpr,blcr->blpc",
            (predecessors * hidden[:, :, None]).float(),
            successors.float(),
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_edges(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        )

    def select_candidates_from_topk(
        self,
        unary_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score pre-selected top-k candidates per block step and walk greedy chain."""
        batch_size = int(anchor_token_ids.numel())
        seq_len = int(hidden_states.shape[0] // max(1, batch_size))
        hidden_b = hidden_states.view(batch_size, seq_len, -1)
        unary_b = unary_logits.view(batch_size, seq_len, self.top_k)
        cand_b = candidate_ids.view(batch_size, seq_len, self.top_k)
        edge_scores = self.score_edges(cand_b, unary_b, hidden_b, anchor_token_ids)
        selected_tokens = greedy_select_path(cand_b, edge_scores)
        return selected_tokens, cand_b, edge_scores

    def select_candidates(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score top-k candidates per block step and walk coherent greedy chain."""
        batch_size = int(anchor_token_ids.numel())
        seq_len = int(hidden_states.shape[0] // max(1, batch_size))
        logits_b = logits.view(batch_size, seq_len, -1)
        unary_logits, candidate_ids = torch.topk(logits_b, self.top_k, dim=-1)
        return self.select_candidates_from_topk(
            unary_logits, candidate_ids, hidden_states, anchor_token_ids
        )


_DFLASH2_MAX_SPLITS = 16


def _select_tagged_attn_inputs(
    attn_or_map: Any, tag: str = FULL_ATTN_KV_TAG
) -> Any:
    if isinstance(attn_or_map, Mapping):
        if tag in attn_or_map:
            return attn_or_map[tag]
        if len(attn_or_map) > 0:
            return next(iter(attn_or_map.values()))
    return attn_or_map


def _is_collective_torch_initialized() -> bool:
    try:
        from rtp_llm.models_py.distributed import collective_torch

        return bool(getattr(collective_torch, "_initialized", False))
    except Exception:
        return False


def _resolve_tp_group(tp_size: int) -> Any:
    if tp_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return None
    if _is_collective_torch_initialized():
        try:
            from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

            return _get_group(Group.TP)
        except Exception:
            pass
    if dist.get_world_size() >= tp_size:
        return dist.group.WORLD
    return None


def _resolve_world_group() -> Any:
    if not dist.is_available() or not dist.is_initialized():
        return None
    if _is_collective_torch_initialized():
        try:
            from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

            return _get_group(Group.WORLD)
        except Exception:
            pass
    return dist.group.WORLD


class _DFlash2SplitKVWorkspace:
    """Static preallocated Split-KV workspace for CUDA-Graph-safe MHA execution."""

    def __init__(
        self,
        max_batch: int = 0,
        num_heads: int = 0,
        query_len: int = 0,
        max_splits: int = _DFLASH2_MAX_SPLITS,
        head_dim: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.max_batch = 0
        self.num_heads = 0
        self.query_len = 0
        self.max_splits = max(1, int(max_splits))
        self.head_dim = 0
        self.device: Optional[torch.device] = None
        self.ws_m: Optional[torch.Tensor] = None
        self.ws_l: Optional[torch.Tensor] = None
        self.ws_acc: Optional[torch.Tensor] = None
        if (
            device is not None
            and device.type == "cuda"
            and max_batch > 0
            and num_heads > 0
            and query_len > 0
            and head_dim > 0
        ):
            self.ensure_capacity(max_batch, num_heads, query_len, self.max_splits, head_dim, device)

    def ensure_capacity(
        self,
        batch_size: int,
        num_heads: int,
        query_len: int,
        num_splits: int,
        head_dim: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        need_alloc = (
            self.ws_m is None
            or self.ws_l is None
            or self.ws_acc is None
            or self.device != device
            or batch_size > self.max_batch
            or num_heads > self.num_heads
            or query_len > self.query_len
            or num_splits > self.max_splits
            or head_dim > self.head_dim
        )
        if need_alloc:
            if torch.cuda.is_available() and device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DFlash2 Split-KV workspace capacity exceeded during CUDA Graph capture; "
                    f"need=({batch_size}, {num_heads}, {query_len}, {num_splits}, {head_dim}) vs "
                    f"cap=({self.max_batch}, {self.num_heads}, {self.query_len}, {self.max_splits}, {self.head_dim})"
                )
            new_b = max(self.max_batch, int(batch_size))
            new_h = max(self.num_heads, int(num_heads))
            new_q = max(self.query_len, int(query_len))
            new_s = max(self.max_splits, int(num_splits))
            new_d = max(self.head_dim, int(head_dim))
            self.ws_m = torch.empty((new_b, new_h, new_q, new_s), dtype=torch.float32, device=device)
            self.ws_l = torch.empty((new_b, new_h, new_q, new_s), dtype=torch.float32, device=device)
            self.ws_acc = torch.empty(
                (new_b, new_h, new_q, new_s, new_d), dtype=torch.float32, device=device
            )
            self.max_batch = new_b
            self.num_heads = new_h
            self.query_len = new_q
            self.max_splits = new_s
            self.head_dim = new_d
            self.device = device
        assert self.ws_m is not None and self.ws_l is not None and self.ws_acc is not None
        return self.ws_m, self.ws_l, self.ws_acc


def _choose_dflash2_tile_and_splits(
    batch_size: int,
    q_len: int,
    num_kv_heads: int,
    gqa_groups: int,
    max_blocks: int,
    page_size: int,
    max_splits: int = _DFLASH2_MAX_SPLITS,
) -> Tuple[int, int, int, int]:
    """Select (BLOCK_Q, BLOCK_G, BLOCK_M, num_splits) with static tensor shapes only."""
    block_g = min(8, max(1, int(triton.next_power_of_2(max(1, gqa_groups)))))
    if q_len <= 1:
        block_g = min(16, max(1, int(triton.next_power_of_2(max(1, gqa_groups)))))
    # For small GQA (e.g. TP=4 gqa=8), BLOCK_M=16 maximizes Split-KV occupancy & SM speed;
    # for large GQA (TP=1 gqa=32), pack up to BLOCK_M=64 to amortize HBM KV loads 8x.
    target_m = 64 if (gqa_groups >= 16 and q_len >= 4) else 16
    max_q_fit = max(1, target_m // block_g)
    block_q = min(max_q_fit, max(1, int(triton.next_power_of_2(max(1, q_len)))))
    if block_q * block_g < 16:
        block_q = max(1, (16 + block_g - 1) // block_g)
    block_m = block_q * block_g
    num_q_tiles = (q_len + block_q - 1) // block_q
    num_g_tiles = (gqa_groups + block_g - 1) // block_g
    max_possible_kv = max(0, max_blocks * page_size)
    if max_possible_kv <= 32 or max_splits <= 1:
        num_splits = 1
    else:
        base_ctas = max(1, batch_size * num_kv_heads * num_q_tiles * num_g_tiles)
        desired = (128 + base_ctas - 1) // base_ctas
        max_useful = max(1, (max_possible_kv + 31) // 32)
        num_splits = max(1, min(int(max_splits), int(desired), int(max_useful)))
    return block_q, block_g, block_m, num_splits


class _DFlash2PagedFMHADispatcher(FMHAImplBase):
    """Paged attention dispatcher supporting SWA + Full non-causal layers on SM100 & CPU."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: Any,
        *,
        use_triton_paged: Optional[bool] = None,
        use_flashinfer: bool = False,
        query_positions: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        active_requests: Optional[torch.Tensor] = None,
        query_width: Optional[int] = None,
        split_kv_workspace: Optional[_DFlash2SplitKVWorkspace] = None,
    ) -> None:
        self.attn_configs = attn_configs
        self.attn_inputs = _select_tagged_attn_inputs(attn_inputs)
        if use_triton_paged is None:
            use_triton_paged = bool(use_flashinfer)
        self.use_triton_paged = bool(use_triton_paged and _HAS_TRITON)
        # Keep legacy attribute false so FlashInfer tmp_v 134MB overflow never triggers.
        self.use_flashinfer = False
        self.query_positions = query_positions
        self.prefix_lengths = prefix_lengths
        self.active_requests = active_requests
        self.query_width = query_width
        self.split_kv_workspace = (
            split_kv_workspace if split_kv_workspace is not None else _DFlash2SplitKVWorkspace()
        )

        self._page_size = max(1, int(attn_configs.kernel_tokens_per_block or 16))
        self._kv_dtype = attn_configs.dtype

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        del attn_inputs
        return not attn_configs.is_causal

    def support_cuda_graph(self) -> bool:
        return bool(self.use_triton_paged)

    def prepare(self, attn_inputs: Any) -> None:
        self.attn_inputs = _select_tagged_attn_inputs(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs: Any) -> None:
        self.attn_inputs = _select_tagged_attn_inputs(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        raise NotImplementedError("Qwen4_5DFlashAttention uses append_and_attend()")

    def _resolve_block_table(self, device: torch.device) -> torch.Tensor:
        ai = self.attn_inputs
        bt = device_metadata_tensor(ai, "kv_cache_kernel_block_id")
        if bt is None:
            bt = _resolve_host_or_device(
                getattr(ai, "kv_cache_kernel_block_id_device", None),
                getattr(ai, "kv_cache_kernel_block_id", None),
            )
        if bt is None:
            bt = _resolve_host_or_device(
                getattr(ai, "kv_cache_block_id_device", None),
                getattr(ai, "kv_cache_block_id", None),
            )
        if bt is None:
            raise ValueError("DFlash2 paged attention requires kv_cache_kernel_block_id")
        return bt if bt.device == device else bt.to(device)

    def append_and_attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        *,
        sliding_window: Optional[int] = None,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Append K/V into paged cache and compute non-causal (SWA/Full) attention."""
        total_tokens = int(q.size(0))
        head_num = int(q.size(1))
        kv_head_num = int(k.size(1))
        head_dim = int(q.size(-1))
        if total_tokens == 0:
            return torch.empty_like(q)

        gqa_groups = head_num // kv_head_num
        sm_scale = 1.0 / math.sqrt(head_dim)

        # Fast SM103 CUDA-Graph-native Triton path (zero host syncs / dynamic shapes).
        if (
            self.use_triton_paged
            and _HAS_TRITON
            and self.query_width is not None
            and self.active_requests is not None
            and self.query_positions is not None
            and self.prefix_lengths is not None
            and kv_cache is not None
            and getattr(kv_cache, "kv_cache_base", None) is not None
            and kv_cache.kv_cache_base.is_cuda
            and total_tokens % int(self.query_width) == 0
        ):
            q_len = int(self.query_width)
            batch_size = int(self.active_requests.numel())
            if batch_size * q_len == total_tokens:
                kv_hnd = _as_hnd_kv_pool(
                    kv_cache.kv_cache_base, kv_head_num, self._page_size, head_dim
                )
                bt = self._resolve_block_table(q.device)
                if bt.dtype != torch.int32:
                    bt = bt.to(torch.int32)
                max_blocks = int(bt.shape[1]) if bt.ndim >= 2 else 0
                num_blocks = int(kv_hnd.shape[0])
                if max_blocks == 0 or num_blocks == 0:
                    return torch.zeros_like(q)

                pos_d = (
                    self.query_positions
                    if self.query_positions.device == q.device
                    and self.query_positions.dtype == torch.int32
                    and self.query_positions.is_contiguous()
                    else self.query_positions.to(device=q.device, dtype=torch.int32).contiguous()
                )
                pref_d = (
                    self.prefix_lengths
                    if self.prefix_lengths.device == q.device
                    and self.prefix_lengths.dtype == torch.int32
                    and self.prefix_lengths.is_contiguous()
                    else self.prefix_lengths.to(device=q.device, dtype=torch.int32).contiguous()
                )
                act_d = (
                    self.active_requests
                    if self.active_requests.device == q.device
                    and self.active_requests.is_contiguous()
                    else self.active_requests.to(device=q.device).contiguous()
                )

                block_d = max(16, int(triton.next_power_of_2(head_dim)))
                _dflash2_scatter_block_kv_triton[(total_tokens, kv_head_num)](
                    k,
                    v,
                    act_d,
                    pos_d,
                    bt,
                    kv_hnd,
                    batch_size,
                    q_len,
                    kv_head_num,
                    self._page_size,
                    head_dim,
                    max_blocks,
                    num_blocks,
                    int(k.stride(0)),
                    int(k.stride(1)),
                    int(k.stride(2)),
                    int(v.stride(0)),
                    int(v.stride(1)),
                    int(v.stride(2)),
                    int(bt.stride(0)),
                    int(bt.stride(1)),
                    int(kv_hnd.stride(0)),
                    int(kv_hnd.stride(1)),
                    int(kv_hnd.stride(2)),
                    int(kv_hnd.stride(3)),
                    int(kv_hnd.stride(4)),
                    BLOCK_D=block_d,
                )

                block_q, block_g, block_m, num_splits = _choose_dflash2_tile_and_splits(
                    batch_size,
                    q_len,
                    kv_head_num,
                    gqa_groups,
                    max_blocks,
                    self._page_size,
                )
                num_q_tiles = (q_len + block_q - 1) // block_q
                num_g_tiles = (gqa_groups + block_g - 1) // block_g
                ws_m, ws_l, ws_acc = self.split_kv_workspace.ensure_capacity(
                    batch_size, head_num, q_len, num_splits, block_d, q.device
                )

                out = torch.empty_like(q)
                has_swa = sliding_window is not None and int(sliding_window) >= 0
                w_left = int(sliding_window) if has_swa else -1
                _dflash2_paged_noncausal_mha_stage1_triton[
                    (num_q_tiles * num_g_tiles, batch_size * kv_head_num, num_splits)
                ](
                    q,
                    kv_hnd,
                    bt,
                    pref_d,
                    pos_d,
                    act_d,
                    ws_m,
                    ws_l,
                    ws_acc,
                    sm_scale,
                    batch_size,
                    q_len,
                    kv_head_num,
                    gqa_groups,
                    num_g_tiles,
                    num_splits,
                    self._page_size,
                    head_dim,
                    max_blocks,
                    num_blocks,
                    w_left,
                    int(q.stride(0)),
                    int(q.stride(1)),
                    int(q.stride(2)),
                    int(bt.stride(0)),
                    int(bt.stride(1)),
                    int(kv_hnd.stride(0)),
                    int(kv_hnd.stride(1)),
                    int(kv_hnd.stride(2)),
                    int(kv_hnd.stride(3)),
                    int(kv_hnd.stride(4)),
                    int(ws_m.stride(0)),
                    int(ws_m.stride(1)),
                    int(ws_m.stride(2)),
                    int(ws_m.stride(3)),
                    int(ws_acc.stride(0)),
                    int(ws_acc.stride(1)),
                    int(ws_acc.stride(2)),
                    int(ws_acc.stride(3)),
                    int(ws_acc.stride(4)),
                    HAS_SWA=has_swa,
                    BLOCK_Q=block_q,
                    BLOCK_G=block_g,
                    BLOCK_M=block_m,
                    BLOCK_N=16,
                    BLOCK_D=block_d,
                    num_warps=4,
                    num_stages=3 if block_m <= 16 else 2,
                )

                has_gate = output_gate is not None
                gate_arg = output_gate.view(total_tokens, head_num, head_dim) if has_gate else out
                split_block = max(1, int(triton.next_power_of_2(num_splits)))
                _dflash2_paged_noncausal_mha_stage2_triton[(total_tokens, head_num)](
                    ws_m,
                    ws_l,
                    ws_acc,
                    act_d,
                    gate_arg,
                    out,
                    batch_size,
                    q_len,
                    head_num,
                    num_splits,
                    head_dim,
                    int(ws_m.stride(0)),
                    int(ws_m.stride(1)),
                    int(ws_m.stride(2)),
                    int(ws_m.stride(3)),
                    int(ws_acc.stride(0)),
                    int(ws_acc.stride(1)),
                    int(ws_acc.stride(2)),
                    int(ws_acc.stride(3)),
                    int(ws_acc.stride(4)),
                    int(gate_arg.stride(0)),
                    int(gate_arg.stride(1)),
                    int(gate_arg.stride(2)),
                    int(out.stride(0)),
                    int(out.stride(1)),
                    int(out.stride(2)),
                    HAS_GATE=has_gate,
                    SPLIT_BLOCK=split_block,
                    BLOCK_D=block_d,
                    num_warps=4,
                )
                return out

        # Eager PyTorch / CPU fallback path.
        prefix_lengths = (
            self.prefix_lengths
            if self.prefix_lengths is not None
            else _resolve_host_or_device(
                self.attn_inputs.prefix_lengths, self.attn_inputs.prefix_lengths_device
            )
        )
        if self.active_requests is not None and self.query_width is not None:
            batch_size = int(self.active_requests.numel())
            input_lengths = torch.full(
                (batch_size,), self.query_width, dtype=torch.int32, device=q.device
            )
        else:
            input_lengths = _resolve_host_or_device(
                self.attn_inputs.input_lengths, self.attn_inputs.input_lengths_device
            )
            batch_size = int(input_lengths.numel()) if input_lengths is not None else 1

        active_mask = (
            self.active_requests
            if self.active_requests is not None
            else torch.ones(batch_size, dtype=torch.bool, device=q.device)
        )
        if self.query_positions is not None:
            positions = self.query_positions.to(q.device)
        elif (
            self.attn_inputs.combo_position_ids is not None
            and self.attn_inputs.combo_position_ids.numel() == total_tokens
        ):
            positions = self.attn_inputs.combo_position_ids.to(q.device)
        elif input_lengths is not None and batch_size > 0:
            pos_parts: List[torch.Tensor] = []
            for b in range(batch_size):
                pref_len = int(prefix_lengths[b].item()) if prefix_lengths is not None else 0
                q_len = max(0, int(input_lengths[b].item()))
                pos_parts.append(
                    pref_len + torch.arange(q_len, device=q.device, dtype=torch.int32)
                )
            positions = (
                torch.cat(pos_parts, dim=0)
                if pos_parts
                else torch.empty((0,), device=q.device, dtype=torch.int32)
            )
            if positions.numel() != total_tokens:
                positions = torch.arange(total_tokens, device=q.device, dtype=torch.int32)
        else:
            positions = torch.arange(total_tokens, device=q.device, dtype=torch.int32)

        if kv_cache is None or getattr(kv_cache, "kv_cache_base", None) is None:
            out = torch.zeros_like(q)
            offset = 0
            for b in range(batch_size):
                q_len = int(input_lengths[b].item()) if input_lengths is not None else total_tokens
                if q_len <= 0 or not bool(active_mask[b].item()):
                    offset += max(q_len, 0)
                    continue
                qb = q[offset : offset + q_len].transpose(0, 1).unsqueeze(0)
                kb = k[offset : offset + q_len].transpose(0, 1).unsqueeze(0)
                vb = v[offset : offset + q_len].transpose(0, 1).unsqueeze(0)
                if gqa_groups > 1:
                    kb = kb.repeat_interleave(gqa_groups, dim=1)
                    vb = vb.repeat_interleave(gqa_groups, dim=1)
                mask = None
                if sliding_window is not None:
                    pos_slice = positions[offset : offset + q_len].long()
                    mask = pos_slice[None, :] >= (pos_slice[:, None] - int(sliding_window))
                ob = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask, is_causal=False)
                out[offset : offset + q_len] = ob.squeeze(0).transpose(0, 1)
                offset += q_len
            if output_gate is not None:
                out = out * torch.sigmoid(output_gate.view(total_tokens, head_num, head_dim))
            return out

        kv_hnd = _as_hnd_kv_pool(
            kv_cache.kv_cache_base, kv_head_num, self._page_size, head_dim
        )
        block_table = self._resolve_block_table(q.device)
        max_blocks = int(block_table.shape[1]) if block_table.ndim >= 2 else 0

        req_ids_list: List[torch.Tensor] = []
        for b in range(batch_size):
            q_len = int(input_lengths[b].item()) if input_lengths is not None else total_tokens
            if q_len > 0:
                rid = b if bool(active_mask[b].item()) else -1
                req_ids_list.append(
                    torch.full((q_len,), rid, dtype=torch.int32, device=q.device)
                )
        req_ids = (
            torch.cat(req_ids_list, dim=0)
            if req_ids_list
            else torch.arange(total_tokens, dtype=torch.int32, device=q.device)
        )
        scatter_kv_to_paged_cache(
            k, v, req_ids, positions, block_table, kv_hnd, self._page_size
        )

        out = torch.zeros_like(q)
        offset = 0
        for b in range(batch_size):
            q_len = int(input_lengths[b].item()) if input_lengths is not None else total_tokens
            if q_len <= 0 or not bool(active_mask[b].item()):
                offset += max(q_len, 0)
                continue
            pref_len = int(prefix_lengths[b].item()) if prefix_lengths is not None else 0
            total_kv = min(pref_len + q_len, max_blocks * self._page_size)
            if total_kv <= 0:
                offset += q_len
                continue
            kv_pos = torch.arange(total_kv, device=q.device, dtype=torch.long)
            pages = block_table[b, kv_pos // self._page_size].long()
            slots = kv_pos % self._page_size
            kb = kv_hnd[pages, 0, :, slots, :].transpose(0, 1).unsqueeze(0).to(q.dtype)
            vb = kv_hnd[pages, 1, :, slots, :].transpose(0, 1).unsqueeze(0).to(q.dtype)
            qb = q[offset : offset + q_len].transpose(0, 1).unsqueeze(0)
            if gqa_groups > 1:
                kb = kb.repeat_interleave(gqa_groups, dim=1)
                vb = vb.repeat_interleave(gqa_groups, dim=1)
            mask = None
            if sliding_window is not None:
                q_pos = positions[offset : offset + q_len].long()
                mask = kv_pos[None, :] >= (q_pos[:, None] - int(sliding_window))
            ob = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask, is_causal=False)
            out[offset : offset + q_len] = ob.squeeze(0).transpose(0, 1)
            offset += q_len
        if output_gate is not None:
            out = out * torch.sigmoid(output_gate.view(total_tokens, head_num, head_dim))
        return out


class Qwen4_5DFlashAttention(nn.Module):
    """Non-causal MQA-1 Attention with TP sharding, fused QKVG GEMM, Gemma +1 norms, SWA/Full."""

    def __init__(
        self,
        spec: DFlash2ArchSpec,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = spec.hidden_size
        self.tp_size = max(1, int(tp_size or 1))
        self.tp_rank = int(tp_rank or 0)
        self.tp_group = tp_group
        self.fast_all_reduce = fast_all_reduce

        self.num_heads_total = int(spec.num_attention_heads)
        self.num_kv_heads_total = int(spec.num_key_value_heads)
        if self.num_heads_total % self.tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads_total}) must be divisible by tp_size ({self.tp_size})"
            )
        self.num_heads = self.num_heads_total // self.tp_size
        if self.num_kv_heads_total >= self.tp_size and self.num_kv_heads_total % self.tp_size == 0:
            self.num_kv_heads = self.num_kv_heads_total // self.tp_size
            self._kv_sharded = self.tp_size > 1
        else:
            # MQA-1 (< tp_size) stays replicated on every TP rank (matches sp_head).
            self.num_kv_heads = self.num_kv_heads_total
            self._kv_sharded = False

        self.head_dim = spec.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.full_q_size = self.num_heads_total * self.head_dim
        self.full_kv_size = self.num_kv_heads_total * self.head_dim
        self.rotary_dim = spec.rotary_dim
        self.eps = spec.rms_norm_eps
        self.use_output_gate = spec.use_attn_output_gate
        self.zero_centered_block_norms = spec.zero_centered_block_norms
        layer_type = spec.layer_types[layer_idx]
        self.sliding_window: Optional[int] = (
            int(spec.swa_window_size) if layer_type == "sliding_attention" else None
        )

        qkv_t = weights[W.attn_qkv_w].t().contiguous()
        if (
            self.tp_size > 1
            and int(qkv_t.shape[0]) == self.full_q_size + 2 * self.full_kv_size
        ):
            q_full, k_full, v_full = torch.split(
                qkv_t, [self.full_q_size, self.full_kv_size, self.full_kv_size], dim=0
            )
            q_w = q_full[self.tp_rank * self.q_size : (self.tp_rank + 1) * self.q_size]
            if self._kv_sharded:
                k_w = k_full[self.tp_rank * self.kv_size : (self.tp_rank + 1) * self.kv_size]
                v_w = v_full[self.tp_rank * self.kv_size : (self.tp_rank + 1) * self.kv_size]
            else:
                k_w, v_w = k_full, v_full
        else:
            q_w, k_w, v_w = torch.split(qkv_t, [self.q_size, self.kv_size, self.kv_size], dim=0)

        o_t = weights[W.attn_o_w].t().contiguous()
        if self.tp_size > 1 and int(o_t.shape[1]) == self.full_q_size:
            o_t = o_t[:, self.tp_rank * self.q_size : (self.tp_rank + 1) * self.q_size].contiguous()

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False, dtype=dtype, device=device)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False, dtype=dtype, device=device)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False, dtype=dtype, device=device)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False, dtype=dtype, device=device)
        with torch.no_grad():
            self.q_proj.weight.copy_(q_w)
            self.k_proj.weight.copy_(k_w)
            self.v_proj.weight.copy_(v_w)
            self.o_proj.weight.copy_(o_t)

        if self.use_output_gate:
            self.output_gate_proj: Optional[nn.Linear] = nn.Linear(
                self.hidden_size, self.q_size, bias=False, dtype=dtype, device=device
            )
            gate_w = weights.get(DFLASH2_W.attn_output_gate)
            if gate_w is not None:
                if self.tp_size > 1 and int(gate_w.shape[0]) == self.full_q_size:
                    gate_w = gate_w[self.tp_rank * self.q_size : (self.tp_rank + 1) * self.q_size]
                with torch.no_grad():
                    self.output_gate_proj.weight.copy_(gate_w)
        else:
            self.output_gate_proj = None

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=self.eps)
        if W.q_ln_gamma in weights:
            self.q_norm.weight = nn.Parameter(weights[W.q_ln_gamma], requires_grad=False)
        if W.k_ln_gamma in weights:
            self.k_norm.weight = nn.Parameter(weights[W.k_ln_gamma], requires_grad=False)

        inv_freq = 1.0 / (
            float(spec.rope_theta)
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._qkvg_versions: Tuple[int, ...] = ()
        self.register_buffer("_fused_qkvg_weight", None, persistent=False)
        self.prepare_weights()

    def _weight_versions(self) -> Tuple[int, ...]:
        if self.output_gate_proj is not None:
            return (
                int(self.q_proj.weight._version),
                int(self.k_proj.weight._version),
                int(self.v_proj.weight._version),
                int(self.output_gate_proj.weight._version),
            )
        return (
            int(self.q_proj.weight._version),
            int(self.k_proj.weight._version),
            int(self.v_proj.weight._version),
        )

    def prepare_weights(self) -> torch.Tensor:
        """Pack [q_proj, k_proj, v_proj(, output_gate_proj)] into a single contiguous weight."""
        parts = [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]
        if self.output_gate_proj is not None:
            parts.append(self.output_gate_proj.weight)
        self._fused_qkvg_weight = torch.cat(parts, dim=0).contiguous()
        self._qkvg_versions = self._weight_versions()
        return self._fused_qkvg_weight

    def _get_fused_qkvg_weight(self) -> torch.Tensor:
        if (
            self._fused_qkvg_weight is None
            or self._fused_qkvg_weight.device != self.q_proj.weight.device
            or self._weight_versions() != self._qkvg_versions
        ):
            return self.prepare_weights()
        return self._fused_qkvg_weight

    def _norm_and_rope_qk(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_2d = positions.unsqueeze(0)
        q_4d = q.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        if self.zero_centered_block_norms:
            q_rot = gemma_rmsnorm_and_partial_rope(
                q_4d, self.q_norm.weight, pos_2d, self.inv_freq, self.rotary_dim, self.eps
            )
            k_rot = gemma_rmsnorm_and_partial_rope(
                k_4d, self.k_norm.weight, pos_2d, self.inv_freq, self.rotary_dim, self.eps
            )
        else:
            qn = _apply_rmsnorm(q_4d, self.q_norm.weight, self.eps)
            kn = _apply_rmsnorm(k_4d, self.k_norm.weight, self.eps)
            q_rot = apply_partial_neox_rope(qn, pos_2d, self.inv_freq, self.rotary_dim)
            k_rot = apply_partial_neox_rope(kn, pos_2d, self.inv_freq, self.rotary_dim)
        return q_rot.squeeze(0), k_rot.squeeze(0)

    def forward_context_kv(
        self, main_x: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_kv_w = self._get_fused_qkvg_weight()[self.q_size : self.q_size + 2 * self.kv_size]
        kv = F.linear(main_x, fused_kv_w)
        k_raw, v_raw = torch.split(kv, [self.kv_size, self.kv_size], dim=-1)
        k = k_raw.reshape(main_x.size(0), self.num_kv_heads, self.head_dim)
        v = v_raw.reshape(main_x.size(0), self.num_kv_heads, self.head_dim)
        pos_2d = positions.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        if self.zero_centered_block_norms:
            k_rot = gemma_rmsnorm_and_partial_rope(
                k_4d, self.k_norm.weight, pos_2d, self.inv_freq, self.rotary_dim, self.eps
            ).squeeze(0)
        else:
            kn = _apply_rmsnorm(k_4d, self.k_norm.weight, self.eps)
            k_rot = apply_partial_neox_rope(kn, pos_2d, self.inv_freq, self.rotary_dim).squeeze(0)
        return k_rot, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        fmha_impl: _DFlash2PagedFMHADispatcher,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        T = hidden_states.size(0)
        qkvg = F.linear(hidden_states, self._get_fused_qkvg_weight())
        if self.use_output_gate:
            q_raw, k_raw, v_raw, gate_raw = torch.split(
                qkvg, [self.q_size, self.kv_size, self.kv_size, self.q_size], dim=-1
            )
            gate: Optional[torch.Tensor] = gate_raw.reshape(T, self.num_heads, self.head_dim)
        else:
            q_raw, k_raw, v_raw = torch.split(
                qkvg, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            gate = None

        q = q_raw.reshape(T, self.num_heads, self.head_dim)
        k = k_raw.reshape(T, self.num_kv_heads, self.head_dim)
        v = v_raw.reshape(T, self.num_kv_heads, self.head_dim)

        q, k = self._norm_and_rope_qk(q, k, positions)
        attn_out = fmha_impl.append_and_attend(
            q,
            k,
            v,
            kv_cache,
            sliding_window=self.sliding_window,
            output_gate=gate,
        )
        attn_flat = attn_out.reshape(T, self.q_size).contiguous()
        out = self.o_proj(attn_flat)
        if self.tp_size > 1:
            if self.fast_all_reduce is not None:
                out = self.fast_all_reduce(out)
            elif self.tp_group is not None and dist.is_available() and dist.is_initialized():
                out = out.contiguous()
                dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        return out


def _copy_if_present(dst: Optional[torch.Tensor], src: Optional[torch.Tensor]) -> None:
    if dst is not None and src is not None:
        with torch.no_grad():
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


class Qwen4_5DFlash2DecoderLayer(nn.Module):
    """DFlash2 layer: 4-stream HC + DFlashGroupedConv + SWA/Full Attn + LatentMoE."""

    def __init__(
        self,
        spec: DFlash2ArchSpec,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group: Any = None,
        fast_all_reduce: Any = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.use_cuda_triton = bool(device is not None and device.type == "cuda")
        self.clip_mlp_output = spec.clip_mlp_output

        self.attn_hyper_connection = Qwen4_5HyperConnection(
            hidden_size=spec.hidden_size,
            hc_count=spec.hc_count,
            hc_lowrank=spec.hc_lowrank,
            eps=spec.rms_norm_eps,
            use_combine=True,
            prev_mix=spec.hc_prev_mix,
            is_head=bool(spec.hc_prev_mix and layer_idx == 0),
            emits_latent=True,
            dtype=dtype,
            device=device,
        )
        _copy_if_present(self.attn_hyper_connection.hc_norm_weight, weights.get(DFLASH2_W.attn_hc_norm_w))
        _copy_if_present(self.attn_hyper_connection.hc_norm_bias, weights.get(DFLASH2_W.attn_hc_norm_b))
        _copy_if_present(self.attn_hyper_connection.block_inject_weight, weights.get(DFLASH2_W.attn_hc_inject))
        _copy_if_present(self.attn_hyper_connection.input_mix_weight_down, weights.get(DFLASH2_W.attn_hc_down))
        _copy_if_present(self.attn_hyper_connection.input_mix_weight_up, weights.get(DFLASH2_W.attn_hc_up))

        self.attention_conv = DFlashGroupedConv(
            hidden_size=spec.hidden_size,
            taps=spec.conv_kernel_size,
            group_size=spec.conv_group_size,
            block_size=spec.query_width,
            identity_folded=spec.conv_base_kernel_identity_folded,
            dtype=dtype,
            device=device,
        )
        _copy_if_present(self.attention_conv.base_kernel, weights.get(DFLASH2_W.attn_conv_base))
        _copy_if_present(self.attention_conv.kernel_projection.weight, weights.get(DFLASH2_W.attn_conv_proj))

        self.self_attn = Qwen4_5DFlashAttention(
            spec,
            layer_idx,
            weights,
            dtype=dtype,
            device=device,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            fast_all_reduce=fast_all_reduce,
        )

        emits_mlp_latent = (not spec.hc_prev_mix) or (layer_idx + 1 < spec.num_hidden_layers)
        self.mlp_hyper_connection = Qwen4_5HyperConnection(
            hidden_size=spec.hidden_size,
            hc_count=spec.hc_count,
            hc_lowrank=spec.hc_lowrank,
            eps=spec.rms_norm_eps,
            use_combine=True,
            prev_mix=spec.hc_prev_mix,
            is_head=False,
            emits_latent=emits_mlp_latent,
            dtype=dtype,
            device=device,
        )
        _copy_if_present(self.mlp_hyper_connection.hc_norm_weight, weights.get(DFLASH2_W.mlp_hc_norm_w))
        _copy_if_present(self.mlp_hyper_connection.hc_norm_bias, weights.get(DFLASH2_W.mlp_hc_norm_b))
        _copy_if_present(self.mlp_hyper_connection.block_inject_weight, weights.get(DFLASH2_W.mlp_hc_inject))
        _copy_if_present(self.mlp_hyper_connection.input_mix_weight_down, weights.get(DFLASH2_W.mlp_hc_down))
        _copy_if_present(self.mlp_hyper_connection.input_mix_weight_up, weights.get(DFLASH2_W.mlp_hc_up))

        self.mlp_conv = DFlashGroupedConv(
            hidden_size=spec.hidden_size,
            taps=spec.conv_kernel_size,
            group_size=spec.conv_group_size,
            block_size=spec.query_width,
            identity_folded=spec.conv_base_kernel_identity_folded,
            dtype=dtype,
            device=device,
        )
        _copy_if_present(self.mlp_conv.base_kernel, weights.get(DFLASH2_W.mlp_conv_base))
        _copy_if_present(self.mlp_conv.kernel_projection.weight, weights.get(DFLASH2_W.mlp_conv_proj))

        moe_extra_kwargs: Dict[str, Any] = {}
        try:
            moe_sig = inspect.signature(Qwen4_5LatentMoE.__init__).parameters
            if "fast_all_reduce" in moe_sig:
                moe_extra_kwargs["fast_all_reduce"] = fast_all_reduce
        except Exception:
            pass
        self.mlp = Qwen4_5LatentMoE(
            hidden_size=spec.hidden_size,
            routed_expert_hidden_size=spec.routed_expert_hidden_size,
            moe_intermediate_size=spec.moe_intermediate_size,
            shared_expert_intermediate_size=spec.shared_expert_intermediate_size,
            num_experts=spec.num_experts,
            num_experts_per_tok=spec.num_experts_per_tok,
            rms_norm_eps=spec.rms_norm_eps,
            gate_beta=spec.hidden_act_gate_beta,
            up_beta=spec.hidden_act_up_beta,
            shared_expert_gate=spec.shared_expert_gate,
            router_precision=spec.router_precision,
            router_normalize_topk_scores=spec.router_normalize_topk_scores,
            moe_router_enable_expert_bias=spec.moe_router_enable_expert_bias,
            use_triton_mxfp4=self.use_cuda_triton,
            dtype=dtype,
            device=device,
            init_mxfp4_weights=False,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
            **moe_extra_kwargs,
        )
        _copy_if_present(self.mlp.latent_down_proj.weight, weights.get(DFLASH2_W.moe_latent_down))
        _copy_if_present(self.mlp.latent_gate_proj.weight, weights.get(DFLASH2_W.moe_latent_gate))
        _copy_if_present(self.mlp.latent_up_proj.weight, weights.get(DFLASH2_W.moe_latent_up))
        _copy_if_present(self.mlp.latent_scale_gate.weight, weights.get(DFLASH2_W.moe_latent_scale_gate))
        _copy_if_present(self.mlp.pre_moe_latent_layernorm.weight, weights.get(DFLASH2_W.moe_pre_latent_ln_w))
        _copy_if_present(self.mlp.gate.weight, weights.get(DFLASH2_W.moe_gate_w))
        _copy_if_present(self.mlp.expert_bias, weights.get(DFLASH2_W.moe_expert_bias))
        _copy_if_present(self.mlp.shared_expert.gate_proj.weight, weights.get(DFLASH2_W.moe_shared_gate_proj))
        _copy_if_present(self.mlp.shared_expert.up_proj.weight, weights.get(DFLASH2_W.moe_shared_up_proj))
        _copy_if_present(self.mlp.shared_expert.down_proj.weight, weights.get(DFLASH2_W.moe_shared_down_proj))
        _copy_if_present(
            self.mlp.shared_expert_gate.weight if self.mlp.shared_expert_gate is not None else None,
            weights.get(DFLASH2_W.moe_shared_expert_gate),
        )

        ep_range = range(self.mlp.ep_start, self.mlp.ep_end)
        if DFLASH2_W.moe_expert_gate_w.format(e=self.mlp.ep_start) in weights:
            w_gate_u8 = torch.stack(
                [weights[DFLASH2_W.moe_expert_gate_w.format(e=e)] for e in ep_range],
                dim=0,
            )
            w_gate_s = torch.stack(
                [weights[DFLASH2_W.moe_expert_gate_s.format(e=e)] for e in ep_range],
                dim=0,
            )
            w_up_u8 = torch.stack(
                [weights[DFLASH2_W.moe_expert_up_w.format(e=e)] for e in ep_range],
                dim=0,
            )
            w_up_s = torch.stack(
                [weights[DFLASH2_W.moe_expert_up_s.format(e=e)] for e in ep_range],
                dim=0,
            )
            w2_u8 = torch.stack(
                [weights[DFLASH2_W.moe_expert_down_w.format(e=e)] for e in ep_range],
                dim=0,
            )
            w2_s = torch.stack(
                [weights[DFLASH2_W.moe_expert_down_s.format(e=e)] for e in ep_range],
                dim=0,
            )
            self.mlp.load_mxfp4_weights(
                w_gate_u8=w_gate_u8,
                w_gate_scale=w_gate_s,
                w_up_u8=w_up_u8,
                w_up_scale=w_up_s,
                w2_u8=w2_u8,
                w2_scale=w2_s,
            )
            self.mlp.prepare_weights(fuse_shared_gate_up=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        fmha_impl: _DFlash2PagedFMHADispatcher,
        kv_cache: Optional[LayerKVCache],
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        latent: Optional[torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        use_cuda = hidden_states.is_cuda
        if deferred is None:
            if use_cuda:
                mixed, residual, inject, latent = self.attn_hyper_connection.mix_with_latent(
                    hidden_states, latent
                )
            else:
                mixed, residual, inject, latent = self.attn_hyper_connection.mix_with_latent_ref(
                    hidden_states, latent
                )
        else:
            if use_cuda:
                mixed, residual, inject, latent = (
                    self.attn_hyper_connection.combine_and_mix_with_latent(
                        *deferred, prev_latent=latent
                    )
                )
            else:
                mixed, residual, inject, latent = (
                    self.attn_hyper_connection.combine_and_mix_with_latent_ref(
                        *deferred, prev_latent=latent
                    )
                )

        mixed, attn_coeff = self.attention_conv.prepare(mixed)
        attended = self.self_attn(mixed, positions, fmha_impl, kv_cache)
        attended = self.attention_conv.finish(attended, attn_coeff)

        if use_cuda:
            mixed, residual, inject, latent = (
                self.mlp_hyper_connection.combine_and_mix_with_latent(
                    attended, residual, inject, prev_latent=latent
                )
            )
        else:
            mixed, residual, inject, latent = (
                self.mlp_hyper_connection.combine_and_mix_with_latent_ref(
                    attended, residual, inject, prev_latent=latent
                )
            )

        mixed, mlp_coeff = self.mlp_conv.prepare(mixed)
        output = self.mlp_conv.finish(
            self.mlp(mixed, use_triton_mxfp4=use_cuda), mlp_coeff
        )
        if self.clip_mlp_output:
            output = output.clamp(-self.clip_mlp_output, self.clip_mlp_output)
        return (output, residual, inject), latent


class Qwen4_5DSparkModel(DSparkProposerMixin, GptModelBase):
    """5-layer hybrid non-causal DFlashQwenDraftModel for Qwen4.5-Flash."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        fmha_config: Optional[Any] = None,
        py_hw_kernel_config: Optional[Any] = None,
    ) -> None:
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
        )
        spec = getattr(config, "_dflash2_spec", None)
        if not isinstance(spec, DFlash2ArchSpec):
            spec = DFlash2ArchSpec(
                num_hidden_layers=int(config.num_layers),
                hidden_size=int(config.hidden_size),
                vocab_size=int(config.vocab_size),
                num_attention_heads=int(config.attn_config.head_num),
                num_key_value_heads=int(config.attn_config.kv_head_num),
                head_dim=int(config.attn_config.size_per_head),
                selector_rank=int(getattr(config, "dspark_markov_rank", 256) or 256),
            )
        self.spec = spec

        tp_size = int(getattr(parallelism_config, "tp_size", 1) or 1)
        tp_rank = int(getattr(parallelism_config, "tp_rank", 0) or 0)
        dp_size = int(getattr(parallelism_config, "dp_size", 1) or 1)
        ep_size_cfg = int(getattr(parallelism_config, "ep_size", 1) or 1)
        self.tp_size = max(1, tp_size)
        self.tp_rank = tp_rank

        global_weights = weights.global_weights
        embed_orig = global_weights[W.embedding]
        dtype = embed_orig.dtype
        device = embed_orig.device
        self._use_cuda_ops = bool(embed_orig.is_cuda)

        # Use Embedding (with TP all_gather) only when W.embedding is already sharded (sp_neg1).
        # If W.embedding is already replicated [V, H], avoid allocating a sliced copy or doing CG all_gather.
        self.embed_weight = embed_orig
        embed_is_sharded = self.tp_size > 1 and int(embed_orig.shape[-1]) != int(spec.hidden_size)
        self.embed_tokens = (
            Embedding(config, parallelism_config, embed_orig)
            if (
                self._use_cuda_ops
                and (not embed_is_sharded or _is_collective_torch_initialized())
                and (self.tp_size == 1 or embed_is_sharded)
            )
            else None
        )

        # Resolve LM head weight (sp_0_pad8 when ModelLoader loaded W.lm_head, or defensive TP row slice).
        raw_lm_head = global_weights.get(W.lm_head)
        if raw_lm_head is None:
            raw_lm_head = embed_orig
        if self.tp_size > 1 and int(raw_lm_head.shape[0]) == int(spec.vocab_size) and int(raw_lm_head.shape[-1]) == int(spec.hidden_size):
            per_rank_pad8 = ((spec.vocab_size + self.tp_size - 1) // self.tp_size + 7) // 8 * 8
            v_start = min(tp_rank * per_rank_pad8, spec.vocab_size)
            v_end = min((tp_rank + 1) * per_rank_pad8, spec.vocab_size)
            lm_slice = raw_lm_head[v_start:v_end]
            if int(lm_slice.shape[0]) < per_rank_pad8:
                pad_rows = per_rank_pad8 - int(lm_slice.shape[0])
                lm_slice = F.pad(lm_slice, (0, 0, 0, pad_rows))
            self.lm_head_weight = lm_slice.contiguous()
            self.lm_head_vocab_start = v_start
            self.lm_head_vocab_end = v_end
            self.lm_head_sharded = True
        elif self.tp_size > 1 and int(raw_lm_head.shape[0]) != int(spec.vocab_size):
            per_rank = int(raw_lm_head.shape[0])
            self.lm_head_weight = raw_lm_head
            self.lm_head_vocab_start = min(tp_rank * per_rank, int(spec.vocab_size))
            self.lm_head_vocab_end = min((tp_rank + 1) * per_rank, int(spec.vocab_size))
            self.lm_head_sharded = True
        else:
            self.lm_head_weight = raw_lm_head
            self.lm_head_vocab_start = 0
            self.lm_head_vocab_end = int(spec.vocab_size)
            self.lm_head_sharded = False

        self.input_embedding_scale = float(spec.input_embedding_scale)

        self.fc = nn.Linear(
            spec.fc_in_features, spec.hidden_size, bias=False, dtype=dtype, device=device
        )
        with torch.no_grad():
            self.fc.weight.copy_(global_weights[W.dspark_fc_w].t().contiguous())
        self.hidden_norm_weight = global_weights[W.dspark_hidden_norm_gamma]
        self.hidden_norm_zero_centered = spec.zero_centered_hidden_norm

        tp_group = _resolve_tp_group(self.tp_size)
        self.tp_group = tp_group

        eff_ep_size = 1
        eff_ep_rank = 0
        ep_group: Any = None
        if self.tp_size > 1 and dp_size == 1:
            eff_ep_size = self.tp_size
            eff_ep_rank = tp_rank
            ep_group = tp_group
        elif ep_size_cfg > 1:
            eff_ep_size = ep_size_cfg
            eff_ep_rank = int(getattr(parallelism_config, "ep_rank", 0) or 0)
            ep_group = _resolve_world_group()

        self.eff_ep_size = eff_ep_size
        self.eff_ep_rank = eff_ep_rank
        self.ep_group = ep_group

        self.fast_all_reduce = None
        if (
            self.tp_size > 1
            and tp_group is not None
            and device.type == "cuda"
            and dist.is_available()
            and dist.is_initialized()
        ):
            self.fast_all_reduce = get_qwen4_5_fast_all_reduce(
                tp_group,
                device=device,
                dtype=dtype,
                hidden_dims=(int(spec.hidden_size), int(spec.routed_expert_hidden_size)),
            )

        self.layers = nn.ModuleList(
            [
                Qwen4_5DFlash2DecoderLayer(
                    spec,
                    layer_idx=layer_idx,
                    weights=weights.weights[layer_idx],
                    dtype=dtype,
                    device=device,
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                    tp_group=tp_group,
                    ep_size=eff_ep_size,
                    ep_rank=eff_ep_rank,
                    ep_group=ep_group,
                    fast_all_reduce=self.fast_all_reduce,
                )
                for layer_idx in range(spec.num_hidden_layers)
            ]
        )

        self.hyper_connection_mixer: Optional[Qwen4_5HyperConnection] = None
        if spec.use_final_hc_mixer:
            self.hyper_connection_mixer = Qwen4_5HyperConnection(
                hidden_size=spec.hidden_size,
                hc_count=spec.hc_count,
                hc_lowrank=spec.hc_lowrank,
                eps=spec.rms_norm_eps,
                use_combine=False,
                prev_mix=False,
                is_head=False,
                emits_latent=True,
                dtype=dtype,
                device=device,
            )
            _copy_if_present(self.hyper_connection_mixer.hc_norm_weight, global_weights.get(DFLASH2_W.hc_mixer_norm_w))
            _copy_if_present(self.hyper_connection_mixer.hc_norm_bias, global_weights.get(DFLASH2_W.hc_mixer_norm_b))
            _copy_if_present(self.hyper_connection_mixer.input_mix_weight_down, global_weights.get(DFLASH2_W.hc_mixer_down))
            _copy_if_present(self.hyper_connection_mixer.input_mix_weight_up, global_weights.get(DFLASH2_W.hc_mixer_up))

        self.candidate_selector = CandidateSelector(
            hidden_size=spec.hidden_size,
            vocab_size=spec.vocab_size,
            rank=spec.selector_rank,
            top_k=spec.selector_top_k,
            predecessor_codebook=global_weights[W.dspark_markov_w1],
            successor_codebook=global_weights[W.dspark_markov_w2],
            dtype=dtype,
            device=device,
        )
        _copy_if_present(
            self.candidate_selector.hidden_projection.weight,
            global_weights.get(DFLASH2_W.selector_hidden_proj),
        )

        self.init_dspark_proposer(
            width=spec.block_size,
            query_width=spec.query_width,
            noise_token_id=spec.mask_token_id,
            aux_feature_dim=spec.fc_in_features,
            hidden_dim=spec.hidden_size,
        )
        local_num_heads = self.layers[0].self_attn.num_heads if self.layers else (spec.num_attention_heads // self.tp_size)
        init_batch_cap = max(8, min(32 if local_num_heads > 16 else 64, int(max_generate_batch_size or 16)))
        self._mha_workspace = _DFlash2SplitKVWorkspace(
            max_batch=init_batch_cap,
            num_heads=local_num_heads,
            query_len=max(1, int(spec.query_width)),
            max_splits=_DFLASH2_MAX_SPLITS,
            head_dim=max(16, int(spec.head_dim)),
            device=device,
        )
        # Freeze inference parameters so Qwen4_5HyperConnection uses cached fused down+inject weights
        # instead of launching torch.cat on every forward pass inside CUDA graphs.
        self.requires_grad_(False)
        for layer in self.layers:
            for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
                if hc.emits_latent and hc.use_combine and hasattr(hc, "_get_fused_down_inject_weight"):
                    hc._get_fused_down_inject_weight()

    def cuda_graph_input_hidden_size(self) -> int:
        return int(self._dspark_aux_feature_dim)

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is not None and token_ids.is_cuda:
            emb = self.embed_tokens(token_ids)
        else:
            emb = F.embedding(token_ids.long(), self.embed_weight)
            if (
                self.tp_size > 1
                and int(emb.shape[-1]) != int(self.spec.hidden_size)
                and self.tp_group is not None
                and dist.is_available()
                and dist.is_initialized()
            ):
                gathered = [torch.empty_like(emb) for _ in range(self.tp_size)]
                dist.all_gather(gathered, emb.contiguous(), group=self.tp_group)
                emb = torch.cat(gathered, dim=-1)
        if self.input_embedding_scale != 1.0:
            emb = emb * self.input_embedding_scale
        return emb

    def dspark_attention_inputs(self, inputs: PyModelInputs) -> Any:
        by_tag = getattr(inputs, "attention_inputs_by_tag", None)
        if isinstance(by_tag, Mapping) and len(by_tag) > 0:
            if FULL_ATTN_KV_TAG in by_tag:
                return by_tag[FULL_ATTN_KV_TAG]
            return next(iter(by_tag.values()))
        return _select_tagged_attn_inputs(inputs.attention_inputs)

    def _get_layer_full_kv(self, layer_idx: int) -> Optional[Any]:
        if self.kv_cache is None:
            return None
        if hasattr(self.kv_cache, "get_layer_cache_groups"):
            groups = self.kv_cache.get_layer_cache_groups(layer_idx)
            for grp in groups:
                if getattr(grp, "tag", None) == FULL_ATTN_KV_TAG:
                    return grp
            if groups:
                return groups[0]
        return self.kv_cache.get_layer_cache(layer_idx)

    def _has_cuda_layer_kv_cache(self) -> bool:
        if self.kv_cache is None or not self.layers:
            return False
        layer_kv = self._get_layer_full_kv(0)
        return (
            layer_kv is not None
            and getattr(layer_kv, "kv_cache_base", None) is not None
            and layer_kv.kv_cache_base.is_cuda
        )

    def _resolve_kv_page_size(self) -> int:
        if self.kv_cache is not None and hasattr(
            self.kv_cache, "get_kernel_seq_size_per_block"
        ):
            raw_ps = self.kv_cache.get_kernel_seq_size_per_block
            if callable(raw_ps):
                try:
                    return max(1, int(raw_ps(FULL_ATTN_KV_TAG)))
                except TypeError:
                    return max(1, int(raw_ps()))
            return max(1, int(raw_ps))
        return max(1, int(self.config.attn_config.kernel_tokens_per_block or 16))

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ) -> _DFlash2PagedFMHADispatcher:
        del is_cuda_graph, cuda_graph_selection_mode
        attn_inputs = self.dspark_attention_inputs(inputs)
        self.config.attn_config.kernel_tokens_per_block = self._resolve_kv_page_size()
        use_triton_paged = bool(
            self._use_cuda_ops and _HAS_TRITON and self._has_cuda_layer_kv_cache()
        )
        return _DFlash2PagedFMHADispatcher(
            self.config.attn_config,
            attn_inputs,
            use_triton_paged=use_triton_paged,
            split_kv_workspace=self._mha_workspace,
        )

    def combine_hidden_states(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.fc(features.to(self.fc.weight.dtype))
        if self.hidden_norm_zero_centered:
            return _apply_gemma_rmsnorm(
                projected, self.hidden_norm_weight, self.spec.rms_norm_eps
            )
        return _apply_rmsnorm(
            projected, self.hidden_norm_weight, self.spec.rms_norm_eps
        )

    def commit_feature_rows(
        self,
        main_x: torch.Tensor,
        context_req_ids: torch.Tensor,
        context_positions: torch.Tensor,
        committed_ends: torch.Tensor,
        inputs: PyModelInputs,
        commit_ctx: Any = None,
    ) -> None:
        del committed_ends, commit_ctx
        if self.kv_cache is None or main_x.size(0) == 0:
            return

        attn_inputs = self.dspark_attention_inputs(inputs)
        block_table = device_metadata_tensor(attn_inputs, "kv_cache_kernel_block_id")
        if block_table is None:
            block_table = _resolve_host_or_device(
                getattr(attn_inputs, "kv_cache_kernel_block_id_device", None),
                getattr(attn_inputs, "kv_cache_kernel_block_id", None),
            )
        if block_table is None:
            raise ValueError("DSpARK commit requires kv_cache_kernel_block_id")
        block_table = block_table if block_table.device == main_x.device else block_table.to(main_x.device)

        page_size = self._resolve_kv_page_size()

        req_ids_dev = context_req_ids if context_req_ids.device == main_x.device else context_req_ids.to(main_x.device)
        pos_dev = context_positions if context_positions.device == main_x.device else context_positions.to(main_x.device)
        safe_pos = pos_dev.clamp(min=0)

        cache_store_writer = None
        if create_write_cache_store_impl is not None:
            cache_store_writer = create_write_cache_store_impl(attn_inputs, self.kv_cache)

        for layer_idx, layer in enumerate(self.layers):
            layer_kv = self._get_layer_full_kv(layer_idx)
            if layer_kv is None or getattr(layer_kv, "kv_cache_base", None) is None:
                continue
            k, v = layer.self_attn.forward_context_kv(main_x, safe_pos)
            scatter_kv_to_paged_cache(
                k,
                v,
                req_ids_dev,
                pos_dev,
                block_table,
                layer_kv.kv_cache_base,
                page_size,
            )
            if cache_store_writer is not None:
                cache_store_writer(layer_kv)

    def forward_query_block(
        self,
        query_ids: torch.Tensor,
        query_positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
        inputs: PyModelInputs,
        fmha_impl: Any,
    ) -> torch.Tensor:
        flat_ids = query_ids.reshape(-1)
        flat_pos = query_positions.reshape(-1)
        input_embeds = self._embed_tokens(flat_ids)
        hidden = input_embeds.repeat(1, self.spec.hc_count)

        attn_inputs = self.dspark_attention_inputs(inputs)
        active_fmha = fmha_impl
        if active_fmha is None or not isinstance(
            active_fmha, _DFlash2PagedFMHADispatcher
        ):
            active_fmha = _DFlash2PagedFMHADispatcher(
                self.config.attn_config,
                attn_inputs,
                use_triton_paged=bool(
                    self._use_cuda_ops
                    and _HAS_TRITON
                    and self._has_cuda_layer_kv_cache()
                ),
                query_positions=flat_pos,
                prefix_lengths=prefix_lengths,
                active_requests=active_requests,
                query_width=self._dspark_query_width,
                split_kv_workspace=self._mha_workspace,
            )
        else:
            active_fmha.attn_inputs = _select_tagged_attn_inputs(attn_inputs)
            active_fmha.query_positions = flat_pos
            active_fmha.prefix_lengths = prefix_lengths
            active_fmha.active_requests = active_requests
            active_fmha.query_width = self._dspark_query_width

        deferred = None
        latent = None
        for layer_idx, layer in enumerate(self.layers):
            layer_kv = self._get_layer_full_kv(layer_idx)
            deferred, latent = layer(
                hidden,
                flat_pos.clamp(min=0),
                active_fmha,
                layer_kv,
                deferred,
                latent,
            )

        assert deferred is not None
        if self.hyper_connection_mixer is not None:
            if hidden.is_cuda:
                mixed, _, _, _ = self.hyper_connection_mixer.combine_and_mix_with_latent(
                    *deferred
                )
            else:
                mixed, _, _, _ = (
                    self.hyper_connection_mixer.combine_and_mix_with_latent_ref(
                        *deferred
                    )
                )
            return mixed
        out, residual, _ = deferred
        return (
            residual.unflatten(-1, (self.spec.hc_count, self.spec.hidden_size)).mean(dim=-2)
            + out
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits_local = F.linear(
            hidden_states.to(self.lm_head_weight.dtype), self.lm_head_weight
        ).float()
        if (
            self.tp_size > 1
            and self.lm_head_sharded
            and self.tp_group is not None
            and dist.is_available()
            and dist.is_initialized()
        ):
            gathered = [torch.empty_like(logits_local) for _ in range(self.tp_size)]
            dist.all_gather(gathered, logits_local.contiguous(), group=self.tp_group)
            logits_full = torch.cat(gathered, dim=-1)
            return logits_full[:, : int(self.spec.vocab_size)].contiguous()
        return logits_local[:, : int(self.spec.vocab_size)]

    def vocab_parallel_topk(
        self, hidden_states: torch.Tensor, k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vocab-parallel distributed Top-K gathering only [T, k] per rank."""
        top_k = int(k if k is not None else self.candidate_selector.top_k)
        logits_local = F.linear(
            hidden_states.to(self.lm_head_weight.dtype), self.lm_head_weight
        ).float()
        valid_len = max(0, min(int(logits_local.shape[-1]), self.lm_head_vocab_end - self.lm_head_vocab_start))
        if valid_len < int(logits_local.shape[-1]):
            logits_local = logits_local.clone()
            logits_local[:, valid_len:] = float("-inf")

        local_k = min(top_k, int(logits_local.shape[-1]))
        local_vals, local_idx = torch.topk(logits_local, local_k, dim=-1)
        local_ids = local_idx + int(self.lm_head_vocab_start)

        if (
            self.tp_size > 1
            and self.lm_head_sharded
            and self.tp_group is not None
            and dist.is_available()
            and dist.is_initialized()
        ):
            gathered_vals = [torch.empty_like(local_vals) for _ in range(self.tp_size)]
            gathered_ids = [torch.empty_like(local_ids) for _ in range(self.tp_size)]
            dist.all_gather(gathered_vals, local_vals.contiguous(), group=self.tp_group)
            dist.all_gather(gathered_ids, local_ids.contiguous(), group=self.tp_group)
            all_vals = torch.cat(gathered_vals, dim=-1)
            all_ids = torch.cat(gathered_ids, dim=-1)
            top_vals, top_pos = torch.topk(all_vals, top_k, dim=-1)
            top_ids = torch.gather(all_ids, -1, top_pos)
            return top_vals, top_ids
        return local_vals, local_ids

    def propose_with_candidate_selector(
        self,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Vocab-Parallel Top-K logits and run DFlash2 CandidateSelector greedy walk."""
        unary_logits, candidate_ids = self.vocab_parallel_topk(
            hidden_states, k=self.candidate_selector.top_k
        )
        return self.candidate_selector.select_candidates_from_topk(
            unary_logits, candidate_ids, hidden_states, anchor_token_ids
        )

    def forward_commit(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        del fmha_impl
        return self.run_commit_step(inputs, self.embed_weight.device)

    def forward_propose(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        return self.run_propose_step(inputs, fmha_impl, self.embed_weight.device)

    def forward(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        raise RuntimeError(
            "Qwen4_5DSparkModel requires forward_commit or forward_propose"
        )


def extract_qwen4_5_aux_hidden_states(
    layer_hidden_states: Union[Sequence[torch.Tensor], Mapping[int, torch.Tensor]],
    target_layer_ids: Sequence[int] = (0, 5, 10, 15, 20),
    *,
    reduction: str = "concat",
    source: str = "post_next_hc_mix",
) -> torch.Tensor:
    """Collect target hidden states (``post_next_hc_mix``) for DFlash2."""
    if source != "post_next_hc_mix":
        logger.warning("Unexpected DFlash2 target_hidden_source=%r", source)
    if not target_layer_ids:
        raise ValueError("target_layer_ids must be non-empty")
    selected: List[torch.Tensor] = []
    is_mapping = isinstance(layer_hidden_states, Mapping)
    for lid in target_layer_ids:
        if is_mapping:
            if lid not in layer_hidden_states:
                raise KeyError(f"Missing target hidden state for layer {lid}")
            selected.append(layer_hidden_states[lid])
        else:
            if lid < 0 or lid >= len(layer_hidden_states):
                raise IndexError(
                    f"Target layer id {lid} out of bounds for {len(layer_hidden_states)} layers"
                )
            selected.append(layer_hidden_states[lid])

    if reduction == "concat":
        return torch.cat(selected, dim=-1).contiguous()
    if reduction == "mean":
        stacked = torch.stack([t.float() for t in selected], dim=0)
        return stacked.mean(dim=0).to(selected[0].dtype).contiguous()
    raise ValueError(f"Unsupported aux hidden reduction: {reduction!r}")


@dataclass
class _OpaquePoolHandle:
    key: str
    tensor: torch.Tensor


def discover_qwen4_5_opaque_pools(
    state_container: Any,
    *,
    include_tags: Sequence[str] = QWEN4_5_OPAQUE_STATE_TAGS,
) -> List[_OpaquePoolHandle]:
    """Extract normalized ``OPAQUE_STATE`` pool handles from KVCache or a dict."""
    allowed = set(include_tags)
    handles: List[_OpaquePoolHandle] = []
    if isinstance(state_container, KVCache):
        raw_tags = (
            state_container.group_tags()
            if callable(state_container.group_tags)
            else state_container.group_tags
        )
        raw_layers = (
            state_container.layer_count()
            if callable(state_container.layer_count)
            else state_container.layer_count
        )
        tags = [t for t in raw_tags if t in allowed]
        for layer_id in range(int(raw_layers)):
            for tag in tags:
                layer_kv = state_container.get_layer_cache(layer_id, tag)
                if layer_kv is not None and layer_kv.kv_cache_base is not None:
                    handles.append(
                        _OpaquePoolHandle(
                            key=f"layer{layer_id}:{tag}",
                            tensor=layer_kv.kv_cache_base,
                        )
                    )
        return handles

    if isinstance(state_container, Mapping):
        for k, v in state_container.items():
            if isinstance(v, torch.Tensor):
                tag = k[1] if isinstance(k, tuple) and len(k) >= 2 else str(k)
                if not allowed or any(tag == t or tag.endswith(f":{t}") for t in allowed):
                    handles.append(_OpaquePoolHandle(key=str(k), tensor=v))
            elif isinstance(v, Mapping):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, torch.Tensor):
                        handles.append(
                            _OpaquePoolHandle(key=f"{k}:{sub_k}", tensor=sub_v)
                        )
        return handles

    raise TypeError(f"Unsupported state_container type: {type(state_container)!r}")


@dataclass
class Qwen4_5OpaqueStateTransaction:
    """Snapshot and partial-acceptance rollback for Qwen4.5 ``OPAQUE_STATE`` pools."""

    pools: List[_OpaquePoolHandle]
    _pre_verify_slots: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    _step_slots: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=dict
    )

    @classmethod
    def from_container(
        cls,
        state_container: Any,
        *,
        include_tags: Sequence[str] = QWEN4_5_OPAQUE_STATE_TAGS,
    ) -> "Qwen4_5OpaqueStateTransaction":
        return cls(
            pools=discover_qwen4_5_opaque_pools(
                state_container, include_tags=include_tags
            )
        )

    def _capture_into(
        self, block_ids: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        snapshots: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        ids = block_ids.reshape(-1).long()
        for handle in self.pools:
            ids_dev = ids.to(handle.tensor.device)
            snapshots[handle.key] = (ids_dev, handle.tensor[ids_dev].clone())
        return snapshots

    def begin_verify(self, block_ids: torch.Tensor) -> None:
        self._step_slots.clear()
        self._pre_verify_slots = self._capture_into(block_ids)

    def capture_step(self, step_idx: int, block_ids: torch.Tensor) -> None:
        self._step_slots[int(step_idx)] = self._capture_into(block_ids)

    def restore_pre_verify(
        self,
        batch_mask: Optional[torch.Tensor] = None,
        target_block_ids: Optional[torch.Tensor] = None,
    ) -> None:
        for handle in self.pools:
            saved_ids, saved_vals = self._pre_verify_slots[handle.key]
            dst_ids = (
                target_block_ids.reshape(-1).long().to(handle.tensor.device)
                if target_block_ids is not None
                else saved_ids
            )
            if batch_mask is None:
                handle.tensor[dst_ids] = saved_vals
            else:
                mask = batch_mask.reshape(-1).to(handle.tensor.device)
                if bool(mask.any().item()):
                    handle.tensor[dst_ids[mask]] = saved_vals[mask]

    def commit_accepted(
        self,
        accepted_lens: torch.Tensor,
        query_lens: Optional[torch.Tensor] = None,
        final_block_ids: Optional[torch.Tensor] = None,
    ) -> None:
        acc = accepted_lens.reshape(-1)
        batch_size = int(acc.numel())
        for b in range(batch_size):
            a_len = int(acc[b].item())
            if query_lens is not None and a_len >= int(query_lens.reshape(-1)[b].item()):
                continue
            if a_len == 0:
                if not self._pre_verify_slots:
                    raise RuntimeError(
                        "accepted_len == 0 rollback requires begin_verify() snapshot"
                    )
                for handle in self.pools:
                    saved_ids, saved_vals = self._pre_verify_slots[handle.key]
                    dst_id = (
                        int(final_block_ids.reshape(-1)[b].item())
                        if final_block_ids is not None
                        else int(saved_ids[b].item())
                    )
                    handle.tensor[dst_id] = saved_vals[b]
                continue

            step_idx = a_len - 1
            if step_idx not in self._step_slots:
                raise KeyError(
                    f"Missing step snapshot {step_idx} for accept_len={a_len}"
                )
            step_snap = self._step_slots[step_idx]
            for handle in self.pools:
                saved_ids, saved_vals = step_snap[handle.key]
                dst_id = (
                    int(final_block_ids.reshape(-1)[b].item())
                    if final_block_ids is not None
                    else int(saved_ids[b].item())
                )
                handle.tensor[dst_id] = saved_vals[b]


def snapshot_qwen4_5_opaque_states(
    state_container: Any,
    block_ids: torch.Tensor,
    *,
    include_tags: Sequence[str] = QWEN4_5_OPAQUE_STATE_TAGS,
) -> Qwen4_5OpaqueStateTransaction:
    tx = Qwen4_5OpaqueStateTransaction.from_container(
        state_container, include_tags=include_tags
    )
    tx.begin_verify(block_ids)
    return tx


def rollback_qwen4_5_opaque_states(
    tx: Qwen4_5OpaqueStateTransaction,
    *,
    accepted_lens: Optional[torch.Tensor] = None,
    query_lens: Optional[torch.Tensor] = None,
    final_block_ids: Optional[torch.Tensor] = None,
    batch_mask: Optional[torch.Tensor] = None,
) -> None:
    if tx._step_slots and accepted_lens is not None:
        tx.commit_accepted(
            accepted_lens=accepted_lens,
            query_lens=query_lens,
            final_block_ids=final_block_ids,
        )
    else:
        tx.restore_pre_verify(
            batch_mask=batch_mask,
            target_block_ids=final_block_ids,
        )


def compute_qsa_indexer_state_lengths(
    prefix_lens: torch.Tensor,
    accepted_lens: torch.Tensor,
    compress_ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    new_total_lens = prefix_lens.to(torch.int64) + accepted_lens.to(torch.int64)
    compressed_kv_lens = torch.div(new_total_lens, compress_ratio, rounding_mode="floor")
    tail_ring_lens = torch.remainder(new_total_lens, compress_ratio)
    return compressed_kv_lens, tail_ring_lens
