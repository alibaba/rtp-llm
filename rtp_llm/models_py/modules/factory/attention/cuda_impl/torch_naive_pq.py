"""Torch Naive PQ Attention Backend.

PQ (Product Quantization) 加速 decode attention：
- Prefill：将 head_dim 切成 num_subspaces 个子空间，每个子空间独立 K-Means 聚类
- Decode：q_sub @ centroids_sub 得到每个子空间的簇分数，按 token 的 cids 累加；
  每个 q-head 选 top_k_tokens，同 KV group 内取 union，再对 union + 新 decode token
  做 full attention。

默认配置：num_subspaces=16, num_clusters=256, top_k_tokens=2000.
"""

import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func
from torch.nn.functional import scaled_dot_product_attention

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.pq_kmeans_triton import (
    batch_kmeans_Euclid,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.torch_naive import (
    TorchNaiveDecodeImpl,
    TorchNaivePrefillImpl,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs

# ============================================================================
# Global PQ cache: key -> {"cids": [S, N], "cents": [S, K, sub_dim], "prefill_len": int}
# ============================================================================
_PQ_CACHE: dict = {}


# ============================================================================
# Triton kernel: fused gather + sum over subspaces
# ============================================================================
@triton.jit
def _pq_aggregate_kernel(
    cluster_scores_ptr,  # [bs, S, K, num_groups]  (q 在 innermost，连续)
    cids_ptr,  # [bs, S, prefill_len] int64
    token_scores_ptr,  # [bs, num_groups, prefill_len]
    cs_stride_b,
    cs_stride_s,
    cs_stride_k,
    cid_stride_b,
    cid_stride_s,
    ts_stride_b,
    ts_stride_q,
    PREFILL_LEN,
    NUM_GROUPS: tl.constexpr,
    S: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """每个 program 算 (b, t_block) 内 [BLOCK_T, NUM_GROUPS] 的 token_scores。

    内存模式：
      - cluster_scores 布局 [bs, S, K, num_groups]：固定 (b, s, k_idx[t]) 时
        num_groups=16 个值连续，每个 t 一次 32-byte coalesced load。
      - 老布局 [bs, num_groups, S, K] 跨 q stride=4096 (8 KB)，每个 (q,t) 单独标量 load。
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < PREFILL_LEN
    q_offsets = tl.arange(0, NUM_GROUPS)

    acc = tl.zeros((BLOCK_T, NUM_GROUPS), dtype=tl.float32)

    cid_base = cids_ptr + pid_b * cid_stride_b
    cs_base = cluster_scores_ptr + pid_b * cs_stride_b

    for s in tl.static_range(S):
        k_idx = tl.load(
            cid_base + s * cid_stride_s + t_offsets,
            mask=t_mask,
            other=0,
        )
        # cluster_scores[b, s, k_idx, q] → [BLOCK_T, NUM_GROUPS]，q 维度 stride=1
        cs_ptrs = (
            cs_base
            + s * cs_stride_s
            + k_idx[:, None] * cs_stride_k
            + q_offsets[None, :]
        )
        cs_vals = tl.load(cs_ptrs, mask=t_mask[:, None], other=0.0)
        acc += cs_vals.to(tl.float32)

    # 输出 token_scores[b, q, t]：q stride = prefill_len，t stride = 1
    # 把 acc[BLOCK_T, NUM_GROUPS] 转置成 [NUM_GROUPS, BLOCK_T] 再写，store 端 t 在 inner 也 coalesced
    acc_t = tl.trans(acc)
    out_ptrs = (
        token_scores_ptr
        + pid_b * ts_stride_b
        + q_offsets[:, None] * ts_stride_q
        + t_offsets[None, :]
    )
    tl.store(
        out_ptrs,
        acc_t.to(token_scores_ptr.dtype.element_ty),
        mask=t_mask[None, :],
    )


def _pq_aggregate_triton(
    cluster_scores: torch.Tensor,  # [bs, S, K, num_groups]
    cids: torch.Tensor,  # [bs, S, prefill_len] int64
    num_groups: int,
) -> torch.Tensor:
    """fused gather + sum: token_scores[b,q,t] = sum_s cluster_scores[b,s,cids[b,s,t],q].

    cluster_scores 这里假定按 [bs, S, K, num_groups] 排（num_groups innermost）。
    """
    bs, S, K, _ = cluster_scores.shape
    _, _, prefill_len = cids.shape

    assert cluster_scores.is_contiguous() and cids.is_contiguous()
    assert cids.dtype == torch.int64

    token_scores = torch.empty(
        bs,
        num_groups,
        prefill_len,
        dtype=cluster_scores.dtype,
        device=cluster_scores.device,
    )

    BLOCK_T = 128
    grid = (bs, triton.cdiv(prefill_len, BLOCK_T))

    _pq_aggregate_kernel[grid](
        cluster_scores,
        cids,
        token_scores,
        cluster_scores.stride(0),
        cluster_scores.stride(1),
        cluster_scores.stride(2),
        cids.stride(0),
        cids.stride(1),
        token_scores.stride(0),
        token_scores.stride(1),
        prefill_len,
        NUM_GROUPS=num_groups,
        S=S,
        BLOCK_T=BLOCK_T,
    )
    return token_scores


def _pq_key(layer_id: int, seq_idx: int, kv_head_idx: int) -> str:
    return f"pq_layer_{layer_id}_seq_{seq_idx}_kv_head_{kv_head_idx}"


def _pq_score_and_select(
    q_group: torch.Tensor,  # [num_q, head_dim]
    cids: torch.Tensor,  # [S, clustered_len]
    cents: torch.Tensor,  # [S, K, sub_dim]
    top_k: int,
) -> torch.Tensor:
    """对该 KV head 的 q_group 计算 PQ 分数，每个 q-head 选 top_k，取 union.

    Returns:
        selected: [num_selected] (long, sorted unique token indices)
    """
    num_q, head_dim = q_group.shape
    num_subspaces, clustered_len = cids.shape
    sub_dim = head_dim // num_subspaces

    q_subs = q_group.reshape(num_q, num_subspaces, sub_dim)  # [Q, S, sub_dim]
    # 每子空间每簇得分: [Q, S, K]
    cluster_scores = torch.einsum("qsd,skd->qsk", q_subs, cents)
    # 查表展开到每 token: [Q, S, N]
    cids_exp = cids.unsqueeze(0).expand(num_q, -1, -1)  # [Q, S, N]
    token_subspace = cluster_scores.gather(2, cids_exp)  # [Q, S, N]
    token_scores = token_subspace.sum(dim=1)  # [Q, N]

    per_head_k = min(top_k, clustered_len)
    topk_ids = token_scores.topk(per_head_k, dim=-1).indices  # [Q, per_head_k]
    return torch.unique(topk_ids.flatten())


# ============================================================================
# Prefill: 子空间 PQ 聚类
# ============================================================================


class TorchNaivePQPrefillImpl(TorchNaivePrefillImpl):
    """带 PQ 子空间聚类的 Prefill 实现."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(attn_configs, attn_inputs, parallelism_config)

        self.num_subspaces = 16
        self.num_clusters = 256
        self.kmeans_iters = 20

        assert (
            self.head_dim % self.num_subspaces == 0
        ), f"head_dim {self.head_dim} must be divisible by num_subspaces {self.num_subspaces}"

        logging.debug(
            f"TorchNaivePQPrefillImpl: S={self.num_subspaces}, K={self.num_clusters}, "
            f"sub_dim={self.head_dim // self.num_subspaces}"
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # 1. RoPE
        if self.need_rope_kv_cache:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # 2. Split QKV
        q, k, v = self._split_qkv(qkv)

        # 3. PQ 聚类（在写 cache 前对 K 做）
        self._perform_pq_clustering(k, kv_cache)

        # 4. Write cache
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 5. 全 K 走标准 flash attention
        output = self._run_attention_extend(q, k, v)
        output = output.reshape(output.shape[0], -1)
        return output

    def _perform_pq_clustering(
        self,
        k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> None:
        if kv_cache is None:
            return

        layer_id = kv_cache.layer_id
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1]
        num_kv_heads = k.shape[1]
        S = self.num_subspaces
        sub_dim = self.head_dim // S

        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            # 把 num_kv_heads × S 合成 batch 维：[H*S, seq_len, sub_dim]
            k_seq = k[start_idx:end_idx]  # [seq_len, H, head_dim]
            k_subs = (
                k_seq.reshape(seq_len, num_kv_heads, S, sub_dim)
                .permute(1, 2, 0, 3)  # [H, S, seq_len, sub_dim]
                .reshape(num_kv_heads * S, seq_len, sub_dim)
                .contiguous()
            )

            cids_b, cents_b, _ = batch_kmeans_Euclid(
                k_subs,
                self.num_clusters,
                max_iters=self.kmeans_iters,
                tol=1e-4,
                init_centroids=None,
                verbose=False,
            )
            cids_b = cids_b.to(torch.int64).reshape(num_kv_heads, S, seq_len)
            cents_b = cents_b.reshape(num_kv_heads, S, self.num_clusters, sub_dim)

            for h in range(num_kv_heads):
                _PQ_CACHE[_pq_key(layer_id, seq_idx, h)] = {
                    "cids": cids_b[h],  # [S, seq_len]
                    "cents": cents_b[h],  # [S, K, sub_dim]
                    "prefill_len": seq_len,
                }

            logging.debug(
                f"[PQ Prefill] layer={layer_id} seq={seq_idx} "
                f"H={num_kv_heads} S={S} K={self.num_clusters} seq_len={seq_len}"
            )


# ============================================================================
# Decode: PQ 打分 + per-q-head top_k union + 全 attention
# ============================================================================


class TorchNaivePQDecodeImpl(TorchNaiveDecodeImpl):
    """带 PQ 加速的 Decode 实现.

    每个 KV head：
      1. q_sub @ centroids_sub 得到 [Q, S, K] 簇分数
      2. 按 cids 查表得到 [Q, N] token 分数
      3. 每个 q-head 选 top_k_tokens，取 union
      4. 加上 prefill_len 之后的所有新 decode token
      5. 用 union + 新 token 做 full attention
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(attn_configs, attn_inputs, parallelism_config)

        self.num_subspaces = int(os.getenv("PQ_NUM_SUBSPACES", "16"))
        self.top_k_tokens = int(os.getenv("PQ_TOP_K_TOKENS", "2000"))

        logging.debug(
            f"TorchNaivePQDecodeImpl: S={self.num_subspaces}, top_k={self.top_k_tokens}"
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # 1. RoPE
        if self.need_rope_kv_cache:
            q = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
            q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        else:
            q, _, _ = self._split_qkv(qkv)

        # 2. Write cache（含本步新 K, V）
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 3. Read full K, V
        k_full, v_full = self._read_kv_from_cache(kv_cache)

        # 4. PQ-guided sparse attention
        output = self._run_pq_attention_decode(q, k_full, v_full, kv_cache)
        output = output.reshape(output.shape[0], -1)
        return output

    def _compute_kv_seq_lens(self, batch_size: int) -> torch.Tensor:
        """每条序列的实际 KV 长度（CPU [batch_size]）— k_full 是 padded 的。"""
        from rtp_llm.ops.compute_ops import fill_mla_params

        ai = self.attn_inputs
        params = fill_mla_params(
            (
                ai.prefix_lengths
                if getattr(ai, "prefix_lengths", None) is not None
                else torch.tensor([], dtype=torch.int32)
            ),
            ai.sequence_lengths,
            ai.input_lengths,
            (
                ai.kv_cache_block_id_host
                if ai.kv_cache_block_id_host is not None
                else torch.tensor([], dtype=torch.int32)
            ),
            self.tokens_per_block,
        )
        return params.kvlen_h[:batch_size]

    def _run_pq_attention_decode(
        self,
        q: torch.Tensor,  # [batch, num_heads, head_dim]
        k_full: torch.Tensor,  # [batch, max_seq_len, num_kv_heads, head_dim] (padded)
        v_full: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """整 batch 化的 PQ-guided sparse attention.

        快路径（所有 batch 在该 kv_head 都命中 PQ cache 且 prefill_len 一致）：
          1. 把各 batch 的 cids/cents 沿 batch 维 stack，单次 einsum + gather + sum
             得到 token_scores [bs, num_groups, prefill_len]
          2. 单次 topk + scatter 到 [bs, max_seq_len] bool mask（取代 unique）
          3. mask | new_decode_token_mask → mask.nonzero() 得到 packed (b, pos)
          4. 单次 K/V index_select + 单次 flash_attn_varlen_func
        慢路径：回退到逐 batch 循环（原 V2b）。
        """
        batch_size = q.shape[0]
        layer_id = kv_cache.layer_id if kv_cache is not None else 0
        num_groups = self.num_heads // self.num_kv_heads if self.enable_gqa else 1
        seq_lens = self._compute_kv_seq_lens(batch_size)
        device = q.device
        max_seq_len = k_full.shape[1]

        output = torch.empty_like(q)

        for kv_head_idx in range(self.num_kv_heads):
            start_h = kv_head_idx * num_groups
            end_h = start_h + num_groups
            q_kv = q[:, start_h:end_h, :].contiguous()  # [bs, num_groups, D]

            # 收集本 kv_head 在各 batch 的 PQ cache 条目
            cids_list = []
            cents_list = []
            prefill_lens = []
            all_cached = True
            for b in range(batch_size):
                entry = _PQ_CACHE.get(_pq_key(layer_id, b, kv_head_idx))
                if entry is None:
                    all_cached = False
                    break
                cids_list.append(entry["cids"])
                cents_list.append(entry["cents"])
                prefill_lens.append(entry["prefill_len"])

            uniform = all_cached and len(set(prefill_lens)) == 1
            if uniform:
                output[:, start_h:end_h, :] = self._batched_pq_decode_one_kv(
                    q_kv,
                    cids_list,
                    cents_list,
                    prefill_lens[0],
                    k_full,
                    v_full,
                    kv_head_idx,
                    seq_lens,
                    max_seq_len,
                )
            else:
                output[:, start_h:end_h, :] = self._per_batch_pq_decode_one_kv(
                    q_kv,
                    layer_id,
                    kv_head_idx,
                    k_full,
                    v_full,
                    seq_lens,
                    max_seq_len,
                )

        return output

    def _batched_pq_decode_one_kv(
        self,
        q_kv: torch.Tensor,  # [bs, num_groups, head_dim]
        cids_list,
        cents_list,
        prefill_len: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        kv_head_idx: int,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        bs, num_groups, head_dim = q_kv.shape
        device = q_kv.device

        # ---- Step A: stack cids/cents 整 batch 化 PQ 打分 ----
        cids = torch.stack(cids_list, dim=0).contiguous()  # [bs, S, prefill_len]
        cents = torch.stack(cents_list, dim=0).contiguous()  # [bs, S, K, sub_dim]
        S = cids.shape[1]
        sub_dim = head_dim // S

        q_subs = q_kv.reshape(bs, num_groups, S, sub_dim)
        # cluster_scores[b, s, k, q] = <q_subs[b,q,s,:], cents[b,s,k,:]>
        # 这里把 q 放到 innermost 维（[bs,S,K,num_groups]），让 Triton kernel 的
        # gather load 一次取 num_groups=16 个连续 bf16（32 B），coalesced。
        cluster_scores = torch.einsum(
            "bqsd,bskd->bskq", q_subs, cents.to(q_subs.dtype)
        ).contiguous()  # [bs, S, K, num_groups]
        token_scores = _pq_aggregate_triton(cluster_scores, cids, num_groups)
        # token_scores: [bs, num_groups, prefill_len]

        # ---- Step B: 整 batch topk → scatter 到 [bs, max_seq_len] mask ----
        per_q_k = min(self.top_k_tokens, prefill_len)
        topk_ids = token_scores.topk(
            per_q_k, dim=-1
        ).indices  # [bs, num_groups, per_q_k]

        mask = torch.zeros(bs, max_seq_len, dtype=torch.bool, device=device)
        mask[:, :prefill_len].scatter_(1, topk_ids.reshape(bs, -1), True)

        # 加上 prefill 之后的新 decode token，并裁掉 padding
        pos = torch.arange(max_seq_len, device=device, dtype=torch.int32)
        seq_lens_gpu = seq_lens.to(device=device, dtype=torch.int32)
        within_seq = pos[None, :] < seq_lens_gpu[:, None]
        new_decode_mask = (pos[None, :] >= prefill_len) & within_seq
        mask = (mask | new_decode_mask) & within_seq

        # ---- Step C: K/V packed gather + flash_attn_varlen ----
        # max_seqlen_k 上界：每条 seq 至多 prefill 内 PQ 选中 + (max_seq_len - prefill_len) 个新 token
        # 简化为 max_seq_len（kernel 端会按 cu_seqlens fast-exit 短序列），避免 .item() 同步。
        return self._packed_attention(
            q_kv,
            mask,
            k_full,
            v_full,
            kv_head_idx,
            max_seq_len,
            max_seqlen_k_bound=max_seq_len,
        )

    def _per_batch_pq_decode_one_kv(
        self,
        q_kv: torch.Tensor,
        layer_id: int,
        kv_head_idx: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """回退路径：原 V2b 的逐 batch 实现，处理 cache miss / 非 uniform prefill_len。"""
        batch_size, num_groups, head_dim = q_kv.shape
        device = q_kv.device

        selected_per_batch = []
        kv_lens_list = []
        for b in range(batch_size):
            total_seq_len = int(seq_lens[b])
            entry = _PQ_CACHE.get(_pq_key(layer_id, b, kv_head_idx))
            q_group = q_kv[b]
            if entry is None:
                sel = torch.arange(total_seq_len, device=device, dtype=torch.long)
            else:
                cids = entry["cids"]
                cents = entry["cents"]
                prefill_len = entry["prefill_len"]
                sel = _pq_score_and_select(q_group, cids, cents, self.top_k_tokens)
                if total_seq_len > prefill_len:
                    new_tokens = torch.arange(
                        prefill_len, total_seq_len, device=device, dtype=sel.dtype
                    )
                    sel = torch.cat([sel, new_tokens])
            selected_per_batch.append(sel)
            kv_lens_list.append(sel.shape[0])

        sel_global = torch.cat(selected_per_batch)
        kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device=device)
        batch_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.long),
            kv_lens.to(torch.long),
        )
        flat_idx = batch_ids * max_seq_len + sel_global

        # 走 _packed_attention 的子路径，但传入显式的 (flat_idx, kv_lens) 而非 mask
        # max_seqlen_k 用 max_seq_len 作上界，避开 .item() 同步
        return self._packed_attention_from_idx(
            q_kv,
            flat_idx,
            kv_lens,
            k_full,
            v_full,
            kv_head_idx,
            max_seqlen_k_bound=max_seq_len,
        )

    def _packed_attention(
        self,
        q_kv: torch.Tensor,
        mask: torch.Tensor,  # [bs, max_seq_len]
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        kv_head_idx: int,
        max_seq_len: int,
        max_seqlen_k_bound: int,
    ) -> torch.Tensor:
        kv_lens = mask.sum(dim=1).to(torch.int32)
        sel_pairs = mask.nonzero(as_tuple=False)  # [total_kv, 2]
        flat_idx = sel_pairs[:, 0] * max_seq_len + sel_pairs[:, 1]
        return self._packed_attention_from_idx(
            q_kv,
            flat_idx,
            kv_lens,
            k_full,
            v_full,
            kv_head_idx,
            max_seqlen_k_bound=max_seqlen_k_bound,
        )

    def _packed_attention_from_idx(
        self,
        q_kv: torch.Tensor,  # [bs, num_groups, head_dim]
        flat_idx: torch.Tensor,  # [total_kv] long
        kv_lens: torch.Tensor,  # [bs] int32
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        kv_head_idx: int,
        max_seqlen_k_bound: int,
    ) -> torch.Tensor:
        """max_seqlen_k_bound 是 CPU 端已知的上界（如 max_seq_len），
        flash_attn 用它配 grid；实际短的序列会按 cu_seqlens fast-exit。
        这样避免 int(kv_lens.max().item()) 触发 GPU↔CPU 同步。"""
        bs = q_kv.shape[0]
        device = q_kv.device

        k_flat = k_full.reshape(-1, self.num_kv_heads, self.head_dim)
        v_flat = v_full.reshape(-1, self.num_kv_heads, self.head_dim)
        k_packed = k_flat.index_select(0, flat_idx)[:, kv_head_idx : kv_head_idx + 1, :]
        v_packed = v_flat.index_select(0, flat_idx)[:, kv_head_idx : kv_head_idx + 1, :]

        cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        cu_seqlens_k[1:] = kv_lens.cumsum(0).to(torch.int32)

        if k_packed.dtype != q_kv.dtype:
            k_packed = k_packed.to(q_kv.dtype)
            v_packed = v_packed.to(q_kv.dtype)

        return flash_attn_varlen_func(
            q_kv,
            k_packed,
            v_packed,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k_bound,
            causal=False,
            softmax_scale=self.scaling,
        )

    def _full_attention_gqa_group(
        self,
        q_group: torch.Tensor,  # [num_q, head_dim]
        k: torch.Tensor,  # [num_selected, head_dim]
        v: torch.Tensor,  # [num_selected, head_dim]
    ) -> torch.Tensor:
        """同 KV head 内多 Q heads 共享 K/V 的 attention."""
        q = q_group.unsqueeze(0).unsqueeze(2)  # [1, num_q, 1, head_dim]
        k = k.unsqueeze(0).unsqueeze(0)  # [1, 1, num_selected, head_dim]
        v = v.unsqueeze(0).unsqueeze(0)

        if not (q.dtype == k.dtype == v.dtype):
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        return out.squeeze(0).squeeze(1)  # [num_q, head_dim]
