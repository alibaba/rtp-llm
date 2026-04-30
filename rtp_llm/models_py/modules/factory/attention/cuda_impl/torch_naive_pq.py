"""Torch Naive PQ Attention Backend.

PQ (Product Quantization) 加速 decode attention：
- Prefill：将 head_dim 切成 num_subspaces 个子空间，每个子空间独立 K-Means 聚类
- Decode：q_sub @ centroids_sub 得到每个子空间的簇分数，按 token 的 cids 累加；
  每个 q-head 选 top_k_tokens，同 KV group 内取 union，再对 union + 新 decode token
  做 full attention。

默认配置：num_subspaces=16, num_clusters=256, top_k_tokens=2000.
"""

import logging
import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl

import flashinfer

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
# PD separation helpers: store / load PQ buffers via PyAttentionInputs
# ============================================================================

def store_pq_to_attn_inputs(
    attn_inputs,
    layer_id: int,
    cids: torch.Tensor,      # [num_kv_heads, S, seq_len]
    cents: torch.Tensor,     # [num_kv_heads, S, K, sub_dim]
) -> None:
    """Store PQ buffers for *layer_id* onto attn_inputs for C++ PD transfer.

    Uses flat format: per_layer_cids[layer_id] = single Tensor per layer.
    """
    try:
        raw = attn_inputs.per_layer_cids
        current_cids = list(raw) if raw is not None else []
    except (TypeError, AttributeError):
        current_cids = []

    try:
        raw = attn_inputs.per_layer_cents
        current_cents = list(raw) if raw is not None else []
    except (TypeError, AttributeError):
        current_cents = []

    while len(current_cids) <= layer_id:
        current_cids.append(torch.empty(0))
        current_cents.append(torch.empty(0))

    current_cids[layer_id] = cids.to(torch.int32) if cids.dtype == torch.int64 else cids
    current_cents[layer_id] = cents.float() if cents.dtype not in (torch.float32, torch.float16, torch.bfloat16) else cents

    try:
        attn_inputs.per_layer_cids = current_cids
        attn_inputs.per_layer_cents = current_cents
    except (TypeError, AttributeError):
        pass


def load_pq_batch_from_attn_inputs(
    attn_inputs,
    layer_id: int,
    kv_head_idx: int,
    batch_size: int,
) -> Optional[dict]:
    """Load PQ data for ALL batch items at a given (layer, kv_head).

    Handles two formats from C++:
      - 3D: [H, S, prefill_len]               (single stream / prefill-side)
      - 4D: [batch, H, S, max_prefill_len]    (merged from NormalModelInputGatherer)

    Returns dict with:
        cids  : [batch, S, prefill_len]  int64, CUDA
        cents : [batch, S, K, sub_dim]   CUDA
        prefill_len : int  (max across batches when padded)
    or None if unavailable.
    """
    per_layer_cids = getattr(attn_inputs, "per_layer_cids", None)
    if per_layer_cids is None or layer_id >= len(per_layer_cids):
        return None

    layer_cids = per_layer_cids[layer_id]
    if not isinstance(layer_cids, torch.Tensor) or layer_cids.numel() == 0:
        return None

    per_layer_cents = getattr(attn_inputs, "per_layer_cents", None)
    layer_cents = per_layer_cents[layer_id]

    device = torch.device("cuda")

    if layer_cids.dim() == 4:
        # 4D: [batch, H, S, prefill_len] from merged decode streams
        if layer_cids.shape[0] < batch_size:
            return None
        cids = layer_cids[:batch_size, kv_head_idx]   # [batch, S, prefill_len]
        cents = layer_cents[:batch_size, kv_head_idx]  # [batch, S, K, sub_dim]
    elif layer_cids.dim() == 3:
        # 3D: [H, S, prefill_len] single stream — expand to batch
        cids = layer_cids[kv_head_idx].unsqueeze(0).expand(batch_size, -1, -1)
        cents = layer_cents[kv_head_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
    else:
        return None

    cids = cids.contiguous()
    cents = cents.contiguous()

    if cids.dtype != torch.int64:
        cids = cids.to(torch.int64)
    if not cids.is_cuda:
        cids = cids.to(device)
    if not cents.is_cuda:
        cents = cents.to(device)

    return {
        "cids": cids,
        "cents": cents,
        "prefill_len": int(cids.shape[-1]),
    }


# ============================================================================
# Triton kernel: fused gather + sum over subspaces
# ============================================================================
@triton.jit
def _pq_aggregate_kernel(
    cluster_scores_ptr,  # [bs, S, K, num_groups]
    cids_ptr,            # [bs, S, prefill_len] int64
    token_scores_ptr,    # [bs, num_groups, prefill_len]
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
        cs_ptrs = (
            cs_base
            + s * cs_stride_s
            + k_idx[:, None] * cs_stride_k
            + q_offsets[None, :]
        )
        cs_vals = tl.load(cs_ptrs, mask=t_mask[:, None], other=0.0)
        acc += cs_vals.to(tl.float32)

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


def _next_po2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _pq_aggregate_triton(
    cluster_scores: torch.Tensor,  # [bs, S, K, num_groups]
    cids: torch.Tensor,            # [bs, S, prefill_len] int64
    num_groups: int,
) -> torch.Tensor:
    bs, S, K, _ = cluster_scores.shape
    _, _, prefill_len = cids.shape

    assert cluster_scores.is_contiguous() and cids.is_contiguous()
    assert cids.dtype == torch.int64

    NUM_GROUPS_PO2 = _next_po2(num_groups)
    if NUM_GROUPS_PO2 != num_groups:
        cluster_scores = torch.nn.functional.pad(
            cluster_scores, (0, NUM_GROUPS_PO2 - num_groups)
        ).contiguous()

    token_scores = torch.empty(
        bs, NUM_GROUPS_PO2, prefill_len,
        dtype=cluster_scores.dtype, device=cluster_scores.device,
    )

    BLOCK_T = 128
    grid = (bs, triton.cdiv(prefill_len, BLOCK_T))

    _pq_aggregate_kernel[grid](
        cluster_scores, cids, token_scores,
        cluster_scores.stride(0), cluster_scores.stride(1), cluster_scores.stride(2),
        cids.stride(0), cids.stride(1),
        token_scores.stride(0), token_scores.stride(1),
        prefill_len,
        NUM_GROUPS=NUM_GROUPS_PO2, S=S, BLOCK_T=BLOCK_T,
    )
    return token_scores[:, :num_groups, :]


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

        logging.info(
            f"TorchNaivePQPrefillImpl: S={self.num_subspaces}, K={self.num_clusters}, "
            f"sub_dim={self.head_dim // self.num_subspaces}"
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        q, k, v = self._split_qkv(qkv)
        self._perform_pq_clustering(k, kv_cache)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        output = self._run_attention_extend(q, k, v)
        return output.reshape(output.shape[0], -1)

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

        # PD separation currently only supports batch_size=1 through the C++ path.
        # For multi-batch, cids/cents per layer are stored for seq_idx=0 only.
        all_cids = []
        all_cents = []

        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            k_seq = k[start_idx:end_idx]  # [seq_len, H, head_dim]
            k_subs = (
                k_seq.reshape(seq_len, num_kv_heads, S, sub_dim)
                .permute(1, 2, 0, 3)        # [H, S, seq_len, sub_dim]
                .reshape(num_kv_heads * S, seq_len, sub_dim)
                .contiguous()
            )

            cids_b, cents_b, _ = batch_kmeans_Euclid(
                k_subs,
                self.num_clusters,
                max_iters=self.kmeans_iters,
                tol=1e-4,
            )
            cids_b = cids_b.to(torch.int64).reshape(num_kv_heads, S, seq_len)
            cents_b = cents_b.reshape(num_kv_heads, S, self.num_clusters, sub_dim)

            all_cids.append(cids_b)
            all_cents.append(cents_b)

        # Store for PD transfer (only seq_idx=0 in current C++ path)
        store_pq_to_attn_inputs(
            self.attn_inputs, layer_id, all_cids[0], all_cents[0],
        )

        logging.debug(
            f"PQ clustering done: layer={layer_id} batch={batch_size} "
            f"H={num_kv_heads} S={S} K={self.num_clusters}"
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

        logging.info(
            f"TorchNaivePQDecodeImpl: S={self.num_subspaces}, top_k={self.top_k_tokens}"
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            q = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
            q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        else:
            q, _, _ = self._split_qkv(qkv)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        k_full, v_full = self._read_kv_from_cache(kv_cache)
        output = self._run_pq_attention_decode(q, k_full, v_full, kv_cache)
        return output.reshape(output.shape[0], -1)

    def _compute_kv_seq_lens(self, batch_size: int) -> torch.Tensor:
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
        q: torch.Tensor,        # [batch, num_heads, head_dim]
        k_full: torch.Tensor,   # [batch, max_seq_len, num_kv_heads, head_dim]
        v_full: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
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
            q_kv = q[:, start_h:end_h, :].contiguous()

            pq = load_pq_batch_from_attn_inputs(
                self.attn_inputs, layer_id, kv_head_idx, batch_size,
            )
            if pq is not None:
                output[:, start_h:end_h, :] = self._batched_pq_decode_one_kv(
                    q_kv,
                    pq["cids"],      # [batch, S, prefill_len]
                    pq["cents"],     # [batch, S, K, sub_dim]
                    pq["prefill_len"],
                    k_full, v_full,
                    kv_head_idx, seq_lens, max_seq_len,
                )
            else:
                output[:, start_h:end_h, :] = self._fallback_full_attn(
                    q_kv, k_full, v_full, kv_head_idx, seq_lens, max_seq_len,
                )

        return output

    def _batched_pq_decode_one_kv(
        self,
        q_kv: torch.Tensor,       # [bs, num_groups, head_dim]
        cids: torch.Tensor,        # [bs, S, max_prefill_len] (padded)
        cents: torch.Tensor,       # [bs, S, K, sub_dim]
        max_prefill_len: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        kv_head_idx: int,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        bs, num_groups, head_dim = q_kv.shape
        device = q_kv.device
        S = cids.shape[1]
        sub_dim = head_dim // S

        # Per-batch actual prefill_len from input_lengths
        input_lens = self.attn_inputs.input_lengths[:bs].to(
            device=device, dtype=torch.int32
        )

        q_subs = q_kv.reshape(bs, num_groups, S, sub_dim)
        cluster_scores = torch.einsum(
            "bqsd,bskd->bskq", q_subs, cents.to(q_subs.dtype)
        ).contiguous()
        token_scores = _pq_aggregate_triton(cluster_scores, cids, num_groups)
        # token_scores: [bs, num_groups, max_prefill_len]

        # Mask padded positions to -inf so topk never selects them.
        # For batch b, positions >= input_lens[b] are padding.
        pq_pos = torch.arange(max_prefill_len, device=device, dtype=torch.int32)
        valid_pq = pq_pos[None, :] < input_lens[:, None]      # [bs, max_prefill_len]
        token_scores.masked_fill_(~valid_pq.unsqueeze(1), float("-inf"))

        per_q_k = min(self.top_k_tokens, max_prefill_len)
        topk_ids = token_scores.topk(per_q_k, dim=-1).indices  # [bs, G, per_q_k]

        # Build per-sequence mask over full KV length
        mask = torch.zeros(bs, max_seq_len, dtype=torch.bool, device=device)
        topk_flat = topk_ids.reshape(bs, -1).clamp(max=max_seq_len - 1)
        mask.scatter_(1, topk_flat, True)

        pos = torch.arange(max_seq_len, device=device, dtype=torch.int32)
        seq_lens_gpu = seq_lens.to(device=device, dtype=torch.int32)
        within_seq = pos[None, :] < seq_lens_gpu[:, None]

        # Add new decode tokens: positions >= per-batch prefill boundary
        prefill_boundary = input_lens[:, None]
        new_decode_mask = (pos[None, :] >= prefill_boundary) & within_seq
        mask = (mask | new_decode_mask) & within_seq

        return self._packed_attention(
            q_kv, mask, k_full, v_full, kv_head_idx, max_seq_len, max_seq_len,
        )

    def _fallback_full_attn(
        self,
        q_kv: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        kv_head_idx: int,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """No PQ cache: use all tokens within each sequence."""
        bs = q_kv.shape[0]
        device = q_kv.device
        pos = torch.arange(max_seq_len, device=device, dtype=torch.int32)
        seq_lens_gpu = seq_lens.to(device=device, dtype=torch.int32)
        mask = pos[None, :] < seq_lens_gpu[:, None]
        return self._packed_attention(
            q_kv, mask, k_full, v_full, kv_head_idx, max_seq_len, max_seq_len,
        )

    def _packed_attention(
        self,
        q_kv: torch.Tensor,
        mask: torch.Tensor,     # [bs, max_seq_len]
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        kv_head_idx: int,
        max_seq_len: int,
        max_seqlen_k_bound: int,
    ) -> torch.Tensor:
        bs, num_groups, head_dim = q_kv.shape
        device = q_kv.device

        kv_lens = mask.sum(dim=1).to(torch.int32)
        max_kv_len = int(kv_lens.max().item())

        if max_kv_len == 0:
            return torch.zeros_like(q_kv)

        cache_dtype = k_full.dtype
        kv_cache = torch.empty(
            bs, 1, 2, 1, max_kv_len, head_dim,
            dtype=cache_dtype, device=device,
        ).contiguous()

        for b in range(bs):
            sel = mask[b].nonzero(as_tuple=True)[0]
            n = sel.shape[0]
            if n > 0:
                kv_cache[b, 0, 0, 0, :n] = k_full[b, sel, kv_head_idx]
                kv_cache[b, 0, 1, 0, :n] = v_full[b, sel, kv_head_idx]

        q_4d = q_kv.unsqueeze(1)                            # [bs, 1, num_groups, head_dim]
        seq_lens_2d = kv_lens.to(torch.uint32).unsqueeze(1)  # [bs, 1]
        output = torch.empty_like(q_4d)

        if not hasattr(self, "_xqa_workspace") or self._xqa_workspace.device != device:
            self._xqa_workspace = torch.empty(64 << 20, dtype=torch.uint8, device=device)

        nb_seq = 1 * bs
        nb_semaphores = ((nb_seq + 1) // 2) * 2 + 2 + nb_seq + 2
        semaphores = torch.zeros(nb_semaphores, dtype=torch.uint32, device=device)

        q_scale = self.scaling * math.sqrt(head_dim)

        xqa_kwargs: dict = dict(
            num_kv_heads=1,
            max_seq_len=max_kv_len,
            q_scale=q_scale,
        )
        if cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            xqa_kwargs["kv_scale"] = torch.tensor(1.0, device=device)

        flashinfer.xqa_continuous(
            q_4d, kv_cache, seq_lens_2d, output,
            self._xqa_workspace, semaphores,
            **xqa_kwargs,
        )

        return output.squeeze(1)
