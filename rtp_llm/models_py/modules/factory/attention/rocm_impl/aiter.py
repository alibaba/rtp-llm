import logging
import math
from typing import Any, List, Optional

import aiter
import torch
from aiter_meta.csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import (
    pa_decode_gluon_aot,
)

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.attention.rocm_impl._attn_utils import (
    reshape_kv_cache_vectorized,
    split_qkv_fp8,
    split_raw_qkv,
    unpad_kv_vectorized,
)
from rtp_llm.ops import AttentionConfigs, FMHAType, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOpAsm,
    FusedRopeKVCacheDecodeOpNonAsm,
    FusedRopeKVCachePrefillOpAsm,
    FusedRopeKVCachePrefillOpNonAsm,
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    paged_attention_atrex,
)


# Pure Python implementation of FMHAParams
class FMHAParams(ParamsBase):
    """Python implementation of FMHAParams for Aiter attention operations."""

    def __init__(
        self,
        attn_inputs: PyAttentionInputs,
        is_prefill: bool = True,
        enable_cuda_graph: bool = True,
        graph_max_seq_len: Optional[int] = None,
        alloc_scale: bool = False,
    ):
        super().__init__()
        self.enable_cuda_graph = enable_cuda_graph
        self.graph_max_seq_len = graph_max_seq_len
        # avoids device alloc in forward under graph capture.
        self.kv_scale: Optional[torch.Tensor] = None

        # Prefill mode
        if is_prefill:
            input_lengths = attn_inputs.input_lengths
            prefix_lengths = (
                attn_inputs.prefix_lengths
                if hasattr(attn_inputs, "prefix_lengths")
                else None
            )

            self.max_seq_len = input_lengths.max().item()
            batch_size = input_lengths.size(0)

            # Create cu_seqlens on GPU directly to avoid per-layer .to(device) copies.
            # On ROCm each hipMemcpyWithStream costs ~1-8ms, so keeping these on GPU
            # from the start eliminates 28-layer × ~3ms/layer = ~84ms of sync overhead.
            # NOTE: input_lengths is CPU pinned memory in production; we must
            # explicitly target CUDA so cumsum and cu_seqlens live on GPU.
            gpu_device = torch.device("cuda")
            input_lengths_gpu = input_lengths.to(gpu_device, non_blocking=True)

            # Create cu_seqlens_q for query (based on input_lengths only)
            self.cu_seqlens_q = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=gpu_device
            )
            self.cu_seqlens_q[1:] = torch.cumsum(input_lengths_gpu, 0)

            # Create cu_seqlens_k for key/value (includes prefix_lengths)
            if prefix_lengths is not None and prefix_lengths.numel() > 0:
                prefix_lengths_gpu = prefix_lengths.to(gpu_device, non_blocking=True)
                kv_lengths_gpu = input_lengths_gpu + prefix_lengths_gpu
                self.cu_seqlens_k = torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=gpu_device
                )
                self.cu_seqlens_k[1:] = torch.cumsum(kv_lengths_gpu, 0)
                # Calculate max sequence length including prefix
                max_prefix_length = (
                    prefix_lengths.max().item() if prefix_lengths.numel() > 0 else 0
                )
                self.max_seqlen_k = self.max_seq_len + max_prefix_length
                kv_lengths = input_lengths + prefix_lengths
                # Hoist FMHA-setup tensor out of the per-layer hot path: with prefix,
                # seqlen_k = input_lengths + prefix_lengths (int32, on GPU).
                self.prefill_seqlen_k_int32 = kv_lengths_gpu.to(torch.int32)
            else:
                # No prefix, kv_lengths equals input_lengths
                kv_lengths = input_lengths
                self.cu_seqlens_k = self.cu_seqlens_q.clone()
                self.max_seqlen_k = self.max_seq_len
                # Hoist FMHA-setup tensor: with no prefix, seqlen_k == input_lengths
                # (int32 on GPU). Saves a per-layer alloc + add + dtype-cast trio.
                self.prefill_seqlen_k_int32 = input_lengths_gpu.to(torch.int32)

            self.max_seqlen_q = self.max_seq_len
            self.seq_lens = None
            self.kv_cache_block_id_device = getattr(
                attn_inputs, "kv_cache_kernel_block_id_device", None
            )
            self.prefix_lengths = prefix_lengths
            self.token_q_num = input_lengths.sum().item()
            self.token_kv_num = kv_lengths.sum().item()

            if alloc_scale:
                self.kv_scale = torch.ones(1, dtype=torch.float32, device=gpu_device)
        # Decode mode
        else:
            input_lengths = attn_inputs.input_lengths
            sequence_lengths = getattr(attn_inputs, "sequence_lengths", None)
            kv_cache_block_id_device = getattr(
                attn_inputs, "kv_cache_kernel_block_id_device", None
            )

            self.sequence_lengths = sequence_lengths
            self.kv_cache_block_id_device = kv_cache_block_id_device

            if (
                self.enable_cuda_graph
                and self.graph_max_seq_len is not None
                and self.graph_max_seq_len > 0
            ):
                self.max_seq_len = self.graph_max_seq_len
            elif sequence_lengths is not None:
                self.max_seq_len = sequence_lengths.max().item() + 1
            else:
                self.max_seq_len = input_lengths.max().item() + 1

            self.max_seqlen_k = self.max_seq_len
            self.max_seqlen_q = 0
            self.cu_seqlens_q = None
            self.cu_seqlens_k = None

            # Create seq_lens on CUDA
            if sequence_lengths is not None:
                self.seq_lens = (sequence_lengths + 1).to(torch.device("cuda"))
            else:
                self.seq_lens = None

            bid = self.kv_cache_block_id_device
            if bid is not None and alloc_scale:
                self.kv_scale = torch.ones(1, dtype=torch.float32, device=bid.device)

    def fillParams(
        self,
        sequence_lengths,
        input_lengths,
        kv_cache_block_id_host=None,
        kv_cache_block_id_device=None,
    ):
        self.sequence_lengths = sequence_lengths
        self.input_lengths = input_lengths
        self.kv_cache_block_id_host = kv_cache_block_id_host
        if kv_cache_block_id_device is not None:
            self.kv_cache_block_id_device = kv_cache_block_id_device
        if self.seq_lens is not None and self.sequence_lengths is not None:
            self.seq_lens.copy_((self.sequence_lengths + 1).to(torch.device("cuda")))
            if (
                self.enable_cuda_graph
                and self.graph_max_seq_len is not None
                and self.graph_max_seq_len > 0
            ):
                self.max_seq_len = self.graph_max_seq_len
            else:
                self.max_seq_len = self.sequence_lengths.max().item() + 1
            self.max_seqlen_k = self.max_seq_len

    def check_recycle(self) -> bool:
        """Check whether the params can be recycled automatically."""
        return True


class AiterPrefillAttnOp:
    def __init__(self, attn_configs: AttentionConfigs, v1_kv_layout: bool = False):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.tokens_per_block = attn_configs.kernel_tokens_per_block
        self.is_causal = attn_configs.is_causal
        self.v1_kv_layout = v1_kv_layout
        self._block_positions: Optional[torch.Tensor] = None
        self._compact_arange: Optional[torch.Tensor] = None

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self.fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
        )
        return self.fmha_params

    def _reshape_kv_cache_vectorized(self, kv_cache_base):
        """Reshape kv_cache_base into 5D VECTORIZED_LAYOUT for mha_batch_prefill.

        Handles both 2D flat buffer and 5D pre-shaped kv_cache_base.

        Returns (k_cache_5d, v_cache_5d):
            K: [num_blocks, num_kv_heads, head_dim/vs, page_size, vs]
            V: [num_blocks, num_kv_heads, page_size/vs, head_dim, vs]

        Note on V layout:
        - V1 non-FP8 (BASE): kernel uses non-template getVLocalIdx → linear [hd, ps].
          Needs permute to convert to vectorized [ps//vs, hd, vs].
        - V1 FP8: kernel uses getVLocalIdx<FP8> → already vectorized [ps//vs, hd, vs].
        - ASM (both BASE and FP8): kernel uses getVLocalIdx<CType> → vectorized.
        """
        block_num = kv_cache_base.shape[0]
        hk = self.head_num_kv
        ps = self.tokens_per_block
        hd = self.head_dim
        vs = 16 // kv_cache_base.element_size()

        # FP8 KV cache always uses vectorized layout (getKLocalIdx<FP8>/getVLocalIdx<FP8>),
        # regardless of v1_kv_layout flag.
        is_fp8 = kv_cache_base.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
        use_v1_linear_v = self.v1_kv_layout and not is_fp8

        if kv_cache_base.ndim >= 4:
            # Already shaped as [block_num, 2, hk, ps, hd] or similar multi-dim format.
            k_4d = kv_cache_base.select(1, 0)  # [block_num, hk, ps, hd]
            v_4d = kv_cache_base.select(1, 1)  # [block_num, hk, ps, hd]
            k_cache = k_4d.view(block_num, hk, hd // vs, ps, vs)
            if use_v1_linear_v:
                v_linear = v_4d.reshape(block_num, hk, hd, ps)
                v_cache = (
                    v_linear.reshape(block_num, hk, hd, ps // vs, vs)
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                )
            else:
                v_cache = v_4d.view(block_num, hk, ps // vs, hd, vs)
            return k_cache, v_cache

        # 2D flat buffer path
        expected_elems = 2 * hk * ps * hd
        flat = kv_cache_base[:, :expected_elems].reshape(block_num, 2, hk, ps * hd)

        # K: kernel writes via getKLocalIdx<CType> → vectorized [hd//vs, ps, vs].
        k_cache = flat[:, 0, :, :].view(block_num, hk, hd // vs, ps, vs)

        if use_v1_linear_v:
            # V1 non-FP8: kernel uses non-template getVLocalIdx → linear [hd, ps].
            v_linear = flat[:, 1, :, :].view(block_num, hk, hd, ps)
            v_cache = (
                v_linear.reshape(block_num, hk, hd, ps // vs, vs)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
        else:
            # ASM or FP8: kernel uses getVLocalIdx<CType> → vectorized [ps//vs, hd, vs].
            v_cache = flat[:, 1, :, :].view(block_num, hk, ps // vs, hd, vs)

        return k_cache, v_cache

    def _sanitize_block_table(
        self, block_table, block_num: int, seqlen_k=None, max_seqlen_k=None
    ):
        """Sanitize + pad block_table via shared helper."""
        if max_seqlen_k is None:
            max_seqlen_k = 0
        result, self._block_positions = _sanitize_and_pad_block_table(
            block_table,
            seqlen_k,
            self.tokens_per_block,
            max_seqlen_k,
            self._block_positions,
        )
        return result

    def _gather_and_reshape_kv_compact(self, kv_cache_base, block_table):
        """Gather referenced blocks once, then reshape to VECTORIZED_LAYOUT.

        For v1_kv_layout=True (non-ASM, non-FP8) path, the V cache needs a
        permute+contiguous to convert from linear [hd, ps] to vectorized
        [ps//vs, hd, vs] layout. Doing this on the full KV cache pool is
        extremely expensive. This method gathers all referenced blocks
        (including duplicates) from the current prefill batch and remaps
        block_table so mha_batch_prefill_func indexes into the compact
        buffer directly. Dedup via torch.unique was removed to eliminate
        stream-0 memset stalls caused by its internal workspace allocation.

        This also avoids the int32 offset overflow in aiter CK kernel when
        block_num * batch_stride_k > INT32_MAX (single-layer K cache > 2 GB).

        Args:
            kv_cache_base: Full pool — [block_num, 2, hk, ps, hd] (5D) or 2D flat.
            block_table:   [batch_size, max_blocks_per_seq] int32 on GPU.

        Returns:
            (k_compact, v_compact, compact_block_table)
        """
        hk = self.head_num_kv
        ps = self.tokens_per_block
        hd = self.head_dim
        vs = 16 // kv_cache_base.element_size()

        # Flatten block_table to get all referenced block indices.
        # NOTE: we intentionally skip torch.unique dedup here — unique's
        # internal workspace allocation triggers stream-0 memset stalls that
        # block the compute stream on every prefill batch. The tradeoff is
        # that prefix-sharing scenarios will gather duplicate blocks, inflating
        # the compact buffer up to block_table.numel() entries (still bounded
        # by the original KV pool size). If this becomes a memory bottleneck
        # under heavy prefix caching, consider a sort-based dedup path
        # (torch.sort + torch.diff) which avoids the memset issue.
        block_indices = block_table.reshape(-1).to(torch.int64)
        num_gathered = block_indices.numel()

        if kv_cache_base.ndim >= 4:
            # 5D path: [block_num, 2, hk, ps, hd]
            k_4d = kv_cache_base.select(1, 0)  # [block_num, hk, ps, hd]
            v_4d = kv_cache_base.select(1, 1)  # [block_num, hk, ps, hd]
            k_used = k_4d.index_select(0, block_indices)  # [n, hk, ps, hd]
            v_used = v_4d.index_select(0, block_indices)  # [n, hk, ps, hd]
            k_compact = k_used.view(num_gathered, hk, hd // vs, ps, vs)
            v_compact = (
                v_used.reshape(num_gathered, hk, hd, ps // vs, vs)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
        else:
            # 2D flat buffer path
            block_num = kv_cache_base.shape[0]
            expected_elems = 2 * hk * ps * hd
            flat = kv_cache_base[:, :expected_elems].reshape(block_num, 2, hk, ps * hd)
            k_used = flat[:, 0, :, :].index_select(0, block_indices)
            v_used = flat[:, 1, :, :].index_select(0, block_indices)
            k_compact = k_used.view(num_gathered, hk, hd // vs, ps, vs).contiguous()
            v_compact = (
                v_used.view(num_gathered, hk, hd, ps // vs, vs)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

        # Build identity block_table: [0, 1, 2, ...] so aiter indexes into
        # the compact buffer instead of the original full pool.
        cached_arange = self._compact_arange
        if (
            cached_arange is None
            or cached_arange.numel() < num_gathered
            or cached_arange.device != block_table.device
        ):
            cached_arange = torch.arange(
                max(num_gathered, 1024), dtype=torch.int32, device=block_table.device
            )
            self._compact_arange = cached_arange
        compact_block_table = cached_arange[:num_gathered].view_as(block_table)

        # Append one dummy zero-block to k/v compact buffers. CK kernel may
        # speculatively read beyond the last valid block_table entry; having a
        # trailing safe block prevents GPU page faults even if the read lands
        # on a padding column that maps to index num_gathered (one past end).
        k_pad = torch.zeros(
            (1, *k_compact.shape[1:]), dtype=k_compact.dtype, device=k_compact.device
        )
        v_pad = torch.zeros(
            (1, *v_compact.shape[1:]), dtype=v_compact.dtype, device=v_compact.device
        )
        k_compact = torch.cat([k_compact, k_pad], dim=0)
        v_compact = torch.cat([v_compact, v_pad], dim=0)

        return k_compact, v_compact, compact_block_table

    def _split_qkv_fp8(self, qkv_fp8):
        return split_qkv_fp8(qkv_fp8, self.head_num, self.head_num_kv, self.head_dim)

    def _split_raw_qkv(self, qkv, token_q_num, token_kv_num):
        return split_raw_qkv(
            qkv,
            self.head_num,
            self.head_num_kv,
            self.head_dim,
            token_q_num,
            token_kv_num,
        )

    def _forward_varlen(self, qkv, fmha_params):
        """Varlen path using flash_attn_varlen_func.

        Accepts either:
          - A packed QKV tensor (from qkv_proj): split into Q/K/V and reshape.
          - A tuple/list of (Q, K, V) tensors where Q is [tokens, H_q, D] and
            K/V may be 4D padded [B, H_kv, max_seqlen, D] (from C++ RoPE op).
            In that case, unpad K/V back to [total_tokens, H_kv, D].
        """
        if isinstance(qkv, (tuple, list)):
            query, key, value = qkv[0], qkv[1], qkv[2]
            # Ensure Q is 3D [tokens, heads, head_dim]
            if query.dim() == 2:
                query = query.view(-1, self.head_num, self.head_dim)
            # K/V from C++ FusedRopeKVCachePrefillOp are 4D padded
            # [B, H_kv, max_seqlen, D]. Unpad on device via vectorized gather to
            # avoid per-layer D2H sync and Python batch loop on the hot path.
            if key.dim() == 4 and value.dim() == 4:
                key, value = unpad_kv_vectorized(key, value, fmha_params.cu_seqlens_k)
            else:
                if key.dim() == 2:
                    key = key.view(-1, self.head_num_kv, self.head_dim)
                if value.dim() == 2:
                    value = value.view(-1, self.head_num_kv, self.head_dim)
        else:
            query, key, value = self._split_raw_qkv(
                qkv, fmha_params.token_q_num, fmha_params.token_kv_num
            )
        cu_seqlens_q = fmha_params.cu_seqlens_q
        cu_seqlens_k = fmha_params.cu_seqlens_k
        # Defensive: ensure cu_seqlens are on the same device as query
        if cu_seqlens_q.device != query.device:
            cu_seqlens_q = cu_seqlens_q.to(query.device, non_blocking=True)
        if cu_seqlens_k.device != query.device:
            cu_seqlens_k = cu_seqlens_k.to(query.device, non_blocking=True)
        res = aiter.flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            fmha_params.max_seqlen_q,
            fmha_params.max_seqlen_k,
            dropout_p=0.0,
            causal=self.is_causal,
        )
        return res.reshape(fmha_params.token_q_num, self.head_num * self.head_dim)

    def _forward_paged(self, q_tensor, kv_cache, fmha_params):
        """Paged prefill attention from paged KV cache using mha_batch_prefill_func.

        Supports BF16/FP16 Q or FP8 Q with FP8 KV cache (with descale).
        mha_batch_prefill_func handles mixed dtype via q_descale/k_descale/v_descale.
        """
        # Ensure Q is 3D [tokens, heads, head_dim] for mha_batch_prefill_func
        if q_tensor.dim() == 2:
            q_tensor = q_tensor.view(q_tensor.size(0), self.head_num, self.head_dim)

        is_fp8 = kv_cache.kv_cache_base.dtype in (
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
        )
        # cu_seqlens are already created on GPU in FMHAParams.__init__
        cu_seqlens_q = fmha_params.cu_seqlens_q
        # Ensure cu_seqlens_q is on the same device as q_tensor
        if cu_seqlens_q.device != q_tensor.device:
            cu_seqlens_q = cu_seqlens_q.to(q_tensor.device, non_blocking=True)

        # FMHA-setup tensor is pre-computed once per prefill in FMHAParams.__init__,
        # so the per-layer hot path skips a kernel/alloc trio (prefix-zeros allocation
        # + cu_seqlens diff + add+to_int32). The cached tensor is bit-exact identical
        # to the per-layer recomputation.
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_k = fmha_params.prefill_seqlen_k_int32
        if seqlen_k.device != q_tensor.device:
            seqlen_k = seqlen_k.to(q_tensor.device, non_blocking=True)

        # Sanitize padding columns + pad for CK speculative prefetch in one call.
        max_seqlen_q = fmha_params.max_seqlen_q
        max_seqlen_k = fmha_params.max_seqlen_k
        block_table = self._sanitize_block_table(
            fmha_params.kv_cache_block_id_device,
            kv_cache.kv_cache_base.shape[0],
            seqlen_k,
            max_seqlen_k,
        )

        use_compact = self.v1_kv_layout and not is_fp8

        if use_compact:
            k_cache, v_cache, block_table = self._gather_and_reshape_kv_compact(
                kv_cache.kv_cache_base, block_table
            )
        else:
            k_cache, v_cache = self._reshape_kv_cache_vectorized(kv_cache.kv_cache_base)

        softmax_scale = 1.0 / math.sqrt(self.head_dim)
        # kv_indptr must be all-zeros when kv_page_indices is empty (block_table
        # is used instead).  Previously cu_seqlens_q was passed here, causing
        # aiter to index into the empty kv_page_indices and trigger an
        # out-of-bounds assert on GPU.
        if (
            not hasattr(self, "_kv_indptr")
            or self._kv_indptr.shape[0] != batch_size + 1
        ):
            self._kv_indptr = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=cu_seqlens_q.device
            )
        kv_indptr = self._kv_indptr
        # Reuse cached empty tensor to avoid per-layer allocation
        if not hasattr(self, "_empty_kv_page_indices"):
            self._empty_kv_page_indices = torch.empty(
                0, dtype=torch.int32, device=cu_seqlens_q.device
            )
        kv_page_indices = self._empty_kv_page_indices

        # FP8 KV cache needs descale parameters for mha_batch_prefill_func
        q_descale = None
        k_descale = None
        v_descale = None
        if kv_cache.kv_cache_base.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            # Reuse cached ones tensor to avoid per-layer allocation
            if not hasattr(self, "_fp8_descale"):
                self._fp8_descale = torch.ones(
                    1, dtype=torch.float32, device=cu_seqlens_q.device
                )
            q_descale = self._fp8_descale
            k_descale = self._fp8_descale
            v_descale = self._fp8_descale

        res = aiter.mha_batch_prefill_func(
            q_tensor,
            k_cache,
            v_cache,
            cu_seqlens_q,
            kv_indptr,
            kv_page_indices,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
            block_table=block_table,
            seqlen_k=seqlen_k,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return res.reshape(fmha_params.token_q_num, self.head_num * self.head_dim)

    def forward(self, qkv, kv_cache, fmha_params):
        q_tensor = qkv[0] if isinstance(qkv, (tuple, list)) else qkv

        # Paged path: only when kv_cache is a real paged cache object (has kv_cache_base).
        # When qkv is a tuple (Q, K_padded, V_padded), we always use the varlen unpad path
        # regardless of kv_cache presence — the caller provides K/V explicitly.
        # NOTE on head_dim=256: aiter.mha_batch_prefill_func supports head_dim in {64,128,256}
        # since aiter 0.1.13.dev4 (commit 9ed3e3490). No varlen fallback needed here.
        if kv_cache is not None and hasattr(kv_cache, "kv_cache_base"):
            return self._forward_paged(q_tensor, kv_cache, fmha_params)

        # FP8 non-paged path: C++ returns full qkv_buf_fp8 (Q+K+V concatenated in FP8).
        # Split into Q/K/V and use flash_attn_varlen_fp8_pertensor_func.
        # Only used when kv_cache is None (e.g., encoder-only models).
        if q_tensor.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            query, key, value = self._split_qkv_fp8(q_tensor)
            # cu_seqlens are already on GPU from FMHAParams.__init__
            cu_seqlens_q = fmha_params.cu_seqlens_q
            cu_seqlens_k = fmha_params.cu_seqlens_k
            res = aiter.flash_attn_varlen_fp8_pertensor_func(
                query,
                key,
                value,
                None,
                None,
                None,
                cu_seqlens_q,
                cu_seqlens_k,
                fmha_params.max_seqlen_q,
                fmha_params.max_seqlen_k,
                causal=self.is_causal,
            )
            return res.reshape(fmha_params.token_q_num, self.head_num * self.head_dim)

        # Non-FP8 non-paged path (e.g., encoder-only BF16 models)
        return self._forward_varlen(qkv, fmha_params)


def _sanitize_and_pad_block_table(
    block_table: Optional[torch.Tensor],
    seqlen_k: Optional[torch.Tensor],
    tokens_per_block: int,
    max_seqlen_k: int,
    block_positions_cache: Optional[torch.Tensor] = None,
):
    """Shared helper: sanitize padding columns + pad for CK speculative prefetch.

    Only padding/speculative columns (beyond valid blocks per sequence) are
    filled with the last valid block id. Valid-mask entries are left untouched
    so that truly invalid block ids fail fast.

    Returns (sanitized_and_padded_block_table, updated_block_positions_cache).
    """
    if block_table is None:
        return None, block_positions_cache

    if seqlen_k is None or block_table.dim() != 2:
        return block_table, block_positions_cache

    if seqlen_k.device != block_table.device:
        seqlen_k = seqlen_k.to(block_table.device, non_blocking=True)

    max_blocks_per_seq = block_table.shape[1]
    if seqlen_k.numel() != block_table.shape[0]:
        return block_table, block_positions_cache

    # Sanitize: fill padding columns with last-valid-block-id
    if (
        block_positions_cache is None
        or block_positions_cache.numel() < max_blocks_per_seq
        or block_positions_cache.device != block_table.device
    ):
        block_positions_cache = torch.arange(
            max(max_blocks_per_seq, 1024),
            dtype=torch.int32,
            device=block_table.device,
        )

    valid_blocks = torch.div(
        seqlen_k + tokens_per_block - 1,
        tokens_per_block,
        rounding_mode="floor",
    ).to(torch.int32)
    positions = block_positions_cache[:max_blocks_per_seq].unsqueeze(0)
    valid_mask = positions < valid_blocks.unsqueeze(1)

    last_valid_col = (valid_blocks - 1).clamp(min=0).unsqueeze(1)
    last_valid_block_id = block_table.gather(1, last_valid_col.to(torch.int64))
    fill_value = last_valid_block_id.expand_as(block_table)
    block_table = torch.where(valid_mask, block_table, fill_value)

    # Pad: CK kernel speculatively prefetches V tiles beyond valid page entries.
    # Use kN0=128 (conservative upper bound for all head_dim configs).
    _CK_KN0 = 128
    extra_pages = (_CK_KN0 + tokens_per_block - 1) // tokens_per_block
    required_cols = (
        max_seqlen_k + tokens_per_block - 1
    ) // tokens_per_block + extra_pages
    if block_table.shape[1] < required_cols:
        pad_cols = required_cols - block_table.shape[1]
        last_col = block_table[:, -1:].expand(-1, pad_cols)
        block_table = torch.cat([block_table, last_col], dim=1)

    return block_table, block_positions_cache


def _infer_cuda_graph_device(
    attn_inputs: PyAttentionInputs,
    fmha_params: FMHAParams,
    fallback_tensor: Optional[torch.Tensor],
) -> torch.device:
    candidates = [
        getattr(attn_inputs, "input_lengths_d", None),
        getattr(attn_inputs, "prefix_lengths_d", None),
        getattr(attn_inputs, "decode_cu_seqlens_d", None),
        getattr(attn_inputs, "sequence_lengths_plus_1_d", None),
        getattr(attn_inputs, "kv_cache_kernel_block_id_device", None),
        getattr(attn_inputs, "kv_cache_block_id_device", None),
        getattr(fmha_params, "cu_seqlens_q", None),
        getattr(fmha_params, "cu_seqlens_k", None),
        fallback_tensor,
    ]
    for tensor in candidates:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            return tensor.device
    raise ValueError(
        "Failed to infer CUDA/HIP graph device from attn_inputs/fmha_params tensors"
    )


class AiterPrefillAttnOpPaged:
    """Paged prefill attention"""

    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.tokens_per_block = attn_configs.kernel_tokens_per_block
        self.enable_cuda_graph = False
        self.cuda_graph_prepared = False
        self.graph_device: Optional[torch.device] = None
        self.seqlen_k_buf: Optional[torch.Tensor] = None
        self.kv_indptr_buf: Optional[torch.Tensor] = None
        self.kv_page_indices_buf: Optional[torch.Tensor] = None
        self.descale_buf: Optional[torch.Tensor] = None
        self._block_positions: Optional[torch.Tensor] = None

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        has_prefix = (
            attn_inputs.prefix_lengths is not None
            and attn_inputs.prefix_lengths.numel() > 0
            and attn_inputs.prefix_lengths.max().item() > 0
        )
        return has_prefix

    def prepare(self, attn_inputs: PyAttentionInputs):
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
        )
        self.enable_cuda_graph = bool(getattr(attn_inputs, "is_cuda_graph", False))
        self.cuda_graph_prepared = False
        if self.enable_cuda_graph:
            self.prepare_cuda_graph(fmha_params, attn_inputs)
        return fmha_params

    def prepare_cuda_graph(
        self, fmha_params: FMHAParams, attn_inputs: PyAttentionInputs
    ) -> None:
        graph_block_table = getattr(
            attn_inputs, "kv_cache_kernel_block_id_device", None
        )
        if graph_block_table is None:
            graph_block_table = getattr(attn_inputs, "kv_cache_block_id_device", None)
        self.graph_device = _infer_cuda_graph_device(
            attn_inputs, fmha_params, graph_block_table
        )
        fmha_params.cu_seqlens_q = fmha_params.cu_seqlens_q.to(
            device=self.graph_device, dtype=torch.int32
        )
        fmha_params.cu_seqlens_k = fmha_params.cu_seqlens_k.to(
            device=self.graph_device, dtype=torch.int32
        )
        fmha_params.kv_cache_block_id_device = fmha_params.kv_cache_block_id_device.to(
            device=self.graph_device, dtype=torch.int32
        )
        batch_size = fmha_params.cu_seqlens_q.shape[0] - 1
        if self.seqlen_k_buf is None or self.seqlen_k_buf.shape[0] < batch_size:
            self.seqlen_k_buf = torch.empty(
                max(1, batch_size), dtype=torch.int32, device=self.graph_device
            )
        if self.kv_indptr_buf is None or self.kv_indptr_buf.shape[0] < batch_size + 1:
            self.kv_indptr_buf = torch.zeros(
                max(1, batch_size + 1), dtype=torch.int32, device=self.graph_device
            )
        if self.kv_page_indices_buf is None:
            self.kv_page_indices_buf = torch.zeros(
                1, dtype=torch.int32, device=self.graph_device
            )
        if self.descale_buf is None:
            self.descale_buf = torch.ones(
                1, dtype=torch.float32, device=self.graph_device
            )
        # Pre-allocate a fixed-address buffer for the sanitized+padded block_table.
        # CUDA graph replay requires stable tensor addresses; sanitize produces a
        # new tensor each call, so we copy_ into this fixed buffer.
        bt = fmha_params.kv_cache_block_id_device
        extra_pages = (128 + self.tokens_per_block - 1) // self.tokens_per_block
        max_cols = bt.shape[1] + extra_pages
        self.sanitized_bt_buf = torch.zeros(
            batch_size, max_cols, dtype=torch.int32, device=self.graph_device
        )
        self.cuda_graph_prepared = True

    def forward(self, qkv, kv_cache, fmha_params) -> torch.Tensor:
        # NOTE: This is a *prefill*-stage operator (handles prefix-cache prefill).
        # The graph_ready branches below are interface-compatible scaffolding for
        # potential future CUDA-graph-captured prefill; in production, CUDA graph
        # capture only happens in the decode stage (AiterDecodeAttnOp), so the
        # graph_ready path here is never triggered and does not require dedicated
        # regression tests.
        q_tensor = qkv[0][: fmha_params.token_q_num]
        device = q_tensor.device

        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)

        x = 16 // key_cache.element_size()
        kv_sizes = key_cache.shape
        key_cache = key_cache.view(
            kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x
        )
        value_cache = value_cache.view(
            kv_sizes[0], kv_sizes[1], kv_sizes[2] // x, kv_sizes[3], x
        )

        graph_ready = self.enable_cuda_graph and self.cuda_graph_prepared
        if graph_ready:
            cu_seqlens_q = fmha_params.cu_seqlens_q
            cu_seqlens_k = fmha_params.cu_seqlens_k
        else:
            cu_seqlens_q = fmha_params.cu_seqlens_q.to(device)
            cu_seqlens_k = fmha_params.cu_seqlens_k.to(device)
        batch_size = cu_seqlens_q.shape[0] - 1

        if graph_ready:
            torch.sub(cu_seqlens_k[1:], cu_seqlens_k[:-1], out=self.seqlen_k_buf)
            seqlen_k = self.seqlen_k_buf
        else:
            seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)

        if graph_ready:
            block_table = fmha_params.kv_cache_block_id_device
        else:
            block_table = fmha_params.kv_cache_block_id_device.to(
                dtype=torch.int32, device=device
            )

        max_seqlen_q = fmha_params.max_seqlen_q
        max_seqlen_k = fmha_params.max_seqlen_k

        # Apply sanitize + pad to prevent CK speculative prefetch OOB.
        sanitized_bt, self._block_positions = _sanitize_and_pad_block_table(
            block_table,
            seqlen_k,
            self.tokens_per_block,
            max_seqlen_k,
            self._block_positions,
        )
        if graph_ready:
            # CUDA graph replay requires stable tensor addresses. Copy the
            # sanitized result into the pre-allocated fixed-address buffer.
            cols = sanitized_bt.shape[1]
            self.sanitized_bt_buf[:, :cols] = sanitized_bt
            block_table = self.sanitized_bt_buf[:, :cols]
        else:
            block_table = sanitized_bt

        if graph_ready:
            self.kv_indptr_buf.zero_()
            kv_indptr = self.kv_indptr_buf
            kv_page_indices = self.kv_page_indices_buf
        else:
            kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            kv_page_indices = torch.zeros(1, dtype=torch.int32, device=device)

        q_descale = None
        k_descale = None
        v_descale = None
        if key_cache.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            if graph_ready:
                q_descale = self.descale_buf
                k_descale = self.descale_buf
                v_descale = self.descale_buf
            else:
                q_descale = torch.ones(1, dtype=torch.float32, device=device)
                k_descale = torch.ones(1, dtype=torch.float32, device=device)
                v_descale = torch.ones(1, dtype=torch.float32, device=device)

        res = aiter.mha_batch_prefill_func(
            q_tensor,
            key_cache,
            value_cache,
            cu_seqlens_q,
            kv_indptr,
            kv_page_indices,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
            block_table=block_table,
            seqlen_k=seqlen_k,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

        token_num = fmha_params.token_q_num
        return res.reshape(token_num, self.head_num * self.head_dim)


def _run_triton_paged_attention(
    query: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    kv_scale_base,
    num_seqs: int,
    query_length: int,
    seq_lens: torch.Tensor,
    block_tables_id_device: torch.Tensor,
    max_seq_len: int,
    num_kv_heads: int,
    context_partition_size: int,
    kv_scale_buf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    key_cache = paged_kv_cache.select(1, 0)
    value_cache = paged_kv_cache.select(1, 1)

    x = 16 // key_cache.element_size()
    kv_sizes = key_cache.shape
    key_cache = key_cache.view(
        kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x
    )
    value_cache = value_cache.view(
        kv_sizes[0], kv_sizes[1], kv_sizes[2] // x, kv_sizes[3], x
    )

    key_scale, value_scale = None, None
    if kv_scale_base is not None:
        kv_b = kv_scale_buf
        if kv_b is None or kv_b.device != query.device:
            kv_b = torch.ones(1, dtype=torch.float32, device=query.device)
        key_scale = kv_b
        value_scale = kv_b

    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    query_group_size = num_query_heads // num_kv_heads

    query_dtype = query.dtype
    compute_type = (
        torch.bfloat16
        if query_dtype
        not in (
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
            torch.bfloat16,
            torch.float16,
        )
        else query_dtype
    )
    output_dtype = (
        torch.bfloat16
        if query_dtype
        in (
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
        )
        else query_dtype
    )

    softmax_scale = 1.0 / (head_size**0.5)
    max_context_partition_num = (
        max_seq_len + context_partition_size - 1
    ) // context_partition_size
    equivalent_query_group_size = query_length * query_group_size

    output = torch.empty(
        (num_seqs * query_length, num_query_heads, head_size),
        dtype=output_dtype,
        device=query.device,
    )
    exp_sums = torch.zeros(
        (
            num_seqs,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
        ),
        dtype=torch.float32,
        device=query.device,
    )
    max_logits = torch.full(
        (
            num_seqs,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
        ),
        -float("inf"),
        dtype=torch.float32,
        device=query.device,
    )
    temporary_output = torch.zeros(
        (
            num_seqs,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
            head_size,
        ),
        dtype=output_dtype,
        device=query.device,
    )

    context_lengths = seq_lens.to(dtype=torch.int32, device=query.device)
    block_tables = block_tables_id_device.to(dtype=torch.int32, device=query.device)

    if query.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
        q_b = kv_scale_buf
        if q_b is None or q_b.device != query.device:
            q_b = torch.ones(1, device=query.device, dtype=torch.float32)
        query_scale = q_b
    else:
        query_scale = None

    pa_decode_gluon_aot(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lengths=context_lengths,
        block_tables=block_tables,
        softmax_scale=softmax_scale,
        query_length=query_length,
        max_context_partition_num=max_context_partition_num,
        context_partition_size=context_partition_size,
        compute_type=compute_type,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        alibi_slopes=None,
        sinks=None,
    )
    return output


class AiterPrefillAttnOpTriton:
    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.context_partition_size = 256
        self.alloc_scale = attn_configs.kv_cache_dtype == KvCacheDataType.FP8

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        has_prefix = (
            attn_inputs.prefix_lengths is not None
            and attn_inputs.prefix_lengths.numel() > 0
            and attn_inputs.prefix_lengths.max().item() > 0
        )
        return has_prefix

    def prepare(self, attn_inputs: PyAttentionInputs):
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
            alloc_scale=self.alloc_scale,
        )
        return fmha_params

    def forward(self, qkv, kv_cache, fmha_params) -> torch.Tensor:
        block_tables_id_device = fmha_params.kv_cache_block_id_device
        num_seqs = (
            block_tables_id_device.shape[0] if block_tables_id_device is not None else 1
        )
        query = qkv[0]
        token_num = query.shape[0]
        device = query.device

        # cu_seqlens are already on GPU from FMHAParams.__init__
        cu_seqlens_q = fmha_params.cu_seqlens_q
        q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
        max_q_len = fmha_params.max_seqlen_q
        real_token_num = fmha_params.token_q_num

        cu_seqlens_k = fmha_params.cu_seqlens_k
        seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

        output = _run_triton_paged_attention(
            query,
            kv_cache.kv_cache_base,
            kv_cache.kv_scale_base,
            num_seqs,
            max_q_len,
            seq_lens,
            block_tables_id_device,
            fmha_params.max_seqlen_k,
            self.head_num_kv,
            self.context_partition_size,
            kv_scale_buf=fmha_params.kv_scale,
        )

        if token_num != real_token_num:
            seq_ids = torch.arange(num_seqs, device=device).repeat_interleave(q_lens)
            within_seq_pos = (
                torch.arange(real_token_num, device=device) - cu_seqlens_q[seq_ids]
            )
            dst_indices = (
                seq_ids * max_q_len + (max_q_len - q_lens[seq_ids]) + within_seq_pos
            )
            output = output[dst_indices]

        return output.view(real_token_num, -1)


class AiterDecodeAttnOpBase:
    """Base class for Aiter decode attention operations."""

    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.tokens_per_block = attn_configs.kernel_tokens_per_block
        self.max_seq_len = attn_configs.max_seq_len
        self.alloc_scale = True
        # Updated per-request in prepare(); keep default false to avoid stale state
        # before first input is prepared.
        self.enable_cuda_graph = False
        # Pre-allocated output tensor for CUDA graph capture/replay stability.
        # Allocated once in prepare() when enable_cuda_graph is True, then reused
        # in every forward() call so that graph replay always writes to the same
        # device address captured during graph recording.
        self._graph_output: Optional[torch.Tensor] = None
        self._default_scale: Optional[torch.Tensor] = None

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        if not self.enable_cuda_graph:
            self._graph_output = None
        # Create decode parameters using pure Python implementation
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=False,
            enable_cuda_graph=self.enable_cuda_graph,
            graph_max_seq_len=self.max_seq_len,
            alloc_scale=self.alloc_scale,
        )
        fmha_params.max_seqlen_k = fmha_params.max_seq_len

        return fmha_params

    def _get_output(self, query: torch.Tensor) -> torch.Tensor:
        """Return a pre-allocated output tensor when running under CUDA graph,
        or a freshly allocated one otherwise.

        On the first call under graph mode the tensor is lazily allocated so
        that dtype/device are derived from the actual query tensor.  Subsequent
        calls reuse the same storage so that graph replay always writes to the
        address captured during recording."""
        if self.enable_cuda_graph:
            if self._graph_output is None or self._graph_output.shape != query.shape:
                self._graph_output = torch.empty_like(query)
            return self._graph_output
        return torch.empty_like(query)

    def reshape_kv_cache(self, paged_kv_cache):
        return common.reshape_paged_kv_cache(
            paged_kv_cache, self.head_num_kv, self.tokens_per_block, self.head_dim
        )


class AiterDecodeAttnOpAsm(AiterDecodeAttnOpBase):
    """Aiter decode attention operation using ASM paged attention."""

    def __init__(self, attn_configs: AttentionConfigs):
        super().__init__(attn_configs)
        self.alloc_scale = False

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[LayerKVCache], fmha_params
    ) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens

        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)
        block_tables_id_device = fmha_params.kv_cache_block_id_device
        max_num_blocks = block_tables_id_device.shape[1]
        K_QScale = None
        V_QScale = None
        if key_cache.dtype in (
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
        ) and value_cache.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            K_QScale = kv_cache.kv_scale_base.select(1, 0)
            V_QScale = kv_cache.kv_scale_base.select(1, 1)
        out_ = self._get_output(query)
        output = aiter.pa_fwd_asm(
            query,  # [num_seqs, num_heads, head_size]
            key_cache,  # [num_blocks, num_kv_heads, block_size, head_size/x, x]
            value_cache,  # [num_blocks, num_kv_heads, block_size, head_size/x, x]
            block_tables_id_device,
            seq_lens,
            max_num_blocks,
            1,
            K_QScale,
            V_QScale,
            out_,
            None,
            0,
        )
        output_reshaped = output.view(output.shape[0], -1)
        return output_reshaped


class AiterDecodeAttnOpNonAsm(AiterDecodeAttnOpBase):
    """Aiter decode attention operation using non-ASM paged attention."""

    def __init__(self, attn_configs: AttentionConfigs):
        super().__init__(attn_configs)
        # Non-ASM fallback scale is only needed for non-FP8 KV cache:
        # - FP8 KV cache use kv_cache.kv_scale_base directly.
        # - non-FP8 KV cache: pass a pre-allocated unit scale (1.0) as fallback.
        self.alloc_scale = attn_configs.kv_cache_dtype != KvCacheDataType.FP8
        # Per-instance so TP ranks on different devices don't share a tensor
        # pinned to the first caller's device.
        self._default_scale: Optional[torch.Tensor] = None

    def _get_default_scale(self, query: torch.Tensor) -> torch.Tensor:
        if self._default_scale is None:
            # paged_attention_rocm requires k_scale / v_scale in float32
            # (matches K_QScale / V_QScale dtype from kv_scale_base).
            self._default_scale = torch.tensor(
                1.0, device=query.device, dtype=torch.float32
            )
        return self._default_scale

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[LayerKVCache], fmha_params
    ) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens
        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)

        K_QScale = None
        V_QScale = None
        using_fp8_kvcache = False
        if key_cache.dtype in (
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
        ) and value_cache.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            K_QScale = kv_cache.kv_scale_base.select(1, 0)
            V_QScale = kv_cache.kv_scale_base.select(1, 1)
            using_fp8_kvcache = True

        block_tables_id_device = fmha_params.kv_cache_block_id_device

        max_seq_len = fmha_params.max_seq_len
        scale = 1.0 / (self.head_dim**0.5)
        alibi_slopes = None
        num_kv_heads = self.head_num_kv
        num_seqs, num_heads, head_size = query.shape
        block_size = value_cache.shape[2]
        output = self._get_output(query).view((num_seqs, num_heads, head_size))
        if max_seq_len <= 16384 and (not using_fp8_kvcache) and head_size <= 128:
            _PARTITION_SIZE_ROCM = 512
            max_num_partitions = (
                max_seq_len + _PARTITION_SIZE_ROCM - 1
            ) // _PARTITION_SIZE_ROCM
            x = 16 // key_cache.element_size()
            grp_size = num_heads // num_kv_heads
            kv_sizes = value_cache.shape
            exp_sums = torch.empty(
                size=(num_seqs, num_kv_heads, max_num_partitions, grp_size),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            # init tmp_output
            tmp_output = torch.empty(
                size=(num_seqs, num_kv_heads, max_num_partitions, grp_size, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            query = query.view((num_seqs, num_heads, head_size))
            key_cache = key_cache.view(
                (kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x)
            )
            value_cache = value_cache.view(
                (kv_sizes[0], kv_sizes[1], kv_sizes[3], kv_sizes[2])
            )
            paged_attention_atrex(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                seq_lens,
                block_tables_id_device,
                scale,
                max_seq_len,
                alibi_slopes,
            )
        else:
            _PARTITION_SIZE_ROCM = 256

            max_num_partitions = (
                max_seq_len + _PARTITION_SIZE_ROCM - 1
            ) // _PARTITION_SIZE_ROCM
            assert _PARTITION_SIZE_ROCM % block_size == 0
            # output already allocated above via _get_output(query); reuse it here.
            # init tmp_output
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )

            # init exp_sums
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            fp8_out_scale = None
            cpa_fp8_out = False
            # init max_logits
            max_logits = torch.ones_like(exp_sums)

            kv_cache_dtype = "fp8" if using_fp8_kvcache else "auto"
            default_scale = self._get_default_scale(query)
            k_scale = K_QScale if kv_cache and K_QScale is not None else default_scale
            v_scale = V_QScale if kv_cache and V_QScale is not None else default_scale
            aiter.paged_attention_rocm(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                float(scale),
                block_tables_id_device,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,  # kv_cache_dtype
                k_scale,
                v_scale,
                fp8_out_scale if cpa_fp8_out else None,
                _PARTITION_SIZE_ROCM,
            )

        output_reshaped = output.view(output.shape[0], -1)
        return output_reshaped


class AiterDecodeAttnOpTriton(AiterDecodeAttnOpBase):

    def __init__(self, attn_configs: AttentionConfigs):
        super().__init__(attn_configs)
        self.alloc_scale = attn_configs.kv_cache_dtype == KvCacheDataType.FP8
        self.context_partition_size = 256

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[LayerKVCache], fmha_params
    ) -> torch.Tensor:
        num_seqs = query.shape[0]
        paged_kv_cache = self.reshape_kv_cache(kv_cache.kv_cache_base)
        output = _run_triton_paged_attention(
            query,
            paged_kv_cache,
            kv_cache.kv_scale_base,
            num_seqs,
            1,
            fmha_params.seq_lens,
            fmha_params.kv_cache_block_id_device,
            fmha_params.max_seq_len,
            self.head_num_kv,
            self.context_partition_size,
            kv_scale_buf=fmha_params.kv_scale,
        )
        return output.view(num_seqs, -1)


class AiterPrefillImplAsm(FMHAImplBase):
    """Aiter prefill attention implementation using ASM."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterPrefillAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpAsm(attn_configs)
        self.rope_kvcache_impl.use_paged_fmha = True

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if kv_cache is None:
            # Embedding models still need positional encoding even without a KV cache.
            if self.need_rope_kv_cache:
                fmha_input = self.rope_kvcache_impl.forward(
                    qkv, kv_cache, self.rope_params
                )
            else:
                fmha_input = qkv
            return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class AiterPrefillImplNonAsm(FMHAImplBase):
    """Aiter prefill attention implementation using non-ASM."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterPrefillAttnOp(attn_configs, v1_kv_layout=True)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpNonAsm(attn_configs)
        self.rope_kvcache_impl.use_paged_fmha = True

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if kv_cache is None:
            # Embedding models still need positional encoding even without a KV cache.
            if self.need_rope_kv_cache:
                fmha_input = self.rope_kvcache_impl.forward(
                    qkv, kv_cache, self.rope_params
                )
            else:
                fmha_input = qkv
            return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class AiterPrefillImplPaged(FMHAImplBase):
    """Paged prefill impl: dispatches between CK batch-prefill and Triton PA at runtime.

    - seq_len <= 4: Triton PA (short query optimization)
    - Otherwise: CK batch-prefill (general paged prefill)
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.head_num_kv = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.tokens_per_block = attn_configs.kernel_tokens_per_block

        self.batch_prefill_impl = AiterPrefillAttnOpPaged(attn_configs)
        self.triton_prefill_impl = AiterPrefillAttnOpTriton(attn_configs)

        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpAsm(attn_configs)
        self.rope_kvcache_impl.use_paged_fmha = True

        self.attn_inputs = attn_inputs
        self.fmha_params = self.batch_prefill_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        pl = attn_inputs.prefix_lengths
        if pl is None or pl.numel() == 0:
            return False
        return int(pl.max().item()) > 0

    def _update_prefill_params_for_cuda_graph(
        self, attn_inputs: PyAttentionInputs
    ) -> None:
        input_lengths = attn_inputs.input_lengths

        fmha_params = self.fmha_params
        expected_batch = fmha_params.cu_seqlens_q.numel() - 1

        live_cu_seqlens_q = getattr(attn_inputs, "cu_seqlens", None)
        live_cu_seqlens_k = getattr(attn_inputs, "cu_kv_seqlens", None)
        use_live_cu_seqlens = (
            live_cu_seqlens_q is not None
            and live_cu_seqlens_k is not None
            and live_cu_seqlens_q.numel() == expected_batch + 1
            and live_cu_seqlens_k.numel() == expected_batch + 1
        )

        if use_live_cu_seqlens:
            fmha_params.cu_seqlens_q.copy_(
                live_cu_seqlens_q.to(
                    device=fmha_params.cu_seqlens_q.device, dtype=torch.int32
                )
            )
            fmha_params.cu_seqlens_k.copy_(
                live_cu_seqlens_k.to(
                    device=fmha_params.cu_seqlens_k.device, dtype=torch.int32
                )
            )
            q_lengths = fmha_params.cu_seqlens_q[1:] - fmha_params.cu_seqlens_q[:-1]
            kv_lengths = fmha_params.cu_seqlens_k[1:] - fmha_params.cu_seqlens_k[:-1]
            prefix_lengths = kv_lengths - q_lengths
        else:
            q_lengths = input_lengths.to(
                device=fmha_params.cu_seqlens_q.device, dtype=torch.int32
            )
            fmha_params.cu_seqlens_q.zero_()
            fmha_params.cu_seqlens_q[1:].copy_(torch.cumsum(q_lengths, dim=0))

            prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
            if prefix_lengths is None:
                prefix_lengths = torch.zeros_like(q_lengths)
            else:
                if prefix_lengths.shape[0] != expected_batch:
                    raise ValueError(
                        "AiterPrefillImplPaged CUDA graph replay prefix length mismatch: "
                        f"capture={expected_batch}, replay={prefix_lengths.shape[0]}"
                    )
                prefix_lengths = prefix_lengths.to(
                    device=fmha_params.cu_seqlens_k.device, dtype=torch.int32
                )

            kv_lengths = q_lengths + prefix_lengths
            fmha_params.cu_seqlens_k.zero_()
            fmha_params.cu_seqlens_k[1:].copy_(torch.cumsum(kv_lengths, dim=0))

        fmha_params.prefix_lengths = prefix_lengths

        q_lens = input_lengths
        pl_src = getattr(attn_inputs, "prefix_lengths", None)
        if pl_src is not None and pl_src.numel() >= expected_batch:
            p_lens = pl_src[:expected_batch]
        else:
            p_lens = torch.zeros_like(q_lens)
        kv_lens = q_lens + p_lens.to(dtype=q_lens.dtype)
        fmha_params.max_seq_len = int(q_lens.max()) if expected_batch > 0 else 0
        fmha_params.max_seqlen_q = fmha_params.max_seq_len
        fmha_params.max_seqlen_k = int(kv_lens.max()) if expected_batch > 0 else 0
        fmha_params.token_q_num = int(q_lens.sum())
        fmha_params.token_kv_num = int(kv_lens.sum())

        # Sync prefill_seqlen_k_int32 from updated cu_seqlens_k so that
        # CUDA graph replay does not reuse stale seqlen_k values.
        fmha_params.prefill_seqlen_k_int32 = (
            fmha_params.cu_seqlens_k[1:] - fmha_params.cu_seqlens_k[:-1]
        ).to(torch.int32)

        kv_block_id = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
        if kv_block_id is None:
            kv_block_id = getattr(attn_inputs, "kv_cache_block_id_device", None)
        if kv_block_id is None:
            raise ValueError(
                "AiterPrefillImplPaged.prepare_cuda_graph requires kv cache block ids"
            )
        fmha_params.kv_cache_block_id_device = kv_block_id

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.attn_inputs = attn_inputs
        self._update_prefill_params_for_cuda_graph(attn_inputs)

        self.batch_prefill_impl.prepare_cuda_graph(self.fmha_params, attn_inputs)

        prepare_in_place = getattr(self.rope_params, "prepare_in_place", None)
        if callable(prepare_in_place):
            prepare_in_place(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        cu_seqlens_q = self.fmha_params.cu_seqlens_q
        batch_size = cu_seqlens_q.shape[0] - 1
        max_q_len = int(self.fmha_params.max_seqlen_q) if batch_size > 0 else 0
        token_num = int(self.fmha_params.token_q_num) if batch_size > 0 else 0
        use_triton = (
            False
            if self.batch_prefill_impl.enable_cuda_graph
            else (batch_size > 0 and 0 < max_q_len <= 4)
        )

        if self.need_rope_kv_cache:
            self.rope_kvcache_impl.pad_query = (
                use_triton and token_num != batch_size * max_q_len
            )
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        kv_cache.kv_cache_base = common.reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            self.head_num_kv,
            self.tokens_per_block,
            self.head_dim,
        )

        if use_triton:
            return self.triton_prefill_impl.forward(
                fmha_input, kv_cache, self.fmha_params
            )
        else:
            return self.batch_prefill_impl.forward(
                fmha_input, kv_cache, self.fmha_params
            )


class AiterDecodeImplBase(FMHAImplBase):
    fmha_params: Any
    rope_params: Any

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        # Replay path must reuse capture-time FMHA params object to keep graph memory stable.
        self.fmha_params.fillParams(
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            attn_inputs.kv_cache_kernel_block_id_device,
        )
        if attn_inputs.kv_cache_kernel_block_id_device is not None:
            update_kv_cache_offset = getattr(
                self.rope_params, "update_kv_cache_offset", None
            )
            if not callable(update_kv_cache_offset):
                raise TypeError(
                    "AiterDecodeImplBase.prepare_cuda_graph expects rope_params to provide "
                    "update_kv_cache_offset(kv_cache_kernel_block_id_device)"
                )
            update_kv_cache_offset(attn_inputs.kv_cache_kernel_block_id_device)


class AiterDecodeImplAsm(AiterDecodeImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterDecodeAttnOpAsm(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOpAsm(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class AiterDecodeImplNonAsm(AiterDecodeImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterDecodeAttnOpNonAsm(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOpNonAsm(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class AiterDecodeImplTriton(AiterDecodeImplBase):
    """Aiter decode attention implementation using Triton."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterDecodeAttnOpTriton(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOpAsm(attn_configs)

        self.attn_inputs = attn_inputs

        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
