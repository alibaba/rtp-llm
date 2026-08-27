from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from rtp_llm.models_py.modules import IndexerOp, LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.hybrid.indexer_compressor import (
    fp8_pool_view,
    fp32_state_pool_view,
)
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, KVCacheRegionName
from rtp_llm.utils.model_weight import W


class Indexer(nn.Module):
    """
    Indexer for DeepSeek-V3.2 DSA (DeepSeek Sparse Attention) mechanism.
    Adapted from sglang's Indexer implementation.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        parallelism_config: Optional[ParallelismConfig] = None,
        scale_fmt: Optional[str] = "none",
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.index_n_heads = attn_config.indexer_head_num
        self.index_head_dim = attn_config.indexer_head_dim
        self.index_topk = attn_config.indexer_topk
        self.compress_ratio = int(getattr(attn_config, "indexer_compress_ratio", 1))

        self.rope_head_dim = attn_config.rope_head_dim
        self.block_size = 128  # quantization block size (128)
        self.head_kv = 1
        self.scale_fmt = scale_fmt  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_scale = self.index_n_heads**-0.5
        self.blocksize = attn_config.kernel_tokens_per_block  # page size, typically 64
        self.indexer_size = self.index_head_dim / 2 + self.index_head_dim / 128 * 2
        self.is_neox_style = attn_config.rope_config.indexer_is_neox_style
        self.parallelism_config = parallelism_config

        self.wq_b = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_qb_w,
            W.mla_indexer_qb_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        if self.compress_ratio > 1:
            if self._prefill_cp_enabled():
                raise ValueError(
                    "GLM-5.3 compressed indexer does not support prefill CP"
                )
            from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8

            # Model-loader linear weights use [in, out]; the DSV4 compressor
            # and its direct F.linear projections use PyTorch [out, in].
            k_weight = weights[W.mla_indexer_k_w].T.contiguous()
            gate_weight = weights[W.mla_indexer_kpool_gate_w].T.contiguous()
            weights_projection = weights[W.mla_indexer_weights_proj_w].T.contiguous()
            self.compressed_indexer = IndexerFP8(
                dim=int(k_weight.shape[1]),
                q_lora_rank=int(weights[W.mla_indexer_qb_w].shape[0]),
                index_n_heads=self.index_n_heads,
                index_head_dim=self.index_head_dim,
                rope_head_dim=self.rope_head_dim,
                index_topk=self.index_topk,
                compress_ratio=self.compress_ratio,
                max_batch_size=1024,
                max_seq_len=int(global_weights[W.rope_cos_sin_cache].shape[0]),
                norm_eps=layernorm_eps,
                q_projection=self.wq_b,
                weights_projection=weights_projection,
                compressor_weights={
                    "ape": weights[W.mla_indexer_kpool_ape],
                    "wkv": k_weight,
                    "wgate": gate_weight,
                    # KPool normalizes raw keys before pooling; this tensor is
                    # unused by the post-pool writer but preserves its API.
                    "norm": torch.ones_like(weights[W.mla_indexer_k_norm_w]),
                },
                compressor_kpool_mode=True,
                compressor_pre_norm_weight=weights[W.mla_indexer_k_norm_w],
                compressor_pre_norm_bias=weights[W.mla_indexer_k_norm_b],
                rotate_q=True,
            )
            return

        self.wk = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_k_w,
            W.mla_indexer_k_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.k_norm = LayerNorm(
            weights[W.mla_indexer_k_norm_w],
            weights[W.mla_indexer_k_norm_b],
            eps=layernorm_eps,
        )

        self.weights_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_weights_proj_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.cos_sin_cache = global_weights[W.rope_cos_sin_cache]

        self.indexer_op = IndexerOp(
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            rope_head_dim=self.rope_head_dim,
            cos_sin_cache=self.cos_sin_cache,
            blocksize=self.blocksize,
            block_size=self.block_size,
            scale_fmt=self.scale_fmt,
            is_neox_style=self.is_neox_style,
        )

    @staticmethod
    def _fp8_pool_view(
        base: torch.Tensor, entry_bytes: int
    ) -> tuple[torch.Tensor, int]:
        return fp8_pool_view(base, entry_bytes)

    @staticmethod
    def _state_pool_view(
        base: torch.Tensor, state_width: int
    ) -> tuple[torch.Tensor, int]:
        return fp32_state_pool_view(base, state_width)

    def _bind_compressed_pools(
        self, global_kv_cache: KVCache, attention_inputs: Any
    ) -> tuple[torch.Tensor, int]:
        from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
            require_pool_tokens_per_block,
        )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            build_block_tables_batched,
        )

        block_tables = build_block_tables_batched(global_kv_cache, attention_inputs)
        if block_tables is None:
            raise RuntimeError(
                "GLM-5.3 compressed indexer block tables are unavailable"
            )
        kv_base = global_kv_cache.get_raw_pool_tensor(
            self.layer_idx, KVCacheRegionName.INDEXER_KV
        )
        state_base = global_kv_cache.get_raw_pool_tensor(
            self.layer_idx, KVCacheRegionName.INDEXER_STATE
        )
        kv_view, kv_eb = self._fp8_pool_view(kv_base, 132)
        state_view, state_eb = self._state_pool_view(
            state_base, 2 * self.index_head_dim
        )
        from rtp_llm.models_py.modules.dsv4.attn_type import INDEXER_KV, INDEXER_STATE

        kv_key = int(INDEXER_KV)
        state_key = int(INDEXER_STATE)
        kv_block_table = block_tables[kv_key]
        state_block_table = block_tables[state_key]
        self.compressed_indexer.set_pool_context(
            kv_view,
            kv_block_table,
            kv_eb,
            state_view,
            state_block_table,
            state_eb,
            state_tokens_per_block=require_pool_tokens_per_block(
                global_kv_cache, region=state_key
            ),
            kv_tokens_per_block=require_pool_tokens_per_block(
                global_kv_cache, region=kv_key
            ),
            kv_owner_tokens_per_block=int(global_kv_cache.seq_size_per_block),
        )
        return kv_block_table, kv_eb

    @staticmethod
    def _has_prefix(attention_inputs: Any) -> bool:
        prefix_host = getattr(attention_inputs, "prefix_lengths_host", None)
        if prefix_host is not None and prefix_host.numel():
            return any(int(value) > 0 for value in prefix_host.tolist())
        prefix = attention_inputs.prefix_lengths
        return bool(prefix.numel() and prefix.max().item() > 0)

    def _forward_compressed(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        fmha_params: Any,
        attention_inputs: Any,
        global_kv_cache: KVCache,
        use_fast_path: bool,
    ) -> Optional[torch.Tensor]:
        kv_block_table, kv_eb = self._bind_compressed_pools(
            global_kv_cache, attention_inputs
        )
        try:
            if attention_inputs.is_prefill:
                batch_size = int(attention_inputs.input_lengths.numel())
                meta = self.compressed_indexer.prepare(
                    bsz=1,
                    seqlen=int(hidden_states.shape[0]),
                    sp_int=0,
                    device=hidden_states.device,
                    kv_block_table=kv_block_table,
                    kv_eb=kv_eb,
                    use_varlen=True,
                    batch_size=batch_size,
                    cu_seqlens=attention_inputs.cu_seqlens,
                    input_lengths=attention_inputs.input_lengths,
                    prefix_lengths=attention_inputs.prefix_lengths,
                    position_ids=fmha_params.positions_d,
                    req_id_per_token=fmha_params.batch_indice_d,
                    has_prefix=self._has_prefix(attention_inputs),
                )
                topk = self.compressed_indexer(
                    hidden_states,
                    q_lora,
                    meta,
                    workspace=None,
                )
            else:
                hidden_3d = hidden_states.reshape(-1, 1, hidden_states.shape[-1])
                q_lora_3d = q_lora.reshape(-1, 1, q_lora.shape[-1])
                positions = fmha_params.positions_d.reshape(-1).to(torch.long)
                topk_buffer = torch.empty(
                    (hidden_3d.shape[0], 1, self.index_topk),
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
                topk = self.compressed_indexer.forward_decode_vectorized(
                    hidden_3d,
                    q_lora_3d,
                    positions,
                    topk_buffer,
                ).reshape(-1, self.index_topk)
            return None if use_fast_path else topk
        finally:
            self.compressed_indexer.clear_pool_context()

    def _prefill_cp_enabled(self) -> bool:
        if self.parallelism_config is None:
            return False
        return self.parallelism_config.prefill_cp_config.is_enabled()

    def _is_sparse_prefill_cp(self, attention_inputs: Any) -> bool:
        return bool(attention_inputs.is_prefill) and self._prefill_cp_enabled()

    # TODO: fuse kernel here
    def _get_logits_head_gate(
        self, x: torch.Tensor, q_scale: torch.Tensor
    ) -> torch.Tensor:
        x = x.float()
        weights = self.weights_proj(x)
        scale = self.softmax_scale * self.weights_scale
        weights = weights.unsqueeze(-1) * q_scale * scale
        return weights

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        flashmla_params: Any,
        cp_params: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        k = self.wk(x)
        k = self.k_norm(k)

        if self._prefill_cp_enabled():
            assert cp_params is not None
            query, key = self.indexer_op.apply_rope_and_rotate_q_k_cp(
                q,
                k,
                cp_params.full_rope_pos_ids,
            )
        else:
            positions = flashmla_params.positions_d
            query, key = self.indexer_op.apply_rope_and_rotate_q_k(q, k, positions)

        return query, key

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        flashmla_params: Any,
    ) -> torch.Tensor:
        k = self.wk(x)
        k = self.k_norm(k)
        return self.indexer_op.apply_rope_and_rotate_k(k, flashmla_params.positions_d)

    def _quantize_q_k(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None
            return self.indexer_op.quant_q_k_cp(
                query,
                key,
                kv_cache,
                fmha_params.slot_mapping,
                cp_params.kv_restore_unpad_indices,
            )
        return self.indexer_op.quant_q_k(query, key, kv_cache, fmha_params.slot_mapping)

    def _compute_topk(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
    ) -> torch.Tensor:
        if not attention_inputs.is_prefill:
            return self.indexer_op._get_topk_paged(
                q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
        if self._prefill_cp_enabled():
            assert cp_params is not None
            return self.indexer_op._get_topk_ragged_cp(
                q_fp8,
                weights,
                kv_cache,
                fmha_params,
                attention_inputs,
                cp_params.total_local_ids,
                cp_params.cu_kv_seqlens_global,
                cp_params.total_kv_len,
                cp_params.precomputed_ks,
                cp_params.precomputed_ke,
                cp_params.precomputed_lengths,
                cp_params.precomputed_topk_off,
            )
        return self.indexer_op._get_topk_ragged(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        use_fast_path: bool,
        cp_params: Any = None,
        global_kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        if self.compress_ratio > 1:
            if global_kv_cache is None:
                raise RuntimeError("GLM-5.3 compressed indexer requires global KVCache")
            return self._forward_compressed(
                hidden_states,
                q_lora,
                fmha_params,
                attention_inputs,
                global_kv_cache,
                use_fast_path,
            )
        if use_fast_path:
            key = self._get_k_bf16(hidden_states, fmha_params)
            self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)
            return None

        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None, "cp_params is required for sparse prefill CP"

        query, key = self._get_q_k_bf16(q_lora, hidden_states, fmha_params, cp_params)
        q_fp8, q_scale = self._quantize_q_k(
            query, key, kv_cache, fmha_params, attention_inputs, cp_params
        )
        weights = self._get_logits_head_gate(hidden_states, q_scale)
        return self._compute_topk(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs, cp_params
        )
