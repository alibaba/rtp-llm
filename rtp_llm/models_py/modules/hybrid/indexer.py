import os
import weakref
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from rtp_llm.models_py.modules import IndexerOp, LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.hybrid.topology_kv_policy import (
    TopologyKvPolicyConfig,
    apply_topology_kv_policy,
    normalize_topology_kv_policy,
)
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache
from rtp_llm.utils.model_weight import W

_TOPOLOGY_KV_STATE = weakref.WeakKeyDictionary()


def _topology_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _topology_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


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

        self.rope_head_dim = attn_config.rope_head_dim
        self.block_size = 128  # quantization block size (128)
        self.head_kv = 1
        self.scale_fmt = scale_fmt  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_scale = self.index_n_heads**-0.5
        self.blocksize = attn_config.kernel_tokens_per_block  # page size, typically 64
        self.topology_kv_policy = normalize_topology_kv_policy(
            os.getenv("RTP_LLM_TOPOLOGY_KV_POLICY", "disabled")
        )
        self.topology_sink_blocks = _topology_env_int("RTP_LLM_TOPOLOGY_SINK_BLOCKS", 1)
        self.topology_local_blocks = _topology_env_int(
            "RTP_LLM_TOPOLOGY_LOCAL_BLOCKS", 1
        )
        self.topology_witness_blocks = _topology_env_int(
            "RTP_LLM_TOPOLOGY_WITNESS_BLOCKS", 1
        )
        self.topology_max_policy_tokens = _topology_env_int(
            "RTP_LLM_TOPOLOGY_MAX_POLICY_TOKENS", 8192
        )
        self.topology_max_structural_fraction = _topology_env_float(
            "RTP_LLM_TOPOLOGY_MAX_STRUCTURAL_FRACTION", 0.5
        )
        self.topology_coordinate_mismatch_action = os.getenv(
            "RTP_LLM_TOPOLOGY_COORDINATE_MISMATCH_ACTION", "fallback_disabled"
        )
        self.topology_stable_scaffold = os.getenv("RTP_LLM_TOPOLOGY_STABLE_SCAFFOLD")
        self.topology_output_contract = os.getenv("RTP_LLM_TOPOLOGY_OUTPUT_CONTRACT")
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

    def _prefill_cp_enabled(self) -> bool:
        if self.parallelism_config is None:
            return False
        return self.parallelism_config.prefill_cp_config.is_enabled()

    def _is_sparse_prefill_cp(self, attention_inputs: Any) -> bool:
        return bool(attention_inputs.is_prefill) and self._prefill_cp_enabled()

    @property
    def latest_topology_kv_counters(self):
        state = _TOPOLOGY_KV_STATE.get(self)
        return state.counters if state is not None else None

    @property
    def latest_topology_kv_fingerprint(self):
        state = _TOPOLOGY_KV_STATE.get(self)
        return state.stable_fingerprint if state is not None else None

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

    def _apply_topology_kv_policy(
        self,
        topk_result: Optional[torch.Tensor],
        lengths: torch.Tensor,
        row_starts: Optional[torch.Tensor] = None,
        topk_indices_offset: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.topology_kv_policy == "disabled" or topk_result is None:
            return topk_result
        if (
            topk_result.is_cuda
            and os.getenv("RTP_LLM_TOPOLOGY_KV_ALLOW_CUDA_SYNC") != "1"
        ):
            return topk_result
        config = TopologyKvPolicyConfig(
            policy=self.topology_kv_policy,
            sink_blocks=self.topology_sink_blocks,
            local_blocks=self.topology_local_blocks,
            witness_blocks=self.topology_witness_blocks,
            block_size=self.blocksize,
            max_policy_tokens=self.topology_max_policy_tokens,
            max_structural_fraction=self.topology_max_structural_fraction,
            coordinate_mismatch_action=self.topology_coordinate_mismatch_action,
        )
        result = apply_topology_kv_policy(
            topk_result,
            lengths,
            config=config,
            row_starts=row_starts,
            topk_indices_offset=topk_indices_offset,
            stable_scaffold=self.topology_stable_scaffold,
            output_contract=self.topology_output_contract,
            previous_fingerprint=self.latest_topology_kv_fingerprint,
        )
        _TOPOLOGY_KV_STATE[self] = SimpleNamespace(
            counters=result.counters,
            stable_fingerprint=result.stable_fingerprint,
        )
        return result.topk_indices

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
            topk_result = self.indexer_op._get_topk_ragged_cp(
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
            return self._apply_topology_kv_policy(
                topk_result,
                cp_params.precomputed_lengths,
                row_starts=cp_params.precomputed_ks,
                topk_indices_offset=cp_params.precomputed_topk_off,
            )
        topk_result = self.indexer_op._get_topk_ragged(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs
        )
        return self._apply_topology_kv_policy(
            topk_result,
            fmha_params.expanded_seq_lens,
            row_starts=fmha_params.ks,
            topk_indices_offset=fmha_params.topk_indices_offset,
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
    ) -> torch.Tensor:
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
