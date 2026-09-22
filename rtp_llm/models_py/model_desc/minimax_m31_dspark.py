"""MiniMax-M3.1 DSpARK draft model.

The first bring-up deliberately supports BF16/FP8 paged MSA cache only.  It
uses the shared :class:`DSparkProposerMixin` contract and keeps MiniMax-owned
K/V plus index-K projection/write details inside this module.
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.minimax_m31 import (
    MiniMaxM31DecoderLayer,
    MiniMaxM31Model,
)
from rtp_llm.models_py.modules import LinearFactory
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.hybrid.msa_attention import (
    MSAAttention,
    _gemma_rmsnorm_per_head,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import (
    DSparkProposerMixin,
    primary_attention_inputs,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class _MiniMaxM31DSparkQueryContext:
    """Signals non-causal fixed-width proposal attention to draft layers."""

    fmha_params = None

    def prepare_cuda_graph(self, _attn_inputs) -> None:
        return None

    def support_cuda_graph(self) -> bool:
        return True


class MiniMaxM31DSparkDecoderLayer(MiniMaxM31DecoderLayer):
    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache],
        prev_topk_indices: Optional[torch.Tensor],
        force_reuse_topk_indices: bool,
        attn_inputs: Optional[Any],
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ):
        if isinstance(fmha_impl, _MiniMaxM31DSparkQueryContext):
            if not isinstance(self.self_attn, MSAAttention):
                raise RuntimeError(
                    "MiniMax-M3.1 DSpARK mock requires a sparse MSA draft layer"
                )
            if kv_cache is None or attn_inputs is None:
                raise RuntimeError(
                    "MiniMax-M3.1 DSpARK proposal requires paged attention state"
                )
            output = self.self_attn.forward_dspark_query_block(
                hidden_states,
                attn_inputs,
                kv_cache,
                x_fp8=x_fp8,
                x_scale=x_scale,
            )
            return output, None
        return super()._forward_attention(
            hidden_states,
            fmha_impl,
            kv_cache,
            prev_topk_indices,
            force_reuse_topk_indices,
            attn_inputs,
            x_fp8,
            x_scale,
        )


class MiniMaxM31DSparkModel(DSparkProposerMixin, MiniMaxM31Model):
    """One or more MiniMax-M3.1 sparse blocks used as a DSpARK proposer."""

    decoder_layer_cls = MiniMaxM31DSparkDecoderLayer
    _captures_aux_hidden = False

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        if not model_config.dspark_target_layer_ids:
            raise ValueError("MiniMax-M3.1 DSpARK requires dspark_target_layer_ids")
        proposal_width = int(model_config.gen_num_per_cycle)
        query_width = proposal_width + int(not model_config.dspark_sample_from_anchor)
        self.init_dspark_proposer(
            width=proposal_width,
            query_width=query_width,
            noise_token_id=int(model_config.dspark_noise_token_id),
            aux_feature_dim=len(model_config.dspark_target_layer_ids)
            * int(model_config.hidden_size),
            hidden_dim=int(model_config.hidden_size),
        )
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.dspark_fc_w
        )
        for layer in self.layers[: self.layer_num]:
            if not isinstance(layer.self_attn, MSAAttention):
                raise ValueError(
                    "MiniMax-M3.1 DSpARK bring-up requires every draft layer to be sparse MSA"
                )

    def cuda_graph_input_hidden_size(self) -> int:
        return self._dspark_aux_feature_dim

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        del inputs
        return _MiniMaxM31DSparkQueryContext()

    def combine_hidden_states(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    @staticmethod
    def _project_commit_kv(
        attn: MSAAttention, hidden: torch.Tensor, positions: torch.Tensor
    ):
        qkv = attn.qkv_proj(hidden)
        if attn.qk_fuse_norm is not None:
            qkv = attn.qk_fuse_norm(qkv)
        _, key, value = torch.split(
            qkv, [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
        )
        rows = int(hidden.shape[0])
        key = key.reshape(rows, attn.kv_head_num, attn.head_dim).contiguous()
        value = value.reshape(rows, attn.kv_head_num, attn.head_dim)
        dummy_q = key.new_zeros((rows, 1, attn.head_dim))
        attn._apply_rope(dummy_q, key, positions)

        idx_k = F.linear(hidden, attn.idx_k_w).reshape(rows, 1, attn.idx_head_dim)
        idx_k = _gemma_rmsnorm_per_head(
            idx_k, attn.idx_k_norm_w, attn.layernorm_eps
        ).contiguous()
        dummy_idx_q = idx_k.new_zeros((rows, 1, attn.idx_head_dim))
        attn._apply_rope(dummy_idx_q, idx_k, positions)
        return key, value, idx_k

    def commit_feature_rows(
        self,
        main_x: torch.Tensor,
        context_req_ids: torch.Tensor,
        context_positions: torch.Tensor,
        committed_ends: torch.Tensor,
        inputs: PyModelInputs,
        commit_ctx: Any = None,
    ) -> None:
        del committed_ends
        if commit_ctx is not None:
            raise RuntimeError("MiniMax-M3.1 DSpARK CP commit is not implemented")
        # MiniMax DSpARK currently enables CUDA Graph only for exact decode
        # buckets, so every fixed-width commit row is payload. Avoid boolean
        # compaction (and valid.any()'s device-to-host sync) in the captured path.
        req_ids = context_req_ids.long()
        positions = context_positions.long()
        features = main_x
        attention_inputs = primary_attention_inputs(inputs.attention_inputs)
        writer = create_write_cache_store_impl(attention_inputs, self.kv_cache)

        for layer_idx, layer in enumerate(self.layers[: self.layer_num]):
            normalized, _ = layer.input_layernorm(features, torch.zeros_like(features))
            attn: MSAAttention = layer.self_attn
            key, value, idx_k = self._project_commit_kv(attn, normalized, positions)
            block_table = attn._physical_block_table(attention_inputs).index_select(
                0, req_ids
            )
            result = attn._write_kv_cache_and_idx_k_for_decode(
                self.kv_cache.get_layer_cache(layer_idx),
                key,
                value,
                idx_k,
                (positions + 1).to(torch.int32),
                block_table,
            )
            if result is None:
                raise RuntimeError(
                    "MiniMax-M3.1 DSpARK commit requires paged K/V and index-K"
                )
            if writer is not None:
                writer(self.kv_cache.get_layer_cache(layer_idx))

    def forward_query_block(
        self,
        query_ids: torch.Tensor,
        query_positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
        inputs: PyModelInputs,
        fmha_impl: Any,
    ) -> torch.Tensor:
        del query_ids, query_positions, prefix_lengths, active_requests
        return super().forward(inputs, fmha_impl).hidden_states

    def _forward_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    @torch.inference_mode()
    def forward_propose(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        if self.kv_cache is None:
            return self.dspark_empty_outputs(0, self._forward_device())
        return self.run_propose_step(inputs, fmha_impl, self._forward_device())

    @torch.inference_mode()
    def forward_commit(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        del fmha_impl
        if self.kv_cache is None:
            return PyModelOutputs(
                torch.empty(
                    (0, self.config.hidden_size),
                    dtype=torch.bfloat16,
                    device=self._forward_device(),
                )
            )
        return self.run_commit_step(inputs, self._forward_device())

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        del inputs, fmha_impl
        raise RuntimeError(
            "MiniMaxM31DSparkModel requires forward_propose/forward_commit"
        )


__all__ = [
    "MiniMaxM31DSparkDecoderLayer",
    "MiniMaxM31DSparkModel",
]
