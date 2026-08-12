"""Qwen3 DSpark backbone on the shared DSpark proposer contract."""

from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.modules import LinearFactory, RMSNorm
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import DSparkCallPhase, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class _RopePositions:
    def __init__(self, positions: torch.Tensor) -> None:
        self.positions_d = positions


class Qwen3DSparkModel(DSparkProposerMixin, Qwen3Model):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config=None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ) -> None:
        if quant_config is not None:
            raise NotImplementedError("Qwen3 DSpark quantization is not supported")
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            quant_config=quant_config,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        proposal_width = int(config.gen_num_per_cycle)
        query_width = proposal_width + 1
        if query_width > int(config.dspark_block_size):
            raise ValueError(
                f"Qwen3 DSpark query width {query_width} exceeds "
                f"checkpoint block_size {config.dspark_block_size}"
            )
        self.init_dspark_proposer(
            width=proposal_width,
            query_width=query_width,
            noise_token_id=config.dspark_noise_token_id,
            aux_feature_dim=len(config.dspark_target_layer_ids)
            * config.hidden_size,
            hidden_dim=config.hidden_size,
        )

        self.attn_configs = config.getAttentionConfigs(
            parallelism_config.get_attn_tp_size()
        )
        if self.attn_configs.is_causal:
            raise ValueError("Qwen3 DSpark proposal attention must be non-causal")
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.dspark_fc_w
        )
        self.hidden_norm = RMSNorm(
            weights.get_global_weight(W.dspark_hidden_norm_gamma),
            eps=config.layernorm_eps,
        )

        heads = self.attn_configs.head_num
        kv_heads = self.attn_configs.kv_head_num
        head_dim = self.attn_configs.size_per_head
        q_cols = heads * head_dim
        self.context_kv_projections = nn.ModuleList()
        self.context_k_norms = nn.ModuleList()
        for layer_weights in weights.weights[: self.layer_num]:
            qkv = layer_weights[W.attn_qkv_w]
            self.context_kv_projections.append(
                LinearFactory.create_linear(
                    qkv[:, q_cols:], None, None, None, py_hw_kernel_config
                )
            )
            self.context_k_norms.append(
                RMSNorm(layer_weights[W.k_ln_gamma], eps=config.layernorm_eps)
            )

        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
            MhaRotaryEmbeddingOp,
        )

        self.context_rope = MhaRotaryEmbeddingOp(self.attn_configs)

    def cuda_graph_input_hidden_size(self) -> int:
        return self._dspark_aux_feature_dim

    def combine_hidden_states(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def _block_table(self, inputs: PyModelInputs) -> torch.Tensor:
        attention = inputs.attention_inputs
        table = attention.kv_cache_block_id_device
        if table is None or table.numel() == 0:
            table = attention.kv_cache_block_id_host.to(
                device=self.embed_tokens.weight.device, non_blocking=True
            )
        return table[0] if table.dim() == 3 else table

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
        table = self._block_table(inputs)
        page_size = self.attn_configs.kernel_tokens_per_block
        positions = context_positions.long()
        pages = table[context_req_ids.long(), positions // page_size].long()
        slots = positions % page_size
        hidden = self.hidden_norm(main_x)
        head_dim = self.attn_configs.size_per_head
        kv_heads = self.attn_configs.kv_head_num
        dummy_q = hidden.new_zeros((hidden.shape[0], 1, head_dim))

        for layer_idx in range(self.layer_num):
            key, value = self.context_kv_projections[layer_idx](hidden).view(
                -1, 2, kv_heads, head_dim
            ).unbind(1)
            key = self.context_k_norms[layer_idx](key.reshape(-1, head_dim)).view(
                -1, kv_heads, head_dim
            )
            self.context_rope._apply_rope(
                dummy_q, key, _RopePositions(context_positions)
            )
            cache = self.kv_cache.get_layer_cache(layer_idx).kv_cache_base
            cache[pages, 0, :, slots, :] = key.to(cache.dtype)
            cache[pages, 1, :, slots, :] = value.to(cache.dtype)

        writer = create_write_cache_store_impl(
            inputs.attention_inputs, self.kv_cache
        )
        if writer is not None:
            for layer_idx in range(self.layer_num):
                writer(self.kv_cache.get_layer_caches(layer_idx))

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

    @torch.inference_mode()
    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        phase = getattr(inputs, "dspark_call_phase", DSparkCallPhase.NONE)
        if phase == DSparkCallPhase.NONE:
            raise RuntimeError("Qwen3 DSpark requires an explicit phase")
        device = self.embed_tokens.weight.device
        if self.kv_cache is None:
            if phase == DSparkCallPhase.COMMIT:
                return PyModelOutputs(
                    torch.empty(
                        (0, self.config.hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
            tokens = int(inputs.input_ids.numel())
            batch = max(
                (tokens + self._dspark_query_width - 1)
                // self._dspark_query_width,
                1,
            )
            return self.dspark_empty_outputs(batch, device)
        if phase == DSparkCallPhase.COMMIT:
            return self.run_commit_step(inputs, device)
        return self.run_propose_step(inputs, fmha_impl, device)


__all__ = ["Qwen3DSparkModel"]
