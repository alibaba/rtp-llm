"""Qwen3 DSpark backbone on the shared DSpark proposer contract."""

from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    select_attention_inputs_for_layer,
)
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.modules import LinearFactory, RMSNorm
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
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
        query_width = proposal_width + int(not config.dspark_sample_from_anchor)
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
        context_kv_weights = []
        self.context_k_norms = nn.ModuleList()
        for layer_weights in weights.weights[: self.layer_num]:
            qkv = layer_weights[W.attn_qkv_w]
            context_kv_weights.append(qkv[:, q_cols:])
            self.context_k_norms.append(
                RMSNorm(layer_weights[W.k_ln_gamma], eps=config.layernorm_eps)
            )
        self.context_kv_projection = LinearFactory.create_linear(
            torch.cat(context_kv_weights, dim=1),
            None,
            None,
            None,
            py_hw_kernel_config,
        )

        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
            MhaRotaryEmbeddingOp,
        )

        self.context_rope = MhaRotaryEmbeddingOp(self.attn_configs)

    def cuda_graph_input_hidden_size(self) -> int:
        return self._dspark_aux_feature_dim

    def combine_hidden_states(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def dspark_attention_inputs(self, inputs: PyModelInputs):
        attention = select_attention_inputs_for_layer(inputs, self.kv_cache, 0)
        if isinstance(attention, list):
            if len(attention) != 1:
                raise RuntimeError(
                    "Qwen3 DSpark requires exactly one KV cache group per layer, "
                    f"got {len(attention)}"
                )
            attention = attention[0]
        return attention

    def _block_table(self, inputs: PyModelInputs) -> torch.Tensor:
        attention = self.dspark_attention_inputs(inputs)
        table = attention.kv_cache_kernel_block_id_device
        if table is None or table.numel() == 0:
            table = attention.kv_cache_kernel_block_id.to(
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

        all_kv = self.context_kv_projection(hidden).view(
            -1, self.layer_num, 2, kv_heads, head_dim
        )
        for layer_idx in range(self.layer_num):
            key, value = all_kv[:, layer_idx].unbind(1)
            key = self.context_k_norms[layer_idx](key.reshape(-1, head_dim)).view(
                -1, kv_heads, head_dim
            )
            self.context_rope._apply_rope(
                dummy_q, key, _RopePositions(context_positions)
            )
            cache = self.kv_cache.get_layer_cache(layer_idx).kv_cache_base
            cache[pages, 0, :, slots, :] = key.to(cache.dtype)
            cache[pages, 1, :, slots, :] = value.to(cache.dtype)

        writer = create_write_cache_store_impl(self.dspark_attention_inputs(inputs))
        if writer is not None:
            for layer_idx in range(self.layer_num):
                layer_caches = self.kv_cache.get_layer_cache_groups(layer_idx)
                if len(layer_caches) != 1:
                    raise RuntimeError(
                        "Qwen3 DSpark requires exactly one KV cache group per layer, "
                        f"got {len(layer_caches)} for layer {layer_idx}"
                    )
                writer(layer_caches[0])

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
        device = self.embed_tokens.weight.device
        return device

    @torch.inference_mode()
    def forward_propose(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        device = self._forward_device()
        if self.kv_cache is None:
            tokens = int(inputs.input_ids.numel())
            batch = max(
                (tokens + self._dspark_query_width - 1)
                // self._dspark_query_width,
                1,
            )
            return self.dspark_empty_outputs(batch, device)
        return self.run_propose_step(inputs, fmha_impl, device)

    @torch.inference_mode()
    def forward_commit(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        del fmha_impl
        device = self._forward_device()
        if self.kv_cache is None:
            return PyModelOutputs(
                torch.empty(
                    (0, self.config.hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
            )
        return self.run_commit_step(inputs, device)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        del inputs, fmha_impl
        raise RuntimeError(
            "Qwen3DSparkModel requires a fixed forward_propose or "
            "forward_commit entrypoint"
        )


__all__ = ["Qwen3DSparkModel"]
