from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import CausalAttention, DenseMLP, Embedding, RMSNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class MiniMaxM3Eagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: dict[str, torch.Tensor],
        layer_idx: int,
        hw_kernel_config: Optional[Any] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.embedding_norm = RMSNorm(
            weights[W.eagle3_input_norm_gamma],
            eps=model_config.layernorm_eps,
        )
        self.hidden_norm = RMSNorm(
            weights[W.eagle3_fc_norm_gamma],
            eps=model_config.layernorm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma],
            eps=model_config.layernorm_eps,
        )
        self.self_attn = CausalAttention(
            model_config.getAttentionConfigs(parallelism_config.get_attn_tp_size()),
            parallelism_config,
            weights,
            model_config.layernorm_eps,
            model_config.quant_config,
            hw_kernel_config,
            layer_idx,
        )
        self.mlp = DenseMLP(
            model_config.activation_type,
            parallelism_config,
            weights,
            model_config.quant_config,
            hw_kernel_config,
        )

    def _build_attention_input(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [self.embedding_norm(input_embeds), self.hidden_norm(hidden_states)],
            dim=-1,
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=self._build_attention_input(input_embeds, hidden_states),
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiniMaxM3Eagle3Model(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        if self.layer_num != 1:
            raise ValueError(
                f"MiniMax-M3 EAGLE3 draft expects one layer, got {self.layer_num}"
            )

        self.hidden_size = int(model_config.hidden_size)
        layer_weights = weights.weights[0]
        aux_norm_weight = layer_weights[W.eagle3_aux_norm_gamma]
        self.num_aux_hidden_states = int(aux_norm_weight.shape[0])
        if self.num_aux_hidden_states != 3:
            raise ValueError(
                "MiniMax-M3 EAGLE3 expects three auxiliary hidden states, "
                f"got {self.num_aux_hidden_states}"
            )

        self.embed_tokens = Embedding(
            model_config,
            parallelism_config,
            weights.get_global_weight(W.embedding),
        )
        self.fc_norms = nn.ModuleList(
            [
                RMSNorm(
                    layer_weights[W.eagle3_aux_norm_gamma][index],
                    eps=model_config.layernorm_eps,
                )
                for index in range(self.num_aux_hidden_states)
            ]
        )
        self.fc = LinearFactory.create_linear_from_weights(
            layer_weights,
            W.eagle3_fc_proj,
            quant_config=model_config.quant_config,
            hw_kernel_config=py_hw_kernel_config,
        )
        self.layers = nn.ModuleList(
            [
                MiniMaxM3Eagle3DecoderLayer(
                    model_config,
                    parallelism_config,
                    layer_weights,
                    0,
                    hw_kernel_config=py_hw_kernel_config,
                )
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma),
            eps=model_config.layernorm_eps,
        )

        fc_weight = layer_weights[W.eagle3_fc_proj]
        expected_fc_width = self.hidden_size * self.num_aux_hidden_states
        if int(fc_weight.shape[0]) != expected_fc_width:
            raise RuntimeError(
                "MiniMax-M3 EAGLE3 FC input width mismatch: "
                f"expected {expected_fc_width}, got {int(fc_weight.shape[0])}"
            )
        if tuple(aux_norm_weight.shape) != (
            self.num_aux_hidden_states,
            self.hidden_size,
        ):
            raise RuntimeError(
                "MiniMax-M3 EAGLE3 auxiliary norm shape mismatch: "
                f"expected {(self.num_aux_hidden_states, self.hidden_size)}, "
                f"got {tuple(aux_norm_weight.shape)}"
            )
        qkv_weight = layer_weights[W.attn_qkv_w]
        if int(qkv_weight.shape[0]) != self.hidden_size * 2:
            raise RuntimeError(
                "MiniMax-M3 EAGLE3 first-layer QKV input width must be 2H, "
                f"got {int(qkv_weight.shape[0])}"
            )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        attn_inputs = inputs.attention_inputs
        if is_cuda_graph and attn_inputs.is_prefill:
            # Compact EAGLE3 draft-prefill graphs replay only an exact batch
            # bucket. Their total Q rows are batch * (proposal_step + 1), so
            # every request has the same captured Q length.
            attn_inputs.prefill_cuda_graph_fixed_q_per_request = True
        return super().prepare_fmha_impl(inputs, is_cuda_graph)

    def clone_for_cuda_graph(self) -> "MiniMaxM3Eagle3Model":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)
        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.weight = self.weight
        clone.fmha_config = self.fmha_config
        clone.py_hw_kernel_config = self.py_hw_kernel_config
        clone.micro_batch_size = self.micro_batch_size
        clone.layer_num = self.layer_num
        clone.vocab_size = self.vocab_size
        clone.kv_cache = None
        clone.device_type = self.device_type
        clone.params_dict = {}
        clone.hidden_size = self.hidden_size
        clone.num_aux_hidden_states = self.num_aux_hidden_states
        clone.embed_tokens = self.embed_tokens
        clone.fc_norms = self.fc_norms
        clone.fc = self.fc
        clone.layers = self.layers
        clone.norm = self.norm
        return clone

    def _combine_aux_hidden_states(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected_width = self.hidden_size * self.num_aux_hidden_states
        if int(target_hidden.shape[-1]) != expected_width:
            raise RuntimeError(
                "MiniMax-M3 EAGLE3 expected concatenated target hidden width "
                f"{expected_width}, got {int(target_hidden.shape[-1])}"
            )
        chunks = target_hidden.split(self.hidden_size, dim=-1)
        normalized = [
            norm(chunk.contiguous()) for norm, chunk in zip(self.fc_norms, chunks)
        ]
        return self.fc(torch.cat(normalized, dim=-1))

    def _prepare_hidden_states(self, target_hidden: torch.Tensor) -> torch.Tensor:
        width = int(target_hidden.shape[-1])
        if width == self.hidden_size:
            return target_hidden
        if width == self.hidden_size * self.num_aux_hidden_states:
            return self._combine_aux_hidden_states(target_hidden)
        raise RuntimeError(
            "MiniMax-M3 EAGLE3 hidden width must be H for an autoregressive "
            f"draft step or 3H for a target step, got {width}"
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        target_hidden = inputs.input_hiddens
        if target_hidden.numel() == 0:
            raise RuntimeError("MiniMax-M3 EAGLE3 draft requires hidden states")
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        input_embeds = self.embed_tokens(inputs.input_ids)
        hidden_states = self._prepare_hidden_states(target_hidden)
        hidden_states = self.layers[0](
            input_embeds,
            hidden_states,
            fmha_impl,
            self.kv_cache.get_layer_cache(0) if self.kv_cache else None,
        )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
