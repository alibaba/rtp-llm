from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, LinearFactory, RMSNorm, RMSResNorm
from rtp_llm.ops import CPRotateMethod, HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class GenericMoeMTPModel(GptModelBase):
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
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.pre_fc_norm_embedding = RMSNorm(
            weights.global_weights[W.multi_tokens_predict_enorm],
            eps=model_config.layernorm_eps,
        )
        self.pre_fc_norm_hidden = RMSNorm(
            weights.global_weights[W.multi_tokens_predict_hnorm],
            eps=model_config.layernorm_eps,
        )
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.multi_tokens_predict_eh_proj
        )

        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.global_weights[W.multi_tokens_predict_final_ln_gamma],
            eps=model_config.layernorm_eps,
        )

        self.register_buffer("_mtp_hidden_buffer", None, persistent=False)
        self._mtp_hidden_valid_tokens = 0
        self.register_buffer("_mtp_last_hidden_buffer", None, persistent=False)
        self._mtp_last_hidden_valid_tokens = 0

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        cp_config = self.parallelism_config.prefill_cp_config
        saved_method = cp_config.method
        saved_tp = self.parallelism_config.tp_size
        if cp_config.is_enabled():
            cp_config.method = CPRotateMethod.DISABLED
            self.parallelism_config.tp_size = 1
        try:
            return super().prepare_fmha_impl(inputs, is_cuda_graph)
        finally:
            cp_config.method = saved_method
            self.parallelism_config.tp_size = saved_tp

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        last_hidden_states = inputs.input_hiddens

        e_norm = self.pre_fc_norm_embedding(inputs_embeds)
        h_norm = self.pre_fc_norm_hidden(last_hidden_states)
        cat_hidden_states = torch.cat([e_norm, h_norm], -1)
        hidden_states = self.fc(cat_hidden_states)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = output.hidden_states
            residual = output.residual

        pre_norm_hidden = hidden_states + residual
        self._write_mtp_hidden_buffer(pre_norm_hidden)
        self._write_mtp_last_hidden(pre_norm_hidden, inputs)

        hidden_states, _ = self.norm(hidden_states, residual)

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)

    def _write_mtp_hidden_buffer(self, flat: torch.Tensor) -> None:
        T, D = flat.size(0), flat.size(1)
        if self._mtp_hidden_buffer is None or self._mtp_hidden_buffer.size(0) < T:
            self.register_buffer(
                "_mtp_hidden_buffer",
                torch.empty(max(T, 1024), D, dtype=flat.dtype, device=flat.device),
                persistent=False,
            )
        self._mtp_hidden_buffer[:T].copy_(flat)
        self._mtp_hidden_valid_tokens = int(T)

    def _write_mtp_last_hidden(self, flat: torch.Tensor, inputs: PyModelInputs) -> None:
        attn = inputs.attention_inputs
        if attn is None:
            return
        input_lengths = attn.input_lengths
        if input_lengths is None or input_lengths.numel() == 0:
            return
        T = flat.size(0)
        if torch.cuda.is_current_stream_capturing():
            last_hidden = flat.contiguous()
        else:
            total_new = int(input_lengths.sum().item())
            if total_new == T:
                cu_seqlens = torch.cumsum(
                    input_lengths.to(dtype=torch.long, device=flat.device), dim=0
                )
                last_indices = cu_seqlens - 1
                last_hidden = flat.index_select(0, last_indices).contiguous()
            else:
                last_hidden = flat.contiguous()

        B, D = last_hidden.size(0), last_hidden.size(1)
        if (
            self._mtp_last_hidden_buffer is None
            or self._mtp_last_hidden_buffer.size(0) < B
        ):
            self.register_buffer(
                "_mtp_last_hidden_buffer",
                torch.empty(max(B, 64), D, dtype=flat.dtype, device=flat.device),
                persistent=False,
            )
        self._mtp_last_hidden_buffer[:B].copy_(last_hidden)
        self._mtp_last_hidden_valid_tokens = int(B)

    def get_mtp_target_hidden_states(self, num_tokens: int):
        buf = self._mtp_hidden_buffer
        if buf is None:
            return None
        requested = int(num_tokens)
        if requested < 0:
            requested = self._mtp_hidden_valid_tokens
            assert requested > 0, "MTP hidden buffer has no written rows"
        assert requested <= buf.size(0), (
            f"requested MTP hidden states exceed buffer capacity: "
            f"requested={requested}, capacity={buf.size(0)}"
        )
        return buf[:requested]

    def get_mtp_last_hidden_states(self, num_tokens: int):
        buf = self._mtp_last_hidden_buffer
        if buf is None:
            return None
        requested = int(num_tokens)
        if requested < 0:
            requested = self._mtp_last_hidden_valid_tokens
        assert requested > 0, "MTP last hidden buffer has no written rows"
        assert requested <= buf.size(0), (
            f"requested MTP last hidden states exceed buffer capacity: "
            f"requested={requested}, capacity={buf.size(0)}"
        )
        return buf[:requested]
