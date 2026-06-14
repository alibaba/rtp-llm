from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, LinearFactory, RMSNorm, RMSResNorm
from rtp_llm.ops import MoeConfig, ParallelismConfig
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
        self.moe_config = moe_config
        self.max_generate_batch_size = max_generate_batch_size
        self.device_resource_config = device_resource_config
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

        self._share_mtp_topk_indices = bool(
            getattr(model_config, "index_share_for_mtp_iteration", False)
        )
        self._mtp_iteration_topk_buffers = [None for _ in range(self.layer_num)]
        self._mtp_iteration_topk_valid_tokens = [0 for _ in range(self.layer_num)]
        self._mtp_iteration_topk_indices = [None for _ in range(self.layer_num)]


    def clone_for_cuda_graph(self) -> "GenericMoeMTPModel":
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
        clone.moe_config = self.moe_config
        clone.max_generate_batch_size = self.max_generate_batch_size
        clone.device_resource_config = self.device_resource_config

        clone.embed_tokens = self.embed_tokens
        clone.pre_fc_norm_embedding = self.pre_fc_norm_embedding
        clone.pre_fc_norm_hidden = self.pre_fc_norm_hidden
        clone.fc = self.fc
        clone.layers = nn.ModuleList(
            [
                (
                    layer.clone_for_cuda_graph()
                    if hasattr(layer, "clone_for_cuda_graph")
                    else layer
                )
                for layer in self.layers
            ]
        )
        clone.norm = self.norm
        clone._share_mtp_topk_indices = self._share_mtp_topk_indices
        clone._mtp_iteration_topk_buffers = self._mtp_iteration_topk_buffers
        clone._mtp_iteration_topk_valid_tokens = self._mtp_iteration_topk_valid_tokens
        clone._mtp_iteration_topk_indices = self._mtp_iteration_topk_indices

        return clone

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self._mask_position_zero_embeddings(inputs_embeds, fmha_impl)
        last_hidden_states = inputs.input_hiddens

        e_norm = self.pre_fc_norm_embedding(inputs_embeds)
        h_norm = self.pre_fc_norm_hidden(last_hidden_states)
        cat_hidden_states = torch.cat([e_norm, h_norm], -1)
        hidden_states = self.fc(cat_hidden_states)

        self._reset_mtp_iteration_topk_if_needed(inputs)
        reuse_mtp_iteration_topk = self._should_reuse_mtp_iteration_topk(inputs)
        residual = torch.zeros_like(hidden_states)
        prev_topk_indices = None
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            cached_topk = (
                self._get_mtp_iteration_topk(i, hidden_states.size(0))
                if reuse_mtp_iteration_topk
                else None
            )
            if reuse_mtp_iteration_topk and cached_topk is None:
                raise RuntimeError(
                    "MTP top-k index sharing requested for draft step "
                    f"{self._mtp_iteration_step(inputs)} at layer {i}, "
                    "but no compatible step-0 top-k indices are cached"
                )
            force_reuse_topk = cached_topk is not None
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                prev_topk_indices=(
                    cached_topk if force_reuse_topk else prev_topk_indices
                ),
                force_reuse_topk_indices=force_reuse_topk,
            )
            hidden_states = output.hidden_states
            residual = output.residual
            prev_topk_indices = output.topk_indices
            self._set_mtp_iteration_topk(i, output.topk_indices)

        hidden_states, _ = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)

    def _reset_mtp_iteration_topk_if_needed(self, inputs: PyModelInputs) -> None:
        if not self._share_mtp_topk_indices:
            return
        step = self._mtp_iteration_step(inputs)
        if step == 0:
            self._clear_mtp_iteration_topk()
            return
        attention_inputs = getattr(inputs, "attention_inputs", None)
        if step < 0 and bool(getattr(attention_inputs, "is_prefill", False)):
            self._clear_mtp_iteration_topk()

    def _should_reuse_mtp_iteration_topk(self, inputs: PyModelInputs) -> bool:
        if not self._share_mtp_topk_indices:
            return False
        return self._mtp_iteration_step(inputs) > 0

    def _mtp_iteration_step(self, inputs: PyModelInputs) -> int:
        attention_inputs = getattr(inputs, "attention_inputs", None)
        step = getattr(attention_inputs, "mtp_iteration_step", -1)
        if step is None:
            return -1
        return int(step)

    def _clear_mtp_iteration_topk(self) -> None:
        self._mtp_iteration_topk_indices[:] = [None for _ in range(int(self.layer_num))]
        self._mtp_iteration_topk_valid_tokens[:] = [
            0 for _ in range(int(self.layer_num))
        ]

    def _get_mtp_iteration_topk(self, layer_idx: int, expected_tokens: int = -1):
        if not self._share_mtp_topk_indices:
            return None
        if layer_idx < 0 or layer_idx >= len(self._mtp_iteration_topk_indices):
            return None
        topk_indices = self._mtp_iteration_topk_indices[layer_idx]
        if torch.is_tensor(topk_indices):
            valid_tokens = self._mtp_iteration_topk_valid_tokens[layer_idx]
            if expected_tokens >= 0 and valid_tokens != expected_tokens:
                return None
            if (
                valid_tokens >= 0
                and topk_indices.dim() > 0
                and topk_indices.size(0) != valid_tokens
            ):
                return topk_indices[:valid_tokens]
        if (
            expected_tokens >= 0
            and torch.is_tensor(topk_indices)
            and topk_indices.dim() > 0
            and topk_indices.size(0) != expected_tokens
        ):
            return None
        return topk_indices

    def _set_mtp_iteration_topk(self, layer_idx: int, topk_indices) -> None:
        if not self._share_mtp_topk_indices or topk_indices is None:
            return
        self._ensure_mtp_iteration_topk_layer(layer_idx)
        if not torch.is_tensor(topk_indices):
            self._mtp_iteration_topk_indices[layer_idx] = topk_indices
            return

        cache_shape = tuple(int(dim) for dim in topk_indices.shape)
        if len(cache_shape) == 0:
            self._mtp_iteration_topk_indices[layer_idx] = topk_indices
            self._mtp_iteration_topk_valid_tokens[layer_idx] = 1
            return

        buffer = self._mtp_iteration_topk_buffers[layer_idx]
        needs_new_buffer = (
            buffer is None
            or not torch.is_tensor(buffer)
            or buffer.dtype != topk_indices.dtype
            or buffer.device != topk_indices.device
            or buffer.dim() != topk_indices.dim()
            or tuple(int(dim) for dim in buffer.shape[1:]) != cache_shape[1:]
            or int(buffer.size(0)) < int(topk_indices.size(0))
        )
        if needs_new_buffer:
            self._mtp_iteration_topk_buffers[layer_idx] = torch.empty_like(topk_indices)
            buffer = self._mtp_iteration_topk_buffers[layer_idx]

        valid_tokens = int(topk_indices.size(0))
        buffer[:valid_tokens].copy_(topk_indices)
        self._mtp_iteration_topk_valid_tokens[layer_idx] = valid_tokens
        self._mtp_iteration_topk_indices[layer_idx] = buffer[:valid_tokens]

    def select_mtp_iteration_topk_cache(
        self, select_indices: torch.Tensor, total_tokens: int = -1
    ) -> None:
        if not self._share_mtp_topk_indices:
            return
        if select_indices is None or not torch.is_tensor(select_indices):
            return
        if select_indices.numel() == 0:
            self._clear_mtp_iteration_topk()
            return
        for layer_idx in range(len(self._mtp_iteration_topk_indices)):
            source = self._mtp_iteration_topk_buffers[layer_idx]
            if source is None:
                source = self._mtp_iteration_topk_indices[layer_idx]
            if source is None or not torch.is_tensor(source) or source.dim() == 0:
                continue
            if total_tokens >= 0 and int(source.size(0)) < int(total_tokens):
                continue
            indices = select_indices.reshape(-1).to(
                device=source.device, dtype=torch.long, non_blocking=True
            )
            selected = torch.index_select(source, 0, indices)
            self._set_mtp_iteration_topk(layer_idx, selected)

    def copy_mtp_iteration_topk_cache_from(self, other) -> None:
        if not self._share_mtp_topk_indices or other is None:
            return
        if other is self:
            return
        other_cache = getattr(other, "_mtp_iteration_topk_indices", None)
        if other_cache is None:
            return
        if other_cache is self._mtp_iteration_topk_indices:
            return
        for layer_idx in range(len(other_cache)):
            topk = None
            if hasattr(other, "_get_mtp_iteration_topk"):
                valid_tokens = -1
                other_valid = getattr(other, "_mtp_iteration_topk_valid_tokens", None)
                if other_valid is not None and layer_idx < len(other_valid):
                    valid_tokens = int(other_valid[layer_idx])
                topk = other._get_mtp_iteration_topk(layer_idx, valid_tokens)
            if topk is None and layer_idx < len(other_cache):
                topk = other_cache[layer_idx]
            self._set_mtp_iteration_topk(layer_idx, topk)

    def _ensure_mtp_iteration_topk_layer(self, layer_idx: int) -> None:
        if layer_idx < len(self._mtp_iteration_topk_indices):
            return
        extend_num = layer_idx + 1 - len(self._mtp_iteration_topk_indices)
        self._mtp_iteration_topk_indices.extend([None for _ in range(extend_num)])
        self._mtp_iteration_topk_buffers.extend([None for _ in range(extend_num)])
        self._mtp_iteration_topk_valid_tokens.extend([0 for _ in range(extend_num)])

    def _mask_position_zero_embeddings(
        self, inputs_embeds: torch.Tensor, fmha_impl: Any
    ) -> torch.Tensor:
        fmha_params = getattr(fmha_impl, "fmha_params", None)
        positions = getattr(fmha_params, "positions_d", None)
        if (
            positions is None
            or not torch.is_tensor(positions)
            or positions.numel() == 0
        ):
            return inputs_embeds
        positions = positions.reshape(-1)
        if positions.size(0) != inputs_embeds.size(0):
            return inputs_embeds
        if positions.device != inputs_embeds.device:
            positions = positions.to(device=inputs_embeds.device)
        return torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
