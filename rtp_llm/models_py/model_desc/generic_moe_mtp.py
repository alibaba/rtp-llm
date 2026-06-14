import os
from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, LinearFactory, RMSNorm, RMSResNorm
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    PyModelInputs,
    PyModelOutputs,
    cuda_graph_capture_forward_enabled,
)
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

        self.register_buffer("_mtp_hidden_buffer", None, persistent=False)
        self._mtp_hidden_valid_tokens = 0
        self.register_buffer("_mtp_normalized_hidden_buffer", None, persistent=False)
        self._mtp_normalized_hidden_valid_tokens = 0
        self.register_buffer("_mtp_final_hidden_buffer", None, persistent=False)
        self._mtp_final_hidden_valid_tokens = 0
        self.register_buffer("_mtp_final_residual_buffer", None, persistent=False)
        self._mtp_final_residual_valid_tokens = 0
        self.register_buffer("_mtp_final_norm_hidden_work", None, persistent=False)
        self.register_buffer("_mtp_final_norm_residual_work", None, persistent=False)
        self.register_buffer("_mtp_last_hidden_buffer", None, persistent=False)
        self._mtp_last_hidden_valid_tokens = 0
        self._mtp_debug_buffers: dict[str, torch.Tensor] = {}
        self._mtp_debug_shapes: dict[str, tuple[int, ...]] = {}
        self._mtp_debug_enabled_flag = (
            os.environ.get("RTP_LLM_DEBUG_MTP_PREFILL_DATA", "0") != "0"
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

        clone.register_buffer("_mtp_hidden_buffer", None, persistent=False)
        clone._mtp_hidden_valid_tokens = 0
        clone.register_buffer("_mtp_normalized_hidden_buffer", None, persistent=False)
        clone._mtp_normalized_hidden_valid_tokens = 0
        clone.register_buffer("_mtp_final_hidden_buffer", None, persistent=False)
        clone._mtp_final_hidden_valid_tokens = 0
        clone.register_buffer("_mtp_final_residual_buffer", None, persistent=False)
        clone._mtp_final_residual_valid_tokens = 0
        clone.register_buffer("_mtp_final_norm_hidden_work", None, persistent=False)
        clone.register_buffer("_mtp_final_norm_residual_work", None, persistent=False)
        clone.register_buffer("_mtp_last_hidden_buffer", None, persistent=False)
        clone._mtp_last_hidden_valid_tokens = 0
        clone._mtp_debug_buffers = {}
        clone._mtp_debug_shapes = {}
        clone._mtp_debug_enabled_flag = self._mtp_debug_enabled_flag
        clone._share_mtp_topk_indices = self._share_mtp_topk_indices
        clone._mtp_iteration_topk_buffers = self._mtp_iteration_topk_buffers
        clone._mtp_iteration_topk_valid_tokens = self._mtp_iteration_topk_valid_tokens
        clone._mtp_iteration_topk_indices = self._mtp_iteration_topk_indices
        return clone

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        mtp_debug_enabled = self._mtp_debug_enabled()
        if mtp_debug_enabled:
            self._write_mtp_debug_buffer("input_ids", input_ids)
            self._write_mtp_debug_buffer("input_hiddens", inputs.input_hiddens)
            self._write_mtp_attention_debug_buffers(inputs)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self._mask_position_zero_embeddings(inputs_embeds, fmha_impl)
        last_hidden_states = inputs.input_hiddens

        e_norm = self.pre_fc_norm_embedding(inputs_embeds)
        h_norm = self.pre_fc_norm_hidden(last_hidden_states)
        cat_hidden_states = torch.cat([e_norm, h_norm], -1)
        hidden_states = self.fc(cat_hidden_states)
        if mtp_debug_enabled:
            self._write_mtp_debug_buffer("fc_hidden", hidden_states)

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
            if mtp_debug_enabled and i == 0:
                self._write_mtp_debug_buffer("layer0_hidden", hidden_states)
                self._write_mtp_debug_buffer("layer0_residual", residual)

        pre_norm_hidden = hidden_states + residual
        if mtp_debug_enabled:
            self._write_mtp_debug_buffer("pre_norm_hidden", pre_norm_hidden)
        self._write_mtp_hidden_buffer(pre_norm_hidden)
        self._write_mtp_last_hidden(pre_norm_hidden, inputs)

        is_sp_prefill_cuda_graph_shape = (
            inputs.attention_inputs is not None
            and inputs.attention_inputs.is_s_padded
            and inputs.attention_inputs.is_prefill
        )
        is_cuda_graph_capture_forward = cuda_graph_capture_forward_enabled()
        is_sp_prefill_cuda_graph = (
            is_sp_prefill_cuda_graph_shape and is_cuda_graph_capture_forward
        )
        if is_sp_prefill_cuda_graph:
            self._write_mtp_final_norm_input_buffers(hidden_states, residual)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
            self._write_mtp_normalized_hidden_buffer(hidden_states)

        output_hidden_states = (
            pre_norm_hidden if is_sp_prefill_cuda_graph else hidden_states
        )
        return PyModelOutputs(output_hidden_states, fmha_impl.fmha_params)

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

    def _write_mtp_normalized_hidden_buffer(self, flat: torch.Tensor) -> None:
        T, D = flat.size(0), flat.size(1)
        if (
            self._mtp_normalized_hidden_buffer is None
            or self._mtp_normalized_hidden_buffer.size(0) < T
        ):
            self.register_buffer(
                "_mtp_normalized_hidden_buffer",
                torch.empty(max(T, 1024), D, dtype=flat.dtype, device=flat.device),
                persistent=False,
            )
        self._mtp_normalized_hidden_buffer[:T].copy_(flat)
        self._mtp_normalized_hidden_valid_tokens = int(T)

    def _write_mtp_final_norm_input_buffers(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> None:
        T, D = hidden_states.size(0), hidden_states.size(1)
        if (
            self._mtp_final_hidden_buffer is None
            or self._mtp_final_hidden_buffer.size(0) < T
        ):
            self.register_buffer(
                "_mtp_final_hidden_buffer",
                torch.empty(
                    max(T, 1024),
                    D,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                ),
                persistent=False,
            )
        if (
            self._mtp_final_residual_buffer is None
            or self._mtp_final_residual_buffer.size(0) < T
        ):
            self.register_buffer(
                "_mtp_final_residual_buffer",
                torch.empty(
                    max(T, 1024), D, dtype=residual.dtype, device=residual.device
                ),
                persistent=False,
            )
        self._mtp_final_hidden_buffer[:T].copy_(hidden_states)
        self._mtp_final_residual_buffer[:T].copy_(residual)
        self._mtp_final_hidden_valid_tokens = int(T)
        self._mtp_final_residual_valid_tokens = int(T)

    def _mtp_debug_enabled(self) -> bool:
        return self._mtp_debug_enabled_flag

    def _write_mtp_attention_debug_buffers(self, inputs: PyModelInputs) -> None:
        if not self._mtp_debug_enabled():
            return
        attn = inputs.attention_inputs
        if attn is None:
            return
        self._write_mtp_debug_buffer("input_lengths", attn.input_lengths)
        self._write_mtp_debug_buffer("prefix_lengths", attn.prefix_lengths)
        self._write_mtp_debug_buffer("cu_seqlens", attn.cu_seqlens)
        self._write_mtp_debug_buffer("cu_kv_seqlens", attn.cu_kv_seqlens)
        self._write_mtp_debug_buffer(
            "kv_cache_kernel_block_id_device", attn.kv_cache_kernel_block_id_device
        )
        self._write_mtp_debug_buffer(
            "kv_cache_block_id_device", attn.kv_cache_block_id_device
        )
        if len(attn.kv_cache_kernel_block_id_device_by_group) > 0:
            self._write_mtp_debug_buffer(
                "kv_cache_kernel_block_id_group0",
                attn.kv_cache_kernel_block_id_device_by_group[0],
            )

    def _write_mtp_debug_buffer(self, name: str, tensor: torch.Tensor) -> None:
        if not self._mtp_debug_enabled():
            return
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        # Keep debug buffers flat so a larger captured graph can serve smaller
        # replay shapes without reallocating during CUDA graph capture.
        flat = tensor.detach().reshape(-1)
        buf = self._mtp_debug_buffers.get(name)
        if (
            buf is None
            or buf.numel() < flat.numel()
            or buf.dtype != flat.dtype
            or buf.device != flat.device
        ):
            buf = torch.empty(
                max(int(flat.numel()), 1),
                dtype=flat.dtype,
                device=flat.device,
            )
            self._mtp_debug_buffers[name] = buf
        buf[: flat.numel()].copy_(flat)
        self._mtp_debug_shapes[name] = tuple(int(dim) for dim in tensor.shape)

    def get_mtp_debug_tensor(self, name: str, num_rows: int = -1):
        buf = self._mtp_debug_buffers.get(name)
        shape = self._mtp_debug_shapes.get(name)
        if buf is None or shape is None:
            return None
        numel = 1
        for dim in shape:
            numel *= int(dim)
        view = buf[:numel].view(shape)
        requested = int(num_rows)
        if requested >= 0 and view.dim() > 0:
            requested = min(requested, int(view.size(0)))
            return view[:requested]
        return view

    def get_mtp_debug_kv_cache(self, layer_idx: int = 0, max_blocks: int = 8):
        if not self._mtp_debug_enabled() or self.kv_cache is None:
            return None
        block_ids = self.get_mtp_debug_tensor("kv_cache_kernel_block_id_group0", -1)
        if block_ids is None:
            block_ids = self.get_mtp_debug_tensor("kv_cache_kernel_block_id_device", -1)
        if block_ids is None:
            return None

        layer_cache = self.kv_cache.get_layer_cache(int(layer_idx))
        kv_base = layer_cache.kv_cache_base
        if kv_base is None or not torch.is_tensor(kv_base) or kv_base.numel() == 0:
            return None
        if kv_base.dim() < 2:
            return None

        input_lengths = self.get_mtp_debug_tensor("input_lengths", -1)
        prefix_lengths = self.get_mtp_debug_tensor("prefix_lengths", -1)
        if input_lengths is None:
            return None
        if prefix_lengths is None:
            prefix_lengths = torch.zeros_like(input_lengths)

        seq_size = int(layer_cache.seq_size_per_block)
        if seq_size <= 0:
            return None
        block_ids = block_ids.to(device=kv_base.device, dtype=torch.long)
        input_lengths = input_lengths.to(
            device=kv_base.device, dtype=torch.long
        ).reshape(-1)
        prefix_lengths = prefix_lengths.to(
            device=kv_base.device, dtype=torch.long
        ).reshape(-1)

        flat_slots = []
        batch_size = min(int(block_ids.size(0)), int(input_lengths.numel()))
        for b in range(batch_size):
            input_len = int(input_lengths[b].item())
            prefix_len = (
                int(prefix_lengths[b].item()) if b < prefix_lengths.numel() else 0
            )
            if input_len <= 0:
                continue
            positions = torch.arange(
                prefix_len,
                prefix_len + input_len,
                dtype=torch.long,
                device=kv_base.device,
            )
            block_offsets = positions // seq_size
            slot_offsets = positions % seq_size
            valid = block_offsets < block_ids.size(1)
            if not bool(valid.any().item()):
                continue
            block_offsets = block_offsets[valid]
            slot_offsets = slot_offsets[valid]
            physical_blocks = block_ids[b].index_select(0, block_offsets)
            valid_blocks = physical_blocks >= 0
            if not bool(valid_blocks.any().item()):
                continue
            flat_slots.append(
                physical_blocks[valid_blocks] * seq_size + slot_offsets[valid_blocks]
            )

        if not flat_slots:
            return None
        slot_ids = torch.cat(flat_slots, dim=0)
        kv_flat = kv_base.reshape(-1, kv_base.size(-1))
        slot_ids = slot_ids[slot_ids < kv_flat.size(0)]
        if slot_ids.numel() == 0:
            return None
        slot_ids = slot_ids[: int(max_blocks) * seq_size]
        return kv_flat.index_select(0, slot_ids).detach().clone()

    def _capture_safe_rmsnorm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_fp32 = hidden_states_fp32 * torch.rsqrt(
            variance + self.norm.variance_epsilon
        )
        return self.norm.weight * hidden_states_fp32.to(input_dtype)

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

    def get_mtp_normalized_hidden_states(self, num_tokens: int):
        buf = self._mtp_normalized_hidden_buffer
        if buf is None:
            return None
        requested = int(num_tokens)
        if requested < 0:
            requested = self._mtp_normalized_hidden_valid_tokens
            assert requested > 0, "MTP normalized hidden buffer has no written rows"
        assert requested <= buf.size(0), (
            f"requested MTP normalized hidden states exceed buffer capacity: "
            f"requested={requested}, capacity={buf.size(0)}"
        )
        return buf[:requested]

    def get_mtp_final_hidden_states(self, num_tokens: int):
        buf = self._mtp_final_hidden_buffer
        if buf is None:
            return None
        requested = int(num_tokens)
        if requested < 0:
            requested = self._mtp_final_hidden_valid_tokens
            assert requested > 0, "MTP final hidden buffer has no written rows"
        assert requested <= buf.size(0), (
            f"requested MTP final hidden states exceed buffer capacity: "
            f"requested={requested}, capacity={buf.size(0)}"
        )
        return buf[:requested]

    def get_mtp_final_residual_states(self, num_tokens: int):
        buf = self._mtp_final_residual_buffer
        if buf is None:
            return None
        requested = int(num_tokens)
        if requested < 0:
            requested = self._mtp_final_residual_valid_tokens
            assert requested > 0, "MTP final residual buffer has no written rows"
        assert requested <= buf.size(0), (
            f"requested MTP final residual states exceed buffer capacity: "
            f"requested={requested}, capacity={buf.size(0)}"
        )
        return buf[:requested]

    def apply_mtp_final_layernorm_from_buffers(self, num_tokens: int):
        hidden = self.get_mtp_final_hidden_states(num_tokens)
        residual = self.get_mtp_final_residual_states(num_tokens)
        if hidden is None or residual is None:
            return None
        T, D = hidden.size(0), hidden.size(1)
        if (
            self._mtp_final_norm_hidden_work is None
            or self._mtp_final_norm_hidden_work.size(0) < T
        ):
            self.register_buffer(
                "_mtp_final_norm_hidden_work",
                torch.empty(max(T, 1024), D, dtype=hidden.dtype, device=hidden.device),
                persistent=False,
            )
        if (
            self._mtp_final_norm_residual_work is None
            or self._mtp_final_norm_residual_work.size(0) < T
        ):
            self.register_buffer(
                "_mtp_final_norm_residual_work",
                torch.empty(
                    max(T, 1024), D, dtype=residual.dtype, device=residual.device
                ),
                persistent=False,
            )
        hidden_work = self._mtp_final_norm_hidden_work[:T]
        residual_work = self._mtp_final_norm_residual_work[:T]
        hidden_work.copy_(hidden)
        residual_work.copy_(residual)
        hidden_work, _ = self.norm(hidden_work, residual_work)
        return hidden_work

    def get_mtp_final_layernorm_weight(self):
        return self.norm.weight

    def get_mtp_final_layernorm_eps(self) -> float:
        return float(self.norm.variance_epsilon)

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
