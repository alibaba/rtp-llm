"""Python model descriptor for the Qwen4-Exp MTP draft."""

import os
from collections.abc import Mapping
from typing import Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import get_attention_inputs_value
from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextMetadata
from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
from rtp_llm.models_py.modules import LinearFactory
from rtp_llm.models_py.modules.qwen4_exp.gated_residual import grouped_rms_norm
from rtp_llm.models_py.modules.qwen4_exp.norm import exact_head_rms_norm
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs
from rtp_llm.utils.model_weight import W


def normalize_mtp_pd_positions(inputs, index_factor: int) -> None:
    """Use the same draft cache coordinates for main and indexer RoPE."""
    groups = get_attention_inputs_value(inputs)
    values = tuple(groups.values()) if isinstance(groups, Mapping) else (groups,)
    if not any(
        getattr(value, "cache_store_inputs", None) is not None for value in values
    ):
        return
    from rtp_llm.models_py.modules.qwen4_exp.pd_cache_store import (
        validate_pd_cache_inputs,
    )

    validate_pd_cache_inputs(values)
    anchor = values[0]
    prefixes = anchor.prefix_lengths_device
    cu = anchor.cu_seqlens_device
    token_count = int(inputs.input_ids.numel())
    if (
        index_factor <= 0
        or token_count <= 0
        or prefixes.ndim != 1
        or prefixes.numel() == 0
        or cu.ndim != 1
        or cu.numel() != prefixes.numel() + 1
        or any(
            tensor.dtype != torch.int32 or tensor.device != inputs.input_ids.device
            for tensor in (prefixes, cu)
        )
    ):
        raise RuntimeError("MTP PD position geometry is invalid")
    tensors = [inputs.combo_position_ids]
    tensors.extend(value.combo_position_ids for value in values)
    if any(
        tensor.dtype != torch.int32
        or tensor.device != inputs.input_ids.device
        or tensor.numel() != token_count * index_factor
        for tensor in tensors
    ):
        raise RuntimeError("MTP PD transported position geometry is invalid")
    rows = torch.arange(token_count, dtype=torch.int32, device=inputs.input_ids.device)
    request = torch.bucketize(rows, cu[1:], right=True).clamp_max(prefixes.numel() - 1)
    # Native metadata is validated by QSA before any cache write. Keep these
    # gathers in range even if corrupted cumulative lengths reach preparation.
    positions = (
        prefixes.index_select(0, request) + rows - cu.index_select(0, request)
    ).repeat_interleave(index_factor)
    copied = set()
    for tensor in tensors:
        if tensor.data_ptr() not in copied:
            tensor.copy_(positions.reshape_as(tensor))
            copied.add(tensor.data_ptr())


class Qwen4ExpMTPInputProjection(nn.Module):
    """Fuse one token embedding into every target hyper-connection branch."""

    def __init__(
        self,
        embedding_gamma: torch.Tensor,
        hidden_gamma: torch.Tensor,
        fc_embedding: nn.Module,
        fc_hidden: nn.Module,
        hidden_size: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.embedding_gamma = embedding_gamma
        self.hidden_gamma = hidden_gamma
        self.fc_embedding = fc_embedding
        self.fc_hidden = fc_hidden
        self.hidden_size = int(hidden_size)
        if int(hidden_gamma.numel()) % self.hidden_size:
            raise ValueError(
                "qwen4_exp MTP hidden norm width must be divisible by hidden_size"
            )
        self.hc_mult = int(hidden_gamma.numel()) // self.hidden_size
        self.hc_hidden_size = self.hc_mult * self.hidden_size
        self.norm_eps = float(norm_eps)

    def forward(
        self, inputs_embeds: torch.Tensor, input_hiddens: torch.Tensor
    ) -> torch.Tensor:
        if inputs_embeds.dim() != 2 or input_hiddens.dim() != 2:
            raise RuntimeError(
                "qwen4_exp MTP input projection expects packed 2-D tensors"
            )
        if int(inputs_embeds.shape[0]) != int(input_hiddens.shape[0]):
            raise RuntimeError(
                "qwen4_exp MTP embedding/hidden row counts differ: "
                f"{inputs_embeds.shape[0]} != {input_hiddens.shape[0]}"
            )
        if int(inputs_embeds.shape[-1]) != self.hidden_size:
            raise RuntimeError(
                "qwen4_exp MTP embedding width mismatch: expected "
                f"{self.hidden_size}, got {inputs_embeds.shape[-1]}"
            )
        if int(input_hiddens.shape[-1]) != self.hc_hidden_size:
            raise RuntimeError(
                "qwen4_exp MTP target hidden width mismatch: expected "
                f"{self.hc_hidden_size}, got {input_hiddens.shape[-1]}"
            )
        if tuple(self.embedding_gamma.shape) != (self.hidden_size,) or tuple(
            self.hidden_gamma.shape
        ) != (self.hc_hidden_size,):
            raise RuntimeError("qwen4_exp MTP pre-FC norm weight shape is invalid")

        embedding = exact_head_rms_norm(
            inputs_embeds, self.embedding_gamma, self.norm_eps
        )
        hidden = grouped_rms_norm(
            input_hiddens,
            self.hidden_gamma,
            self.hidden_size,
            self.norm_eps,
        ).unflatten(-1, (self.hc_mult, self.hidden_size))

        # fc_embedding/fc_hidden are hidden_size-wide matrices.  Apply them to
        # every branch independently; pooling the target stream before either
        # projection destroys the released head's trained branch semantics.
        projected_embedding = self.fc_embedding(embedding).unsqueeze(-2)
        projected_hidden = self.fc_hidden(hidden.reshape(-1, self.hidden_size)).reshape(
            int(hidden.shape[0]), self.hc_mult, self.hidden_size
        )
        return (projected_embedding + projected_hidden).flatten(-2)


class Qwen4ExpMTPModel(Qwen4ExpModel):
    """One-layer draft model with target-hidden input fusion and QSA."""

    def supports_cuda_graph_draft_prefill(self) -> bool:
        return os.environ.get("RTP_LLM_QWEN4_DRAFT_PREFILL_GRAPH", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        groups = get_attention_inputs_value(inputs)
        values = tuple(groups.values()) if isinstance(groups, Mapping) else (groups,)
        if any(
            getattr(value, "cache_store_inputs", None) is not None for value in values
        ):
            normalize_mtp_pd_positions(
                inputs, int(self.config.attn_config.rope_config.index_factor)
            )
        impls = super().prepare_fmha_impl(inputs, is_cuda_graph)
        groups = get_attention_inputs_value(inputs)
        if (
            not is_cuda_graph
            or not groups
            or not next(iter(groups.values())).is_prefill
        ):
            return impls
        from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
            INDEXER_KV_TAG,
            INDEXER_STATE_TAG,
        )
        from rtp_llm.models_py.modules.qwen4_exp.draft_prefill_graph import (
            DraftPrefillGraphReplay,
        )

        bounds = []
        for layer_idx, layer in enumerate(self.layers):
            tag = self._attention_tag(layer_idx)
            cache = self.kv_cache.get_layer_cache(layer_idx, tag)
            bounds.append(
                (
                    tag,
                    int(cache.kv_cache_base.shape[0]),
                    int(impls[tag].attn_configs.kernel_tokens_per_block),
                    False,
                )
            )
            for side_tag in (INDEXER_KV_TAG, INDEXER_STATE_TAG):
                cache = self.kv_cache.get_layer_cache(layer_idx, side_tag)
                page = int(cache.seq_size_per_block)
                bounds.append(
                    (
                        side_tag,
                        int(cache.kv_cache_base.shape[0]),
                        page // 4 if side_tag == INDEXER_KV_TAG else page,
                        side_tag == INDEXER_KV_TAG,
                    )
                )
        for impl in impls.values():
            impl.set_mtp_draft_mode(True)
        return DraftPrefillGraphReplay(
            impls,
            bounds,
            self.cuda_graph_position_id_len_factor(),
            int(inputs.input_ids.numel()),
        )

    def _build_attn_meta(self, inputs, device, is_cuda_graph=False):
        groups = get_attention_inputs_value(inputs)
        if is_cuda_graph and next(iter(groups.values())).is_prefill:
            if any(
                layer.layer_type == HybridAttentionType.LINEAR for layer in self.layers
            ):
                raise RuntimeError(
                    "Qwen4 MTP draft Graph requires full-attention layers"
                )
            # The one-layer draft has no GDN convolution. Its dynamic metadata
            # builder is unnecessary and performs capture-unsafe host copies.
            return Qwen3NextMetadata(is_cuda_graph=True)
        return super()._build_attn_meta(inputs, device, is_cuda_graph)

    def cuda_graph_position_id_len_factor(self) -> int:
        # The draft uses Base RoPE but its QSA indexer consumes the engine's
        # explicit text positions, including their configured axis width.
        return int(self.config.attn_config.rope_config.index_factor)

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
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
        if self.layer_num != 1:
            raise RuntimeError("qwen4_exp MTP model descriptor requires one layer")
        if getattr(model_config, "enable_qwen4_ple", False):
            raise RuntimeError("qwen4_exp MTP model descriptor does not support PLE")
        self.mtp_input_projection = Qwen4ExpMTPInputProjection(
            weights.get_global_weight(W.multi_tokens_predict_enorm),
            weights.get_global_weight(W.multi_tokens_predict_hnorm),
            LinearFactory.create_linear_from_weights(
                weights.global_weights,
                W.qwen4_mtp_fc_embedding_w,
                hw_kernel_config=py_hw_kernel_config,
            ),
            LinearFactory.create_linear_from_weights(
                weights.global_weights,
                W.qwen4_mtp_fc_hidden_w,
                hw_kernel_config=py_hw_kernel_config,
            ),
            hidden_size=model_config.hidden_size,
            norm_eps=model_config.layernorm_eps,
        )

    def word_embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        input_hiddens: Optional[torch.Tensor] = getattr(inputs, "input_hiddens", None)
        if not isinstance(input_hiddens, torch.Tensor) or not input_hiddens.numel():
            raise RuntimeError(
                "qwen4_exp MTP requires hc_mult*hidden_size-wide target input_hiddens"
            )
        # The MTP draft consumes text ids only; do not enter Qwen35's multimodal
        # embedding injector from this override.
        inputs_embeds = self.embed_tokens(inputs.input_ids)
        return self.mtp_input_projection(inputs_embeds, input_hiddens)

    def _initial_hyper_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = int(self.config.hc_mult) * int(self.config.hidden_size)
        if int(hidden_states.shape[-1]) != expected:
            raise RuntimeError(
                "qwen4_exp MTP projected residual width mismatch: "
                f"expected {expected}, got {hidden_states.shape[-1]}"
            )
        return hidden_states
