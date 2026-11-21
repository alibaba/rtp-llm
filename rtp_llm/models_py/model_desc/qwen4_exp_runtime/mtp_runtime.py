"""MTP graph/cache behavior shared by legacy and direct-parameter constructors."""

from collections.abc import Mapping
from typing import Optional

import torch

from rtp_llm.models_py.model_desc.block_map import get_attention_inputs_value
from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextMetadata
from rtp_llm.ops import HybridAttentionType
from rtp_llm.ops.compute_ops import PyModelInputs


def normalize_mtp_prefill_positions(inputs, index_factor: int) -> None:
    """Use logical draft-cache coordinates for every eager prefill.

    Native prompt prefill shifts the token and position rows together. Cold
    requests and prefix hits must write the same RoPE origin, including when
    they share a batch. Accepted-token commits already use these coordinates.
    Decode and CUDA Graph inputs retain their own position preparation.
    """
    groups = get_attention_inputs_value(inputs)
    values = tuple(groups.values()) if isinstance(groups, Mapping) else (groups,)
    if not values:
        return
    if any(getattr(value, "cache_store_inputs", None) is not None for value in values):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.pd_publication import (
            validate_pd_cache_inputs,
        )

        validate_pd_cache_inputs(values)
    elif any(
        not value.is_prefill or value.is_target_verify or value.is_cuda_graph
        for value in values
    ):
        return
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
        raise RuntimeError("MTP prefill position geometry is invalid")
    tensors = [inputs.combo_position_ids]
    tensors.extend(value.combo_position_ids for value in values)
    if any(
        tensor.dtype != torch.int32
        or tensor.device != inputs.input_ids.device
        or tensor.numel() != token_count * index_factor
        for tensor in tensors
    ):
        raise RuntimeError("MTP prefill transported position geometry is invalid")
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


def normalize_mtp_decode_positions(inputs, index_factor: int) -> None:
    """Keep draft decode in the logical Base-RoPE coordinates of its prompt KV.

    Native stream positions retain the target's media-dependent MRoPE offset.
    Build from live device lengths inside the captured forward so replay also
    refreshes these positions without reading a device scalar on the host.
    """
    groups = get_attention_inputs_value(inputs)
    values = tuple(groups.values()) if isinstance(groups, Mapping) else (groups,)
    if not values or any(
        value.is_prefill or value.is_target_verify for value in values
    ):
        return
    anchor = values[0]
    lengths = (
        anchor.sequence_lengths_plus_1_device - 1
        if anchor.is_cuda_graph
        else anchor.sequence_lengths
    )
    if (
        index_factor <= 0
        or lengths.ndim != 1
        or lengths.numel() != inputs.input_ids.numel()
    ):
        raise RuntimeError("MTP decode position geometry is invalid")
    positions = lengths.to(
        device=inputs.input_ids.device, dtype=torch.int32, non_blocking=True
    ).repeat_interleave(index_factor)
    tensors = [
        inputs.combo_position_ids,
        *(value.combo_position_ids for value in values),
    ]
    copied = set()
    for tensor in tensors:
        if (
            tensor.dtype != torch.int32
            or tensor.device != positions.device
            or tensor.numel() != positions.numel()
        ):
            raise RuntimeError("MTP decode transported position geometry is invalid")
        if tensor.data_ptr() not in copied:
            tensor.copy_(positions.reshape_as(tensor))
            copied.add(tensor.data_ptr())


class Qwen4ExpMTPRuntime:
    """One-layer draft model with target-hidden input fusion and QSA."""

    def cuda_graph_draft_prefill_uses_request_lanes(self) -> bool:
        # QSA reserves speculative token lanes independently of the HC width.
        return True

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        if not is_cuda_graph:
            normalize_mtp_prefill_positions(
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
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
            make_mtp_draft_replay,
        )

        return make_mtp_draft_replay(
            self, impls, token_capacity=int(inputs.input_ids.numel())
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

    def forward(self, inputs: PyModelInputs, fmha_impl=None):
        normalize_mtp_decode_positions(
            inputs, int(self.config.attn_config.rope_config.index_factor)
        )
        return super().forward(inputs, fmha_impl)

    def word_embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        input_hiddens: Optional[torch.Tensor] = getattr(inputs, "input_hiddens", None)
        if not isinstance(input_hiddens, torch.Tensor) or not input_hiddens.numel():
            raise RuntimeError(
                "qwen4_exp MTP requires hc_mult*hidden_size-wide target input_hiddens"
            )
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen35Model

        # Inject the shifted prompt's visual rows before the draft projection.
        mm = getattr(inputs, "multimodal_inputs", None)
        inputs_embeds = (
            Qwen35Model.word_embedding(self, inputs)
            if mm is not None and len(mm.multimodal_features)
            else self.embed_tokens(inputs.input_ids)
        )
        return self.mtp_input_projection(inputs_embeds, input_hiddens)

    def _initial_hyper_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = int(self.config.hc_mult) * int(self.config.hidden_size)
        if int(hidden_states.shape[-1]) != expected:
            raise RuntimeError(
                "qwen4_exp MTP projected residual width mismatch: "
                f"expected {expected}, got {hidden_states.shape[-1]}"
            )
        return hidden_states
