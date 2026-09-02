"""Qwen3 DSpark backbone on the shared DSpark proposer contract."""

from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import is_hip
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_attention_inputs_for_layer
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.modules import LinearFactory, RMSNorm
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin
from rtp_llm.ops import ParallelismConfig, check_rope_cache, get_rope_cache_once
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class _RopePositions:
    def __init__(self, positions: torch.Tensor) -> None:
        self.positions_d = positions


def _apply_non_interleaved_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> None:
    """Apply LLaMA/Qwen-style RoPE to the leading dimensions in-place."""
    if query.numel() == 0:
        return
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be positive and even, got {rope_dim}")
    if rope_dim > query.size(-1) or rope_dim > key.size(-1):
        raise ValueError(
            f"RoPE dimension {rope_dim} exceeds Q/K head dimensions "
            f"{query.size(-1)}/{key.size(-1)}"
        )

    positions = positions.narrow(0, 0, query.size(0)).long()
    cos_sin = cos_sin_cache.index_select(0, positions)
    half_dim = rope_dim // 2
    cos = cos_sin[:, :half_dim].unsqueeze(1)
    sin = cos_sin[:, half_dim:rope_dim].unsqueeze(1)

    def rotate(tensor: torch.Tensor) -> None:
        rope_input = tensor[..., :rope_dim].float()
        first = rope_input[..., :half_dim]
        second = rope_input[..., half_dim:]
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
        tensor[..., :rope_dim].copy_(rotated)

    rotate(query)
    rotate(key)


def _write_rocm_paged_kv_cache(
    cache: torch.Tensor,
    pages: torch.Tensor,
    slots: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    vectorized_value: bool,
) -> None:
    """Write semantic K/V rows using the physical layout consumed by AITER."""
    if cache.dim() != 5 or cache.size(1) != 2:
        raise ValueError(
            "Qwen3 DSpark ROCm context cache must be "
            f"[blocks,2,heads,page,dim], got {tuple(cache.shape)}"
        )

    block_count, _, kv_heads, page_size, head_dim = cache.shape
    del block_count
    element_size = cache.element_size()
    if element_size <= 0 or 16 % element_size:
        raise ValueError(f"unsupported ROCm KV-cache element size: {element_size}")
    vector_size = 16 // element_size
    if head_dim % vector_size:
        raise ValueError(
            f"ROCm K head dimension {head_dim} must be divisible by {vector_size}"
        )

    key_rows = (
        key.to(cache.dtype)
        .contiguous()
        .view(-1, kv_heads, head_dim // vector_size, vector_size)
    )
    key_physical = cache[:, 0].view(
        cache.shape[0], kv_heads, head_dim // vector_size, page_size, vector_size
    )
    key_physical[pages, :, :, slots, :] = key_rows

    value_rows = value.to(cache.dtype).contiguous()
    if vectorized_value:
        if page_size % vector_size:
            raise ValueError(
                f"ROCm V page size {page_size} must be divisible by {vector_size}"
            )
        value_physical = cache[:, 1].view(
            cache.shape[0], kv_heads, page_size // vector_size, head_dim, vector_size
        )
        value_physical[pages, :, slots // vector_size, :, slots % vector_size] = (
            value_rows
        )
    else:
        value_physical = cache[:, 1].view(cache.shape[0], kv_heads, head_dim, page_size)
        value_physical[pages, :, :, slots] = value_rows


class _TorchMhaRotaryEmbeddingOp:
    """ROCm fallback for the FlashInfer-only DSpark context RoPE path."""

    def __init__(self, attn_config) -> None:
        self.rope_config = attn_config.rope_config
        self.rope_dim = self.rope_config.dim
        rope_cache = get_rope_cache_once(
            self.rope_config,
            attn_config.max_seq_len + attn_config.gen_num_per_cycle + 1,
            # This API flag means "GPU cache"; torch::kCUDA maps to HIP in a
            # ROCm PyTorch build.
            is_cuda=True,
            interleave=False,
        )
        if not check_rope_cache(self.rope_config, rope_cache):
            raise RuntimeError(
                "Qwen3 DSpark ROCm context RoPE requires a supported GPU cache"
            )
        self.cos_sin_cache = rope_cache.data

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rope_params: Any,
    ) -> None:
        _apply_non_interleaved_rope(
            query,
            key,
            self.cos_sin_cache,
            rope_params.positions_d,
            self.rope_dim,
        )


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
            aux_feature_dim=len(config.dspark_target_layer_ids) * config.hidden_size,
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

        if is_hip():
            self.context_rope = _TorchMhaRotaryEmbeddingOp(self.attn_configs)
        else:
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
            if is_hip():
                is_fp8_cache = cache.dtype in (
                    torch.float8_e4m3fnuz,
                    torch.float8_e4m3fn,
                )
                vectorized_value = (
                    self.fmha_config is None
                    or self.fmha_config.use_asm_pa
                    or is_fp8_cache
                )
                _write_rocm_paged_kv_cache(
                    cache,
                    pages,
                    slots,
                    key,
                    value,
                    vectorized_value=vectorized_value,
                )
            else:
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
        del query_ids, prefix_lengths, active_requests

        # The target owns the engine input geometry, so an MRoPE target can
        # publish three position ids per token into the shared attention input.
        # This Qwen3 draft uses scalar RoPE. The FMHA implementation has already
        # retained the shared tensor during prepare_fmha_impl(), therefore rebind
        # is too late here: update the consumed prefix in-place for the duration
        # of the draft layers, then restore it for the target verify path.
        attention = self.dspark_attention_inputs(inputs)
        position_ids = getattr(attention, "combo_position_ids", None)
        restore_position_ids = None
        position_slice = None
        draft_positions = query_positions.reshape(-1)
        if (
            isinstance(position_ids, torch.Tensor)
            and position_ids.numel() > 0
            and draft_positions.numel() > 0
        ):
            if (
                position_ids.dim() != 1
                or position_ids.numel() < draft_positions.numel()
            ):
                raise RuntimeError(
                    "Qwen3 DSpark position ids must be a flat buffer with at least "
                    f"one entry per query token: got shape={tuple(position_ids.shape)}, "
                    f"query_tokens={draft_positions.numel()}"
                )
            position_slice = position_ids.narrow(0, 0, draft_positions.numel())
            restore_position_ids = position_slice.clone()
            position_slice.copy_(
                draft_positions.to(
                    device=position_ids.device,
                    dtype=position_ids.dtype,
                )
            )

        try:
            return super().forward(inputs, fmha_impl).hidden_states
        finally:
            if restore_position_ids is not None:
                position_slice.copy_(restore_position_ids)

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
                (tokens + self._dspark_query_width - 1) // self._dspark_query_width,
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
