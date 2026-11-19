"""Qwen3 DSpark backbone on the shared DSpark proposer contract."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import is_hip, is_ppu
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_group_tags_for_layers,
    select_attention_inputs_for_layer,
)
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.modules import LinearFactory, RMSNorm
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import (
    DSparkProposerMixin,
    map_context_rows,
)
from rtp_llm.ops import ParallelismConfig, check_rope_cache, get_rope_cache_once
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class _RopePositions:
    def __init__(self, positions: torch.Tensor) -> None:
        self.positions_d = positions


class _PpuDraftAttention:
    """Keep one graph-stable FA3 schedule per distinct proposal mask."""

    def __init__(self, configs, inputs, windows):
        from rtp_llm.models_py.modules.factory.attention.ppu_impl.fa3_mha import (
            FA3PrefillPagedImpl,
        )

        if not FA3PrefillPagedImpl.support(configs, inputs):
            raise RuntimeError(
                "native PPU DSpark proposals require paged FA3; check the attention "
                "layout and PPU_USE_FA3_PREFILL/PPU_USE_FA3_VERIFY settings"
            )
        self.plans = {
            window: FA3PrefillPagedImpl(
                configs,
                inputs,
                causal=window is not None,
                window_size=(-1, -1) if window is None else (window - 1, 0),
            )
            for window in dict.fromkeys(windows)
        }
        self.layers = tuple(self.plans[window] for window in windows)

    def forward(self, qkv, kv_cache, layer_idx):
        return self.layers[layer_idx].forward(qkv, kv_cache, layer_idx)

    def support_cuda_graph(self):
        return True

    def prepare_cuda_graph(self, inputs):
        for plan in self.plans.values():
            plan.prepare_cuda_graph(inputs)


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
    # Reinterpret the contiguous backing storage before selecting K/V.  The
    # cache[:, plane] slice is strided across blocks and cannot be viewed.
    key_physical = cache.view(
        cache.shape[0],
        2,
        kv_heads,
        head_dim // vector_size,
        page_size,
        vector_size,
    )[:, 0]
    key_physical[pages, :, :, slots, :] = key_rows

    value_rows = value.to(cache.dtype).contiguous()
    if vectorized_value:
        if page_size % vector_size:
            raise ValueError(
                f"ROCm V page size {page_size} must be divisible by {vector_size}"
            )
        value_physical = cache.view(
            cache.shape[0],
            2,
            kv_heads,
            page_size // vector_size,
            head_dim,
            vector_size,
        )[:, 1]
        value_physical[pages, :, slots // vector_size, :, slots % vector_size] = (
            value_rows
        )
    else:
        value_physical = cache.view(cache.shape[0], 2, kv_heads, head_dim, page_size)[
            :, 1
        ]
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
        *,
        commit_only: bool = False,
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
            build_decoder_layers=not commit_only,
        )
        self._commit_only = commit_only
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

        layer_weights = weights.weights[: self.layer_num]
        self.context_k_norms = nn.ModuleList(
            RMSNorm(layer[W.k_ln_gamma], eps=config.layernorm_eps)
            for layer in layer_weights
        )
        context_kv = weights.get_global_weight_or_none(W.dspark_context_kv_w)
        if context_kv is None:
            q_cols = self.attn_configs.head_num * self.attn_configs.size_per_head
            context_kv = torch.cat(
                [layer[W.attn_qkv_w][:, q_cols:] for layer in layer_weights], dim=1
            )
        self.context_kv_projection = LinearFactory.create_linear(
            context_kv,
            None,
            None,
            None,
            py_hw_kernel_config,
        )

        if (
            config.dspark_layer_windows
            and any(config.dspark_layer_windows)
            and not is_ppu()
        ):
            raise NotImplementedError(
                "native DSpark sliding proposal attention currently requires PPU FA3"
            )
        if (
            parallelism_config.prefill_cp_config.is_enabled()
            and parallelism_config.prefill_cp_config.kv_cache_sharded
        ):
            raise NotImplementedError(
                "Qwen3 DSpark CP commit requires replicated prompt KV caches"
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

    @staticmethod
    def prepare_commit_fmha_impl(inputs, is_cuda_graph=False):
        # COMMIT projects verified features into KV; it never issues a query.
        return None

    def prepare_fmha_impl(
        self, inputs, is_cuda_graph=False, cuda_graph_selection_mode=None
    ):
        # A CP prefill worker only seeds the draft's KV, never proposes a block.
        if self.parallelism_config.prefill_cp_config.is_enabled():
            return None
        if is_ppu() and self.config.dspark_layer_windows is not None:
            attention = inputs.attention_inputs
            windows = self.config.dspark_layer_windows
            if isinstance(attention, Mapping):
                # The graph runner refreshes each dict entry with that tag's
                # block table. The GLM target's indexer table is not draft KV.
                return {
                    tag: _PpuDraftAttention(self.attn_configs, attention[tag], windows)
                    for tag in get_group_tags_for_layers(
                        self.kv_cache, range(self.layer_num)
                    )
                }
            return _PpuDraftAttention(self.attn_configs, attention, windows)
        return super().prepare_fmha_impl(
            inputs, is_cuda_graph, cuda_graph_selection_mode
        )

    def map_commit_rows(self, starts, lengths, committed_ends, row_count, inputs):
        if not self.parallelism_config.prefill_cp_config.is_enabled():
            return super().map_commit_rows(
                starts, lengths, committed_ends, row_count, inputs
            )
        info = self.dspark_attention_inputs(inputs).context_parallel_info
        if info is None or info.prefill_qkv_padding_mask_cpu is None:
            raise RuntimeError(
                "DSpark CP commit requires host row restoration metadata"
            )
        # restore maps global padded order to rank-major order from all_gather.
        # Construct the unpadded index on the CPU where CP produces the plan.
        mask = info.prefill_qkv_padding_mask_cpu
        if mask.numel() != self.parallelism_config.tp_size * row_count:
            raise ValueError(
                "DSpark commit feature rows do not match CP padding metadata"
            )
        order = info.prefill_qkv_restore_indice_cpu[mask == 1].to(
            device=starts.device, dtype=torch.long, non_blocking=True
        )
        global_lengths = info.prefill_actual_input_lengths_cpu.to(
            device=starts.device, dtype=torch.long, non_blocking=True
        )
        req, positions = map_context_rows(
            global_lengths.cumsum(0) - global_lengths,
            global_lengths,
            committed_ends,
            order.numel(),
        )
        return req, positions, order

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
        del committed_ends
        if commit_ctx is not None:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

            # All prompt ranks own every KV head. Project the local 5H features
            # down to H before exchanging rows, then drop padding in global order.
            main_x = all_gather(main_x.contiguous(), group=Group.TP).index_select(
                0, commit_ctx
            )
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
        if self._commit_only:
            raise RuntimeError(
                "a dedicated DSpark prefill worker cannot propose tokens"
            )
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
