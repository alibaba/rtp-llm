"""Qwen3 DFlash V1 block drafter.

Draft residual and Q-projection widths can differ (5120 vs 4096 in the 27B
pairings). Attention geometry and the causal sliding-window / non-causal
full layer policy therefore come from each draft checkpoint.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import is_hip
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_attention_inputs_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3_dspark_model import (
    _RopePositions,
    _TorchMhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules import (
    DenseMLP,
    Embedding,
    FusedQKRMSNorm,
    LinearFactory,
    RMSNorm,
)
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin
from rtp_llm.models_py.triton_kernels.common.dflash_attention import (
    DFlashCacheLayout,
    dflash_paged_attention,
    dflash_write_paged_kv,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


def _single_attention_input(model, inputs: PyModelInputs, layer_idx: int):
    attention = select_attention_inputs_for_layer(inputs, model.kv_cache, layer_idx)
    if isinstance(attention, list):
        if len(attention) != 1:
            raise RuntimeError(
                "DFlash V1 requires exactly one cache group per draft layer; "
                f"layer={layer_idx}, groups={len(attention)}"
            )
        attention = attention[0]
    return attention


def _block_table(attention, device: torch.device) -> torch.Tensor:
    table = getattr(attention, "kv_cache_kernel_block_id_device", None)
    if table is None or table.numel() == 0:
        table = getattr(attention, "kv_cache_kernel_block_id", None)
        if table is None or table.numel() == 0:
            raise RuntimeError("DFlash requires a non-empty KV kernel block table")
        table = table.to(device=device, non_blocking=True)
    if table.dim() == 3:
        if table.shape[0] != 1:
            raise RuntimeError("DFlash V1 supports one cache group per draft layer")
        table = table[0]
    if table.dim() != 2:
        raise RuntimeError(
            f"DFlash block table must be [B,pages], got {tuple(table.shape)}"
        )
    # Triton receives explicit row/column strides.  Returning this stable
    # framework-owned view avoids an accidental graph-capture allocation.
    return table


def _cache_layout(cache: torch.Tensor, fmha_config: Any) -> DFlashCacheLayout:
    if not is_hip():
        return DFlashCacheLayout.CUDA
    is_fp8 = cache.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
    vectorized_value = (
        fmha_config is None or bool(getattr(fmha_config, "use_asm_pa", False)) or is_fp8
    )
    return (
        DFlashCacheLayout.AITER_VECTOR if vectorized_value else DFlashCacheLayout.AITER
    )


class _DFlashAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: dict[str, torch.Tensor],
        layer_idx: int,
        layer_type: str,
        quant_config: Any,
        hw_kernel_config: Any,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.parallelism_config = parallelism_config
        self.attn_configs = config.getAttentionConfigs(
            parallelism_config.get_attn_tp_size()
        )
        self.q_heads = self.attn_configs.head_num
        self.kv_heads = self.attn_configs.kv_head_num
        self.head_dim = self.attn_configs.size_per_head
        self.q_size = self.q_heads * self.head_dim
        self.kv_size = self.kv_heads * self.head_dim
        self.window_size = int(config.dflash_sliding_window or 0)
        if layer_type == "sliding_attention" and self.window_size <= 0:
            raise ValueError("DFlash sliding_attention needs a positive window")
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError(f"unsupported DFlash layer type {layer_type!r}")
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_qkv_w,
            W.attn_qkv_s,
            W.attn_qkv_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_qkv_s2,
            input_scale_key=W.attn_qkv_i_s,
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_o_w,
            W.attn_o_s,
            W.attn_o_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_o_s2,
            input_scale_key=W.attn_o_i_s,
        )
        self.qk_norm = FusedQKRMSNorm(
            weights[W.q_ln_gamma],
            weights[W.k_ln_gamma],
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            config.layernorm_eps,
        )
        self._out: torch.Tensor | None = None

    def _output_buffer(self, query: torch.Tensor) -> torch.Tensor:
        rows, heads, dim = query.shape
        if (
            self._out is None
            or self._out.device != query.device
            or self._out.dtype != query.dtype
            or self._out.shape[0] < rows
            or self._out.shape[1:] != (heads, dim)
        ):
            raise RuntimeError(
                "DFlash attention output was not prepared for this input shape; "
                "prepare_fmha_impl must run before forward"
            )
        return self._out[:rows]

    def reserve_output(
        self, rows: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Allocate graph-stable attention output before the model forward."""
        if rows < 0:
            raise ValueError(f"DFlash output rows must be non-negative, got {rows}")
        if (
            self._out is None
            or self._out.device != device
            or self._out.dtype != dtype
            or self._out.shape[0] < rows
        ):
            self._out = torch.empty(
                (rows, self.q_heads, self.head_dim), device=device, dtype=dtype
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
        request_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        query_width: int,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        rope: _TorchMhaRotaryEmbeddingOp,
        fmha_config: Any,
    ) -> torch.Tensor:
        qkv = self.qk_norm(self.qkv_proj(hidden_states))
        query = qkv[:, : self.q_size].view(-1, self.q_heads, self.head_dim)
        key = qkv[:, self.q_size : self.q_size + self.kv_size].view(
            -1, self.kv_heads, self.head_dim
        )
        value = qkv[:, self.q_size + self.kv_size :].view(
            -1, self.kv_heads, self.head_dim
        )
        # Inactive graph-padding rows are not persisted.  Clamp only for the
        # RoPE lookup; the writer masks them by their negative request/position.
        rope._apply_rope(query, key, _RopePositions(query_positions.clamp_min(0)))
        layout = _cache_layout(cache, fmha_config)
        dflash_write_paged_kv(
            key,
            value,
            cache,
            block_table,
            request_ids,
            query_positions,
            cache_layout=layout,
        )
        attn = dflash_paged_attention(
            query,
            cache,
            block_table,
            sequence_lengths,
            query_width,
            causal=self.layer_type == "sliding_attention",
            window_size=(
                self.window_size if self.layer_type == "sliding_attention" else 0
            ),
            cache_layout=layout,
            out=self._output_buffer(query),
        )
        output = self.o_proj(attn.reshape(hidden_states.shape[0], -1).contiguous())
        if self.parallelism_config.get_attn_tp_size() > 1:
            output = all_reduce(output, group=Group.TP)
        return output


class _DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: dict[str, torch.Tensor],
        layer_idx: int,
        layer_type: str,
        quant_config: Any,
        hw_kernel_config: Any,
    ) -> None:
        super().__init__()
        self.self_attn = _DFlashAttention(
            config,
            parallelism_config,
            weights,
            layer_idx,
            layer_type,
            quant_config,
            hw_kernel_config,
        )
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3DFlashModel(DSparkProposerMixin, GptModelBase):
    """DFlash V1 using target-owned embedding and lm-head weights."""

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
            raise NotImplementedError("Qwen3 DFlash quantization is not supported")
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        layer_types = config.dflash_layer_types or []
        if len(layer_types) != self.layer_num:
            raise ValueError("DFlash layer policy does not match draft layer count")
        self.init_dspark_proposer(
            width=int(config.gen_num_per_cycle),
            query_width=int(config.gen_num_per_cycle) + 1,
            noise_token_id=int(config.dflash_mask_token_id),
            aux_feature_dim=len(config.dflash_target_layer_ids or [])
            * config.hidden_size,
            hidden_dim=config.hidden_size,
        )
        self.attn_configs = config.getAttentionConfigs(
            parallelism_config.get_attn_tp_size()
        )
        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.dspark_fc_w
        )
        self.hidden_norm = RMSNorm(
            weights.get_global_weight(W.dspark_hidden_norm_gamma),
            eps=config.layernorm_eps,
        )
        q_cols = self.attn_configs.head_num * self.attn_configs.size_per_head
        context_kv_weights = []
        self.context_k_norms = nn.ModuleList()
        for layer_weights in weights.weights[: self.layer_num]:
            context_kv_weights.append(layer_weights[W.attn_qkv_w][:, q_cols:])
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
        self.layers = nn.ModuleList(
            _DFlashDecoderLayer(
                config,
                parallelism_config,
                weights.weights[index],
                index,
                layer_types[index],
                quant_config,
                py_hw_kernel_config,
            )
            for index in range(self.layer_num)
        )

    def cuda_graph_input_hidden_size(self) -> int:
        return self._dspark_aux_feature_dim

    def combine_hidden_states(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> None:
        """DFlash owns its mixed-mask paged-attention metadata and kernels."""
        # Commit/prefill writes context KV only.  Its token count can be a full
        # prompt, while DFlash attention is never evaluated on those rows, so
        # reserving [rows,Hq,D] per layer here would pin multi-GB dead buffers.
        # CUDA graph setup supplies a nonempty, graph-stable input_hiddens
        # buffer for every role, including proposal.  It therefore cannot use
        # input_hiddens to distinguish the commit role.
        input_hiddens = getattr(inputs, "input_hiddens", None)
        if (
            not is_cuda_graph
            and isinstance(input_hiddens, torch.Tensor)
            and input_hiddens.numel() > 0
        ):
            return None
        input_ids = getattr(inputs, "input_ids", None)
        if not isinstance(input_ids, torch.Tensor):
            return None
        rows = int(input_ids.numel())
        device = self._forward_device()
        dtype = self.embed_tokens.weight.dtype
        for layer in self.layers:
            layer.self_attn.reserve_output(rows, device, dtype)
        return None

    def _layer_cache_and_table(self, inputs: PyModelInputs, layer_idx: int):
        attention = _single_attention_input(self, inputs, layer_idx)
        cache = self.kv_cache.get_layer_cache(layer_idx).kv_cache_base
        return cache, _block_table(attention, cache.device)

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
        hidden = self.hidden_norm(main_x)
        head_dim = self.attn_configs.size_per_head
        kv_heads = self.attn_configs.kv_head_num
        dummy_q = hidden.new_zeros((hidden.shape[0], 1, head_dim))
        all_kv = self.context_kv_projection(hidden).view(
            -1,
            self.layer_num,
            2,
            kv_heads,
            head_dim,
        )
        for layer_idx in range(self.layer_num):
            key, value = all_kv[:, layer_idx].unbind(1)
            # Selecting one layer retains the layer stride in the leading
            # dimensions.  RMSNorm's CUDA binding requires a contiguous
            # input, while the paged writer accepts the original strided V.
            key = self.context_k_norms[layer_idx](
                key.contiguous().reshape(-1, head_dim)
            ).view(-1, kv_heads, head_dim)
            self.context_rope._apply_rope(
                dummy_q, key, _RopePositions(context_positions.clamp_min(0))
            )
            cache, table = self._layer_cache_and_table(inputs, layer_idx)
            dflash_write_paged_kv(
                key,
                value,
                cache,
                table,
                context_req_ids,
                context_positions,
                cache_layout=_cache_layout(cache, self.fmha_config),
            )

        attention = _single_attention_input(self, inputs, 0)
        writer = create_write_cache_store_impl(attention)
        if writer is not None:
            for layer_idx in range(self.layer_num):
                layer_caches = self.kv_cache.get_layer_cache_groups(layer_idx)
                if len(layer_caches) != 1:
                    raise RuntimeError(
                        "DFlash V1 requires one KV cache group per draft layer, "
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
        del fmha_impl
        width = query_positions.shape[1]
        batch = query_positions.shape[0]
        # The shared fixed-block buffer may retain arbitrary values in proposal
        # slots.  Only column zero is a genuine anchor; the mixin builds the
        # remaining columns from DFlash's configured mask token.
        hidden_states = self.embed_tokens(query_ids.reshape(-1))
        request_ids = torch.arange(
            batch, device=hidden_states.device, dtype=torch.int32
        ).repeat_interleave(width)
        positions = query_positions.reshape(-1).to(
            device=hidden_states.device, dtype=torch.int32
        )
        live = active_requests.to(device=hidden_states.device).repeat_interleave(width)
        request_ids = torch.where(live, request_ids, torch.full_like(request_ids, -1))
        positions = torch.where(live, positions, torch.full_like(positions, -1))
        sequence_lengths = (
            prefix_lengths.to(device=hidden_states.device, dtype=torch.int32) + width
        )
        active_requests = active_requests.to(
            device=hidden_states.device, dtype=torch.bool
        )
        sequence_lengths = torch.where(
            active_requests, sequence_lengths, torch.zeros_like(sequence_lengths)
        )
        for layer_idx, layer in enumerate(self.layers):
            cache, table = self._layer_cache_and_table(inputs, layer_idx)
            hidden_states = layer(
                hidden_states,
                query_positions=positions,
                request_ids=request_ids,
                sequence_lengths=sequence_lengths,
                query_width=width,
                cache=cache,
                block_table=table,
                rope=self.context_rope,
                fmha_config=self.fmha_config,
            )
        return self.norm(hidden_states).contiguous()

    def _forward_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    @torch.inference_mode()
    def forward_propose(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        device = self._forward_device()
        if self.kv_cache is None:
            tokens = int(inputs.input_ids.numel())
            return self.dspark_empty_outputs(
                max(tokens // self._dspark_query_width, 1), device
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
                    (0, self.config.hidden_size), dtype=torch.bfloat16, device=device
                )
            )
        return self.run_commit_step(inputs, device)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        del inputs, fmha_impl
        raise RuntimeError(
            "Qwen3DFlashModel requires forward_propose or forward_commit"
        )


__all__ = ["Qwen3DFlashModel"]
