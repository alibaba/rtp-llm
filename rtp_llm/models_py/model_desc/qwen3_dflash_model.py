"""Qwen3 DFlash V1 block drafter.

Draft residual and Q-projection widths can differ (5120 vs 4096 in the 27B
pairings). Attention geometry and the causal sliding-window / non-causal
full layer policy therefore come from each draft checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from rtp_llm.device.device_type import is_hip
from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_desc.block_map import select_attention_inputs_for_layer
from rtp_llm.models_py.model_desc.qwen3_dspark_runtime import (
    Qwen3DSparkRuntime,
    _RopePositions,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen3.language import (
    Qwen3Attention, Qwen3ForCausalLM, Qwen3MLP,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
    from rtp_llm.models_py.triton_kernels.common.dflash_attention import DFlashCacheLayout


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
    from rtp_llm.models_py.triton_kernels.common.dflash_attention import DFlashCacheLayout

    if not is_hip():
        return DFlashCacheLayout.CUDA
    is_fp8 = cache.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
    vectorized_value = (
        fmha_config is None or bool(getattr(fmha_config, "use_asm_pa", False)) or is_fp8
    )
    return (
        DFlashCacheLayout.AITER_VECTOR if vectorized_value else DFlashCacheLayout.AITER
    )


class _DFlashAttention(Qwen3Attention):
    def __init__(self, *, layer_type, window_size, **kwargs):
        super().__init__(**kwargs)
        self.layer_type = layer_type
        self.window_size = window_size
        self.q_heads = self.num_heads_per_partition
        self.kv_heads = self.num_kv_heads_per_partition
        if layer_type == "sliding_attention" and window_size <= 0:
            raise ValueError("DFlash sliding_attention needs a positive window")
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError(f"unsupported DFlash layer type {layer_type!r}")
        # _out is the CUDA-graph-stable attention output buffer: captured
        # graphs hold raw pointers into it, so it must never be reallocated
        # after the first capture. _eager_out backs eager steps whose rows
        # exceed the graph buffer (decode batches above the largest captured
        # bucket under --max_generate_batch_size, or warmup before capture);
        # no graph captures it, so it may be replaced freely. _active_out is
        # the buffer prepare_fmha_impl selected for the upcoming forward.
        self._out: torch.Tensor | None = None
        self._eager_out: torch.Tensor | None = None
        self._active_out: torch.Tensor | None = None

    def _output_buffer(self, query: torch.Tensor) -> torch.Tensor:
        rows, heads, dim = query.shape
        out = self._active_out
        if (
            out is None
            or out.device != query.device
            or out.dtype != query.dtype
            or out.shape[0] < rows
            or out.shape[1:] != (heads, dim)
        ):
            raise RuntimeError(
                "DFlash attention output was not prepared for this input shape; "
                "prepare_fmha_impl must run before forward"
            )
        return out[:rows]

    def reserve_output(
        self,
        rows: int,
        device: torch.device,
        dtype: torch.dtype,
        for_graph: bool = False,
    ) -> None:
        """Prepare the graph-stable or eager attention output buffer.

        for_graph=True (CUDA graph capture): _out is allocated once at the
        first capture bucket — capture runs largest bucket first, so that
        reservation is the process maximum — and never replaced afterwards:
        captured graphs hold raw pointers into it, and a reallocation would
        free memory they still write on every replay. for_graph=False (eager
        steps: decode batches above the largest captured bucket when
        --max_generate_batch_size exceeds the graph capacity, or warmup
        before capture): prefer _out when it already covers rows, otherwise
        grow the eager buffer, which no graph holds.
        """
        if rows < 0:
            raise ValueError(f"DFlash output rows must be non-negative, got {rows}")
        if for_graph:
            if self._out is None:
                self._out = torch.empty(
                    (rows, self.q_heads, self.head_dim), device=device, dtype=dtype
                )
            elif (
                self._out.device != device
                or self._out.dtype != dtype
                or self._out.shape[0] < rows
            ):
                raise RuntimeError(
                    "DFlash graph attention output capacity is fixed at the "
                    f"first capture reservation ({tuple(self._out.shape)}); "
                    f"refusing to reallocate for {rows} rows: captured CUDA "
                    "graphs hold raw pointers into this buffer"
                )
            self._active_out = self._out
            return
        if (
            self._out is not None
            and self._out.device == device
            and self._out.dtype == dtype
            and self._out.shape[0] >= rows
        ):
            self._active_out = self._out
            # The oversized eager fallback buffer is no longer needed while
            # the graph buffer covers eager batches; drop it instead of
            # keeping two live buffers. Freeing is stream-ordered safe: any
            # pending eager kernels writing it were enqueued on the same
            # stream as all later reuse.
            self._eager_out = None
            return
        if (
            self._eager_out is None
            or self._eager_out.device != device
            or self._eager_out.dtype != dtype
            or self._eager_out.shape[0] < rows
        ):
            self._eager_out = torch.empty(
                (rows, self.q_heads, self.head_dim), device=device, dtype=dtype
            )
        self._active_out = self._eager_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
        request_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        query_width: int,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        rope: Any,
        fmha_config: Any,
    ) -> torch.Tensor:
        from rtp_llm.models_py.triton_kernels.common.dflash_attention import (
            dflash_paged_attention, dflash_write_paged_kv,
        )

        qkv = self._apply_qk_norm(self.qkv_proj(hidden_states))
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
        # RowParallelLinear owns the attention TP reduction.
        return self.o_proj(attn.reshape(hidden_states.shape[0], -1).contiguous())


class _DFlashDecoderLayer(RtpModule):
    def __init__(
        self, hidden_size, num_heads, num_kv_heads, intermediate_size, head_dim,
        layer_idx, attn_tp_size, attn_tp_rank, ffn_tp_size, ffn_tp_rank,
        quant_config, params_dtype, rms_norm_eps, prefix, *, layer_types, window_size,
    ):
        super().__init__()
        self.self_attn = _DFlashAttention(
            hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, layer_idx=layer_idx, tp_size=attn_tp_size, tp_rank=attn_tp_rank,
            params_dtype=params_dtype, rms_norm_eps=rms_norm_eps, prefix=f"{prefix}.self_attn",
            layer_type=layer_types[layer_idx], window_size=window_size,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, params_dtype=params_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, params_dtype=params_dtype)
        self.mlp = Qwen3MLP(
            hidden_size, intermediate_size, tp_size=ffn_tp_size, tp_rank=ffn_tp_rank,
            params_dtype=params_dtype, prefix=f"{prefix}.mlp",
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


class Qwen3DFlashModel(DSparkProposerMixin, Qwen3ForCausalLM):
    """DFlash V1: the runtime consumes the very parameters NewLoader fills."""

    _decoder_layer = _DFlashDecoderLayer
    runtime_parameter_paths = {"embedding": "embed_tokens.weight", "lm_head": "lm_head.weight"}

    def _decoder_kwargs(self, parallelism_config):
        return dict(layer_types=self.config.dflash_layer_types,
                    window_size=int(self.config.dflash_sliding_window or 0))

    def __init__(self, model_config, load_config):
        quant = getattr(load_config, "quant_config", None)
        if getattr(quant, "quant_type", "none") != "none":
            raise ValueError("Qwen3 DFlash V1 requires unquantized weights")
        if len(model_config.dflash_layer_types or ()) != model_config.num_layers:
            raise ValueError("DFlash layer policy does not match draft layer count")
        if load_config.parallelism_config.pp_size > 1:
            raise ValueError("DFlash V1 requires pp_size=1")
        super().__init__(model_config, load_config)
        width = model_config.gen_num_per_cycle
        target_layers = model_config.dflash_target_layer_ids
        self.init_dspark_proposer(
            width=width, query_width=width + 1, noise_token_id=model_config.dflash_mask_token_id,
            aux_feature_dim=len(target_layers) * model_config.hidden_size,
            hidden_dim=model_config.hidden_size,
        )
        self.attn_configs = model_config.getAttentionConfigs(load_config.attn_tp_size)
        self.fc = ColumnParallelLinear(
            self._dspark_aux_feature_dim, model_config.hidden_size,
            params_dtype=load_config.compute_dtype, prefix="fc",
        )
        self.hidden_norm = RMSNorm(model_config.hidden_size, eps=model_config.layernorm_eps,
                                   params_dtype=load_config.compute_dtype)
        self.context_rope = None

    def prepare_borrowed_parameters(self, aliases):
        if set(aliases) != {"lm_head.weight", "embed_tokens.weight"}:
            raise ValueError("DFlash borrows both the target embedding and lm-head")
        for name, binding in aliases.items():
            leaf = self.get_submodule(name.rsplit(".", 1)[0])
            owner = binding.owner.get_submodule(binding.source_path.rsplit(".", 1)[0])
            if (leaf.tp_size, leaf.tp_rank) != (owner.tp_size, owner.tp_rank):
                raise ValueError("DFlash parameter loan requires the same local TP partition")
        # The owner already normalized/scaled the head. Never apply a second
        # postprocess or let dense Qwen's missing-head fallback copy into it.
        self._lm_head_postprocessed = True
        self.tie_word_embeddings = False

    def process_weights_after_loading(self):
        super().process_weights_after_loading()
        if self.embed_tokens.weight.is_cuda:
            Qwen3DSparkRuntime.initialize_context_rope(self)

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
            # for_graph routes capture-time reservations to the graph-stable
            # buffer and oversized eager batches to the growable buffer; see
            # _DFlashAttention.reserve_output.
            layer.self_attn.reserve_output(rows, device, dtype, is_cuda_graph)
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
        from rtp_llm.models_py.modules.factory.attention.common import create_write_cache_store_impl
        from rtp_llm.models_py.triton_kernels.common.dflash_attention import dflash_write_paged_kv

        for layer_idx, layer in enumerate(self.layers):
            # Slice the rank-local packed projection. Avoid a second set of KV
            # parameters that could become stale after a weight update.
            attn = layer.self_attn
            kv = torch.nn.functional.linear(hidden, attn.qkv_proj.weight[attn.q_size:])
            key, value = kv.reshape(-1, 2, kv_heads, head_dim).unbind(1)
            key = attn.k_norm(key.contiguous().reshape(-1, head_dim)).view(-1, kv_heads, head_dim)
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
        # The DSpark proposer mixin forwards alignment-only steps with an empty
        # batch through the draft model on every rank (see
        # dspark_proposer_mixin.run_propose_step, which keeps executing the
        # collective layers for empty DP ranks). DFlash layers only own
        # TP-group collectives (attention output all-reduce and the dense MLP),
        # and TP ranks share the same batch, so an all-empty alignment step can
        # skip the layers uniformly on every rank at once and return an empty
        # hidden block instead of requiring a non-empty KV block table; the
        # caller skips the non-collective head for empty batches regardless.
        if batch == 0:
            weight = self.embed_tokens.weight
            return torch.empty(
                (0, self.config.hidden_size),
                dtype=weight.dtype,
                device=weight.device,
            )
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
            from rtp_llm.ops.compute_ops import PyModelOutputs

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
