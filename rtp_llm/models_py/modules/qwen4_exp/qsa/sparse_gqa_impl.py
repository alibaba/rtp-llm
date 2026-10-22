"""Sparse GQA FMHA impl for qwen4's full-attention layers.

The indexer picks which compressed-block tokens each query may read; this impl
consumes that selection and runs :func:`sparse_prefill_attn
<rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_fmha.sparse_prefill_attn>` instead
of the dense attention.

Routing uses checkpoint-derived ``attn_configs.is_sparse`` and the ordinary
attention metadata. Layout and publication preflights run before cache writes.

Scope of this restricted production bridge:

* the model-side indexer hands over packed ``[tokens, selected]`` indices;
* main Q/K RoPE and K/V cache writes use the same fused production op as TRT
  prefill, including the common cache-store hook;
* ordinary prefill sparse attention reads this invocation's rotated K/V;
* MTP draft incremental prefill reads the full visible
  prefix and candidate window from the main paged K/V cache;
* ordinary decode can read selected tokens from the main paged K/V cache.

Target verification is bounded by one compression group. Exact-batch decode,
target verification and bounded draft prefill support CUDA Graph. Layout and
publication preflights complete before either side or main-cache writes.
"""

from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQKVOut,
)

from . import fmha_inputs, fmha_validation
from .fmha_plans import IncrementalPrefillInputs, PagedInputs


class SparseGqaFmhaImpl(FMHAImplBase):
    """Block-sparse GQA attention for qwen4 prefill and paged decode."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.parallelism_config = parallelism_config
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        if self.head_num % self.kv_head_num != 0:
            raise ValueError(
                f"head_num {self.head_num} not divisible by kv_head_num "
                f"{self.kv_head_num}"
            )
        self.q_width = self.head_num * self.head_dim
        self.kv_width = self.kv_head_num * self.head_dim
        self.is_prefill = bool(attn_inputs.is_prefill)
        self.is_target_verify = bool(attn_inputs.is_target_verify)
        self._is_mtp_draft = False
        self._selected_indices: Optional[torch.Tensor] = None
        self._trusted_selected_indices = False
        self._qsa_main_cache_mutation_started = False
        self._qsa_target_visible_lengths = None
        self._validate_runtime_mode()
        rope_writer_cls = (
            FusedRopeKVCachePrefillOpQKVOut
            if self.is_prefill
            else FusedRopeKVCacheDecodeOp
        )
        self.rope_kvcache_impl = rope_writer_cls(attn_configs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def _paged_kv_scale_args(self, cache):
        from .sparse_paged_fmha import main_kv_scale_view

        if cache is None:
            return {}
        scale = getattr(cache, "kv_scale_base", None)
        view = main_kv_scale_view(
            cache.kv_cache_base,
            scale,
            kv_heads=self.kv_head_num,
            page_size=int(self.attn_configs.kernel_tokens_per_block),
        )
        return {} if view is None else {"kv_scale": view}

    def _validate_runtime_mode(self) -> None:
        """Keep the correctness bridge inside its deliberately narrow scope."""
        inputs = self.attn_inputs
        if inputs.is_target_verify and not self.is_prefill:
            raise RuntimeError(
                "qwen4_exp sparse GQA target verification must use context-style prefill"
            )
        graph_paged = bool(inputs.is_cuda_graph)
        if graph_paged and not bool(
            getattr(inputs, "is_exact_cuda_graph_batch", False)
        ):
            raise RuntimeError(
                "qwen4_exp sparse GQA Graph decode requires an exact batch graph"
            )
        if inputs.is_s_padded and not graph_paged:
            raise RuntimeError("qwen4_exp sparse GQA does not support padded execution")
        if inputs.context_parallel_info is not None:
            raise RuntimeError(
                "qwen4_exp sparse GQA does not support context parallelism"
            )
        if self.is_prefill and inputs.cache_store_inputs is not None:
            from rtp_llm.models_py.model_desc.qwen4_exp_runtime.pd_publication import (
                validate_pd_cache_inputs,
            )

            validate_pd_cache_inputs((inputs,))

    def set_mtp_draft_mode(self, enabled: bool) -> None:
        """Explicitly select the MTP draft incremental-prefill bridge.

        The flag lives on this model-local implementation rather than on the
        pybind ``PyAttentionInputs`` object.  It must be set by the Qwen4 MTP
        model before side-cache validation/writes; a non-zero prefix alone is
        deliberately not treated as sufficient evidence that this mode is
        safe. Prefix-free draft prefill deliberately keeps the ordinary local
        sparse-prefill path; the paged bridge is selected only once a prefix
        exists.
        """
        if not isinstance(enabled, bool):
            raise TypeError("qwen4_exp MTP draft mode must be a bool")
        if enabled and (not self.is_prefill or self.is_target_verify):
            raise RuntimeError(
                "qwen4_exp MTP draft mode only supports ordinary prefill"
            )
        if self._selected_indices is not None:
            raise RuntimeError(
                "qwen4_exp MTP draft mode cannot change with an unconsumed selection"
            )
        self._is_mtp_draft = enabled

    def begin_qsa_cache_transaction(self) -> None:
        """Reset the main-cache phase before the paired side-cache write."""
        self._qsa_main_cache_mutation_started = False
        self._qsa_target_visible_lengths = None

    def qsa_main_cache_mutation_started(self) -> bool:
        """Whether the fused main KV writer may have mutated its cache."""
        return self._qsa_main_cache_mutation_started

    def _is_mtp_incremental_prefill(self) -> bool:
        return (
            self.is_prefill
            and not self.is_target_verify
            and self._is_mtp_draft
            and self._has_nonzero_prefill_prefix()
        )

    def _is_prefix_reuse_prefill(self) -> bool:
        """Ragged prefill batch containing at least one reused prefix.

        Covers both the MTP draft's post-rejection incremental commit and the
        target's page-aligned prefix-reuse prefill: both carry an explicit
        prefix and consume the ragged paged bridge. Cold rows with prefix zero
        may share the same target batch and are handled by that bridge as well.
        """
        return (
            self.is_prefill
            and not self.is_target_verify
            and self._has_nonzero_prefill_prefix()
        )

    def _has_nonzero_prefill_prefix(self) -> bool:
        prefixes = self.attn_inputs.prefix_lengths
        return bool(prefixes.numel()) and bool(torch.any(prefixes != 0).item())

    def _validate_main_paged_geometry(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        sequence_bases: torch.Tensor,
        query_lengths: torch.Tensor,
        device: torch.device,
        graph_capture: bool = False,
    ) -> int:
        return fmha_validation._validate_main_paged_geometry(
            self,
            cache,
            block_table,
            sequence_bases,
            query_lengths,
            device,
            graph_capture,
        )

    def validate_qsa_before_side_write(
        self,
        qsa_runtime,
        indexer,
        hidden_states: torch.Tensor,
    ) -> None:
        return fmha_validation.validate_qsa_before_side_write(
            self, qsa_runtime, indexer, hidden_states
        )

    def set_selected_indices(self, selected_indices: torch.Tensor) -> None:
        """Set one layer's packed selection without widening the global FMHA ABI."""
        if self._selected_indices is not None:
            raise RuntimeError("qwen4_exp sparse GQA has an unconsumed selection")
        self._selected_indices = selected_indices
        self._trusted_selected_indices = False

    def set_trusted_selected_indices(self, selected_indices: torch.Tensor) -> None:
        """Accept a selection produced by this model's causal QSA indexer."""
        self.set_selected_indices(selected_indices)
        self._trusted_selected_indices = True

    def _prepare_paged_inputs(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        selected_indices: torch.Tensor,
    ) -> dict:
        return fmha_inputs._prepare_paged_inputs(self, qkv, kv_cache, selected_indices)

    def _prepare_incremental_prefill_inputs(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        selected_indices: torch.Tensor,
    ) -> dict:
        return fmha_inputs._prepare_incremental_prefill_inputs(
            self, qkv, kv_cache, selected_indices
        )

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Sparse GQA owns explicitly enabled sparse, non-MLA attention."""
        return bool(
            attn_configs.is_sparse
            and not attn_configs.use_mla
            and attn_configs.use_sparse_gqa_fmha
        )

    def _is_draft_graph(self) -> bool:
        return bool(
            self.attn_inputs.is_cuda_graph
            and self.is_prefill
            and not self.is_target_verify
            and self._is_mtp_draft
        )

    def _forward_draft_graph(self, qkv, kv_cache, selected_indices, output_gate=None):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
            draft_graph_lanes,
            draft_graph_rows,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
            sparse_paged_gqa_attn,
        )

        from .draft_layout_triton import try_pack_inputs, try_unpack_output
        from .draft_paged_fmha import try_draft_paged_attention

        if kv_cache is None or kv_cache.kv_cache_base.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise RuntimeError(
                "qwen4_exp draft Graph requires BF16 or FP8 E4M3 main KV cache"
            )
        tokens = int(qkv.shape[0])
        batch = int(self.attn_inputs.input_lengths_device.numel())
        lanes = draft_graph_lanes(self.attn_inputs, tokens)
        if (
            selected_indices.ndim != 2
            or selected_indices.shape[0] != tokens
            or selected_indices.dtype != torch.int32
        ):
            raise RuntimeError("qwen4_exp draft Graph query/selection capacity changed")
        self._qsa_main_cache_mutation_started = True
        qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        direct = try_draft_paged_attention(
            qkv,
            selected_indices,
            self.attn_inputs.input_lengths_device,
            self.attn_inputs.prefix_lengths_device,
            self.attn_inputs.cu_seqlens_device,
            kv_cache.kv_cache_base,
            self.attn_inputs.kv_cache_kernel_block_id_device,
            heads=self.head_num,
            kv_heads=self.kv_head_num,
            head_dim=self.head_dim,
            page_size=int(self.attn_configs.kernel_tokens_per_block),
            gate=output_gate,
            **self._paged_kv_scale_args(kv_cache),
        )
        if direct is not None:
            return direct
        sources, inverse, valid, _, visible = draft_graph_rows(self.attn_inputs, tokens)
        packed = try_pack_inputs(
            qkv,
            selected_indices,
            sources,
            visible,
            heads=self.head_num,
            head_dim=self.head_dim,
        )
        if packed is None:
            query = (
                qkv[:, : self.q_width]
                .index_select(0, sources)
                .view(batch, lanes, self.head_num, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            selected = selected_indices.index_select(0, sources).view(batch, lanes, -1)
            selected = torch.where(visible.unsqueeze(-1) > 0, selected, -1)
        else:
            query, selected = packed
        output = sparse_paged_gqa_attn(
            query,
            kv_cache.kv_cache_base,
            self.attn_inputs.kv_cache_kernel_block_id_device,
            visible.contiguous(),
            selected.contiguous(),
            page_size=int(self.attn_configs.kernel_tokens_per_block),
            kv_head_num=self.kv_head_num,
            **self._paged_kv_scale_args(kv_cache),
            graph_capture=True,
        )
        unpacked = try_unpack_output(output, inverse, valid, gate=output_gate)
        if unpacked is not None:
            return unpacked
        output = (
            output.transpose(1, 2)
            .reshape(tokens, self.q_width)
            .index_select(0, inverse)
        )
        output = torch.where(valid.unsqueeze(1), output, 0)
        return output if output_gate is None else output * torch.sigmoid(output_gate)

    def support_cuda_graph(self) -> bool:
        return bool(
            self.attn_inputs.is_cuda_graph
            and (not self.is_prefill or self.is_target_verify or self._is_draft_graph())
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        """Refresh the captured RoPE page offsets for this replay's block table."""
        if not self.support_cuda_graph():
            raise RuntimeError(
                "qwen4_exp sparse GQA graph prepare requires a graph role"
            )
        if self.is_target_verify or self._is_draft_graph():
            offset = self.rope_kvcache_impl.prepare_kv_cache_offset(attn_inputs)
        else:
            # Decode also refreshes its stable sequence-length buffer.
            offset = self.rope_kvcache_impl.prepare(
                attn_inputs, forbid_reallocation=True
            ).kv_cache_offset
        captured_offset = self.rope_params.kv_cache_offset
        if (
            offset is None
            or captured_offset is None
            or offset.shape != captured_offset.shape
            or offset.dtype != captured_offset.dtype
        ):
            raise RuntimeError("qwen4_exp sparse GQA graph page geometry changed")
        captured_offset.copy_(offset, non_blocking=True)

    def forward_with_qk_norm_and_output_gate(
        self, qkv, kv_cache, layer_idx, gate, norm
    ):
        if (
            norm is not None
            and self.is_target_verify
            and not self._is_mtp_draft
            and not self.attn_configs.use_logn_attn
        ):
            return self.forward(
                qkv, kv_cache, layer_idx, qk_norm=norm, output_gate=gate
            )
        if not (
            norm is not None
            and self.is_prefill
            and not self.is_target_verify
            and not self._is_mtp_draft
            and not self.attn_inputs.is_cuda_graph
            and 4093 <= qkv.shape[0] <= 4096
            and self.attn_inputs.input_lengths.numel() == 1
            and self.rope_params.position_ids is not None
            and self.rope_params.max_prefix_length == 0
            and not self.attn_configs.use_logn_attn
        ):
            return self.forward_with_output_gate(
                norm(qkv) if norm is not None else qkv, kv_cache, layer_idx, gate
            )
        return self.forward(qkv, kv_cache, layer_idx, qk_norm=norm) * torch.sigmoid(
            gate
        )

    def forward_with_output_gate(self, qkv, kv_cache, layer_idx, gate):
        if (
            (
                self.is_target_verify
                or not self.is_prefill
                or (
                    self._is_draft_graph()
                    and gate.shape == (qkv.shape[0], self.q_width)
                    and gate.device == qkv.device
                    and not gate.requires_grad
                )
            )
            and gate.dtype == torch.bfloat16
            and gate.is_contiguous()
        ):
            return self.forward(qkv, kv_cache, layer_idx, output_gate=gate)
        return self.forward(qkv, kv_cache, layer_idx) * torch.sigmoid(gate)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
        layer_idx: int = 0,
        selected_indices: Optional[torch.Tensor] = None,
        output_gate: Optional[torch.Tensor] = None,
        qk_norm=None,
    ) -> torch.Tensor:
        """Rotate/write packed QKV, then run per-request sparse prefill attention.

        Production calls :meth:`set_selected_indices` immediately before the
        standard three-argument FMHA call. Focused component tests may pass the
        tensor directly. Both forms use packed ``[total_tokens, K]`` indices;
        the legacy uniform ``[B, S, K]`` form is accepted after validation.
        """
        if output_gate is not None and not (
            output_gate.shape == (qkv.shape[0], self.q_width)
            and output_gate.dtype == torch.bfloat16
            and output_gate.device == qkv.device
            and output_gate.is_contiguous()
            and (self.is_target_verify or not self.is_prefill or self._is_draft_graph())
        ):
            raise ValueError(
                "fused QSA output gate requires a packed paged decode plan"
            )
        if selected_indices is not None and self._selected_indices is not None:
            raise RuntimeError("qwen4_exp sparse GQA received two selections")
        uses_stored_selection = selected_indices is None
        trusted_selected = uses_stored_selection and self._trusted_selected_indices
        self._trusted_selected_indices = False
        if selected_indices is None:
            selected_indices = self._selected_indices
        if selected_indices is None:
            raise ValueError(
                "SparseGqaFmhaImpl requires the indexer's selected_indices; "
                "the attention module must run the indexer and pass it in"
            )

        self._paged_kv_scale_args(kv_cache)
        if self._is_draft_graph():
            output = self._forward_draft_graph(
                qkv, kv_cache, selected_indices, output_gate
            )
            if uses_stored_selection:
                self._selected_indices = None
            return output
        is_incremental_prefill = self._is_prefix_reuse_prefill()

        paged_plan = None
        if is_incremental_prefill:
            paged_plan = self._prepare_incremental_prefill_inputs(
                qkv, kv_cache, selected_indices
            )
        elif self.is_target_verify or not self.is_prefill:
            paged_plan = self._prepare_paged_inputs(qkv, kv_cache, selected_indices)

        writer = self.rope_kvcache_impl
        fused_writer = None
        if qk_norm is not None and self.is_target_verify:
            from .main_qk_cache_triton import prepare_main_qk_cache

            fused_writer = prepare_main_qk_cache(
                qkv,
                self.rope_params.position_ids,
                qk_norm,
                self.attn_configs.rope_config,
                kv_cache.kv_cache_base,
                kv_cache.kv_scale_base,
                paged_plan,
                self.rope_params,
            )
            if fused_writer is None:
                qkv = qk_norm(qkv)
        elif qk_norm is not None:
            from .main_qk_norm_rope import try_main_qk_norm_rope

            position_ids = self.rope_params.position_ids
            # The MRoPE fast path needs one three-axis id per token; anything
            # else must fall back to the plain norm instead of raising here.
            if int(position_ids.numel()) % 3 == 0:
                prepared = try_main_qk_norm_rope(
                    qkv,
                    position_ids.reshape(-1, 3),
                    qk_norm,
                    self.attn_configs.rope_config,
                )
            else:
                prepared = None
            if prepared is None:
                qkv = qk_norm(qkv)
            else:
                qkv = prepared
                writer = getattr(self, "_norm_rope_plain_writer", None)
                if writer is None:
                    from rtp_llm.ops import RopeStyle

                    config = AttentionConfigs()
                    for name in (
                        "head_num",
                        "kv_head_num",
                        "size_per_head",
                        "kernel_tokens_per_block",
                        "max_seq_len",
                        "use_logn_attn",
                    ):
                        setattr(config, name, getattr(self.attn_configs, name))
                    config.rope_config.style = RopeStyle.No
                    writer = FusedRopeKVCachePrefillOpQKVOut(config)
                    self._norm_rope_plain_writer = writer
        # Preparation above is read-only. Mark mutation immediately before the
        # native or fused KV/FP32-scale writer; partial launch failures retain
        # the existing fail-stop transaction contract.
        self._qsa_main_cache_mutation_started = True
        pre_writer_shape = tuple(qkv.shape)
        if fused_writer is not None:
            from libth_transformer_config import get_rope_cache_once

            # The native writer initializes a process-wide RoPE cache policy
            # even for MRoPE, which cannot use that cache. Preserve its first
            # call here: otherwise a later Base-style MTP layer can initialize
            # the cache instead and silently switch its rotary math path.
            get_rope_cache_once(
                self.attn_configs.rope_config, self.attn_configs.max_seq_len
            )
        qkv = (
            fused_writer.run()
            if fused_writer is not None
            else writer.forward(qkv, kv_cache, self.rope_params)
        )
        # The production decode writer returns [tokens, q_heads, head_dim] while
        # the paged kernels consume the flat [tokens, width] view.
        if qkv.dim() == 3 and int(qkv.numel()) == int(qkv.shape[0]) * self.q_width:
            qkv = qkv.reshape(int(qkv.shape[0]), self.q_width)
        if self.is_target_verify:
            if int(qkv.shape[1]) < self.q_width:
                raise RuntimeError(
                    "qwen4_exp target-verify RoPE writer returned an undersized tensor"
                )
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )
            output = self._forward_paged(
                qkv[:, : self.q_width], kv_cache, paged_plan, output_gate
            )
            if uses_stored_selection:
                self._selected_indices = None
            return output
        if is_incremental_prefill:
            if int(qkv.shape[1]) < self.q_width:
                raise RuntimeError(
                    "qwen4_exp incremental-prefill RoPE writer returned an "
                    "undersized tensor"
                )
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )
            output = self._forward_ragged_paged(
                qkv[:, : self.q_width], kv_cache, paged_plan
            )
            if uses_stored_selection:
                self._selected_indices = None
            return output
        if not self.is_prefill:
            if qkv.dim() != 2 or int(qkv.shape[1]) != self.q_width:
                prefixes = getattr(self.attn_inputs, "prefix_lengths", None)
                prefix_head = prefixes[:4].tolist() if prefixes is not None else None
                raise RuntimeError(
                    "qwen4_exp decode RoPE writer returned invalid query geometry"
                    f" [qkv={tuple(qkv.shape)} in={pre_writer_shape}"
                    f" q_width={self.q_width} kv_width={self.kv_width}"
                    f" dim={qkv.dim()} writer={type(self.rope_kvcache_impl).__name__}"
                    f" verify={self.is_target_verify} draft={self._is_mtp_draft}"
                    f" prefix_head={prefix_head}]"
                )
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )
            output = self._forward_paged(qkv, kv_cache, paged_plan, output_gate)
            if uses_stored_selection:
                self._selected_indices = None
            return output

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_fmha import (
            sparse_prefill_attn,
        )

        tokens = int(qkv.shape[0])
        q, k, v = torch.split(qkv, [self.q_width, self.kv_width, self.kv_width], dim=-1)

        lengths = [int(length) for length in self.attn_inputs.input_lengths.tolist()]
        if (
            not lengths
            or any(length <= 0 for length in lengths)
            or sum(lengths) != tokens
        ):
            raise ValueError(
                f"qwen4_exp sparse GQA input_lengths={lengths} do not partition "
                f"the {tokens} packed tokens"
            )
        if selected_indices.dim() == 3:
            batch, seq_len, width = selected_indices.shape
            if batch != len(lengths) or any(length != seq_len for length in lengths):
                raise ValueError(
                    "rank-3 selected_indices require uniform input lengths matching "
                    f"[{batch}, {seq_len}], got {lengths}"
                )
            selected_indices = selected_indices.reshape(tokens, width)
        if selected_indices.dim() != 2 or int(selected_indices.shape[0]) != tokens:
            raise ValueError(
                "selected_indices must be packed [total_tokens, K], got "
                f"{tuple(selected_indices.shape)} for total_tokens={tokens}"
            )

        outputs = []
        offset = 0
        validation_kwargs = {"validate_selected": False} if trusted_selected else {}
        for seq_len in lengths:
            end = offset + seq_len
            seq_q = q[offset:end].reshape(1, seq_len, self.head_num, self.head_dim)
            seq_k = k[offset:end].reshape(1, seq_len, self.kv_head_num, self.head_dim)
            seq_v = v[offset:end].reshape(1, seq_len, self.kv_head_num, self.head_dim)
            seq_out = sparse_prefill_attn(
                seq_q.transpose(1, 2).contiguous(),
                seq_k.transpose(1, 2).contiguous(),
                seq_v.transpose(1, 2).contiguous(),
                selected_indices[offset:end].unsqueeze(0).contiguous(),
                **validation_kwargs,
            )
            outputs.append(seq_out.transpose(1, 2).reshape(seq_len, self.q_width))
            offset = end
        output = torch.cat(outputs, dim=0)
        if uses_stored_selection:
            self._selected_indices = None
        return output

    def _forward_paged(
        self,
        q: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        plan: PagedInputs | IncrementalPrefillInputs,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run paged sparse GQA after the production decode RoPE/KV writer."""
        assert kv_cache is not None
        batch = int(plan.batch)
        query_len = int(plan.query_len)
        tokens = int(q.shape[0])
        if tokens != int(plan.tokens) or int(q.shape[1]) != self.q_width:
            raise ValueError(
                "decode RoPE writer returned query geometry inconsistent with preflight"
            )
        selected_indices = plan.selected_indices
        kv_lens = plan.kv_lens
        block_table = plan.block_table
        page_size = int(plan.page_size)

        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
            sparse_paged_gqa_attn,
        )

        q = q.view(batch, query_len, self.head_num, self.head_dim).transpose(1, 2)
        output = sparse_paged_gqa_attn(
            q.contiguous(),
            kv_cache.kv_cache_base,
            block_table,
            kv_lens,
            selected_indices.contiguous(),
            page_size=page_size,
            kv_head_num=self.kv_head_num,
            **(
                {
                    "output_gate": output_gate.view(
                        batch, query_len, self.head_num, self.head_dim
                    )
                }
                if output_gate is not None
                else {}
            ),
            **self._paged_kv_scale_args(kv_cache),
            **({"graph_capture": True} if self.attn_inputs.is_cuda_graph else {}),
        )
        return output.transpose(1, 2).reshape(tokens, self.q_width)

    def _forward_ragged_paged(
        self,
        q: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        plan: PagedInputs | IncrementalPrefillInputs,
    ) -> torch.Tensor:
        """Run paged attention over packed incremental-prefill requests."""
        assert kv_cache is not None
        tokens = int(plan.tokens)
        if q.dim() != 2 or tuple(q.shape) != (tokens, self.q_width):
            raise ValueError(
                "incremental-prefill RoPE writer returned query geometry "
                "inconsistent with preflight"
            )

        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
            sparse_paged_gqa_attn,
        )

        block_table = plan.block_table
        page_size = int(plan.page_size)
        requests = plan.requests
        max_rows = max(int(req.end) - int(req.start) for req in requests)
        batch_rows = self._is_mtp_draft and len(requests) > 1 and max_rows <= 4
        if batch_rows:
            # The draft's short, ragged windows share a single paged launch.
            # Invalid padded rows cannot read KV: their visible length is zero
            # and every selected index is -1. Restore packed order afterwards.
            row_map = []
            valid = []
            packed_positions = []
            for req_idx, request in enumerate(requests):
                start, end = int(request.start), int(request.end)
                length = end - start
                for row in range(max_rows):
                    row_map.append(start + min(row, length - 1))
                    valid.append(row < length)
                    if row < length:
                        packed_positions.append(req_idx * max_rows + row)
            row_map = torch.tensor(row_map, dtype=torch.long, device=q.device)
            valid = torch.tensor(valid, dtype=torch.bool, device=q.device).view(
                len(requests), max_rows
            )
            packed_positions = torch.tensor(
                packed_positions, dtype=torch.long, device=q.device
            )
            query = q.index_select(0, row_map).view(
                len(requests), max_rows, self.head_num, self.head_dim
            )
            query = query.transpose(1, 2).contiguous()
            selected = plan.selected_indices.index_select(0, row_map)
            selected = selected.view(len(requests), max_rows, -1)
            selected = torch.where(valid.unsqueeze(-1), selected, -1)
            row_offsets = torch.arange(
                1, max_rows + 1, dtype=torch.int32, device=q.device
            )
            kv_lens = plan.prefixes_device.unsqueeze(1) + row_offsets
            kv_lens = torch.where(valid, kv_lens, 0)
            output = sparse_paged_gqa_attn(
                query,
                kv_cache.kv_cache_base,
                block_table,
                kv_lens.contiguous(),
                selected.contiguous(),
                page_size=page_size,
                kv_head_num=self.kv_head_num,
                **self._paged_kv_scale_args(kv_cache),
            )
            return (
                output.transpose(1, 2)
                .reshape(len(requests) * max_rows, self.q_width)
                .index_select(0, packed_positions)
            )

        outputs = []
        for request_idx, request in enumerate(requests):
            start, end = int(request.start), int(request.end)
            query_len = end - start
            request_q = q[start:end].view(1, query_len, self.head_num, self.head_dim)
            request_q = request_q.transpose(1, 2).contiguous()
            request_output = sparse_paged_gqa_attn(
                request_q,
                kv_cache.kv_cache_base,
                block_table[request_idx : request_idx + 1].contiguous(),
                request.kv_lens,
                request.selected_indices,
                page_size=page_size,
                kv_head_num=self.kv_head_num,
                **self._paged_kv_scale_args(kv_cache),
            )
            outputs.append(
                request_output.transpose(1, 2).reshape(query_len, self.q_width)
            )
        return torch.cat(outputs, dim=0)
