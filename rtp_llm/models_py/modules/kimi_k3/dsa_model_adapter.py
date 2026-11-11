"""Optional DSA metadata and MTP handoff hooks for the shared K3 model."""

import copy
from collections.abc import Mapping
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value, get_primary_attention_inputs,
)
from rtp_llm.ops import HybridAttentionType


class KimiK31ModelAdapter:
    """No decoder loop: K3 keeps its shared layers, MoE and output handling."""

    def _setup_dsa(self):
        self._query_layout = None
        self._query_layout_buffers = {}
        self._outer_layout = None
        self._outer_layout_buffers = {}
        self._row_exchange = None
        self._routing_plan = None
        self._natural_layout_cache = {}
        self._natural_validation_cache = {}
        self._outer_natural_enabled = self.prefill_query_cp and not self.config.is_mtp
        self.dsa_workspace = None
        self.register_buffer("_mtp_prefill_hidden_buffer", None, persistent=False)
        self._mtp_prefill_hidden_valid_rows = 0
        if getattr(self.config.k3_runtime_config, "dsa", None) is None:
            return
        from rtp_llm.models_py.utils.flash_mla_policy import configure_flash_mla_l2_persist

        configure_flash_mla_l2_persist()
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import DsaWorkspaceOwner

        self.dsa_workspace = DsaWorkspaceOwner.from_model(self.config, self.parallelism_config)
        self.dsa_workspace.graph_max_requests = self._max_generate_batch_size
        self.dsa_workspace.graph_max_queries = max(1, int(self.config.gen_num_per_cycle) + 1)
        self.dsa_workspace.bind_value_weights(
            (i, self.weight.weights[i]) for i, layer in enumerate(self.layers)
            if getattr(layer.attention, "indexer", None) is not None
        )

    def _initialize_natural_outer(self, init_resource):
        if not self._outer_natural_enabled or self._row_exchange is not None:
            return
        from rtp_llm.models_py.distributed.collective_torch import Group, get_process_group
        from rtp_llm.models_py.distributed.zigzag_row_exchange import DirectedRowExchange
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import CudaFp8GEMMLinear
        from .natural_dsa import route_allocation_budget

        fp8_output_width = 0
        for layer in self.layers:
            if layer.layer_type == HybridAttentionType.LINEAR:
                attention = layer.attention
                if any(
                    not ((isinstance(p, CudaF16Linear) and p.weight.dtype == torch.bfloat16)
                         or (isinstance(p, CudaFp8GEMMLinear) and p.weight.dtype == torch.float8_e4m3fn)
                         or (getattr(p, "dynamic_input_scale", False) and p.weight.dtype == torch.uint8))
                    or p.bias is not None for p in (attention.input, attention.f_b, attention.output)
                ):
                    raise ValueError("natural outer requires supported CUDA KDA projections")
                if attention._fp8_output_norm and not (
                    isinstance(attention.output, CudaFp8GEMMLinear) and attention.output.scale_ue8m0
                ):
                    raise ValueError("natural outer FP8 KDA output requires UE8M0 scales")
            elif getattr(layer.attention, "indexer", None) is not None:
                attention = layer.attention
                if not hasattr(attention, "natural_q_input"):
                    raise ValueError("natural outer requires a supported SFA split projection backend")
                if self.tp_size == 8 and attention._fp8_output_gate:
                    width = attention.heads * attention.v_dim
                    if fp8_output_width not in (0, width):
                        raise ValueError("natural outer requires one shared FP8 DSA output width")
                    fp8_output_width = width
        if self.norm.weight.dtype != torch.bfloat16:
            raise ValueError("natural outer requires BF16 hidden states")
        # The model owner already includes chunking, per-request CP padding
        # and unchunked startup work in this global physical-row bound.
        capacity = self._max_moe_tokens // self.tp_size
        self._row_exchange = DirectedRowExchange(
            get_process_group(Group.TP), self.norm.weight.device,
            capacity, self.config.hidden_size, chunk_capacity=8192,
            fp8_output_width=fp8_output_width,
        )
        owner = self.dsa_workspace
        owner.row_exchange = self._row_exchange
        owner.route_capacity = capacity
        owner.route_chunk_rows = self._row_exchange.chunk_capacity
        owner.route_bytes, owner.route_allocation_granularity = route_allocation_budget(self._row_exchange)

    def _prepare_dsa(self, inputs, is_cuda_graph, *, prepare_host_metadata=None):
        """Prepare a batch; pass local-only host preparation inside this boundary.

        The model caller must not run that callback before entering this method:
        otherwise a malformed tag can make one rank exit before the agreement.
        QueryCP and Graph paths retain their existing preparation order.
        """
        # Select the boundary from model/call-site state, never from a local
        # tag or cache hit: even an empty/malformed mapping must enter the vote.
        if not self.prefill_query_cp and not is_cuda_graph:
            return self._prepare_dsa_tp_batch(inputs, prepare_host_metadata)
        if prepare_host_metadata is not None:
            prepare_host_metadata(inputs)
        from rtp_llm.models_py.modules.hybrid.query_layout import QueryLayout
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import SparseMlaPageRRImpl

        primary = get_primary_attention_inputs(inputs, self.kv_cache)
        self._query_layout = None
        self._outer_layout = None
        self._routing_plan = None
        if self.prefill_query_cp and primary.is_prefill and not primary.is_target_verify:
            info = primary.context_parallel_info
            if info is None:
                raise ValueError("K3.1 requires the shared padded zigzag input plan")
            outer = self._outer_natural_enabled and not primary.is_mtp_draft_update
            if outer:
                lengths = info.prefill_actual_input_lengths_cpu.tolist()
                if not lengths or any(length <= 0 for length in lengths):
                    raise ValueError("natural outer Prefill requires nonempty real requests")
            lengths = tuple(info.prefill_actual_input_lengths_cpu.tolist())
            prefixes = tuple(info.prefill_prefix_lengths_cpu.tolist())
            cacheable = (outer and primary.valid_token_mask is None
                         and sum(lengths) == info.prefill_qkv_restore_indice.numel())
            key = (lengths, prefixes, self.tp_size, self.tp_rank, str(inputs.input_ids.device))
            if not hasattr(self, "_natural_layout_cache"):
                self._natural_layout_cache, self._natural_validation_cache = {}, {}
            cached = self._natural_layout_cache.pop(key, None) if cacheable else None
            if cached is not None:
                layout, self._outer_layout, self._routing_plan = cached
                layout._natural_attention_views.clear()
                self._outer_layout._natural_attention_views.clear()
            else:
                # A native producer may recycle its metadata in-place. Cached
                # layouts must own their snapshots rather than alias that pool.
                owned_info = (SimpleNamespace(
                    prefill_actual_input_lengths_cpu=info.prefill_actual_input_lengths_cpu.clone(),
                    prefill_prefix_lengths_cpu=info.prefill_prefix_lengths_cpu.clone(),
                    prefill_qkv_restore_indice=info.prefill_qkv_restore_indice.clone(),
                    prefill_qkv_padding_mask=info.prefill_qkv_padding_mask.clone(),
                ) if cacheable else info)
                layout = QueryLayout(
                    owned_info, self.tp_size, self.tp_rank, inputs.input_ids.device,
                    primary.valid_token_mask, self._query_layout_buffers,
                )
            if outer:
                if is_cuda_graph:
                    raise ValueError("natural outer Query-CP Prefill requires eager execution")
                if self._row_exchange is None:
                    raise ValueError("natural outer transport must be initialized collectively before preparation")
                from rtp_llm.models_py.modules.hybrid.natural_outer_layout import NaturalOuterLayout
                from rtp_llm.models_py.modules.hybrid.natural_zigzag_plan import NaturalZigzagPlan
                from .natural_dsa import validate_shared_restore

                plan = (self._routing_plan if cached is not None else
                        NaturalZigzagPlan(layout.lengths, self.tp_size, self.tp_rank))
                current = SimpleNamespace(
                    physical_rows=info.prefill_qkv_restore_indice.numel(), local_rows=layout.local_rows,
                    restore=info.prefill_qkv_restore_indice, padding_mask=info.prefill_qkv_padding_mask,
                )
                validate_shared_restore(current, plan, self._natural_validation_cache)
                if cached is None:
                    self._outer_layout = NaturalOuterLayout(layout, self._outer_layout_buffers)
                    self._routing_plan = plan
                if cacheable:
                    self._natural_layout_cache[key] = (layout, self._outer_layout, plan)
                    while len(self._natural_layout_cache) > 8:
                        self._natural_layout_cache.pop(next(iter(self._natural_layout_cache)))
                self._row_exchange.prepare_plan(plan, chunk_rows=self.dsa_workspace.route_chunk_rows)
            kda_layout = self._outer_layout if outer else layout
            for layer in self.layers:
                if layer.layer_type == HybridAttentionType.LINEAR:
                    a = layer.attention
                    kda_layout.prepare_kda_buffers(a.input.weight.shape[0], a.width,
                                               quantized=a._fp8_output_norm)
            self._query_layout = layout
        self._update_dsa_workspace_layout(inputs)
        owner = self.dsa_workspace

        def create(attention_inputs, phase=None):
            parallel = self.parallelism_config
            if self._query_layout is not None:
                parallel = copy.deepcopy(parallel)
                parallel.prefill_cp_config.preserve_tp_layout = False
            return SparseMlaPageRRImpl(
                self.config,
                parallel,
                self.weight,
                attention_inputs,
                self.fmha_config,
                is_cuda_graph,
                workspace_owner=owner,
                execution_phase=phase,
            )

        tagged = get_attention_inputs_value(inputs)
        if isinstance(tagged, Mapping):
            tags = self._get_fmha_group_tags()
            tags = tuple(tagged if tags is None else tags)
            owner.graph_cache_groups = len(tags)
            return {
                tag: create(tagged[tag], "indexer" if tag == "indexer_kv" else "sfa")
                for tag in tags
            }
        return create(tagged)


    def _update_dsa_workspace_layout(self, inputs):
        owner = self.dsa_workspace
        owner.query_layout = self._query_layout
        owner.producer_layout = self._outer_layout if self._outer_layout is not None else self._query_layout
        owner.routing_plan = self._routing_plan
        owner.row_exchange = self._row_exchange
        # Storage identity, rather than tensor identity, counts views only once.
        storages = {}
        tensors = [*self._query_layout_buffers.values(), *self._outer_layout_buffers.values()]
        for layout in (self._query_layout, self._outer_layout):
            if layout is not None:
                tensors.extend(x for x in layout.__dict__.values()
                               if isinstance(x, torch.Tensor) and x.device == layout.device)
        for layouts in getattr(self, "_natural_layout_cache", {}).values():
            for layout in layouts[:2]:
                tensors.extend(x for x in layout.__dict__.values() if isinstance(x, torch.Tensor))
        for entry in getattr(self, "_natural_validation_cache", {}).values():
            tensors.extend(entry)
        for tensor in tensors:
            if tensor.device != inputs.input_ids.device:
                continue
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        owner.layout_bytes = sum(storages.values())


    def _prepare_dsa_tp_batch(self, inputs, prepare_host_metadata=None):
        from rtp_llm.models_py.distributed.collective_torch import Group, get_process_group
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
            SparseMlaPageRRImpl, agree_tp_prefill_batch,
        )

        # This region must remain local-only. In particular, constructing an
        # attention implementation can initialize its communicator, so defer
        # constructors (and any layout scatter) until the one final agreement.
        records, contracts, error = [], [], None
        mapping = False
        try:
            # block_map uses RuntimeError for malformed mappings. Validate this
            # recoverable input explicitly, without classifying RuntimeErrors
            # from CUDA/native operations by their message text.
            if isinstance(inputs.attention_inputs, Mapping) and not inputs.attention_inputs:
                raise ValueError("DSA batch requires a nonempty tag mapping")
            tagged = get_attention_inputs_value(inputs)
            mapping = isinstance(tagged, Mapping)
            if mapping:
                tags = self._get_fmha_group_tags()
                tags = tuple(tagged if tags is None else tags)
                if not tags or any(not isinstance(tag, str) or not tag for tag in tags):
                    raise ValueError("DSA batch requires nonempty cache tags")
                if len(set(tags)) != len(tags):
                    raise ValueError("DSA batch contains duplicate cache tags")
                for tag in tags:
                    if tag not in tagged or tagged[tag] is None:
                        raise ValueError(f"DSA attention input tag {tag!r} is missing")
            else:
                tags = (None,)
            if prepare_host_metadata is not None:
                prepare_host_metadata(inputs)
            self._query_layout = None
            self._outer_layout = None
            self._routing_plan = None
            self._update_dsa_workspace_layout(inputs)
            mode, shared = None, None
            for tag in tags:
                attention_inputs = tagged[tag] if mapping else tagged
                current_mode = (
                    bool(attention_inputs.is_prefill),
                    bool(attention_inputs.is_target_verify),
                    bool(attention_inputs.is_mtp_draft_update),
                )
                if mode is not None and current_mode != mode:
                    raise ValueError("DSA cache tags disagree on batch execution mode")
                mode = current_mode
                phase = ("indexer" if tag == "indexer_kv" else "sfa") if mapping else None
                prepared = None
                if mode == (True, False, False):
                    prepared = SparseMlaPageRRImpl.prepare_tp_prefill(
                        self.config, self.parallelism_config, self.dsa_workspace,
                        attention_inputs, phase,
                    )
                    # Request geometry must agree across tags too. Per-phase
                    # KV representation/capacity is compared across ranks below.
                    current = prepared.contract[:3]
                    if shared is not None and current != shared:
                        raise ValueError("DSA cache tags disagree on Prefill query geometry")
                    shared = current
                    contract = prepared.contract
                else:
                    # Eager decode shares this fixed boundary; its existing
                    # scheduler/metadata preparation still runs after agreement.
                    contract = (
                        int(attention_inputs.physical_token_count),
                        int(attention_inputs.logical_token_count),
                    )
                records.append((tag, attention_inputs, phase, prepared))
                contracts.append((tag, mode, contract))
        except ValueError as exc:
            # Only metadata/geometry/capacity validation rejections participate
            # in agreement. MemoryError, CUDA OOM/fatal RuntimeError and unknown
            # exceptions propagate unchanged to the existing fatal boundary.
            # A poisoned CUDA context is not made recoverable by an NCCL vote.
            error = f"{type(exc).__name__}: {exc}"
        agree_tp_prefill_batch(
            (mapping, tuple(contracts)), error,
            get_process_group(Group.TP), self.parallelism_config.tp_size,
        )
        self.dsa_workspace.graph_cache_groups = len(records)
        result = {}
        for tag, attention_inputs, phase, prepared in records:
            result[tag] = SparseMlaPageRRImpl(
                self.config, self.parallelism_config, self.weight,
                attention_inputs, self.fmha_config, False,
                workspace_owner=self.dsa_workspace, execution_phase=phase,
                prepared_tp_prefill=prepared,
            )
        return result if mapping else result[None]


    def _initialize_mtp_hidden_handoff(self, init_resource):
        # CP's public output keeps only LM-head rows. Recurrent MTP must retain
        # every rank-local pre-norm row, including the per-request padding.
        if not (
            getattr(self, "prefill_query_cp", False)
            and getattr(init_resource, "is_speculative", False)
        ):
            return
        capacity = self._max_moe_tokens // self.tp_size
        if capacity <= 0:
            raise ValueError("MTP Prefill hidden capacity must be positive")
        buffer = self._mtp_prefill_hidden_buffer
        shape = (capacity, self.config.hidden_size)
        if buffer is None:
            self._mtp_prefill_hidden_buffer = torch.empty(
                shape, dtype=self.norm.weight.dtype, device=self.norm.weight.device
            )
        elif (
            tuple(buffer.shape) != shape
            or buffer.dtype != self.norm.weight.dtype
            or buffer.device != self.norm.weight.device
        ):
            raise ValueError("Cannot replace an initialized MTP hidden hand-off buffer")



    def has_mtp_hidden_buffer(self):
        return getattr(self, "_mtp_prefill_hidden_buffer", None) is not None



    def get_mtp_target_hidden_states(self, num_tokens):
        buffer = getattr(self, "_mtp_prefill_hidden_buffer", None)
        if buffer is None:
            return super().get_mtp_target_hidden_states(num_tokens)
        rows = self._mtp_prefill_hidden_valid_rows
        requested = rows if num_tokens < 0 else int(num_tokens)
        if rows <= 0 or requested < 0 or requested > rows:
            raise ValueError(
                f"MTP Prefill hidden rows requested={requested}, written={rows}"
            )
        return buffer[:requested]



    def _publish_mtp_prefill_hidden(self, hidden):
        buffer = getattr(self, "_mtp_prefill_hidden_buffer", None)
        if buffer is None or getattr(self, "_query_layout", None) is None:
            return hidden
        layout = self._query_layout
        rows = int(hidden.shape[0])
        if (
            hidden.ndim != 2
            or rows != layout.local_rows
            or rows > buffer.shape[0]
            or hidden.shape[1] != buffer.shape[1]
            or hidden.dtype != buffer.dtype
            or hidden.device != buffer.device
        ):
            raise ValueError("MTP Prefill hidden does not match rank-local CP storage")
        # Prefill CP is eager. Copy on the existing model stream, then publish
        # the exact local extent; target and draft own different storage.
        buffer[:rows].copy_(hidden)
        self._mtp_prefill_hidden_valid_rows = rows
        return buffer[:rows]
