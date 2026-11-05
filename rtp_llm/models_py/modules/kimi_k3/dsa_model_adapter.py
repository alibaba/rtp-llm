"""Optional DSA metadata and MTP handoff hooks for the shared K3 model."""

import copy
from collections.abc import Mapping

import torch

from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value, get_primary_attention_inputs,
)
from rtp_llm.ops import HybridAttentionType


class KimiK31ModelAdapter:
    """No decoder loop: K3 keeps its shared layers, MoE and output handling."""

    def _setup_dsa(self):
        from .natural_dsa import outer_natural_enabled

        self._query_layout = None
        self._query_layout_buffers = {}
        self._outer_layout = None
        self._outer_layout_buffers = {}
        self._row_exchange = None
        self._routing_plan = None
        self._outer_natural_enabled = (
            outer_natural_enabled() and self.prefill_query_cp and not self.config.is_mtp
        )
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
        from .natural_dsa import route_allocation_budget

        for layer in self.layers:
            if layer.layer_type == HybridAttentionType.LINEAR:
                attention = layer.attention
                if attention._fp8_output_norm or any(
                    not isinstance(p, CudaF16Linear) or p.weight.dtype != torch.bfloat16
                    or p.bias is not None for p in (attention.input, attention.f_b, attention.output)
                ):
                    raise ValueError("natural outer requires BF16 KDA projections and output norm")
            elif getattr(layer.attention, "indexer", None) is not None:
                if not hasattr(layer.attention, "natural_q_input"):
                    raise ValueError("natural outer requires the supported BF16 SFA split projection backend")
        if self.norm.weight.dtype != torch.bfloat16:
            raise ValueError("natural outer requires BF16 hidden states")
        global_rows = self.config.moe_prefill_max_tokens_per_rank
        if global_rows is None:
            global_rows = self.config.max_seq_len
        global_rows = max(int(global_rows), self._max_moe_tokens)
        capacity = (global_rows + self.tp_size - 1) // self.tp_size + 2 * int(init_resource.max_context_batch_size)
        self._row_exchange = DirectedRowExchange(
            get_process_group(Group.TP), self.norm.weight.device,
            capacity, self.config.hidden_size, chunk_capacity=8192,
        )
        owner = self.dsa_workspace
        owner.row_exchange = self._row_exchange
        owner.route_capacity = capacity
        owner.route_chunk_rows = self._row_exchange.chunk_capacity
        owner.route_bytes, owner.route_allocation_granularity = route_allocation_budget(self._row_exchange)

    def _prepare_dsa(self, inputs, is_cuda_graph):
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
            layout = QueryLayout(
                info, self.tp_size, self.tp_rank, inputs.input_ids.device,
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

                plan = NaturalZigzagPlan(layout.lengths, self.tp_size, self.tp_rank)
                validate_shared_restore(layout, plan)
                self._outer_layout = NaturalOuterLayout(layout, self._outer_layout_buffers)
                self._routing_plan = plan
                self._row_exchange.prepare_plan(plan, chunk_rows=self.dsa_workspace.route_chunk_rows)
            kda_layout = self._outer_layout if outer else layout
            for layer in self.layers:
                if layer.layer_type == HybridAttentionType.LINEAR:
                    a = layer.attention
                    kda_layout.prepare_kda_buffers(a.input.weight.shape[0], a.width,
                                               quantized=a._fp8_output_norm)
            self._query_layout = layout
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
        for tensor in tensors:
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        owner.layout_bytes = sum(storages.values())

        def create(attention_inputs):
            parallel = self.parallelism_config
            if self._query_layout is not None:
                parallel = copy.deepcopy(parallel)
                parallel.prefill_cp_config.preserve_tp_layout = False
            return SparseMlaPageRRImpl(
                self.config, parallel, self.weight, attention_inputs,
                self.fmha_config, is_cuda_graph, workspace_owner=owner,
            )

        tagged = get_attention_inputs_value(inputs)
        if isinstance(tagged, Mapping):
            tags = self._get_fmha_group_tags()
            tags = tuple(tagged if tags is None else tags)
            owner.graph_cache_groups = len(tags)
            return {tag: create(tagged[tag]) for tag in tags}
        return create(tagged)


    def _initialize_mtp_hidden_handoff(self, init_resource):
        # CP's public output keeps only LM-head rows. Recurrent MTP must retain
        # every rank-local pre-norm row, including the per-request padding.
        if not (
            getattr(self, "prefill_query_cp", False)
            and getattr(init_resource, "is_speculative", False)
        ):
            return
        global_rows = self.config.moe_prefill_max_tokens_per_rank
        if global_rows is None:
            global_rows = self.config.max_seq_len
        requests = int(init_resource.max_context_batch_size)
        capacity = (int(global_rows) + self.tp_size - 1) // self.tp_size + 2 * requests
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
