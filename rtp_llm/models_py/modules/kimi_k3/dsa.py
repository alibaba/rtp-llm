"""K3.1 sparse policy composed from shared K3 projections and Indexer kernels."""

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group, all_gather, reduce_scatter_bf16,
)
from rtp_llm.models_py.mla_routes import resolve_mla_routes, run_sparse_indexer, run_prepared_sparse_indexer
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.models_py.modules.kimi_k3.attention import KimiK3MLA
from rtp_llm.models_py.triton_kernels.common.activation import gate_sigmoid_mul
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
    fused_q_kv_rmsnorm, grouped_q_kv_rmsnorm, independent_q_kv_rmsnorm,
)
from rtp_llm.models_py.utils.module_profiler import profile_scope
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


def split_bf16x3(weight):
    """FP32 [k, n] -> BF16 [k, 3n] (hi | mid | lo) with hi + mid + lo == weight for normal values.

    Each BF16 part takes 8 of the 24 FP32 significand bits, and BF16 x BF16 products are exact in
    FP32, so x @ parts summed over the three column blocks differs from the FP32 x @ weight only in
    the accumulation order.
    """
    hi = weight.bfloat16()
    mid = (weight - hi.float()).bfloat16()
    lo = (weight - hi.float() - mid.float()).bfloat16()
    return torch.cat((hi, mid, lo), dim=1).contiguous()


class KimiK31DSA(KimiK3MLA):
    """Local-Q Prefill and replicated-Q Decode over owner-local cache."""

    # Indexer K/head-gate projections also consume the BF16 hidden rows.
    # AttnRes must not replace those rows with an FP8-only activation.
    supports_fp8_activation = False

    def __init__(self, config, parallelism, weights, layer_idx, hardware=None):
        super().__init__(config, parallelism, weights, layer_idx, hardware)
        policy = config.k3_runtime_config.dsa
        cp = parallelism.prefill_cp_config
        self.prefill_query_cp = (
            parallelism.role_type == RoleType.PREFILL
            and cp.is_enabled() and cp.preserve_tp_layout
        )
        self.heads = config.attn_config.head_num // (1 if self.prefill_query_cp else self.tp_size)
        self.query_heads = (
            config.attn_config.head_num
            if self.prefill_query_cp or parallelism.role_type == RoleType.DECODE
            else self.heads
        )
        qkv_rows = self.q_rank + self.kv_rank + self.suffix_dim
        self._input_padding = self.input.weight.shape[0] - qkv_rows - self.heads * self.v_dim
        expected_padding = (
            (-qkv_rows) % 128 if self.input.weight.dtype == torch.float8_e4m3fn else 0
        )
        if self._input_padding != expected_padding:
            raise ValueError("K3.1 input projection does not match its Q/KV/gate FP8 block layout")
        self._gate_stream = None
        if policy.indexer_use_fp8:
            from rtp_llm.models_py.modules.base.cuda.fp8_indexer_op import Fp8IndexerOp

            op = Fp8IndexerOp(policy)
            quant = config.indexer_projection_quant_config
            if quant is None:
                raise ValueError("FP8 target Indexer requires its projection precision policy")
        else:
            if not config.is_mtp:
                raise ValueError("BF16 Indexer is reserved for synthetic K3.1 draft")
            from rtp_llm.models_py.modules.base.cuda.bf16_indexer_op import Bf16DraftIndexerOp

            op, quant = Bf16DraftIndexerOp(policy), None
        self.indexer = Indexer(
            config.attn_config, weights, {}, layer_idx, config.layernorm_eps,
            quant, hardware, parallelism, dsa_config=policy, indexer_op=op,
        )
        self._qkv_norm = (
            grouped_q_kv_rmsnorm if self.prefill_query_cp else fused_q_kv_rmsnorm
        )
        if self.prefill_query_cp and not config.is_mtp:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import CudaFp8GEMMLinear

            projections = (self.input, self.q_b, self.output)
            if (not policy.indexer_use_fp8 or any(
                not ((isinstance(p, CudaF16Linear) and p.weight.dtype == torch.bfloat16)
                     or (isinstance(p, CudaFp8GEMMLinear) and p.weight.dtype == torch.float8_e4m3fn))
                or p.bias is not None for p in projections
            ) or not isinstance(self.indexer.weights_proj, CudaF16Linear)
                    or self.indexer.weights_proj.weight.dtype != torch.float32
                    or self.indexer.weights_proj.bias is not None
                    or any(not isinstance(p, CudaFp8GEMMLinear) for p in (self.indexer.wk, self.indexer.wq_b))):
                raise ValueError("natural outer DSA requires supported CUDA SFA projections and FP8 Indexer with FP32 head weights")
            width = self.q_rank + self.kv_rank + self.suffix_dim + self._input_padding + self.heads * self.v_dim
            if (self.input.weight.ndim != 2 or self.input.weight.shape != (width, config.hidden_size)
                    or self.q_b.weight.shape != (self.heads * self.q_dim, self.q_rank)
                    or self.output.weight.shape != (config.hidden_size, self.heads * self.v_dim)):
                raise ValueError("natural outer DSA fused input must contain the complete full-head gate")
            from .natural_dsa import NaturalQGateProjection, split_natural_projection

            self.natural_q_input, self.natural_kv_input, self.natural_gate_input = split_natural_projection(
                self.input, self.q_rank, self.kv_rank + self.suffix_dim,
                self.heads * self.v_dim, self._input_padding,
            )
            self.natural_q_gate_input = NaturalQGateProjection(
                self.natural_q_input, self.natural_gate_input,
            )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_dcp_options import (
            dcp_prologue_mode,
        )

        self.prologue = (
            dcp_prologue_mode()
            if policy.indexer_use_fp8 and parallelism.role_type == RoleType.DECODE and self.tp_size > 1
            else "dev"
        )
        if self.prologue != "dev":
            self._init_prologue(config, weights)

    def _init_prologue(self, config, weights):
        """Decode prologue of DSA_DCP_PROLOGUE: every query head on every rank, BF16 head gate."""
        heads = config.attn_config.head_num
        if self.q_b.weight.shape[0] != heads * self.q_dim or weights[W.mla_kc].shape[0] != heads:
            raise ValueError("DSA_DCP_PROLOGUE needs q_b and kc loaded for every head")
        self.query_heads = heads
        gate = weights[W.mla_indexer_weights_proj_w]
        if gate.dtype != torch.float32 or gate.shape != (config.hidden_size, self.indexer.index_n_heads):
            raise ValueError("the Indexer head gate weight must be FP32 [hidden, heads]")
        self._head_gate = split_bf16x3(gate)
        self._head_gate_scale = self.indexer.softmax_scale * self.indexer.weights_scale
        from rtp_llm.models_py.utils.stream_dag import StreamDag

        dag = StreamDag(torch.cuda.current_device(), side_streams=("index_k", "head_gate", "sfa_query"))
        dag.add("gather", self._gather)
        dag.add("input", self._input_projection, after=("gather",))
        dag.add("index_k", self._index_keys, stream="index_k", after=("gather",))
        dag.add("head_gate", self._head_gate_projection, stream="head_gate", after=("gather",))
        dag.add("norm", self._norm, after=("input",))
        dag.add("sfa_query", self._sfa_query, stream="sfa_query", after=("norm",))
        dag.add("index_q", self._index_queries, after=("norm", "head_gate"))
        dag.add("topk", self._topk, after=("index_q", "index_k"))
        dag.add("sfa", self._sfa, after=("topk", "sfa_query"))
        dag.finalize()
        self._dag = dag

    # DAG nodes over one state namespace per call.

    def _gather(self, s):
        s.full = all_gather(s.hidden, Group.TP)

    def _input_projection(self, s):
        s.qkv, _, s.gate = self.input(s.full).split(
            [self.q_rank + self.kv_rank + self.suffix_dim, self._input_padding, self.heads * self.v_dim], dim=-1,
        )

    def _index_keys(self, s):
        indexer = self.indexer
        key = indexer.k_norm(indexer.wk(s.full))
        indexer.indexer_op.write_key(key, s.indexer_cache, s.indexer.fmha_params)

    def _head_gate_projection(self, s):
        s.head_gate = torch.mm(s.full, self._head_gate, out_dtype=torch.float32)

    def _norm(self, s):
        q, kv = s.qkv.split([self.q_rank, self.kv_rank + self.suffix_dim], dim=-1)
        latent, s.suffix = kv.split([self.kv_rank, self.suffix_dim], dim=-1)
        s.q, s.latent = fused_q_kv_rmsnorm(
            q, latent, self.q_norm.weight, self.kv_norm.weight, self.q_norm.variance_epsilon,
        )

    def _sfa_query(self, s):
        query = self.q_b(s.q).view(-1, self.query_heads, self.q_dim)
        s.sfa_query = s.sfa.decode_query(query, self.layer_idx)
        s.sfa.write_decode_cache(s.latent, s.suffix, s.sfa_cache)

    def _index_queries(self, s):
        indexer = self.indexer
        query = indexer.wq_b(s.q).view(-1, indexer.index_n_heads, indexer.index_head_dim)
        s.index_query, scale = indexer.indexer_op.encode_query(query)
        gate = s.head_gate.view(-1, 3, indexer.index_n_heads)
        gate = gate[:, 0] + gate[:, 1] + gate[:, 2]
        s.index_weights = gate.unsqueeze(-1) * scale * self._head_gate_scale

    def _topk(self, s):
        fmha = s.indexer
        s.indices = self.indexer._compute_topk(
            s.index_query, s.index_weights, s.indexer_cache, fmha.fmha_params, fmha.attn_inputs, fmha.cp_params,
        )

    def _sfa(self, s):
        s.output = s.sfa.forward_decode_prepared(s.sfa_query, s.sfa_cache, self.layer_idx, s.indices)

    def _decode_prologue(self, hidden, fmha, cache):
        """The dev path's decode from the input all-gather to sparse MLA, every query head on every rank
        and the head gate as one BF16x3 GEMM; side streams under multi_stream."""
        from types import SimpleNamespace

        sfa, sfa_cache, indexer, indexer_cache = resolve_mla_routes(
            fmha, cache, sparse=True, attention_tag="full",
        )
        if sfa.page_rr_params.is_prefill:
            raise ValueError("DSA_DCP_PROLOGUE covers decode only")
        state = SimpleNamespace(hidden=hidden, sfa=sfa, sfa_cache=sfa_cache, indexer=indexer,
                                indexer_cache=indexer_cache)
        with profile_scope("RTP::attention.dsa.prologue"):
            self._dag.run(state, multi_stream=self.prologue == "multi_stream")
        return state.output.reshape(-1, self.heads * self.v_dim), state.gate

    def _attend(self, qkv, fmha, cache, hidden):
        sfa, sfa_cache, indexer, indexer_cache = resolve_mla_routes(
            fmha, cache, sparse=True, attention_tag="full",
        )
        q, kv = qkv.split([self.q_rank, self.kv_rank + self.suffix_dim], dim=-1)
        latent, suffix = kv.split([self.kv_rank, self.suffix_dim], dim=-1)
        packed = self.prefill_query_cp and sfa.page_rr_params.is_prefill
        with profile_scope("RTP::attention.mla.qkv_norm"):
            if q.is_cuda:
                q, latent = self._qkv_norm(
                    q, latent, self.q_norm.weight, self.kv_norm.weight,
                    self.q_norm.variance_epsilon,
                    **({"kv_suffix": suffix} if packed else {}),
                )
                if packed:
                    suffix = latent.as_strided(
                        suffix.shape, (latent.stride(0), 1),
                        latent.storage_offset() + self.kv_rank,
                    )
            else:
                q, latent = self.q_norm(q.contiguous()), self.kv_norm(latent.contiguous())
        with profile_scope("RTP::attention.mla.q_proj"):
            query = self.q_b(q).reshape(-1, self.query_heads, self.q_dim)
        with profile_scope("RTP::attention.dsa.indexer"):
            indices = run_sparse_indexer(self.indexer, hidden, q, indexer_cache, indexer)
        with profile_scope("RTP::attention.mla.core"):
            output = sfa.forward(query, latent, suffix, sfa_cache, self.layer_idx, indices)
        if output is None:
            raise RuntimeError("K3.1 sparse attention returned no output")
        return output.reshape(-1, self.heads * self.v_dim)

    def _forward_natural(self, hidden, fmha, cache, metadata):
        from rtp_llm.models_py.modules.kimi_k3.natural_dsa import Fp8NaturalProjection

        sfa, sfa_cache, index_fmha, index_cache = resolve_mla_routes(
            fmha, cache, sparse=True, attention_tag="full",
        )
        owner, layout = sfa.workspace_owner, metadata.query_layout
        rows = layout.local_rows
        single_chunk = 0 < rows <= owner.route_chunk_rows
        latent = suffix = None

        def packed_suffix(value):
            return value.as_strided(
                (rows, self.suffix_dim), (value.stride(0), 1),
                value.storage_offset() + self.kv_rank,
            )

        def prepare_kv():
            nonlocal latent, suffix
            with profile_scope("RTP::attention.dsa.natural_kv"):
                packed = self.natural_kv_input(hidden)
                latent, suffix = packed.split([self.kv_rank, self.suffix_dim], -1)
                if not single_chunk:
                    latent = independent_q_kv_rmsnorm(
                        latent, self.kv_norm.weight, self.kv_norm.variance_epsilon,
                        q_width=self.q_rank, kv_width=self.kv_rank, full_rows=rows, suffix=suffix,
                    )
                    suffix = packed_suffix(latent)
                if index_cache is not None:
                    key = self.indexer.project_key(hidden)
                    self.indexer.indexer_op.write_key(key, index_cache, index_fmha.fmha_params)
        # A complete chunk can retain the projection/encoding outputs directly:
        # they own storage independent of the recyclable received H slot.
        query = gate = None
        if not single_chunk:
            query = hidden.new_empty((rows, self.heads, self.q_dim))
            gate = hidden.new_empty((rows, self.heads * self.v_dim))
        prepare_index_query = index_cache is not None and index_fmha.is_sparse()
        index_query = head_weights = None
        if prepare_index_query and not single_chunk:
            index_shape = (rows, self.indexer.index_n_heads, self.indexer.index_head_dim)
            index_query = torch.empty(index_shape, dtype=torch.float8_e4m3fn, device=hidden.device)
            head_weights = torch.empty((*index_shape[:2], 1), dtype=torch.float32, device=hidden.device)

        def receive(start, chunk):
            nonlocal query, gate, index_query, head_weights, latent, suffix
            end = start + chunk.shape[0]
            if single_chunk and (start != 0 or end != rows):
                raise RuntimeError("single-chunk query route must cover all local rows")
            # Q and gate consume the same received H rows with the same FP8
            # policy. Keep their encoding alive until both GEMMs are enqueued.
            quantized = None
            q_gate = getattr(self, "natural_q_gate_input", None)
            if q_gate is not None:
                q_input, projected_gate = q_gate(chunk)
            elif isinstance(self.natural_q_input, Fp8NaturalProjection):
                quantized = self.natural_q_input.quantize_input(chunk)
                q_input = self.natural_q_input.forward_quantized(*quantized)
            else:
                q_input = self.natural_q_input(chunk)
            if single_chunk:
                q, latent = grouped_q_kv_rmsnorm(
                    q_input, latent, self.q_norm.weight, self.kv_norm.weight,
                    self.q_norm.variance_epsilon, kv_suffix=suffix,
                )
                suffix = packed_suffix(latent)
            else:
                q = independent_q_kv_rmsnorm(
                    q_input, self.q_norm.weight, self.q_norm.variance_epsilon,
                    q_width=self.q_rank, kv_width=self.kv_rank, full_rows=rows,
                )
            del q_input
            projected = self.q_b(q).reshape(-1, self.heads, self.q_dim)
            if single_chunk:
                query = projected
            else:
                query[start:end].copy_(projected)
            del projected
            if prepare_index_query:
                encoded, scale = self.indexer.indexer_op.encode_query(
                    self.indexer.project_query(q, original_rows=rows))
                weights = self.indexer.project_head_weights(
                    chunk, scale, is_prefill=True, original_rows=rows,
                    tile_decode_output=not index_fmha.fmha_params.is_prefill,
                )
                if single_chunk:
                    index_query, head_weights = encoded, weights
                else:
                    index_query[start:end].copy_(encoded)
                    head_weights[start:end].copy_(weights)
                del encoded, scale, weights
            if q_gate is None:
                projected_gate = (self.natural_gate_input.forward_quantized(*quantized)
                                  if quantized is not None else self.natural_gate_input(chunk))
            del quantized
            if single_chunk:
                gate = projected_gate
            else:
                gate[start:end].copy_(projected_gate)

        with profile_scope("RTP::attention.dsa.natural_h_to_zigzag"):
            owner.row_exchange.consume_chunks(
                hidden, owner.routing_plan, "to_zigzag",
                chunk_rows=owner.route_chunk_rows, consumer=receive, before_consume=prepare_kv,
            )
        with profile_scope("RTP::attention.dsa.indexer"):
            indices = run_prepared_sparse_indexer(
                self.indexer, index_query, head_weights, index_cache, index_fmha,
            )
        del index_query, head_weights
        with profile_scope("RTP::attention.mla.core"):
            output = sfa.forward(query, latent, suffix, sfa_cache, self.layer_idx, indices)
        if output is None:
            raise RuntimeError("K3.1 sparse attention returned no output")
        del query, latent, suffix, indices
        mask = None if layout.all_valid else layout.local_valid
        with profile_scope("RTP::attention.mla.output_gate_quant"):
            output = output.reshape(rows, self.heads * self.v_dim)
            if self._fp8_output_gate:
                from rtp_llm.models_py.triton_kernels.common.activation import sigmoid_mul_per_token_group_quant_fp8

                if mask is not None:
                    output = torch.where(mask[:, None], output, 0)
                    gate = torch.where(mask[:, None], gate, 0)
                values, scales = sigmoid_mul_per_token_group_quant_fp8(output, gate)
            else:
                output = gate_sigmoid_mul(output, gate, mask)
        del gate
        publish = getattr(owner.row_exchange, "produce_and_exchange", None)
        if self._fp8_output_gate and publish is not None:
            def produce(target):
                with profile_scope("RTP::attention.mla.output_proj"):
                    self.output.forward_quantized(values, scales, out=target)
            with profile_scope("RTP::attention.dsa.zigzag_o_to_natural"):
                return publish(owner.routing_plan, "to_natural", produce)
        with profile_scope("RTP::attention.mla.output_proj"):
            output = (self.output.forward_quantized(values, scales)
                      if self._fp8_output_gate else self.output(output))
        # Full O is complete before publication. No reverse-route overlap or RS.
        with profile_scope("RTP::attention.dsa.zigzag_o_to_natural"):
            return owner.row_exchange.exchange(output, owner.routing_plan, "to_natural")

    def forward(self, hidden, fmha, cache, attention_inputs=None, metadata=None):
        if getattr(metadata, "outer_layout", None) is not None:
            output = self._forward_natural(hidden, fmha, cache, metadata)
            next(iter(fmha.values())).workspace_owner.release_eager_buffers()
            return output
        local_query = (
            self.prefill_query_cp
            and getattr(metadata, "query_layout", None) is not None
        )
        use_bf16_collective = False
        if self.prologue != "dev":
            output, gate = self._decode_prologue(hidden, fmha, cache)
        else:
            collective = self._bf16_collective
            use_bf16_collective = (
                not local_query
                and not self._fp8_output_gate
                and collective is not None
                and collective.eligible(hidden.shape[0], attention_inputs, metadata)
            )
            if use_bf16_collective:
                with profile_scope("RTP::attention.bf16_ag_gemm"):
                    projected, full = collective.all_gather_gemm(
                        hidden,
                        self.input,
                        return_input=True,
                    )
            else:
                with profile_scope("RTP::attention.input_all_gather"):
                    full = (
                        hidden
                        if local_query or self.tp_size == 1
                        else (
                            self._mtp_bf16_collectives.all_gather(hidden)
                            if self._mtp_bf16_collectives is not None
                            else all_gather(hidden, Group.TP)
                        )
                    )
                with profile_scope("RTP::attention.mla.qkv_gate_input_proj"):
                    projected = self.input(full)
            qkv, _, gate = projected.split(
                [self.q_rank + self.kv_rank + self.suffix_dim,
                 self._input_padding, self.heads * self.v_dim],
                dim=-1,
            )
            del projected
            output = self._attend(qkv, fmha, cache, full)
            del full, qkv
        layout = getattr(metadata, "query_layout", None)
        mask = (
            None
            if layout is not None and layout.all_valid
            else (
                layout.local_valid if local_query else attention_inputs.valid_token_mask
            )
        )
        with profile_scope("RTP::attention.mla.output_gate_quant"):
            if self._fp8_output_gate:
                from rtp_llm.models_py.triton_kernels.common.activation import (
                    sigmoid_mul_per_token_group_quant_fp8,
                )

                # Mask both operands: an inactive gate may contain NaN, and
                # zero * sigmoid(NaN) must not enter the FP8 scale reduction.
                if mask is not None:
                    output = torch.where(mask[:, None], output, 0)
                    gate = torch.where(mask[:, None], gate, 0)
                values, scales = sigmoid_mul_per_token_group_quant_fp8(output, gate)
            else:
                output = gate_sigmoid_mul(output, gate, mask)
        use_fused_rs = (
            not local_query
            and self._fp8_output_gate
            and self._fp8_collective is not None
            and self._fp8_collective.eligible_rs(
                values.shape[0], attention_inputs, metadata, True
            )
        )
        if use_fused_rs:
            with profile_scope("RTP::attention.fp8_gemm_rs"):
                output = self._fp8_collective.gemm_reduce_scatter(
                    values, scales, self.output
                )
        elif use_bf16_collective:
            with profile_scope("RTP::attention.bf16_gemm_rs"):
                output = collective.gemm_reduce_scatter(output, self.output)
        else:
            with profile_scope("RTP::attention.mla.output_proj"):
                output = (
                    self.output.forward_quantized(values, scales)
                    if self._fp8_output_gate else self.output(output)
                )
        if not (use_fused_rs or use_bf16_collective) and not local_query and self.tp_size > 1:
            with profile_scope("RTP::attention.output_reduce_scatter"):
                output = (
                    self._mtp_bf16_collectives.reduce_scatter(output)
                    if self._mtp_bf16_collectives is not None
                    else reduce_scatter_bf16(output, Group.TP)
                )
        owner = getattr(next(iter(fmha.values())), "workspace_owner", None)
        if owner is not None:
            owner.release_eager_buffers()
        return output
