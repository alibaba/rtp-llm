"""K3.1 sparse policy composed from shared K3 projections and Indexer kernels."""

import torch
import triton
import triton.language as tl

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


@triton.jit
def _fold_head_gate_kernel(
    Gate, PackedScale, Out, Scale, G_ROW: tl.constexpr, S_ROW: tl.constexpr, HEADS: tl.constexpr
):
    row = tl.program_id(0)
    h = tl.arange(0, HEADS)
    g0 = tl.load(Gate + row * G_ROW + h)
    g1 = tl.load(Gate + row * G_ROW + HEADS + h)
    g2 = tl.load(Gate + row * G_ROW + 2 * HEADS + h)
    packed = tl.load(PackedScale + (row * HEADS + h) * S_ROW)
    q_scale = ((packed & 0xFF) << 23).to(tl.float32, bitcast=True)
    tl.store(Out + row * HEADS + h, (g0 + g1 + g2) * q_scale * Scale)


def fold_head_gate_scales(head_gate, packed_scale, scale, heads):
    rows = head_gate.shape[0]
    out = torch.empty((rows, heads, 1), dtype=torch.float32, device=head_gate.device)
    if rows:
        _fold_head_gate_kernel[(rows,)](
            head_gate, packed_scale, out, float(scale),
            head_gate.stride(0), packed_scale.stride(0), heads, num_warps=1,
        )
    return out


class KimiK31DSA(KimiK3MLA):
    """Local-Q Prefill and replicated-Q Decode over owner-local cache."""

    # Indexer head weights need the local BF16 hidden alongside quantized inputs.
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
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import CudaFp8GEMMLinear

        def compatible_fp8_inputs(first, second, width):
            return all(
                isinstance(projection, CudaFp8GEMMLinear)
                and projection.scale_ue8m0 and projection.K == width
                for projection in (first, second)
            )

        share_fp8_q_latent = self.q_rank == 1536 and compatible_fp8_inputs(
            self.q_b, self.indexer.wq_b, self.q_rank,
        )
        self._verify_shared_q_min_rows = (
            CudaFp8GEMMLinear.FLASHINFER_M_THRESHOLD
            if not config.is_mtp and share_fp8_q_latent else None
        )
        self._verify_shared_hidden_min_rows = (
            CudaFp8GEMMLinear.FLASHINFER_M_THRESHOLD
            if not config.is_mtp and compatible_fp8_inputs(self.input, self.indexer.wk, config.hidden_size)
            else None
        )
        self.natural_kv_indexer_input = None
        self._natural_nvfp4_enabled = False
        self._share_fp8_q_latent = False
        if self.prefill_query_cp and not config.is_mtp:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear

            projections = (self.input, self.q_b, self.output)
            if (not policy.indexer_use_fp8 or any(
                not ((isinstance(p, CudaF16Linear) and p.weight.dtype == torch.bfloat16)
                     or (isinstance(p, CudaFp8GEMMLinear) and p.weight.dtype == torch.float8_e4m3fn)
                     or (getattr(p, "dynamic_input_scale", False) and p.weight.dtype == torch.uint8))
                or p.bias is not None for p in projections
            ) or not isinstance(self.indexer.weights_proj, CudaF16Linear)
                    or self.indexer.weights_proj.weight.dtype != torch.float32
                    or self.indexer.weights_proj.bias is not None
                    or any(not isinstance(p, CudaFp8GEMMLinear) for p in (self.indexer.wk, self.indexer.wq_b))):
                raise ValueError("natural outer DSA requires supported CUDA SFA projections and FP8 Indexer with FP32 head weights")
            width = self.q_rank + self.kv_rank + self.suffix_dim + self._input_padding + self.heads * self.v_dim
            def logical_shape(p):
                return ((p.output_size, p.hidden_size) if getattr(p, "dynamic_input_scale", False)
                        else p.weight.shape)
            if (self.input.weight.ndim != 2 or logical_shape(self.input) != (width, config.hidden_size)
                    or logical_shape(self.q_b) != (self.heads * self.q_dim, self.q_rank)
                    or logical_shape(self.output) != (config.hidden_size, self.heads * self.v_dim)):
                raise ValueError("natural outer DSA fused input must contain the complete full-head gate")
            from .natural_dsa import (
                Fp8NaturalProjection, NaturalPairedProjection, split_natural_projection,
            )

            self.natural_q_input, self.natural_kv_input, self.natural_gate_input = split_natural_projection(
                self.input, self.q_rank, self.kv_rank + self.suffix_dim,
                self.heads * self.v_dim, self._input_padding,
            )
            self.natural_q_gate_input = NaturalPairedProjection(
                self.natural_q_input, self.natural_gate_input,
            )
            wk = self.indexer.wk
            can_quantize_indexer = (
                wk.scale_ue8m0
                and self.indexer.index_n_heads == 32
                and self.indexer.index_head_dim == 128
                and self.indexer._fp32_head_weight is not None
                and config.hidden_size == 7168
            )
            if (can_quantize_indexer and isinstance(self.natural_kv_input, Fp8NaturalProjection)
                    and self.input.scale_ue8m0):
                index_k = Fp8NaturalProjection(wk, 0, wk.N, wk.N)
                self.natural_kv_indexer_input = NaturalPairedProjection(self.natural_kv_input, index_k)
                self._share_fp8_q_latent = share_fp8_q_latent
            self._natural_nvfp4_enabled = (
                can_quantize_indexer and self.natural_q_gate_input.nvfp4 is not None
            )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_dcp_options import (
            dcp_prologue_mode,
        )

        self.prologue = (
            dcp_prologue_mode()
            if policy.indexer_use_fp8 and parallelism.role_type == RoleType.DECODE and self.tp_size > 1
            else "dev"
        )
        self._decode_nvfp4_ag = None
        packed_decode = (
            not config.is_mtp and policy.indexer_use_fp8
            and parallelism.role_type == RoleType.DECODE and self.tp_size > 1
            and getattr(self.input, "dynamic_input_scale", False)
            and config.k3_runtime_config.attn_res_block_size
        )
        self._decode_head_gate_columns = 3 * self.indexer.index_n_heads if packed_decode else 0
        if self.prologue != "dev" or packed_decode:
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
        s.fp4 = None
        if self._decode_nvfp4_ag is not None and not isinstance(s.hidden, torch.Tensor):
            _, _, head_gate = s.hidden.views()
            head_gate.copy_(torch.mm(s.hidden.local_hidden, self._head_gate, out_dtype=torch.float32))
            s.fp4, s.fp8, s.head_gate = self._decode_nvfp4_ag(s.hidden)
        else:
            s.full = all_gather(s.hidden, Group.TP)

    def _input_projection(self, s):
        projected = self.input(s.full) if s.fp4 is None else self.input.forward_quantized(s.fp4)
        s.qkv, _, s.gate = projected.split(
            [self.q_rank + self.kv_rank + self.suffix_dim, self._input_padding, self.heads * self.v_dim], dim=-1,
        )

    def _index_keys(self, s):
        indexer = self.indexer
        raw = indexer.wk(s.full) if s.fp4 is None else indexer.wk.forward_quantized(*s.fp8)
        key = indexer.k_norm(raw)
        indexer.indexer_op.write_key(key, s.indexer_cache, s.indexer.fmha_params)

    def _head_gate_projection(self, s):
        if s.fp4 is None:
            s.head_gate = torch.mm(s.full, self._head_gate, out_dtype=torch.float32)

    def _norm(self, s):
        q, kv = s.qkv.split([self.q_rank, self.kv_rank + self.suffix_dim], dim=-1)
        latent, s.suffix = kv.split([self.kv_rank, self.suffix_dim], dim=-1)
        s.q_fp8 = None
        quantize_q = (
            s.fp4 is not None and self.q_rank % 512 == 0
            and getattr(self.q_b, "dynamic_input_scale", False) and self.indexer.wq_b.scale_ue8m0
        )
        normalized = fused_q_kv_rmsnorm(
            q, latent, self.q_norm.weight, self.kv_norm.weight, self.q_norm.variance_epsilon,
            quantize_q=quantize_q,
        )
        if quantize_q:
            s.q, s.latent, s.q_fp8, s.q_amax = normalized
        else:
            s.q, s.latent = normalized

    def _sfa_query(self, s):
        if s.q_fp8 is None:
            query = self.q_b(s.q)
        else:
            from rtp_llm.models_py.triton_kernels.common.nvfp4 import quantize_nvfp4_from_partials

            query = self.q_b.forward_quantized(quantize_nvfp4_from_partials(s.q, s.q_amax))
        query = query.view(-1, self.query_heads, self.q_dim)
        s.sfa_query = s.sfa.decode_query(query, self.layer_idx)
        s.sfa.write_decode_cache(s.latent, s.suffix, s.sfa_cache)

    def _index_queries(self, s):
        indexer = self.indexer
        query = indexer.wq_b(s.q) if s.q_fp8 is None else indexer.wq_b.forward_quantized(*s.q_fp8)
        query = query.view(-1, indexer.index_n_heads, indexer.index_head_dim)
        s.index_query, scale = indexer.indexer_op.encode_query(query, unpack_scale=False)
        s.index_weights = fold_head_gate_scales(
            s.head_gate, scale, self._head_gate_scale, indexer.index_n_heads
        )

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

    def _attend(self, qkv, fmha, cache, hidden, *, indexer_key=None):
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
        encoded_q = None
        with profile_scope("RTP::attention.mla.q_proj"):
            if (
                self._verify_shared_q_min_rows is not None
                and q.shape[0] >= self._verify_shared_q_min_rows
                and indexer_cache is not None
                and indexer.attn_inputs.is_target_verify
                and indexer.is_sparse()
            ):
                # Both projections already select DeepGEMM for this row count.
                encoded_q = self.q_b.quantize_collective_input(q)
                query = self.q_b.forward_quantized(*encoded_q)
            else:
                query = self.q_b(q)
            query = query.reshape(-1, self.query_heads, self.q_dim)
        with profile_scope("RTP::attention.dsa.indexer"):
            indices = run_sparse_indexer(
                self.indexer, hidden, q, indexer_cache, indexer,
                encoded_q_lora=encoded_q, prepared_key=indexer_key,
            )
        with profile_scope("RTP::attention.mla.core"):
            output = sfa.forward(query, latent, suffix, sfa_cache, self.layer_idx, indices)
        if output is None:
            raise RuntimeError("K3.1 sparse attention returned no output")
        return output.reshape(-1, self.heads * self.v_dim)

    def _forward_natural(self, hidden, fmha, cache, metadata, natural_fp8=None, natural_nvfp4=None):
        from rtp_llm.models_py.modules.kimi_k3.natural_dsa import Fp8NaturalProjection

        sfa, sfa_cache, index_fmha, index_cache = resolve_mla_routes(
            fmha, cache, sparse=True, attention_tag="full",
        )
        owner, layout = sfa.workspace_owner, metadata.query_layout
        rows = layout.local_rows
        joint_q_kv_norm = natural_fp8 is None and 0 < rows <= owner.route_chunk_rows
        share_q_latent = natural_fp8 is not None and self._share_fp8_q_latent
        q_gate = getattr(self, "natural_q_gate_input", None)
        latent = suffix = None

        def packed_suffix(value):
            return value.as_strided(
                (rows, self.suffix_dim), (value.stride(0), 1),
                value.storage_offset() + self.kv_rank,
            )

        def prepare_kv():
            nonlocal latent, suffix
            key = raw_key = None
            if natural_nvfp4 is not None:
                with profile_scope("RTP::attention.dsa.sfa.kv_proj"):
                    packed = self.natural_kv_input.forward_quantized(
                        natural_nvfp4.unpack(natural_nvfp4.wire),
                    )
                if index_cache is not None:
                    with profile_scope("RTP::attention.dsa.indexer.k_proj"):
                        raw_key = self.indexer.wk.forward_quantized(
                            natural_fp8.values, natural_fp8.scale_wire[:, :rows].T,
                        )
            elif natural_fp8 is None:
                with profile_scope("RTP::attention.dsa.sfa.kv_proj"):
                    packed = self.natural_kv_input(hidden)
            else:
                scales = natural_fp8.scale_wire[:, :rows].T
                if index_cache is not None:
                    with profile_scope("RTP::attention.dsa.sfa_kv_and_indexer_k_proj"):
                        packed, raw_key = self.natural_kv_indexer_input.forward_quantized(
                            natural_fp8.values, scales,
                        )
                    packed = packed[:, :self.kv_rank + self.suffix_dim]
                else:
                    with profile_scope("RTP::attention.dsa.sfa.kv_proj"):
                        packed = self.natural_kv_input.forward_quantized(natural_fp8.values, scales)
            latent, suffix = packed.split([self.kv_rank, self.suffix_dim], -1)
            if (raw_key is not None and rows >= 128
                    and (self.q_rank, self.kv_rank, raw_key.shape[1]) == (1536, 512, 128)
                    and self.suffix_dim <= self.kv_rank):
                from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
                    grouped_indexer_k_sfa_kv_norm,
                )

                norm = self.indexer.k_norm
                with profile_scope("RTP::attention.dsa.indexer_k_and_sfa_kv_norm"):
                    key, latent, suffix = grouped_indexer_k_sfa_kv_norm(
                        raw_key, latent, suffix, norm.weight, norm.beta, self.kv_norm.weight,
                        key_eps=norm.variance_epsilon, kv_eps=self.kv_norm.variance_epsilon,
                    )
            else:
                if raw_key is not None:
                    from rtp_llm.models_py.triton_kernels.common.layernorm_gated import layer_norm_fwd

                    norm = self.indexer.k_norm
                    with profile_scope("RTP::attention.dsa.indexer.k_layernorm"):
                        key, _, _ = layer_norm_fwd(
                            raw_key, norm.weight, norm.beta, norm.variance_epsilon,
                            out=torch.empty(raw_key.shape, dtype=raw_key.dtype, device=raw_key.device),
                        )
                if not joint_q_kv_norm:
                    with profile_scope("RTP::attention.dsa.sfa.kv_rmsnorm"):
                        latent = independent_q_kv_rmsnorm(
                            latent, self.kv_norm.weight, self.kv_norm.variance_epsilon,
                            q_width=self.q_rank, kv_width=self.kv_rank, full_rows=rows, suffix=suffix,
                        )
                    suffix = packed_suffix(latent)
            if index_cache is not None:
                if key is None:
                    with profile_scope("RTP::attention.dsa.indexer.k_proj_and_layernorm"):
                        key = self.indexer.project_key(hidden)
                with profile_scope("RTP::attention.dsa.indexer_k_cache_write"):
                    self.indexer.indexer_op.write_key(key, index_cache, index_fmha.fmha_params)
        prepare_index_query = index_cache is not None and index_fmha.is_sparse()
        query = gate = index_query = head_weights = None

        def project_queries(zigzag_hidden, activation_scales=None, *, nvfp4_input=None):
            nonlocal query, gate, index_query, head_weights, latent, suffix
            quantized = None
            with profile_scope(
                "RTP::attention.dsa.q_latent_and_output_gate_proj"
                if q_gate is not None else "RTP::attention.dsa.q_latent_proj"
            ):
                if nvfp4_input is not None:
                    q_input, gate = q_gate.forward_nvfp4(nvfp4_input)
                elif activation_scales is not None:
                    q_input, gate = q_gate.forward_quantized(zigzag_hidden, activation_scales)
                elif q_gate is not None:
                    q_input, gate = q_gate(zigzag_hidden)
                elif isinstance(self.natural_q_input, Fp8NaturalProjection):
                    quantized = self.natural_q_input.quantize_input(zigzag_hidden)
                    q_input = self.natural_q_input.forward_quantized(*quantized)
                else:
                    q_input = self.natural_q_input(zigzag_hidden)
            shared_q = fp4_q = None
            if (nvfp4_input is not None and getattr(self.q_b, "dynamic_input_scale", False)
                    and rows >= self.indexer.wq_b.FLASHINFER_M_THRESHOLD
                    and self.indexer.wq_b.scale_ue8m0):
                from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import (
                    rmsnorm_fp8_quant_ue8m0,
                )
                from rtp_llm.models_py.triton_kernels.common.nvfp4 import quantize_nvfp4_from_partials

                with profile_scope("RTP::attention.dsa.q_rmsnorm_fp8_nvfp4_amax"):
                    amax = q_input.new_empty(((q_input.shape[0] + 3) // 4,), dtype=torch.float32)
                    q, values, scales = rmsnorm_fp8_quant_ue8m0(
                        q_input, self.q_norm.weight, eps=self.q_norm.variance_epsilon,
                        amax_partials=amax,
                    )
                    shared_q = values, scales
                with profile_scope("RTP::attention.dsa.q_nvfp4_quant"):
                    fp4_q = quantize_nvfp4_from_partials(q, amax)
            elif joint_q_kv_norm:
                with profile_scope("RTP::attention.dsa.q_latent_and_sfa_kv_rmsnorm"):
                    q, latent = grouped_q_kv_rmsnorm(
                        q_input, latent, self.q_norm.weight, self.kv_norm.weight,
                        self.q_norm.variance_epsilon, kv_suffix=suffix,
                    )
                suffix = packed_suffix(latent)
            elif share_q_latent:
                from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import (
                    rmsnorm_fp8_quant_ue8m0,
                )

                with profile_scope("RTP::attention.dsa.q_latent_rmsnorm_quant"):
                    shared_q = rmsnorm_fp8_quant_ue8m0(
                        q_input, self.q_norm.weight, eps=self.q_norm.variance_epsilon,
                        store_norm=False,
                    )[1:]
            else:
                with profile_scope("RTP::attention.dsa.q_latent_rmsnorm"):
                    q = independent_q_kv_rmsnorm(
                        q_input, self.q_norm.weight, self.q_norm.variance_epsilon,
                        q_width=self.q_rank, kv_width=self.kv_rank, full_rows=rows,
                    )
            del q_input
            with profile_scope("RTP::attention.dsa.sfa.q_proj"):
                if fp4_q is not None:
                    query = self.q_b.forward_quantized(fp4_q).view(-1, self.heads, self.q_dim)
                elif shared_q is not None:
                    query = self.q_b.forward_quantized(*shared_q).view(-1, self.heads, self.q_dim)
                else:
                    query = self.q_b(q).reshape(-1, self.heads, self.q_dim)
            if prepare_index_query:
                with profile_scope("RTP::attention.dsa.indexer.q_proj"):
                    if shared_q is not None:
                        index_q = self.indexer.wq_b.forward_quantized(*shared_q).view(
                            -1, self.indexer.index_n_heads, self.indexer.index_head_dim,
                        )
                    else:
                        index_q = self.indexer.project_query(q, original_rows=rows)
            del shared_q, fp4_q
            if prepare_index_query and (activation_scales is not None or nvfp4_input is not None):
                index_query = torch.empty(index_q.shape, dtype=torch.float8_e4m3fn, device=hidden.device)
                head_weights = torch.empty((*index_q.shape[:2], 1), dtype=torch.float32, device=hidden.device)
                with profile_scope("RTP::attention.dsa.indexer.q_quant"):
                    self.indexer.indexer_op.encode_query_into(index_q, index_query, head_weights)
            elif prepare_index_query:
                with profile_scope("RTP::attention.dsa.indexer.q_quant"):
                    index_query, scale = self.indexer.indexer_op.encode_query(index_q)
                with profile_scope("RTP::attention.dsa.indexer.head_weights_proj_and_scale"):
                    head_weights = self.indexer.project_head_weights(
                        zigzag_hidden, scale, is_prefill=True, original_rows=rows,
                        tile_decode_output=not index_fmha.fmha_params.is_prefill,
                    )
            if q_gate is None:
                with profile_scope("RTP::attention.dsa.output_gate_proj"):
                    gate = (self.natural_gate_input.forward_quantized(*quantized)
                            if quantized is not None else self.natural_gate_input(zigzag_hidden))

        if natural_fp8 is None:
            project_queries(owner.row_exchange.exchange(
                hidden, owner.routing_plan, "to_zigzag", before_consume=prepare_kv,
            ))
        else:
            from rtp_llm.models_py.kernels.cuda.dsa_head_weights_fp32 import project_prefill_head_weights

            def project_heads(out):
                hidden.record_stream(torch.cuda.current_stream(hidden.device))
                with profile_scope("RTP::attention.dsa.indexer.head_weights_fp32_proj"):
                    project_prefill_head_weights(hidden, self.indexer._fp32_head_weight, out)

            if natural_nvfp4 is not None:
                consume = owner.row_exchange.consume_nvfp4
                activation = natural_nvfp4

                def project_nvfp4_queries(packed):
                    project_queries(None, nvfp4_input=packed)

                project = project_nvfp4_queries
            else:
                consume = owner.row_exchange.consume_fp8
                activation = natural_fp8
                project = project_queries
            raw_weights = consume(
                activation, owner.routing_plan, project,
                before_consume=prepare_kv,
                project_heads=project_heads if prepare_index_query else None,
            )
            if prepare_index_query:
                with profile_scope("RTP::attention.dsa.indexer.apply_q_scale_to_head_weights"):
                    raw_weights = raw_weights.unsqueeze(-1)
                    raw_weights.mul_(head_weights).mul_(self.indexer.softmax_scale * self.indexer.weights_scale)
                head_weights = raw_weights
        indices = run_prepared_sparse_indexer(
            self.indexer, index_query, head_weights, index_cache, index_fmha,
        )
        del index_query, head_weights
        with profile_scope("RTP::attention.dsa.sfa.sparse_attention"):
            output = sfa.forward(query, latent, suffix, sfa_cache, self.layer_idx, indices)
        if output is None:
            raise RuntimeError("K3.1 sparse attention returned no output")
        del query, latent, suffix, indices
        mask = None if layout.all_valid else layout.local_valid
        output = output.reshape(rows, self.heads * self.v_dim)
        exchange = owner.row_exchange
        if self._fp8_output_gate and getattr(exchange, "fp8_output_width", 0):
            from rtp_llm.models_py.triton_kernels.common.activation import sigmoid_mul_per_token_group_quant_fp8

            def quantize(target):
                with profile_scope("RTP::attention.dsa.output_gate_quant"):
                    sigmoid_mul_per_token_group_quant_fp8(output, gate, valid_rows=mask, out=target)

            def project(values, scales, target):
                with profile_scope("RTP::attention.dsa.output_proj"):
                    self.output.forward_quantized(values, scales, out=target)

            return exchange.route_fp8_output_and_project(
                owner.routing_plan, quantize, project,
            )
        with profile_scope(
            "RTP::attention.dsa.output_gate_quant"
            if self._fp8_output_gate else "RTP::attention.dsa.output_gate"
        ):
            if self._fp8_output_gate:
                from rtp_llm.models_py.triton_kernels.common.activation import sigmoid_mul_per_token_group_quant_fp8

                if natural_fp8 is not None:
                    values, scales = sigmoid_mul_per_token_group_quant_fp8(output, gate, valid_rows=mask)
                else:
                    if mask is not None:
                        output = torch.where(mask[:, None], output, 0)
                        gate = torch.where(mask[:, None], gate, 0)
                    values, scales = sigmoid_mul_per_token_group_quant_fp8(output, gate)
            else:
                output = gate_sigmoid_mul(output, gate, mask)
        del gate
        publish = getattr(exchange, "produce_and_exchange", None)
        if self._fp8_output_gate and publish is not None:
            tile_rows = 1024
            if (rows >= 8192 and rows % tile_rows == 0
                    and getattr(exchange, "can_pipeline_producer", False)):
                # Keep both packed scale and output views aligned, with at most
                # four submissions even for long Prefills.
                group_rows = 4 * tile_rows
                chunk_rows = ((rows + group_rows - 1) // group_rows) * tile_rows

                def produce_chunk(start, target):
                    end = start + target.shape[0]
                    with profile_scope("RTP::attention.dsa.output_proj"):
                        self.output.forward_quantized(values[start:end], scales[start:end], out=target)

                return exchange.produce_chunks_and_exchange(owner.routing_plan, chunk_rows, produce_chunk)

            def produce(target):
                with profile_scope("RTP::attention.dsa.output_proj"):
                    self.output.forward_quantized(values, scales, out=target)
            return publish(owner.routing_plan, "to_natural", produce)
        with profile_scope("RTP::attention.dsa.output_proj"):
            output = (self.output.forward_quantized(values, scales)
                      if self._fp8_output_gate else self.output(output))
        return exchange.exchange(output, owner.routing_plan, "to_natural")

    def forward(
        self, hidden, fmha, cache, attention_inputs=None, metadata=None, *,
        natural_fp8=None, natural_nvfp4=None,
    ):
        if getattr(metadata, "outer_layout", None) is not None:
            output = self._forward_natural(hidden, fmha, cache, metadata, natural_fp8, natural_nvfp4)
            next(iter(fmha.values())).workspace_owner.release_eager_buffers()
            return output
        local_query = (
            self.prefill_query_cp
            and getattr(metadata, "query_layout", None) is not None
        )
        use_bf16_collective = False
        if self.prologue != "dev" or (
            self._decode_nvfp4_ag is not None and not isinstance(hidden, torch.Tensor)
        ):
            output, gate = self._decode_prologue(hidden, fmha, cache)
        else:
            collective = self._bf16_collective
            indexer_key = None
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
                shared_min_rows = getattr(self, "_verify_shared_hidden_min_rows", None)
                if (
                    getattr(metadata, "is_target_verify", False)
                    and shared_min_rows is not None and full.shape[0] >= shared_min_rows
                    and cache is not None and cache.get("indexer_kv") is not None
                    and fmha["indexer_kv"].is_sparse()
                ):
                    with profile_scope("RTP::attention.mla.qkv_gate_input_proj"):
                        values, scales = self.input.quantize_collective_input(full)
                        projected = self.input.forward_quantized(values, scales)
                    with profile_scope("RTP::attention.dsa.indexer.k_proj_and_layernorm"):
                        indexer_key = self.indexer.k_norm(
                            self.indexer.wk.forward_quantized(values, scales)
                        )
                    del values, scales
                else:
                    with profile_scope("RTP::attention.mla.qkv_gate_input_proj"):
                        projected = self.input(full)
            qkv, _, gate = projected.split(
                [self.q_rank + self.kv_rank + self.suffix_dim,
                 self._input_padding, self.heads * self.v_dim],
                dim=-1,
            )
            del projected
            output = self._attend(
                qkv, fmha, cache, full,
                **({"indexer_key": indexer_key} if indexer_key is not None else {}),
            )
            del full, qkv, indexer_key
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
                if mask is not None and getattr(metadata, "is_target_verify", False):
                    values, scales = sigmoid_mul_per_token_group_quant_fp8(output, gate, valid_rows=mask)
                else:
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
