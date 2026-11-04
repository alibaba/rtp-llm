"""K3.1 sparse policy composed from shared K3 projections and Indexer kernels."""

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group, all_gather, reduce_scatter_bf16,
)
from rtp_llm.models_py.mla_routes import resolve_mla_routes, run_sparse_indexer
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.models_py.modules.kimi_k3.attention import KimiK3MLA
from rtp_llm.models_py.triton_kernels.common.activation import gate_sigmoid_mul
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
    fused_q_kv_rmsnorm, grouped_q_kv_rmsnorm,
)
from rtp_llm.models_py.utils.module_profiler import profile_scope
from rtp_llm.ops import RoleType


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

    def forward(self, hidden, fmha, cache, attention_inputs=None, metadata=None):
        local_query = (
            self.prefill_query_cp
            and getattr(metadata, "query_layout", None) is not None
        )
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
