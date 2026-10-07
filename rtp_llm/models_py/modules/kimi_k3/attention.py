"""K3 projections over RTP's MLA and paged KDA implementations."""

from functools import lru_cache
from contextlib import nullcontext
import os

import torch
from torch import nn

from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.ops import RoleType
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    all_gather,
)
from rtp_llm.models_py.distributed.fp8_collective_projection import Fp8Activation
from rtp_llm.models_py.model_desc.kimi_linear import (
    KimiLinearKDADecode,
    KimiLinearKDAPrefill,
)
from rtp_llm.models_py.modules import LinearFactory, RMSNorm
from rtp_llm.models_py.modules.kimi_k3.collectives import reduce_scatter
from rtp_llm.models_py.modules.kimi_k3.linear import KimiK3Bf16Linear, KimiK3MlaLinear
from rtp_llm.models_py.modules.kimi_k3.native_gated_norm import KimiK3GatedNorm
from rtp_llm.models_py.modules.kimi_k3.native_mla_ops import (
    fused_q_kv_rmsnorm,
    gate_sigmoid_mul,
)
from rtp_llm.utils.model_weight import W

_PROFILE_MODEL_MODULES = os.environ.get("RTP_LLM_PROFILE_MODEL_MODULES", "0") == "1"
_NO_PROFILE_SCOPE = nullcontext()


def profile_scope(name):
    """Label model calls only during an explicitly requested profiling run."""
    return (
        torch.profiler.record_function(name)
        if _PROFILE_MODEL_MODULES
        else _NO_PROFILE_SCOPE
    )


def linear(weights, name, hardware=None):
    weight = weights[name]
    if weight.is_cuda and weight.dtype == torch.bfloat16:
        return KimiK3Bf16Linear(weight)
    if weight.dtype == torch.float8_e4m3fn:
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.models.kimi_k3.fp8_weight import KimiK3LoadFp8Weight

        scale_name = KimiK3LoadFp8Weight.w8a8_weight_list.get(name)
        if scale_name is None or scale_name not in weights:
            raise ValueError(f"K3 FP8 projection {name} requires its block scales")
        return LinearFactory.create_linear_from_weights(
            weights, name, scale_name, None,
            quant_config=Fp8BlockWiseQuantConfig(), hw_kernel_config=hardware,
        )
    return LinearFactory.create_linear_from_weights(
        weights, name, None, None, quant_config=None, hw_kernel_config=hardware
    )


class KimiK3KDA(nn.Module):
    def __init__(self, config, parallelism, weights, hardware=None):
        super().__init__()
        self._fp8_collective = None
        cfg = config.linear_attention_config
        runtime = config.k3_runtime_config
        if not runtime.kda_use_full_rank_gate:
            raise ValueError("K3 requires a full-rank output gate")
        self.tp_size = parallelism.tp_size
        self.heads = cfg.linear_num_value_heads // self.tp_size
        self.dim = cfg.linear_value_head_dim
        self.width = self.heads * self.dim
        self.input = linear(weights, K3W.KDA_INPUT, hardware)
        self._ag_local_fp8_overlap = (
            os.environ.get("RTP_LLM_NCCL_FP8_AG_LOCAL_OVERLAP", "0") == "1"
        )
        self._ag_local_fp8_overlap_min_rows = int(
            os.environ.get("RTP_LLM_NCCL_FP8_AG_LOCAL_OVERLAP_MIN_ROWS", "4096")
        )
        if self._ag_local_fp8_overlap:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
                CudaFp8GEMMLinear,
            )

            if not isinstance(self.input, CudaFp8GEMMLinear):
                raise ValueError("BF16 NCCL/FP8 local overlap requires a grouped FP8 input projection")
        self.f_b = linear(weights, W.linear_attn_f_b_w, hardware)
        self.output = linear(weights, W.linear_attn_out_w, hardware)
        self._fp8_output_norm = False
        if weights[W.linear_attn_out_w].dtype == torch.float8_e4m3fn:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
                CudaFp8GEMMLinear,
            )

            if not isinstance(self.output, CudaFp8GEMMLinear) or not self.output.scale_ue8m0:
                raise ValueError("FP8 KDA output requires grouped E4M3 GEMM with UE8M0 scales")
            self._fp8_output_norm = True
        forget_weight = weights[W.linear_attn_f_b_w]
        self.fa_width = forget_weight.shape[
            1 if forget_weight.dtype == torch.float8_e4m3fn else 0
        ]
        self._project_forget = self._project_forget_contiguous
        if parallelism.role_type == RoleType.DECODE and self.fa_width == 128:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
                CudaFp8GEMMLinear,
            )

            if isinstance(self.f_b, CudaFp8GEMMLinear) and self.f_b.scale_ue8m0:
                self._project_forget = self._project_forget_strided_fp8
        self.norm = KimiK3GatedNorm(
            weights[W.linear_attn_norm_w],
            eps=config.layernorm_eps,
        )
        backend = getattr(runtime, "kda_prefill_backend", "rtp")
        if backend in {"flashkda", "vllm_triton", "cula"}:
            from rtp_llm.models_py.modules.kimi_k3.native_kda_prefill import (
                KimiK3NativeKDAPrefill,
            )

            self.prefill = KimiK3NativeKDAPrefill(
                cfg, parallelism, weights, backend,
                use_paged_conv=self._fp8_output_norm and backend in {"flashkda", "cula"},
            )
        elif backend == "rtp":
            self.prefill = KimiLinearKDAPrefill(cfg, parallelism, weights)
        else:
            raise ValueError(f"Unsupported K3 KDA prefill backend: {backend}")
        # Native K3 convolves BF16 input/cache with FP32 checkpoint weights.
        # Keep the activation storage dtype instead of widening to weight dtype.
        self.prefill.preserve_conv_input_dtype = True
        # K3 padding rows reserve block zero; generic conv callers may use it.
        self.prefill.conv_reserved_cache_block_id = 0
        self.decode = KimiLinearKDADecode(cfg, parallelism, weights)
        # Preserve FP32 recurrence in block checkpoints instead of widening BF16 snapshots.
        self.prefill.intermediate_states_in_fp32 = True
        self.prefill.gate_lower_bound = runtime.kda_gate_lower_bound
        self.decode.gate_lower_bound = runtime.kda_gate_lower_bound

    def _project_forget_contiguous(self, latent):
        return self.f_b(latent.contiguous())

    def _project_forget_strided_fp8(self, latent):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel.fused_activation import (
            quantize_strided_group128_fp8,
        )

        values, scales = quantize_strided_group128_fp8(latent)
        return self.f_b.forward_quantized(values, scales)

    def forward(self, hidden, fmha, cache, attention_inputs, metadata):
        use_fused_ag = (
            self._fp8_collective is not None
            and self._fp8_collective.eligible_ag(hidden, attention_inputs, metadata)
        )
        use_overlap = (
            self._ag_local_fp8_overlap
            and self.tp_size > 1
            and hidden.shape[0] >= self._ag_local_fp8_overlap_min_rows
            and hidden.is_cuda
            and hidden.dtype == torch.bfloat16
            and attention_inputs.is_prefill
            and not metadata.is_target_verify
            and not torch.cuda.is_current_stream_capturing()
        )
        if isinstance(hidden, Fp8Activation) and not use_fused_ag:
            raise RuntimeError("prequantized FP8 input requires the FP8 AG/GEMM path")
        if use_fused_ag:
            with profile_scope("RTP::attention.fp8_ag_gemm"):
                fused = self._fp8_collective.all_gather_gemm(hidden, self.input)
        elif use_overlap:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.nccl_fp8_projection_overlap import (
                all_gather_project_local_overlap,
            )

            with profile_scope("RTP::attention.kda.ag_local_fp8_overlap"):
                fused = all_gather_project_local_overlap(
                    hidden, self.input, _get_group(Group.TP)
                )
        else:
            with profile_scope("RTP::attention.input_all_gather"):
                full_hidden = all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden
            with profile_scope("RTP::attention.kda.input_proj"):
                fused = self.input(full_hidden)
        logical_width = 4 * self.width + self.fa_width + self.heads
        if fused.shape[-1] < logical_width:
            raise ValueError(
                f"KDA input projection returned {fused.shape[-1]} columns, "
                f"expected at least {logical_width}"
            )
        fused = fused[..., :logical_width]
        qkv, gate, fa, beta = fused.split(
            [3 * self.width, self.width, self.fa_width, self.heads], dim=-1
        )
        with profile_scope("RTP::attention.kda.forget_proj"):
            forget = self._project_forget(fa)
        kernel = (
            self.prefill
            if attention_inputs.is_prefill and not metadata.is_target_verify
            else self.decode
        )
        with profile_scope("RTP::attention.kda.core"):
            output = kernel(
                qkv if kernel is self.prefill else qkv.contiguous(),
                forget, beta, attention_inputs, cache, metadata
            )
        valid_mask = attention_inputs.valid_token_mask
        if valid_mask is not None:
            # Paged KDA skips null-block rows, leaving their output unspecified.
            output = torch.where(valid_mask[:, None], output.reshape(-1, self.width), 0)
        with profile_scope("RTP::attention.kda.output_norm_quant"):
            if self._fp8_output_norm:
                from rtp_llm.models_py.kernels.cuda.fp8_kernel.fused_activation import (
                    rmsnorm_sigmoid_gate_per_token_group_quant_fp8,
                )

                values, scales = rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
                    output.reshape(-1, self.heads, self.dim),
                    gate.reshape(-1, self.heads, self.dim),
                    self.norm.weight,
                    self.norm.eps,
                )
            else:
                output = self.norm(
                    output.reshape(-1, self.dim), gate.reshape(-1, self.dim)
                )
        use_fused_rs = self._fp8_collective is not None and self._fp8_collective.eligible_rs(
            values.shape[0] if self._fp8_output_norm else 0,
            attention_inputs, metadata, self._fp8_output_norm,
        )
        if use_fused_rs:
            with profile_scope("RTP::attention.fp8_gemm_rs"):
                return self._fp8_collective.gemm_reduce_scatter(
                    values, scales, self.output
                )
        with profile_scope("RTP::attention.kda.output_proj"):
            output = (
                self.output.forward_quantized(values, scales)
                if self._fp8_output_norm
                else self.output(output.reshape(-1, self.width))
            )
        with profile_scope("RTP::attention.output_reduce_scatter"):
            return reduce_scatter(output, Group.TP) if self.tp_size > 1 else output


@lru_cache(None)
def _mla_aux_stream(device):
    return torch.cuda.Stream(device=device)


class KimiK3MLA(nn.Module):
    def __init__(self, config, parallelism, weights, layer_idx, hardware=None):
        super().__init__()
        self._fp8_collective = None
        cfg = config.attn_config
        if (
            not config.k3_runtime_config.mla_use_nope
            or not config.k3_runtime_config.mla_use_output_gate
        ):
            raise ValueError("K3 requires NoPE MLA with its output gate")
        self.tp_size = parallelism.tp_size
        self.heads = cfg.head_num // self.tp_size
        self.q_rank, self.kv_rank = cfg.q_lora_rank, cfg.kv_lora_rank
        self.suffix_dim = (
            cfg.rope_head_dim
        )  # Physical suffix remains present under NoPE.
        self.q_dim = cfg.nope_head_dim + cfg.rope_head_dim
        self.v_dim = cfg.v_head_dim
        self.layer_idx = layer_idx
        self.input = linear(weights, W.mla_fusedqkrope_w, hardware)
        self.q_b = linear(weights, W.mla_q_b_w, hardware)
        self.output = linear(weights, W.attn_o_w, hardware)
        self.q_norm = RMSNorm(weights[W.mla_q_a_ln_gamma], config.layernorm_eps)
        self.kv_norm = RMSNorm(weights[W.mla_kv_a_ln_gamma], config.layernorm_eps)
        self._fp8_output_gate = False
        if weights[W.attn_o_w].is_cuda and weights[W.attn_o_w].dtype == torch.float8_e4m3fn:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
                CudaFp8GEMMLinear,
            )

            self._fp8_output_gate = (
                isinstance(self.output, CudaFp8GEMMLinear)
                and self.output.scale_ue8m0
            )
        self._gate_stream = None
        weight = weights[W.mla_fusedqkrope_w]
        if (
            isinstance(self.input, KimiK3Bf16Linear)
            and weight.is_cuda and weight.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(weight.device) in ((10, 3), (10, 7))
            and (self.q_rank + self.kv_rank + self.suffix_dim,
                 self.heads * self.v_dim, weight.shape[0]) == (2112, 1536, 7168)
        ):
            # One allocation supports fused prefill and contiguous split views.
            self.input.weight = self.input.weight.contiguous()
            qkv_rows = self.q_rank + self.kv_rank + self.suffix_dim
            self.qkv_input = KimiK3MlaLinear(self.input.weight[:qkv_rows].t())
            self.gate_input = KimiK3MlaLinear(self.input.weight[qkv_rows:].t())
            self.q_b = KimiK3MlaLinear(weights[W.mla_q_b_w])
            self._gate_stream = _mla_aux_stream(weight.device)
            self._gate_start = torch.cuda.Event()
            self._gate_done = torch.cuda.Event()


    def _attend(self, qkv, fmha, cache):
        q, kv = qkv.split(
            [self.q_rank, self.kv_rank + self.suffix_dim], dim=-1
        )
        latent, suffix = kv.split([self.kv_rank, self.suffix_dim], dim=-1)
        with profile_scope("RTP::attention.mla.qkv_norm"):
            if q.is_cuda:
                q, latent = fused_q_kv_rmsnorm(
                    q,
                    latent,
                    self.q_norm.weight,
                    self.kv_norm.weight,
                    self.q_norm.variance_epsilon,
                )
            else:
                q, latent = self.q_norm(q.contiguous()), self.kv_norm(
                    latent.contiguous()
                )
        with profile_scope("RTP::attention.mla.q_proj"):
            q = self.q_b(q).reshape(-1, self.heads, self.q_dim)
        with profile_scope("RTP::attention.mla.core"):
            output = fmha.forward(q, latent, suffix, cache, self.layer_idx, None)
        if output is None:
            raise RuntimeError("K3 MLA backend returned no attention output")
        return output.reshape(-1, self.heads * self.v_dim)

    def forward(self, hidden, fmha, cache, attention_inputs=None, metadata=None):
        use_fused_ag = (
            self._fp8_collective is not None
            and self._fp8_collective.eligible_ag(hidden, attention_inputs, metadata)
        )
        if isinstance(hidden, Fp8Activation) and not use_fused_ag:
            raise RuntimeError("prequantized FP8 input requires the FP8 AG/GEMM path")
        if use_fused_ag:
            with profile_scope("RTP::attention.fp8_ag_gemm"):
                fused_input = self._fp8_collective.all_gather_gemm(hidden, self.input)
            full_hidden = None
        else:
            with profile_scope("RTP::attention.input_all_gather"):
                full_hidden = all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden
        qkv_rows = self.q_rank + self.kv_rank + self.suffix_dim
        if full_hidden is not None and self._gate_stream is not None and full_hidden.shape[0] < 512:
            # Native event fork/join: attention on current stream, gate on aux.
            self._gate_start.record()
            with profile_scope("RTP::attention.mla.qkv_input_proj"):
                qkv = self.qkv_input(full_hidden)
            output = self._attend(qkv, fmha, cache)
            with torch.cuda.stream(self._gate_stream):
                self._gate_start.wait()
                with profile_scope("RTP::attention.mla.gate_input_proj"):
                    gate = self.gate_input(full_hidden)
                self._gate_done.record()
            self._gate_done.wait()
        else:
            with profile_scope("RTP::attention.mla.qkv_gate_input_proj"):
                qkv, gate = (fused_input if use_fused_ag else self.input(full_hidden)).split(
                    [qkv_rows, self.heads * self.v_dim], dim=-1
                )
            # The projected tensors no longer depend on the gathered input.
            # Release it before MLA expands a historical KV chunk, as the
            # feat/k3_dev projection path does before entering attention.
            del full_hidden
            if use_fused_ag:
                del fused_input
            output = self._attend(qkv, fmha, cache)
        valid_mask = attention_inputs.valid_token_mask
        if valid_mask is not None:
            output = torch.where(valid_mask[:, None], output, 0)
        with profile_scope("RTP::attention.mla.output_gate_quant"):
            if self._fp8_output_gate:
                from rtp_llm.models_py.kernels.cuda.fp8_kernel.fused_activation import (
                    sigmoid_mul_per_token_group_quant_fp8,
                )

                values, scales = sigmoid_mul_per_token_group_quant_fp8(output, gate)
            else:
                output = (
                    gate_sigmoid_mul(output, gate)
                    if output.is_cuda
                    else output * gate.sigmoid()
                )
        use_fused_rs = self._fp8_collective is not None and self._fp8_collective.eligible_rs(
            values.shape[0] if self._fp8_output_gate else 0,
            attention_inputs, metadata, self._fp8_output_gate,
        )
        if use_fused_rs:
            with profile_scope("RTP::attention.fp8_gemm_rs"):
                return self._fp8_collective.gemm_reduce_scatter(
                    values, scales, self.output
                )
        with profile_scope("RTP::attention.mla.output_proj"):
            output = (
                self.output.forward_quantized(values, scales)
                if self._fp8_output_gate else self.output(output)
            )
        with profile_scope("RTP::attention.output_reduce_scatter"):
            return reduce_scatter(output, Group.TP) if self.tp_size > 1 else output
