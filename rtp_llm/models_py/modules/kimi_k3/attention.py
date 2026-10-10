"""K3 projections over RTP's MLA and paged KDA implementations."""

import os
from functools import lru_cache

import torch
from torch import nn

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    all_gather,
    Fp8Activation,
    reduce_scatter_bf16 as reduce_scatter,
)
from rtp_llm.models_py.modules import KdaAttention, KdaDecode, KdaPrefill, RMSNorm
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.triton_kernels.common.activation import gate_sigmoid_mul
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
    FusedRMSNormGated,
    fused_q_kv_rmsnorm,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


class KimiK3KDA(KdaAttention):
    """Bind K3 weights and cache policy to the shared packed KDA pipeline."""

    def __init__(self, config, parallelism, weights, hardware=None):
        cfg = config.linear_attention_config
        runtime = config.k3_runtime_config
        if not runtime.kda_use_full_rank_gate:
            raise ValueError("K3 requires a full-rank output gate")
        projections = []
        for weight_key, scale_key in (
            (K3W.KDA_INPUT, K3W.KDA_INPUT_S),
            (W.linear_attn_f_b_w, K3W.KDA_FORGET_S),
            (W.linear_attn_out_w, W.linear_attn_out_s),
        ):
            weight = weights[weight_key]
            fp8 = weight.dtype == torch.float8_e4m3fn
            projections.append(
                LinearFactory.create_linear(
                    weight,
                    bias=None,
                    weight_scales=weights[scale_key] if fp8 else None,
                    quant_config=Fp8BlockWiseQuantConfig() if fp8 else None,
                    hw_kernel_config=hardware,
                )
            )
        input_projection, forget_projection, output_projection = projections
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )

        overlap = os.environ.get("RTP_LLM_NCCL_FP8_AG_LOCAL_OVERLAP", "0") == "1"
        if overlap and not isinstance(input_projection, CudaFp8GEMMLinear):
            raise ValueError(
                "BF16 NCCL/FP8 local overlap requires a grouped FP8 input projection"
            )
        fp8_output = weights[W.linear_attn_out_w].dtype == torch.float8_e4m3fn
        if fp8_output and (
            not isinstance(output_projection, CudaFp8GEMMLinear)
            or not output_projection.scale_ue8m0
        ):
            raise ValueError("FP8 KDA output requires grouped E4M3 GEMM with UE8M0 scales")
        forget_weight = weights[W.linear_attn_f_b_w]
        fa_width = forget_weight.shape[
            1 if forget_weight.dtype == torch.float8_e4m3fn else 0
        ]
        prefill = KdaPrefill(
            cfg,
            parallelism,
            weights,
            use_paged_conv=True,
        )
        decode = KdaDecode(cfg, parallelism, weights)
        # K3 keeps BF16 convolution storage and FP32 recurrent checkpoints.
        prefill.gate_lower_bound = runtime.kda_gate_lower_bound
        decode.gate_lower_bound = runtime.kda_gate_lower_bound
        super().__init__(
            input_projection=input_projection,
            forget_projection=forget_projection,
            output_projection=output_projection,
            fa_width=fa_width,
            norm=FusedRMSNormGated(
                weights[W.linear_attn_norm_w], eps=config.layernorm_eps
            ),
            prefill=prefill,
            decode=decode,
            fp8_output_norm=fp8_output,
            strided_fp8_forget=(
                parallelism.role_type == RoleType.DECODE
                and fa_width == 128
                and isinstance(forget_projection, CudaFp8GEMMLinear)
                and forget_projection.scale_ue8m0
            ),
            ag_local_fp8_overlap=overlap,
            ag_local_fp8_overlap_min_rows=int(
                os.environ.get("RTP_LLM_NCCL_FP8_AG_LOCAL_OVERLAP_MIN_ROWS", "4096")
            ),
        )


@lru_cache(None)
def _mla_aux_stream(device):
    return torch.cuda.Stream(device=device)


class KimiK3MLA(nn.Module):
    def __init__(self, config, parallelism, weights, layer_idx, hardware=None):
        super().__init__()
        self._fp8_collective = None
        self._mtp_bf16_collectives = None
        cfg = config.attn_config
        if (
            not config.k3_runtime_config.mla_use_nope
            or not config.k3_runtime_config.mla_use_output_gate
        ):
            raise ValueError("K3 requires NoPE MLA with its output gate")
        self.tp_size = parallelism.tp_size
        self.heads = cfg.head_num // self.tp_size
        self.query_heads = (
            cfg.head_num if parallelism.role_type == RoleType.DECODE else self.heads
        )
        self.q_rank, self.kv_rank = cfg.q_lora_rank, cfg.kv_lora_rank
        self.suffix_dim = (
            cfg.rope_head_dim
        )  # Physical suffix remains present under NoPE.
        self.q_dim = cfg.nope_head_dim + cfg.rope_head_dim
        self.v_dim = cfg.v_head_dim
        self.layer_idx = layer_idx
        projections = []
        for weight_key, scale_key in (
            (W.mla_fusedqkrope_w, W.mla_fusedqkrope_s),
            (W.mla_q_b_w, W.mla_q_b_s),
            (W.attn_o_w, W.attn_o_s),
        ):
            weight = weights[weight_key]
            fp8 = weight.dtype == torch.float8_e4m3fn
            projections.append(
                LinearFactory.create_linear(
                    weight,
                    bias=None,
                    weight_scales=weights[scale_key] if fp8 else None,
                    quant_config=Fp8BlockWiseQuantConfig() if fp8 else None,
                    hw_kernel_config=hardware,
                )
            )
        self.input, self.q_b, self.output = projections
        self.q_norm = RMSNorm(weights[W.mla_q_a_ln_gamma], config.layernorm_eps)
        self.kv_norm = RMSNorm(weights[W.mla_kv_a_ln_gamma], config.layernorm_eps)
        self._fp8_output_gate = False
        if (
            weights[W.attn_o_w].is_cuda
            and weights[W.attn_o_w].dtype == torch.float8_e4m3fn
        ):
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
                CudaFp8GEMMLinear,
            )

            self._fp8_output_gate = (
                isinstance(self.output, CudaFp8GEMMLinear) and self.output.scale_ue8m0
            )
        self._gate_stream = None
        weight = weights[W.mla_fusedqkrope_w]
        if (
            isinstance(self.input, CudaF16Linear)
            and weight.is_cuda
            and weight.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(weight.device) in ((10, 3), (10, 7))
            and (
                self.q_rank + self.kv_rank + self.suffix_dim,
                self.heads * self.v_dim,
                weight.shape[0],
            )
            == (2112, 1536, 7168)
        ):
            # One allocation supports fused prefill and contiguous split views.
            self.input.weight = self.input.weight.contiguous()
            qkv_rows = self.q_rank + self.kv_rank + self.suffix_dim
            self.qkv_input = CudaF16Linear(self.input.weight[:qkv_rows].t())
            self.gate_input = CudaF16Linear(self.input.weight[qkv_rows:].t())
            self.q_b = CudaF16Linear(weights[W.mla_q_b_w])
            self._gate_stream = _mla_aux_stream(weight.device)
            self._gate_start = torch.cuda.Event()
            self._gate_done = torch.cuda.Event()

    def _attend(self, qkv, fmha, cache):
        q, kv = qkv.split([self.q_rank, self.kv_rank + self.suffix_dim], dim=-1)
        latent, suffix = kv.split([self.kv_rank, self.suffix_dim], dim=-1)
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
        q = self.q_b(q).reshape(-1, self.query_heads, self.q_dim)
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
            fused_input = self._fp8_collective.all_gather_gemm(hidden, self.input)
            full_hidden = None
        else:
            full_hidden = (
                self._mtp_bf16_collectives.all_gather(hidden)
                if self._mtp_bf16_collectives is not None
                else all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden
            )
        qkv_rows = self.q_rank + self.kv_rank + self.suffix_dim
        if (
            full_hidden is not None
            and self._gate_stream is not None
            and full_hidden.shape[0] < 512
        ):
            # Native event fork/join: attention on current stream, gate on aux.
            self._gate_start.record()
            qkv = self.qkv_input(full_hidden)
            output = self._attend(qkv, fmha, cache)
            with torch.cuda.stream(self._gate_stream):
                self._gate_start.wait()
                gate = self.gate_input(full_hidden)
                self._gate_done.record()
            self._gate_done.wait()
        else:
            qkv, gate = (
                fused_input if use_fused_ag else self.input(full_hidden)
            ).split([qkv_rows, self.heads * self.v_dim], dim=-1)
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
        if self._fp8_output_gate:
            from rtp_llm.models_py.triton_kernels.common.activation import (
                sigmoid_mul_per_token_group_quant_fp8,
            )

            values, scales = sigmoid_mul_per_token_group_quant_fp8(output, gate)
        else:
            output = (
                gate_sigmoid_mul(output, gate)
                if output.is_cuda
                else output * gate.sigmoid()
            )
        use_fused_rs = (
            self._fp8_collective is not None
            and self._fp8_collective.eligible_rs(
                values.shape[0] if self._fp8_output_gate else 0,
                attention_inputs,
                metadata,
                self._fp8_output_gate,
            )
        )
        if use_fused_rs:
            return self._fp8_collective.gemm_reduce_scatter(
                values, scales, self.output
            )
        output = (
            self.output.forward_quantized(values, scales)
            if self._fp8_output_gate
            else self.output(output)
        )
        return (
            self._mtp_bf16_collectives.reduce_scatter(output)
            if self._mtp_bf16_collectives is not None
            else reduce_scatter(output, Group.TP) if self.tp_size > 1 else output
        )
