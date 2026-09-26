"""K3 projections over RTP's MLA and paged KDA implementations."""

from functools import lru_cache

import torch
from torch import nn

from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
)
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
        cfg = config.linear_attention_config
        runtime = config.k3_runtime_config
        if not runtime.kda_use_full_rank_gate:
            raise ValueError("K3 requires a full-rank output gate")
        self.tp_size = parallelism.tp_size
        self.heads = cfg.linear_num_value_heads // self.tp_size
        self.dim = cfg.linear_value_head_dim
        self.width = self.heads * self.dim
        self.input = linear(weights, K3W.KDA_INPUT, hardware)
        self.f_b = linear(weights, W.linear_attn_f_b_w, hardware)
        self.output = linear(weights, W.linear_attn_out_w, hardware)
        forget_weight = weights[W.linear_attn_f_b_w]
        self.fa_width = forget_weight.shape[
            1 if forget_weight.dtype == torch.float8_e4m3fn else 0
        ]
        self.norm = KimiK3GatedNorm(
            weights[W.linear_attn_norm_w],
            eps=config.layernorm_eps,
        )
        backend = getattr(runtime, "kda_prefill_backend", "rtp")
        if backend in {"flashkda", "vllm_triton", "cula"}:
            from rtp_llm.models_py.modules.kimi_k3.native_kda_prefill import (
                KimiK3NativeKDAPrefill,
            )

            self.prefill = KimiK3NativeKDAPrefill(cfg, parallelism, weights, backend)
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

    def forward(self, hidden, fmha, cache, attention_inputs, metadata):
        full_hidden = all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden
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
        forget = self.f_b(fa.contiguous())
        kernel = (
            self.prefill
            if attention_inputs.is_prefill and not metadata.is_target_verify
            else self.decode
        )
        output = kernel(
            qkv if kernel is self.prefill else qkv.contiguous(),
            forget, beta, attention_inputs, cache, metadata
        )
        valid_mask = attention_inputs.valid_token_mask
        if valid_mask is not None:
            # Paged KDA skips null-block rows, leaving their output unspecified.
            output = torch.where(valid_mask[:, None], output.reshape(-1, self.width), 0)
        output = self.norm(output.reshape(-1, self.dim), gate.reshape(-1, self.dim))
        output = self.output(output.reshape(-1, self.width))
        return reduce_scatter(output, Group.TP) if self.tp_size > 1 else output


@lru_cache(None)
def _mla_aux_stream(device):
    return torch.cuda.Stream(device=device)


class KimiK3MLA(nn.Module):
    def __init__(self, config, parallelism, weights, layer_idx, hardware=None):
        super().__init__()
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
        if q.is_cuda:
            q, latent = fused_q_kv_rmsnorm(
                q,
                latent,
                self.q_norm.weight,
                self.kv_norm.weight,
                self.q_norm.variance_epsilon,
            )
        else:
            q, latent = self.q_norm(q.contiguous()), self.kv_norm(latent.contiguous())
        q = self.q_b(q).reshape(-1, self.heads, self.q_dim)
        output = fmha.forward(q, latent, suffix, cache, self.layer_idx, None)
        if output is None:
            raise RuntimeError("K3 MLA backend returned no attention output")
        return output.reshape(-1, self.heads * self.v_dim)

    def forward(self, hidden, fmha, cache, attention_inputs=None, metadata=None):
        full_hidden = all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden
        qkv_rows = self.q_rank + self.kv_rank + self.suffix_dim
        if self._gate_stream is not None and full_hidden.shape[0] < 512:
            # Native event fork/join: attention on current stream, gate on aux.
            self._gate_start.record()
            output = self._attend(self.qkv_input(full_hidden), fmha, cache)
            with torch.cuda.stream(self._gate_stream):
                self._gate_start.wait()
                gate = self.gate_input(full_hidden)
                self._gate_done.record()
            self._gate_done.wait()
        else:
            qkv, gate = self.input(full_hidden).split(
                [qkv_rows, self.heads * self.v_dim], dim=-1
            )
            output = self._attend(qkv, fmha, cache)
        valid_mask = attention_inputs.valid_token_mask
        if valid_mask is not None:
            output = torch.where(valid_mask[:, None], output, 0)
        output = self.output(
            gate_sigmoid_mul(output, gate) if output.is_cuda else output * gate.sigmoid()
        )
        return reduce_scatter(output, Group.TP) if self.tp_size > 1 else output
