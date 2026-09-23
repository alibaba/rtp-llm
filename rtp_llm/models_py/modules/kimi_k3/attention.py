"""K3 projections over RTP's MLA and paged KDA implementations."""

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
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.utils.model_weight import W


def linear(weights, name, hardware=None):
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
        self.fa_width = weights[W.linear_attn_f_b_w].shape[0]
        self.norm = RmsNormGated(
            weights[W.linear_attn_norm_w],
            eps=config.layernorm_eps,
            group_size=self.dim,
            activation="sigmoid",
        )
        backend = getattr(runtime, "kda_prefill_backend", "rtp")
        if backend == "flashkda":
            from rtp_llm.models_py.modules.kimi_k3.native_kda_prefill import (
                KimiK3FlashKDAPrefill,
            )

            self.prefill = KimiK3FlashKDAPrefill(cfg, parallelism, weights)
        elif backend == "rtp":
            self.prefill = KimiLinearKDAPrefill(cfg, parallelism, weights)
        else:
            raise ValueError(f"Unsupported K3 KDA prefill backend: {backend}")
        self.decode = KimiLinearKDADecode(cfg, parallelism, weights)
        # Preserve FP32 recurrence in block checkpoints instead of widening BF16 snapshots.
        self.prefill.intermediate_states_in_fp32 = True
        self.prefill.gate_lower_bound = runtime.kda_gate_lower_bound
        self.decode.gate_lower_bound = runtime.kda_gate_lower_bound

    def forward(self, hidden, fmha, cache, attention_inputs, metadata):
        full_hidden = all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden
        fused = self.input(full_hidden)
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
            qkv.contiguous(), forget, beta, attention_inputs, cache, metadata
        )
        valid_mask = attention_inputs.valid_token_mask
        if valid_mask is not None:
            # Paged KDA skips null-block rows, leaving their output unspecified.
            output = torch.where(valid_mask[:, None], output.reshape(-1, self.width), 0)
        output = self.norm(output.reshape(-1, self.dim), gate.reshape(-1, self.dim))
        output = self.output(output.reshape(-1, self.width))
        return reduce_scatter(output, Group.TP) if self.tp_size > 1 else output


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

    def forward(self, hidden, fmha, cache, attention_inputs=None, metadata=None):
        full_hidden = all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden
        q, kv, gate = self.input(full_hidden).split(
            [self.q_rank, self.kv_rank + self.suffix_dim, self.heads * self.v_dim],
            dim=-1,
        )
        q = self.q_b(self.q_norm(q.contiguous())).reshape(-1, self.heads, self.q_dim)
        latent, suffix = kv.split([self.kv_rank, self.suffix_dim], dim=-1)
        output = fmha.forward(
            q, self.kv_norm(latent.contiguous()), suffix, cache, self.layer_idx, None
        )
        if output is None:
            raise RuntimeError("K3 MLA backend returned no attention output")
        output = output.reshape(-1, self.heads * self.v_dim)
        valid_mask = attention_inputs.valid_token_mask
        if valid_mask is not None:
            output = torch.where(valid_mask[:, None], output, 0)
        output = self.output(output * gate.sigmoid())
        return reduce_scatter(output, Group.TP) if self.tp_size > 1 else output
