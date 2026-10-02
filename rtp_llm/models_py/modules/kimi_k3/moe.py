"""K3 latent and shared branches composed with RTP's fused-MoE factory."""

import torch
from torch import nn

from rtp_llm.models.kimi_k3.kimi_k3_weight import (
    KimiK3WeightNames as K3W,
    shared_expert_weight_shard_enabled,
)
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from .norm import KimiK3LatentRMSNorm
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.ops import MoeConfig
from .attention import linear, profile_scope
from .router import KimiK3RouterProjection
from .routing import grouped_topk
from .moe_backend import get_k3_moe_backend
from .linear import (
    KimiK3Bf16Linear,
    KimiK3LatentDownLinear,
    bf16_linear,
    bf16_linear_add_inplace,
)
from rtp_llm.models_py.triton_kernels.common.activation import situ_and_mul


def situ(gate, up, beta, linear_beta, *, inplace=False):
    if gate.is_cuda:
        from rtp_llm.models_py.triton_kernels.common.situ import situ as fused_situ

        return fused_situ(
            gate, up, beta, linear_beta, inplace=inplace, native_k3=True
        )
    g, u = gate.float(), up.float()
    g = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    result = (g * u).to(gate.dtype)
    return gate.copy_(result) if inplace else result


class KimiK3LatentMoE(nn.Module):
    def __init__(
        self,
        config,
        parallelism,
        weights,
        layer_idx,
        moe_config,
        hardware,
        moe_capacity,
    ):
        super().__init__()
        runtime = config.k3_runtime_config
        self.beta, self.linear_beta = (
            runtime.activation_situ_beta,
            runtime.activation_situ_linear_beta,
        )
        self.top_k, self.groups, self.top_groups = (
            config.moe_k,
            config.moe_n_group,
            config.moe_topk_group,
        )
        self.renormalize, self.route_scale = (
            config.has_moe_norm,
            config.routed_scaling_factor,
        )
        self.router = KimiK3RouterProjection(weights[K3W.MOE_GATE])
        self.correction = weights[K3W.MOE_CORRECTION_BIAS].float()
        down_weight = weights[K3W.MOE_ROUTED_DOWN]
        self.down = (
            KimiK3LatentDownLinear(down_weight)
            if down_weight.is_cuda and down_weight.dtype == torch.bfloat16
            else linear(weights, K3W.MOE_ROUTED_DOWN, hardware)
        )
        self.up = linear(weights, K3W.MOE_ROUTED_UP, hardware)
        self.norm = (
            KimiK3LatentRMSNorm(weights[K3W.MOE_ROUTED_NORM], config.layernorm_eps)
            if runtime.latent_moe_use_norm
            else None
        )
        self.shared_gate_up = weights[K3W.MOE_SHARED_GATE_UP]
        self.shared_expert_weight_shard = shared_expert_weight_shard_enabled(
            parallelism.role_type
        )
        self.shared_weight_tp_size = int(parallelism.tp_size)
        if self.shared_expert_weight_shard:
            if self.shared_weight_tp_size <= 0 or self.shared_weight_tp_size % 2:
                raise ValueError("shared expert sharding requires an even TP size")
            if config.inter_size % self.shared_weight_tp_size:
                raise ValueError("shared expert intermediate size must divide TP")
            expected_gate_up = (2 * config.inter_size // self.shared_weight_tp_size, config.hidden_size)
            expected_down = (config.inter_size // self.shared_weight_tp_size, config.hidden_size)
            self.shared_down_weight = weights[K3W.MOE_SHARED_DOWN]
            self.shared_down = None
        else:
            expected_gate_up = (2 * config.inter_size, config.hidden_size)
            expected_down = (config.inter_size, config.hidden_size)
            self.shared_down_weight = None
            self.shared_down = linear(weights, K3W.MOE_SHARED_DOWN, hardware)
        if self.shared_gate_up.shape != expected_gate_up or weights[K3W.MOE_SHARED_DOWN].shape != expected_down:
            raise ValueError(
                "K3 shared projection layout does not match the FFN TP placement: "
                f"gate/up={tuple(self.shared_gate_up.shape)} expected={expected_gate_up}, "
                f"down={tuple(weights[K3W.MOE_SHARED_DOWN].shape)} expected={expected_down}"
            )
        cfg = MoEConfigAdapter(
            config,
            parallelism,
            moe_config or MoeConfig(),
            enable_cuda_graph=bool(hardware and hardware.enable_cuda_graph),
            expert_activation="situ",
            activation_beta=self.beta,
            activation_linear_beta=self.linear_beta,
            mega_moe_backend=get_k3_moe_backend(),
        )
        if cfg.moe_strategy not in ("auto", "mega_moe"):
            raise ValueError("K3 native MXFP4 requires the MegaMoE executor")
        cfg.moe_strategy = "mega_moe"
        cfg.max_tokens_per_rank = max(int(moe_capacity), 1)
        cfg.moe_quant_method = "FP8_FP4"
        cfg.dim = cfg.hidden_size = runtime.routed_expert_hidden_size
        cfg.layer_id = layer_idx
        cfg.moe_w1_layout = "gate_up"
        packed = {
            W.moe_w1: torch.cat(
                (weights.pop(K3W.MOE_W1_PACKED), weights.pop(K3W.MOE_W3_PACKED)), dim=1
            ).view(torch.int8),
            W.moe_s1: torch.cat(
                (weights.pop(K3W.MOE_W1_SCALE), weights.pop(K3W.MOE_W3_SCALE)), dim=1
            ).view(torch.float8_e8m0fnu),
            W.moe_w2: weights.pop(K3W.MOE_W2_PACKED).view(torch.int8),
            W.moe_s2: weights.pop(K3W.MOE_W2_SCALE).view(torch.float8_e8m0fnu),
        }
        self.experts = FusedMoeFactory().create_fused_moe(cfg, packed)

    def forward(self, hidden, valid_mask=None):
        with profile_scope("RTP::moe.router"):
            routing, ids = grouped_topk(
                self.router(hidden),
                self.correction,
                top_k=self.top_k,
                groups=self.groups,
                top_groups=self.top_groups,
                renormalize=self.renormalize,
                scale=self.route_scale,
            )
        if valid_mask is not None:
            ids = torch.where(valid_mask[:, None], ids, 0)
            routing = torch.where(valid_mask[:, None], routing, 0)
        with profile_scope("RTP::moe.routed_down_proj"):
            routed_input = self.down(hidden)
        with profile_scope("RTP::moe.routed_experts"):
            routed = self.experts(routed_input, routing, ids, activation="situ")
        if self.norm is not None:
            with profile_scope("RTP::moe.routed_norm"):
                routed = self.norm(routed.contiguous())
        with profile_scope("RTP::moe.shared_gate_up_proj"):
            gate_up_weight = self.shared_gate_up
            if self.shared_expert_weight_shard:
                gate_up_weight = all_gather(gate_up_weight.contiguous(), group=Group.TP)
            gate_up = bf16_linear(hidden, gate_up_weight)
            if self.shared_expert_weight_shard:
                del gate_up_weight
            gate, up = gate_up.chunk(2, dim=-1)
        with profile_scope("RTP::moe.shared_activation"):
            shared_input = situ_and_mul(gate, up, self.beta, self.linear_beta)
            del gate, up, gate_up
        with profile_scope("RTP::moe.shared_down_proj"):
            if self.shared_expert_weight_shard:
                down_weight = all_gather(self.shared_down_weight.contiguous(), group=Group.TP)
                shared = torch.matmul(shared_input, down_weight)
                del down_weight
            else:
                shared = self.shared_down(shared_input)
            del shared_input
        if isinstance(self.up, KimiK3Bf16Linear):
            # Match native K3: combine the routed projection and shared
            # output in a single addmm GEMM call.
            with profile_scope("RTP::moe.routed_up_proj_add_shared"):
                if (
                    routed.is_cuda
                    and routed.ndim == 2
                    and routed.shape[0] >= 4096
                    and routed.dtype == self.up.weight.dtype == shared.dtype == torch.bfloat16
                    and shared.is_contiguous()
                    and self.up.bias is None
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    return bf16_linear_add_inplace(routed, self.up.weight, shared)
                return self.up(routed, residual=shared)
        with profile_scope("RTP::moe.routed_up_proj"):
            routed_up = self.up(routed)
        return routed_up + shared
