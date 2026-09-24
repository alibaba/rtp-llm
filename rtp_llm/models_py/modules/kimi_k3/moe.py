"""K3 latent and shared branches composed with RTP's fused-MoE factory."""

import torch
from torch import nn

from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from .norm import KimiK3LatentRMSNorm
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.ops import MoeConfig
from .attention import linear
from .router import KimiK3RouterProjection
from .routing import grouped_topk
from .moe_backend import get_k3_moe_backend
from .linear import KimiK3Bf16Linear, bf16_linear


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
        self.down = linear(weights, K3W.MOE_ROUTED_DOWN, hardware)
        self.up = linear(weights, K3W.MOE_ROUTED_UP, hardware)
        self.norm = (
            KimiK3LatentRMSNorm(weights[K3W.MOE_ROUTED_NORM], config.layernorm_eps)
            if runtime.latent_moe_use_norm
            else None
        )
        self.shared_gate_up = weights[K3W.MOE_SHARED_GATE_UP]
        self.shared_down = linear(weights, K3W.MOE_SHARED_DOWN, hardware)
        if self.shared_gate_up.shape != (2 * config.inter_size, config.hidden_size):
            raise ValueError(
                "K3 shared projections must be replicated over SP token owners"
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
        routed = self.experts(self.down(hidden), routing, ids, activation="situ")
        if self.norm is not None:
            routed = self.norm(routed.contiguous())
        gate, up = bf16_linear(hidden, self.shared_gate_up).chunk(2, dim=-1)
        shared = self.shared_down(situ(gate, up, self.beta, self.linear_beta))
        if isinstance(self.up, KimiK3Bf16Linear):
            # Match native K3: combine the routed projection and shared
            # output in a single addmm GEMM call.
            return self.up(routed, residual=shared)
        return self.up(routed) + shared
