"""Explicit local V4.1 expert strategy; distributed transports are separate."""

import torch

from rtp_llm.models_py.modules.dsv4.v41_expert import V41_EXPERT_RECIPE
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import _scale_bytes
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts, _fp4_bytes

from .base import RoutedExpertsStrategy, register_strategy


@register_strategy
class V41GroupedFP4Strategy(RoutedExpertsStrategy):
    name = "v41_grouped_fp4"
    expert_recipe = V41_EXPERT_RECIPE

    @classmethod
    def can_handle(cls, cfg):
        return (
            getattr(cfg, "expert_recipe", None) == cls.expert_recipe
            and cfg.ep_size == 1
            and cfg.n_local_experts == cfg.n_routed_experts
            and cfg.dim % 128 == cfg.moe_inter_dim % 128 == 0
            and min(cfg.dim, cfg.moe_inter_dim) >= 128
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (12, 0)
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.ep_size != 1 or cfg.n_local_experts != cfg.n_routed_experts:
            raise ValueError(
                "V4.1 grouped strategy is local; EP requires an explicit transport"
            )
        if cfg.max_tokens_per_rank <= 0 or cfg.n_activated_experts <= 0:
            raise ValueError(
                "V4.1 grouped strategy requires positive token/top-k bounds"
            )
        self.local = None

    def setup_weights(self, layer_weights):
        from rtp_llm.utils.model_weight import W

        if self.local is not None:
            raise RuntimeError("V4.1 routed weights are already initialized")
        cfg = self.cfg
        names = {
            "w1_w": W.v4_routed_w1_w,
            "w1_s": W.v4_routed_w1_s,
            "w3_w": W.v4_routed_w3_w,
            "w3_s": W.v4_routed_w3_s,
            "w2_w": W.v4_routed_w2_w,
            "w2_s": W.v4_routed_w2_s,
        }
        tensors = {name: layer_weights[key] for name, key in names.items()}
        device = tensors["w1_w"].device
        for projection, n, k in (
            ("w1", cfg.moe_inter_dim, cfg.dim),
            ("w3", cfg.moe_inter_dim, cfg.dim),
            ("w2", cfg.dim, cfg.moe_inter_dim),
        ):
            weight, scale = tensors[projection + "_w"], tensors[projection + "_s"]
            if (
                weight.shape != (cfg.n_local_experts, n, k // 2)
                or scale.shape != (cfg.n_local_experts, n, k // 32)
                or weight.device != device
                or scale.device != device
                or not weight.is_contiguous()
                or not scale.is_contiguous()
            ):
                raise ValueError(
                    "V4.1 routed loader tensors disagree with the local expert shard"
                )
            tensors[projection + "_w"] = _fp4_bytes(weight)
            tensors[projection + "_s"] = _scale_bytes(scale)
        experts = V41RoutedExperts(
            {
                "w13_w": torch.cat((tensors["w1_w"], tensors["w3_w"]), dim=1),
                "w13_s": torch.cat((tensors["w1_s"], tensors["w3_s"]), dim=1),
                "w2_w": tensors["w2_w"],
                "w2_s": tensors["w2_s"],
            },
            swiglu_limit=cfg.swiglu_limit,
        )
        local = V41LocalMoE(
            experts,
            expert_start_id=cfg.local_expert_start,
            max_tokens=cfg.max_tokens_per_rank,
            masked_token_limit=min(128, cfg.max_tokens_per_rank),
            max_topk=cfg.n_activated_experts,
        )
        for key in names.values():
            layer_weights.pop(key)
        self.local = local

    def forward(self, x, weights, indices):
        if self.local is None:
            raise RuntimeError("V4.1 routed weights are not initialized")
        layout = (
            "masked" if x.shape[0] <= self.local.masked_token_limit else "contiguous"
        )
        return self.local(x, weights, indices, layout=layout)
