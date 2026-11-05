"""Shared NVFP4 weight views without backend runner initialization."""

import torch

from rtp_llm.utils.model_weight import W


class Nvfp4WeightViews:
    def __init__(self, config, quant_config, weights):
        self.config = config
        self.quant_config = quant_config
        self.weights = weights
        # RTP's stacked MoE descriptor already emits [up, gate].
        self.w1 = weights[W.moe_w1].contiguous().view(torch.int64)
        self.w2 = weights[W.moe_w2].contiguous().view(torch.int64)
        self.block_scales = (
            weights[W.moe_s1].view(torch.int32),
            weights[W.moe_s2].view(torch.int32),
        )
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        pc = getattr(config, "parallelism_config", None)
        self.tp_size = pc.get_ffn_tp_size() if pc is not None else 1
        self.tp_rank = pc.get_ffn_tp_rank() if pc is not None else 0

    def cutlass_quant_scales(self):
        weights = self.weights
        for name in (W.moe_w1_i_s, W.moe_w2_i_s):
            scale = weights[name]
            if (
                not bool(torch.isfinite(scale).all())
                or not bool((scale > 0).all())
                or not bool((scale == scale.flatten()[0]).all())
            ):
                raise ValueError(
                    "CUTLASS online NVFP4 requires uniform finite positive activation scales"
                )
        return [
            weights[W.moe_w1_i_s].reciprocal().min(),
            self.block_scales[0],
            weights[W.moe_w1_i_s] * weights[W.moe_w1_s2],
            weights[W.moe_w2_i_s].reciprocal().min(),
            self.block_scales[1],
            weights[W.moe_w2_i_s] * weights[W.moe_w2_s2],
        ]
