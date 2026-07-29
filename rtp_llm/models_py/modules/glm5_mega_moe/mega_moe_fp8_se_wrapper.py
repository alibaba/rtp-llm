"""FusedMoe-compatible FP8 MegaMoE wrapper with fused shared experts."""

from typing import Dict

import torch

from rtp_llm.utils.model_weight import W

from .mega_moe_fp8_se import GLM5MegaMoEFP8SE
from .mega_moe_fp8_wrapper import MegaMoeFp8Wrapper


class MegaMoeFp8SEWrapper(MegaMoeFp8Wrapper):
    """Route both routed and shared experts through ``fp8_fp8_mega_moe``."""

    def _get_mega_moe_cls(self):
        return GLM5MegaMoEFP8SE

    def __init__(
        self,
        config,
        parallelism_config,
        weights: Dict[str, torch.Tensor],
        moe_config=None,
        layer_idx: int = 0,
        max_generate_batch_size: int = 0,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            moe_config=moe_config,
            layer_idx=layer_idx,
            max_generate_batch_size=max_generate_batch_size,
        )

        w1 = weights.get(W.ffn_w13, None)
        s1 = weights.get(W.ffn_s13, None)
        w2 = weights.get(W.ffn_w2, None)
        s2 = weights.get(W.ffn_s2, None)
        if w1 is None or s1 is None or w2 is None or s2 is None:
            raise ValueError(
                "MegaMoeFp8SEWrapper requires FP8 shared-expert weights and "
                "scales (ffn_w13, ffn_w2, ffn_s13, ffn_s2)"
            )
        if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "MegaMoeFp8SEWrapper only accepts FP8 e4m3 shared weights; "
                f"got ffn_w13={w1.dtype}, ffn_w2={w2.dtype}"
            )
        self.mega_moe.setup_shared_expert_from_fp8(
            w1_fp8=w1,
            w1_scale=s1,
            w2_fp8=w2,
            w2_scale=s2,
        )
