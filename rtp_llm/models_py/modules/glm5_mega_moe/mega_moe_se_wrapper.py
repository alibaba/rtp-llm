"""FusedMoe-compatible FP8xFP4 MegaMoE wrapper with fused shared expert."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from rtp_llm.utils.model_weight import W

from .mega_moe_se import GLM5MegaMoESE
from .mega_moe_wrapper import MegaMoeWrapper


class MegaMoeSEWrapper(MegaMoeWrapper):
    """Consume routed MXFP4 and shared MXFP8 checkpoint weights."""

    def _get_mega_moe_cls(self):
        return GLM5MegaMoESE

    def __init__(
        self,
        config,
        parallelism_config,
        weights: Dict[str, torch.Tensor],
        moe_config=None,
        layer_idx: int = 0,
        max_generate_batch_size: int = 0,
    ):
        if parallelism_config.get_ffn_tp_size() != 1:
            raise ValueError(
                "mega_moe_se requires full shared-expert weights on every EP "
                "rank (ffn_tp_size == 1)"
            )
        super().__init__(
            config,
            parallelism_config,
            weights,
            moe_config=moe_config,
            layer_idx=layer_idx,
            max_generate_batch_size=max_generate_batch_size,
        )

        shared_keys = (W.ffn_w13, W.ffn_s13, W.ffn_w2, W.ffn_s2)
        missing = [key for key in shared_keys if weights.get(key) is None]
        if missing:
            raise ValueError(
                "MegaMoeSEWrapper requires MXFP8 shared weights/scales "
                f"ffn_w13, ffn_s13, ffn_w2 and ffn_s2; missing {missing}"
            )

        w1 = weights.pop(W.ffn_w13)
        s1 = weights.pop(W.ffn_s13)
        w2 = weights.pop(W.ffn_w2)
        s2 = weights.pop(W.ffn_s2)
        self.mega_moe.setup_shared_expert_from_fp8(
            w1_fp8=w1,
            w1_scale=s1,
            w2_fp8=w2,
            w2_scale=s2,
        )
        del w1, s1, w2, s2
        torch.cuda.empty_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        expert_map: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        extra_expert_args: Optional[Dict[str, Any]] = None,
        extra_finalize_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return self._forward_chunked(
            hidden_states,
            topk_weights,
            topk_ids,
            lambda h, w, i: self.mega_moe(
                h,
                w,
                i,
                activation=activation,
                extra_expert_args=extra_expert_args,
            ),
        )


__all__ = ["MegaMoeSEWrapper"]
