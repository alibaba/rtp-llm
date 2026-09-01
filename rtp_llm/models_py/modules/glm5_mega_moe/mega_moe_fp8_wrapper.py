"""FusedMoe-compatible wrapper for GLM-5 FP8xFP8 MegaMoE."""

from typing import Any, Dict, Optional

import torch

from .mega_moe_fp8 import GLM5MegaMoEFP8
from .mega_moe_wrapper import MegaMoeWrapper


class MegaMoeFp8Wrapper(MegaMoeWrapper):
    """Route GLM-5 MoE through DeepGEMM ``fp8_fp8_mega_moe``."""

    def _get_mega_moe_cls(self):
        return GLM5MegaMoEFP8

    def clone_for_cuda_graph(self) -> "MegaMoeFp8Wrapper":
        clone = object.__new__(type(self))
        torch.nn.Module.__init__(clone)
        clone.mega_moe = self.mega_moe.clone_for_cuda_graph()
        clone.expert_num = self.expert_num
        return clone

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
        del (
            inplace,
            activation,
            expert_map,
            a1_scale,
            a2_scale,
            apply_router_weight_on_input,
            extra_finalize_args,
        )
        clamp = float((extra_expert_args or {}).get("swiglu_limit", 0.0))
        activation_clamp = clamp if clamp > 0.0 else None
        return self._forward_chunked(
            hidden_states,
            topk_weights,
            topk_ids,
            lambda h, w, i: self.mega_moe(
                h,
                w,
                i,
                activation_clamp=activation_clamp,
            ),
        )
