"""FusedMoe wrapper for FP8 MegaMoE with a fused shared expert."""

from typing import Any, Dict, Optional

import torch

from rtp_llm.utils.model_weight import W

from .mega_moe_fp8_se import GLM5MegaMoEFP8SE
from .mega_moe_wrapper import MegaMoeWrapper


class MegaMoeFp8SEWrapper(MegaMoeWrapper):
    """Route GLM-5 routed and shared experts through FP8 MegaMoE."""

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

        shared_weight_keys = (W.ffn_w13, W.ffn_s13, W.ffn_w2, W.ffn_s2)
        missing_keys = [key for key in shared_weight_keys if weights.get(key) is None]
        if missing_keys:
            raise ValueError(
                "MegaMoeFp8SEWrapper requires FP8 shared expert weights "
                "ffn_w13, ffn_s13, ffn_w2, and ffn_s2; "
                f"missing {missing_keys}"
            )

        # The fused path replaces DenseMLP with a sentinel, so these checkpoint
        # tensors have no consumer after DeepGEMM creates its transformed copies.
        # Consume them just like the routed weights to avoid retaining both layouts.
        w1 = weights.pop(W.ffn_w13)
        s1 = weights.pop(W.ffn_s13)
        w2 = weights.pop(W.ffn_w2)
        s2 = weights.pop(W.ffn_s2)
        self.mega_moe.setup_shared_expert_from_fp8(
            w1_w=w1,
            w1_s=s1,
            w2_w=w2,
            w2_s=s2,
        )
        del w1, s1, w2, s2
        torch.cuda.empty_cache()
        self.mega_moe.maybe_warmup_fused_shared_jit_once()

    def clone_for_cuda_graph(self) -> "MegaMoeFp8SEWrapper":
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
        return self._forward_chunked(
            hidden_states,
            topk_weights,
            topk_ids,
            self.mega_moe.forward,
        )
