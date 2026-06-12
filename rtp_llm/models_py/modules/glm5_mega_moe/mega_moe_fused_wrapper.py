"""Fused routed MegaMoE plus GLM-5 shared expert wrapper."""

from typing import Any, Dict, Optional

import torch

from rtp_llm.utils.model_weight import W

from .mega_moe_fused import GLM5MegaMoEFused
from .mega_moe_wrapper import MegaMoeWrapper


class MegaMoeFusedWrapper(MegaMoeWrapper):
    """FusedMoe-compatible wrapper for ``fp8_fp4_mega_moe_fused``.

    The routed expert weights use the same load-time FP4 layout as
    :class:`MegaMoeWrapper`. The shared expert weights are pre-quantized FP4
    tensors loaded from ``shared_experts.*.{weight,weight_scale}``.
    """

    def _get_mega_moe_cls(self):
        return GLM5MegaMoEFused

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

        # Keep shared-expert entries in the layer weight dict. The routed
        # MegaMoE path historically consumes its own weights, but shared expert
        # weights are still part of the generic FFN weight surface that the
        # backend may inspect during engine initialization.
        w1 = weights.get(W.ffn_w13, None)
        s1 = weights.get(W.ffn_s13, None)
        w2 = weights.get(W.ffn_w2, None)
        s2 = weights.get(W.ffn_s2, None)

        if w1 is None or s1 is None or w2 is None or s2 is None:
            raise ValueError(
                "MegaMoeFusedWrapper requires load-time FP4 shared expert weights "
                "(ffn_w13, ffn_w2, ffn_s13, ffn_s2)."
            )
        if w1.dtype != torch.int8 or w2.dtype != torch.int8:
            raise ValueError(
                "MegaMoeFusedWrapper only accepts load-time FP4 int8 shared "
                f"expert weights. Got ffn_w13 dtype={w1.dtype}, "
                f"ffn_w2 dtype={w2.dtype}."
            )

        self.mega_moe.setup_shared_expert_from_fp4(
            w1_w=w1,
            w1_s=s1,
            w2_w=w2,
            w2_s=s2,
        )
        del w1, s1, w2, s2
        torch.cuda.empty_cache()
        self.mega_moe.maybe_warmup_fused_shared_jit_once()

    def clone_for_cuda_graph(self) -> "MegaMoeFusedWrapper":
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
            self.mega_moe.forward_with_shared_expert,
        )
