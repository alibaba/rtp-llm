"""Qwen4 routing admission and native router fallback."""

import triton
from torch import nn

from rtp_llm.models_py.triton_kernels.moe.qwen4_topk import _moe_topk, is_supported


class Qwen4ExpMoeTopk(nn.Module):
    def __init__(self, native, *, batch_invariant=False):
        super().__init__()
        self.native = native
        self.batch_invariant = batch_invariant
        config = native.config
        self.geometry_supported = (
            config.expert_num in ((512, 768) if batch_invariant else (512,))
            and config.moe_k == 10
            and bool(config.has_moe_norm)
        )

    def forward(self, router_logits, topk_ids, topk_weights):
        if self.batch_invariant and router_logits.shape[0] == 0:
            return
        if not self.geometry_supported or not is_supported(
            router_logits, topk_ids, topk_weights, batch_invariant=self.batch_invariant
        ):
            if self.batch_invariant:
                raise RuntimeError(
                    "M11 batch-invariant MoE requires the ordered router"
                )
            return self.native(router_logits, topk_ids, topk_weights)
        experts = router_logits.shape[1]
        _moe_topk[(router_logits.shape[0],)](
            router_logits,
            topk_ids,
            topk_weights,
            STRIDE=router_logits.stride(0),
            EXPERTS=experts,
            BLOCK=triton.next_power_of_2(experts),
            num_warps=4 if experts == 768 else 1,
            enable_fp_fusion=False,
        )
