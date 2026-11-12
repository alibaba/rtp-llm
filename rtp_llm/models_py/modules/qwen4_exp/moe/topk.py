"""Qwen4 routing admission and native router fallback."""

from torch import nn

from rtp_llm.models_py.triton_kernels.moe.qwen4_topk import _moe_topk, is_supported


class Qwen4ExpMoeTopk(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        config = native.config
        self.geometry_supported = (
            config.expert_num == 512
            and config.moe_k == 10
            and bool(config.has_moe_norm)
        )

    def forward(self, router_logits, topk_ids, topk_weights):
        if not self.geometry_supported or not is_supported(
            router_logits, topk_ids, topk_weights
        ):
            return self.native(router_logits, topk_ids, topk_weights)
        _moe_topk[(router_logits.shape[0],)](
            router_logits,
            topk_ids,
            topk_weights,
            STRIDE=router_logits.stride(0),
            num_warps=1,
            enable_fp_fusion=False,
        )
