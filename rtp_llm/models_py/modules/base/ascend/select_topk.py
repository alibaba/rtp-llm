import torch
import torch.nn as nn

from rtp_llm.config.model_config import ModelConfig


class SelectTopk(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.top_k = config.moe_k

    def forward(
        self,
        router_logits_fp32: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Run CANN's softmax top-k routing kernel.

        ``norm_type=0`` makes the operator apply softmax over all experts before
        selecting top-k.  Qwen's ``norm_topk_prob`` is a separate operation: it
        renormalizes only the selected experts, so it is intentionally handled
        below instead of being folded into the operator arguments.
        """
        try:
            import torch_npu
        except ImportError as error:
            raise RuntimeError(
                "Ascend MoE top-k requires the pinned torch_npu package"
            ) from error
        if not hasattr(torch_npu, "npu_moe_gating_top_k"):
            raise RuntimeError(
                "torch_npu is missing required CANN operator npu_moe_gating_top_k"
            )

        topk_weights_fp32, topk_ids_int32, _ = torch_npu.npu_moe_gating_top_k(
            router_logits_fp32,
            k=self.top_k,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=0,
            norm_type=0,
            out_flag=False,
        )

        if self.config.has_moe_norm:
            denominator = topk_weights_fp32.sum(dim=-1, keepdim=True)
            topk_weights_fp32 = topk_weights_fp32 / denominator.clamp_min(
                torch.finfo(topk_weights_fp32.dtype).tiny
            )
        topk_weights.copy_(topk_weights_fp32.to(topk_weights.dtype))
        topk_ids.copy_(topk_ids_int32.to(topk_ids.dtype))
