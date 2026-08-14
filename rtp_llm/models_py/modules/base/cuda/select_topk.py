from typing import Any, Optional

import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.utils.fuse_config import glm5_prefill_refine_enabled


class SelectTopk(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.select_topk_op = compute_ops.SelectTopkOp(self.config)

    def forward(
        self,
        router_logits_fp32: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        self.select_topk_op.forward(router_logits_fp32, topk_ids, topk_weights)


class GroupTopK(nn.Module):
    def __init__(
        self,
        use_fused: Optional[bool] = None,
        hw_kernel_config: Optional[Any] = None,
    ):
        super().__init__()
        self.group_topk_op = compute_ops.GroupTopKOp()
        if use_fused is None:
            use_fused = glm5_prefill_refine_enabled(hw_kernel_config)
        self.use_fused = use_fused

    @staticmethod
    def _fused_supported(
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scores: torch.Tensor,
        correction_bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
    ) -> bool:
        return (
            scores.is_cuda
            and scores.is_contiguous()
            and scores.ndim == 2
            and scores.shape[1] == 256
            and scores.dtype == torch.bfloat16
            and correction_bias.is_cuda
            and correction_bias.is_contiguous()
            and correction_bias.dtype == torch.float32
            and correction_bias.ndim == 1
            and correction_bias.numel() == 256
            and topk_weights.is_cuda
            and topk_weights.is_contiguous()
            and topk_weights.dtype == torch.float32
            and topk_weights.shape == (scores.shape[0], topk)
            and topk_ids.is_cuda
            and topk_ids.is_contiguous()
            and topk_ids.dtype in (torch.int32, torch.int64)
            and topk_ids.shape == (scores.shape[0], topk)
            and scores.device
            == correction_bias.device
            == topk_weights.device
            == topk_ids.device
            and (n_group, topk_group) in ((8, 4), (1, 1))
            and topk == 8
        )

    def forward_legacy(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scores: torch.Tensor,
        correction_bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        renormalize: bool,
        routed_scaling_factor: float,
    ):
        # The legacy C++ kernel unconditionally reads both score tensors as FP32.
        scores = scores.float().sigmoid()
        scores_with_bias = scores + correction_bias.unsqueeze(0)
        self.group_topk_op.forward(
            topk_weights,
            topk_ids,
            scores,
            scores_with_bias,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
        )

    def can_use_fused(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scores: torch.Tensor,
        correction_bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        use_fused: Optional[bool] = None,
    ) -> bool:
        """Whether the strict fused op can consume these BF16 logits."""
        if use_fused is None:
            use_fused = self.use_fused
        else:
            use_fused = self.use_fused and use_fused
        return use_fused and self._fused_supported(
            topk_weights,
            topk_ids,
            scores,
            correction_bias,
            n_group,
            topk_group,
            topk,
        )

    def forward_fused(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scores: torch.Tensor,
        correction_bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        renormalize: bool,
        routed_scaling_factor: float,
    ):
        if not self._fused_supported(
            topk_weights,
            topk_ids,
            scores,
            correction_bias,
            n_group,
            topk_group,
            topk,
        ):
            raise ValueError(
                "fused GroupTopK requires contiguous BF16 [T, 256] logits, "
                "FP32 [256] bias, (n_group, topk_group, topk)=(8,4,8) or (1,1,8)"
            )
        self.group_topk_op.forward_fused(
            topk_weights,
            topk_ids,
            scores,
            correction_bias,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
        )

    def forward(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scores: torch.Tensor,
        correction_bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        renormalize: bool,
        routed_scaling_factor: float,
        use_fused: Optional[bool] = None,
    ):
        args = (
            topk_weights,
            topk_ids,
            scores,
            correction_bias,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
        )
        if self.can_use_fused(
            topk_weights,
            topk_ids,
            scores,
            correction_bias,
            n_group,
            topk_group,
            topk,
            use_fused=use_fused,
        ):
            self.forward_fused(*args)
        else:
            self.forward_legacy(*args)


class FakeBalanceExpert(nn.Module):
    def __init__(
        self,
        expert_num: int,
        moe_k: int,
        dp_rank: int,
        dp_size: int,
        ep_size: int,
    ):
        super().__init__()
        self.fake_balance_expert_op = compute_ops.FakeBalanceExpertOp(
            expert_num, moe_k, dp_rank, dp_size, ep_size
        )

    def forward(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        self.fake_balance_expert_op.forward(topk_ids, topk_weights)
