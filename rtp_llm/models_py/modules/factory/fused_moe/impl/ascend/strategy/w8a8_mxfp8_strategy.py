"""Ascend NPU W8A8_MXFP8 MoE strategy (W8A8_MXFP8: Weight & Activation FP8 Quantization).

Reuses the pure-torch ``BatchedDataRouter`` (its ``prepare`` asserts
``a1_scale/a2_scale is None``, which holds here because the activation MX
quantization happens inside the executor) with relaxed quantization
conditions, and pairs it with ``AscendW8A8MXFP8Executor``.

The existing ``AscendBf16FallbackStrategy`` requires "no quantization", so
without this strategy a quantized config would find no candidate and
``StrategyRegistry.get_strategy`` would raise.
"""

from typing import Any, Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.router.batched_data_router import (
    BatchedDataRouter,
)


class AscendW8A8MXFP8BatchedDataRouter(BatchedDataRouter):
    """MXFP8 quantized MoE router (single GPU / tp==ep).

    prepare/finalize are vectorized bucketing: the inherited per-expert
    loops sync the device ~2x per expert (>1s/step at E=256). Slot
    semantics unchanged (expert-major, token-ascending per expert).
    """

    def __init__(self, config: Any, quant_config: FusedMoEQuantConfig) -> None:
        super().__init__(config, quant_config)
        # ep_size == 1: all topk ids are local; skip the in-range filter
        # (its nonzero() would force a device sync).
        self._needs_filter = config.expert_num != self.num_local_experts

    def _bucket_slots(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bucket flattened (token, topk) slots by local expert id.

        Returns (sorted_e, slot_idx, sorted_rows, weights_order, num_tokens):
        expert-ascending slot order; slot_idx == the row the executor's
        mask-compaction reads back; weights_order == permutation of
        flattened topk_weights.
        """
        num_tokens, topk_k = topk_ids.shape
        device = topk_ids.device
        local_ids = topk_ids.flatten() - self.num_local_experts * self.ep_rank
        if self._needs_filter:
            in_range = (local_ids >= 0) & (local_ids < self.num_local_experts)
            keep = torch.nonzero(in_range, as_tuple=False).flatten()
            local_ids = local_ids[keep]
            # row-major flatten -> global slot n belongs to token n // topk_k
            slot_rows = keep // topk_k
            weights_order = keep
        else:
            slot_rows = (
                torch.arange(num_tokens * topk_k, device=device) // topk_k
            )
            weights_order = torch.arange(
                num_tokens * topk_k, device=device
            )

        order = torch.argsort(local_ids, stable=True)
        sorted_e = local_ids[order].long()
        sorted_rows = slot_rows[order].long()

        counts = torch.bincount(local_ids, minlength=self.num_local_experts)
        offsets = torch.cumsum(counts, dim=0) - counts
        slot_idx = torch.arange(sorted_e.numel(), device=device) - offsets[sorted_e]
        slot_idx = slot_idx.long()
        return sorted_e, slot_idx, sorted_rows, weights_order[order], num_tokens

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        assert a1.dim() == 2
        assert topk_ids.dim() == 2
        assert a1.size(0) == topk_ids.size(0)
        assert a1_scale is None and a2_scale is None, "not support quanted moe"

        _, hidden_dim = a1.size()
        sorted_e, slot_idx, sorted_rows, _, _ = self._bucket_slots(topk_ids)
        counts = torch.bincount(
            sorted_e, minlength=self.num_local_experts
        ).to(torch.int)

        # Dynamic t_cap: fixed [E, max_tokens, H] costs >1GB zero-fill per
        # layer per step. zeros (not empty): NPU index_put may touch untouched
        # rows, uninitialized memory would leak NaNs.
        t_cap = max(int(counts.max().item()), 1)
        b_a1 = torch.zeros(
            (self.num_local_experts, t_cap, hidden_dim),
            dtype=a1.dtype,
            device=a1.device,
        )
        # 1D flatten indexing: 2D combined indices crash NPU AdvancedIndex
        # (aivec 259) under some topk distributions.
        flat_idx = sorted_e * t_cap + slot_idx
        b_a1.view(-1, hidden_dim)[flat_idx] = a1[sorted_rows]

        return ExpertForwardPayload(
            expert_x=b_a1,
            expert_x_scale=None,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=counts,
                expert_num_tokens_cpu=None,
            ),
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        sorted_e, slot_idx, sorted_rows, weights_order, num_tokens = (
            self._bucket_slots(topk_ids)
        )
        fused = payload.fused_expert_output
        # 1D flattened gather (same AdvancedIndex avoidance as prepare)
        flat_idx = sorted_e * fused.size(1) + slot_idx
        compact = fused.view(-1, fused.size(-1))[flat_idx]
        if not apply_router_weight_on_input:
            # fp32 topk_weights would promote the output dtype; compute in
            # fp32 and cast back so hidden states stay bf16 for downstream.
            compact = (
                compact.float()
                * topk_weights.flatten()[weights_order].unsqueeze(-1)
            ).to(fused.dtype)
        output = torch.zeros(
            (num_tokens, fused.size(-1)), dtype=compact.dtype, device=fused.device
        )
        output.index_add_(0, sorted_rows, compact)
        if self.tp_size > 1:
            output = all_reduce(output, Group.TP)
        return output

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        # Unlike the base router, quantization IS expected here (ModelSlim /
        # W8A8_MXFP8); activation quantization happens inside the executor so
        # prepare()'s a1_scale/a2_scale=None assertion still holds.
        checker.check(
            resolver.get_quant_method(config) in ("ASCEND_W8A8_MXFP8",)
        )
        checker.check(resolver.is_single_gpu(config) or resolver.is_tp_equal_ep(config))


class AscendW8A8MXFP8MoeStrategy(MoeStrategy):
    """Ascend W8A8_MXFP8 MoE strategy."""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.ascend.executors.w8a8_mxfp8_executor import (
            AscendW8A8MXFP8Executor,
        )

        return StrategyAttributes(
            router_class=AscendW8A8MXFP8BatchedDataRouter,
            executor_class=AscendW8A8MXFP8Executor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        # Static (ModelSlim) path only in this stage; the dynamic load-quant
        # path (FP8_PER_BLOCK) is enabled by a follow-up TODO.
        checker.check(resolver.get_quant_method(config) in ("ASCEND_W8A8_MXFP8",))
