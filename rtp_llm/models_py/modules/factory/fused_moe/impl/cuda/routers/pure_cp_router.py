"""Pure CP router using allgather + reduce_scatter.

For Context Parallel (CP) MoE: each rank holds the same experts but processes
different chunks of the context sequence. The router:

  prepare:  allgather(scattered_tokens, TP) -> full_tokens
  execute:  MoE on full_tokens (each rank computes partial sums)
  finalize: reduce_scatter(partial_output, TP) -> scattered_output

Unlike the DP router, no padding is needed because CP splits the context
evenly across ranks, guaranteeing equal token counts on every rank.
Shared expert handling is done at the layer level, not in this router.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    reduce_scatter,
)
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


class PureCpRouterBase(FusedMoeDataRouter):
    """Base class for pure CP routers using allgather + reduce_scatter.

    In CP mode, tp_size == ep_size and each rank has all experts but only a
    portion of the context tokens. This router:
    - In prepare(): allgathers scattered tokens across TP group to get full tokens
    - In finalize(): reduce_scatters the partial MoE output back to scattered form

    No padding is needed because CP guarantees equal token counts across ranks.
    """

    @classmethod
    def router_type(cls):
        return RouterType.PURE_TP

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        # Pure CP + EP only: dp_size must be 1, physical tp == ep > 1, and
        # prefill CP must actually be enabled (CudaFp8PerBlockPureCPStrategy
        # also asserts this — keep both as a defensive check so the router
        # can be selected via paths that bypass the strategy gate).
        # is_cp_equal_ep reads raw parallelism_config.tp_size so it works
        # whether or not CP is enabled (in CP mode adapter tp_size==1).
        # Mixed tp>1+dp>1 is intentionally routed back to DeepEP.
        checker.check(config.dp_size == 1)
        checker.check(resolver.is_cp_equal_ep(config))
        checker.check(config.ep_size > 1)
        checker.check(config.parallelism_config.prefill_cp_config.is_enabled())
        checker.check(resolver.use_all_gather(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

        # Use physical tp_size to match the all_gather(group=Group.TP) below.
        self.tp_size = config.parallelism_config.tp_size
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.expert_num = config.expert_num
        self.expert_num_per_rank = self.expert_num // self.ep_size
        self.expert_start_id = self.ep_rank * self.expert_num_per_rank

    @abstractmethod
    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        assert a1_scale is None and a2_scale is None, "not support quanted moe"

        a1_full = all_gather(a1, group=Group.TP)
        topk_weights_full = all_gather(topk_weights, group=Group.TP)
        topk_ids_full = all_gather(topk_ids, group=Group.TP)

        expert_x, expert_x_scale = self._do_quant(a1_full)

        adjusted_topk_ids, num_recv_tokens_per_expert = (
            recompute_topk_ids_sum_expert_count(
                topk_ids_full, self.expert_start_id, self.expert_num_per_rank
            )
        )

        return ExpertForwardPayload(
            expert_x,
            a1.dtype,
            expert_x_scale,
            ExpertTokensMetadata(None, num_recv_tokens_per_expert, None),
            adjusted_topk_ids,
            topk_weights_full,
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        return reduce_scatter(payload.fused_expert_output, group=Group.TP)


class PureCpRouterFp8PerBlock(PureCpRouterBase):
    """Pure CP router with FP8 per-block quantization."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if is_deep_gemm_e8m0_used():
            return sgl_per_token_group_quant_fp8(
                a1,
                128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
        else:
            return trt_fp8_quantize_128(a1, False)
