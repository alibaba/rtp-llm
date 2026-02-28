from abc import abstractmethod
from typing import Any, Optional, Tuple
import aiter
import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    scaled_fp8_per_token_quant,
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

class PureTpRouterBase(FusedMoeDataRouter):
    """Base class for Pure TP routers.

    This base class handles initialization for routers that use tensor parallelism
    with all-gather or all-reduce patterns.
    """

    @classmethod
    def router_type(cls):
        return RouterType.PURE_TP

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if PureTpRouter can handle the configuration"""
        resolver = MoeConfigResolver()
        checker.check(resolver.is_single_gpu(config) or resolver.is_tp_equal_ep(config))
        checker.check(resolver.use_all_gather(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        do_recompute_topk: bool,
    ):
        super().__init__(config, quant_config)

        self.tp_size = config.tp_size
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.expert_num = config.expert_num
        self.expert_num_per_rank = self.expert_num // self.ep_size
        self.expert_start_id = self.ep_rank * self.expert_num_per_rank
        self.do_recompute_topk = do_recompute_topk

    @abstractmethod
    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Execute quantization on input tensor.

        Args:
            a1: Input tensor to quantize

        Returns:
            Tuple of (quantized_tensor, scale_tensor)
        """
        raise NotImplementedError

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        # Apply quantization
        assert a1_scale is None and a2_scale is None, "not support quanted moe"
        expert_x, expert_x_scale = self._do_quant(a1)

        adjusted_topk_ids = topk_ids
        num_recv_tokens_per_expert = None
        # Recompute topk if needed
        if self.do_recompute_topk:
            adjusted_topk_ids, num_recv_tokens_per_expert = (
                recompute_topk_ids_sum_expert_count(
                    topk_ids, self.expert_start_id, self.expert_num_per_rank
                )
            )
        return ExpertForwardPayload(
            expert_x,
            a1.dtype,
            expert_x_scale,
            ExpertTokensMetadata(None, num_recv_tokens_per_expert, None),
            adjusted_topk_ids,
            topk_weights,
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        fused_expert_output = payload.fused_expert_output
        if self.tp_size > 1:
            fused_expert_output = all_reduce(fused_expert_output, group=Group.TP)
        return fused_expert_output


class PureTpRouterFusedQuant(PureTpRouterBase):
    """Pure TP router (currently only for bf16-fp8 ptpc aiter moe)."""

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, do_recompute_topk=False)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if PureTpRouterFusedQuant can handle the configuration"""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_CHANNEL_COMPRESSED")

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Quantization is fused to executor, just pass through"""
        M, model_dim = a1.shape
        a8_type = self.quant_config.quant_dtype
        a8 = torch.empty((M, model_dim), dtype=a8_type, device=a1.device)
        a8_scale = torch.empty(M, dtype=torch.float32, device=a1.device)
        aiter.dynamic_per_token_scaled_quant(a8, a1, a8_scale)
        return a8, a8_scale
