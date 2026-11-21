from abc import abstractmethod
from typing import Any, Optional, Tuple

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
    FinalizeArgs,
    FusedMoeDataRouter,
    should_skip_tp_allreduce,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
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

    @property
    def supports_skip_tp_allreduce(self) -> bool:
        return True

    def _maybe_all_reduce_tp(
        self, output: torch.Tensor, extra_finalize_args: Optional[FinalizeArgs]
    ) -> torch.Tensor:
        if self.tp_size > 1 and not should_skip_tp_allreduce(extra_finalize_args):
            return getattr(self, "tp_all_reduce", all_reduce)(output, group=Group.TP)
        return output

    def _finalize_eplb_topk_outputs(
        self,
        output: torch.Tensor,
        topk_weights: torch.Tensor,
        extra_finalize_args: Optional[FinalizeArgs],
    ) -> torch.Tensor:
        if should_skip_tp_allreduce(extra_finalize_args):
            raise RuntimeError(
                "Deterministic EPLB aggregation requires the TP all-reduce"
            )
        if output.numel() == 0 and topk_weights.shape[0] == 0:
            # FusedMoe can bypass the executor on an empty input.
            return output.reshape(0, output.shape[-1])
        if output.dim() != 3 or tuple(output.shape[:2]) != tuple(topk_weights.shape):
            raise ValueError(
                "EPLB expert outputs must have shape [tokens, top_k, hidden]"
            )
        args = extra_finalize_args or {}
        logical_topk_ids = args.get("eplb_logical_topk_ids")
        logical_per_rank = args.get("eplb_logical_experts_per_rank")
        if logical_topk_ids is None or not isinstance(logical_per_rank, int):
            raise ValueError("Missing deterministic EPLB aggregation metadata")
        if tuple(logical_topk_ids.shape) != tuple(topk_weights.shape):
            raise ValueError("EPLB logical top-k ids do not match top-k weights")

        logical_topk_ids = logical_topk_ids.to(device=output.device)
        owners = torch.div(
            logical_topk_ids.to(torch.int64),
            logical_per_rank,
            rounding_mode="floor",
        )
        valid = (logical_topk_ids >= 0) & (owners >= 0) & (owners < self.ep_size)
        input_index = torch.arange(
            output.shape[0] * output.shape[1],
            device=output.device,
            dtype=logical_topk_ids.dtype,
        ).reshape(output.shape[0], output.shape[1])
        flat_output = output.reshape(-1, output.shape[2])
        owner_topk_ids = torch.where(
            valid & (owners == self.ep_rank),
            torch.zeros_like(logical_topk_ids),
            torch.full_like(logical_topk_ids, -1),
        )
        owner_output = torch.empty(
            (output.shape[0], output.shape[2]),
            device=output.device,
            dtype=torch.bfloat16,
        )
        ep_gather(
            flat_output,
            owner_topk_ids,
            topk_weights,
            input_index,
            owner_output,
        )
        # The first reduction collects raw expert outputs from their dynamic
        # physical owners.  This second reduction follows the original static
        # path: each rank contributes only its canonical logical-owner sum.
        # Keeping that BF16 collective preserves the static rounding order for
        # every EP size, rather than reproducing it only when ep_size == 2.
        return self._maybe_all_reduce_tp(owner_output, extra_finalize_args)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if PureTpRouter can handle the configuration"""
        resolver = MoeConfigResolver()
        # Keep the existing CUDA strategy coverage for EP-equivalent layouts.
        # The unified TP all-reduce optimization is independently restricted to
        # ep_size == 1 in GenericMoeLayer, so this router selection predicate
        # must not narrow the pre-existing CUDA policy here.
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
        self.use_deterministic_eplb_gather = False

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
        extra_finalize_args: Optional[FinalizeArgs],
    ) -> torch.Tensor:
        output = self._maybe_all_reduce_tp(
            payload.fused_expert_output, extra_finalize_args
        )
        if getattr(self, "use_deterministic_eplb_gather", False):
            output = self._finalize_eplb_topk_outputs(
                output, topk_weights, extra_finalize_args
            )
        return output


class PureTpRouterNoQuant(PureTpRouterBase):
    """Pure TP router without quantization (for f16/bf16)."""

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, do_recompute_topk=False)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if PureTpRouterNoQuant can handle the configuration"""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """No quantization, just pass through"""
        return a1, None


class PureTpRouterFp8PerTensor(PureTpRouterBase):
    """Per-token activation quantization for FP8 expert GEMMs."""

    supported_methods = ("FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR")
    recompute_topk = True

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, do_recompute_topk=self.recompute_topk)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check both topology and the accepted weight scale contract."""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method in cls.supported_methods)

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FP8 per-token quantization"""
        return scaled_fp8_per_token_quant(a1, None)


class PureTpRouterFp8PerChannel(PureTpRouterFp8PerTensor):
    # Single rank: IDs are already local and the number of assignments is
    # known from their shape; no CPU copy of per-expert counts is needed.
    recompute_topk = False
    supported_methods = ("FP8_PER_CHANNEL_COMPRESSED",)


class PureTpRouterFp8PerBlock(PureTpRouterBase):
    """Pure TP router with FP8 per-block quantization."""

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, do_recompute_topk=True)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if PureTpRouterFp8PerBlock can handle the configuration"""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FP8 per-block quantization with DeepGemm"""
        if is_deep_gemm_e8m0_used():
            return sgl_per_token_group_quant_fp8(
                a1,
                128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
        else:
            from rtp_llm.ops.compute_ops import trt_fp8_quantize_128

            return trt_fp8_quantize_128(a1, False)


class PureTpRouterW4a8Int4PerChannel(PureTpRouterBase):
    """Pure TP router with W4A8 INT4 per-channel quantization."""

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, do_recompute_topk=True)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if PureTpRouterW4a8Int4PerChannel can handle the configuration"""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method
            in ["W4A8_INT4_PER_CHANNEL", "W4A8_INT4_PER_CHANNEL_COMPRESSED"]
        )

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """W4A8 INT4 per-channel quantization"""
        return scaled_fp8_per_token_quant(a1, None)


class PureTpRouterFp4PerGroup(PureTpRouterBase):
    """Pure TP router with FP4 per-group quantization."""

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, do_recompute_topk=True)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if PureTpRouterFp4PerGroup can handle the configuration"""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "modelopt_fp4")
        checker.check(not config.moe_config.use_deepep_moe)

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FP4 per-group quantization"""
        return a1, None


class PureTpRouterNvfp4Cutlass(PureTpRouterFp4PerGroup):
    """Replicated tokens and global IDs for CUTLASS expert/tensor parallelism."""

    def __init__(self, config, quant_config):
        PureTpRouterBase.__init__(self, config, quant_config, do_recompute_topk=False)
