from abc import abstractmethod
from typing import Any, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
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
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


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
    """Pure TP router with FP8 per-tensor quantization."""

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, do_recompute_topk=True)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if PureTpRouterFp8PerTensor can handle the configuration"""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method in ["FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR"]
        )

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FP8 per-token quantization"""
        return scaled_fp8_per_token_quant(a1, None)


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
            return trt_fp8_quantize_128(a1, False)


class PureTpRouterFp8PerBlockTriton(PureTpRouterFp8PerBlock):
    """FP8 per-block router for the Triton fused-MoE executor.

    Unlike :class:`PureTpRouterFp8PerBlock`, which produces scales in the
    column-major / TMA-aligned / UE8M0 layout that DeepGEMM consumes, this
    variant returns row-major fp32 scales that match what the Triton
    ``invoke_fused_moe_kernel`` expects (the same layout produced by
    ``sgl_per_token_group_quant_fp8(A, block_k)`` with default arguments).

    In addition to the original PureTp topologies (single-GPU and tp==ep), this
    router supports a pure DP+EP topology where ``tp_size==1`` and
    ``ep_size==dp_size``. In that case it all-gathers tokens across the EP
    group before running the local PureTp filter trick, and all-reduces the
    output back across the same group, then slices to the local DP shard.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        # Same FP8_PER_BLOCK gate as the parent's quant check, but relax the
        # parallelism check so that pure DP+EP (tp==1, dp==ep>1) is also
        # accepted in addition to single-GPU and tp==ep.
        resolver = MoeConfigResolver()
        is_dp_ep = (
            config.tp_size == 1
            and config.dp_size > 1
            and config.dp_size == config.ep_size
        )
        checker.check(
            resolver.is_single_gpu(config)
            or resolver.is_tp_equal_ep(config)
            or is_dp_ep
        )
        # The DP+EP path performs its own all-gather/all-reduce, so it does
        # not require the use_all_gather flag to be set.
        checker.check(resolver.use_all_gather(config) or is_dp_ep)
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)
        self.dp_size = config.dp_size
        self.dp_rank = config.dp_rank
        # Pure DP+EP path needs cross-rank gather/reduce.
        self._needs_dp_gather = (
            self.dp_size > 1 and self.tp_size == 1 and self.ep_size == self.dp_size
        )
        # State carried from prepare() to finalize() for the DP+EP path.
        self._dp_local_num_tokens: Optional[int] = None
        self._dp_padded_size: Optional[int] = None

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Default args: row-major fp32 scales, group_size = block_k = 128.
        return sgl_per_token_group_quant_fp8(a1, 128)

    def _gather_dp_inputs(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """All-gather (a1, topk_ids, topk_weights) across the EP group.

        Each rank pads its local tensors to ``max_size`` (the max number of
        tokens across DP ranks) before all_gather so that all ranks
        contribute equally-sized buffers. Padding rows have ``topk_ids = -1``
        which the PureTp filter trick treats as "skip this token", so they
        produce zero contribution and do not affect correctness.

        Returns: (gathered_a1, gathered_topk_ids, gathered_topk_weights,
                  padded_size_per_rank).
        """
        local_n = a1.size(0)
        # During CUDA graph capture we cannot do GPU->CPU sync. Decode
        # captures graphs at fixed batch sizes via decode_capture_config and
        # the framework keeps DP shards balanced, so it is safe to assume
        # equal sizes per rank inside a captured graph.
        if torch.cuda.is_current_stream_capturing():
            padded = local_n
        else:
            local_size = torch.tensor([local_n], device=a1.device, dtype=torch.long)
            sizes = all_gather(local_size, group=Group.DP_AND_TP)
            padded = int(sizes.max().item())

        if padded != local_n:
            pad_n = padded - local_n
            a1 = torch.cat(
                [
                    a1,
                    torch.zeros((pad_n, a1.size(1)), dtype=a1.dtype, device=a1.device),
                ],
                dim=0,
            )
            topk_ids = torch.cat(
                [
                    topk_ids,
                    torch.full(
                        (pad_n, topk_ids.size(1)),
                        -1,
                        dtype=topk_ids.dtype,
                        device=topk_ids.device,
                    ),
                ],
                dim=0,
            )
            topk_weights = torch.cat(
                [
                    topk_weights,
                    torch.zeros(
                        (pad_n, topk_weights.size(1)),
                        dtype=topk_weights.dtype,
                        device=topk_weights.device,
                    ),
                ],
                dim=0,
            )

        a1_g = all_gather(a1.contiguous(), group=Group.DP_AND_TP)
        ti_g = all_gather(topk_ids.contiguous(), group=Group.DP_AND_TP)
        tw_g = all_gather(topk_weights.contiguous(), group=Group.DP_AND_TP)
        return a1_g, ti_g, tw_g, padded

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> "ExpertForwardPayload":
        if self._needs_dp_gather:
            self._dp_local_num_tokens = a1.size(0)
            a1, topk_ids, topk_weights, padded = self._gather_dp_inputs(
                a1, topk_ids, topk_weights
            )
            self._dp_padded_size = padded
        return super().prepare(a1, a1_scale, a2_scale, topk_weights, topk_ids)

    def finalize(
        self,
        payload: "CombineForwardPayload",
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        if not self._needs_dp_gather:
            return super().finalize(
                payload,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                extra_finalize_args,
            )
        # In the DP+EP path the parent finalize would not all_reduce because
        # tp_size == 1; do the EP-group all_reduce ourselves and slice back
        # to this rank's DP shard.
        out = payload.fused_expert_output
        out = all_reduce(out, group=Group.DP_AND_TP)
        padded = self._dp_padded_size
        local_n = self._dp_local_num_tokens
        assert padded is not None and local_n is not None
        offset = self.dp_rank * padded
        out = out[offset : offset + local_n]
        self._dp_local_num_tokens = None
        self._dp_padded_size = None
        return out


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
        checker.check(quant_method in ["W4A8_INT4_PER_CHANNEL"])

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
