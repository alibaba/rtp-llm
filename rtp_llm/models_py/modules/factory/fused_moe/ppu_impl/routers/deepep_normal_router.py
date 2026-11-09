"""PPU DeepEP Normal router implementations."""

import os
from typing import Any, Optional

import torch

from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper
from rtp_llm.models_py.kernels.ppu.deepgemm_wrapper import (
    has_deep_gemm_fp8_grouped_nopad,
)
from rtp_llm.models_py.kernels.ppu.fp8_quant import fp8_quantize_128
from rtp_llm.models_py.kernels.ppu.int8_kernel.int8_quant import per_token_quant_int8
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
    DeepepNormalRouterFp8PerBlock,
    DeepepNormalRouterNoQuant,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    recompute_topk_ids_sum_expert_count,
)


class PpuDeepepNormalRouterNoQuant(DeepepNormalRouterNoQuant):
    """DeepEP Normal BF16 router without CUDA SM capability checks."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.use_low_latency(config))
        checker.check(resolver.get_quant_method(config) is None)
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(config, quant_config)

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        token_num = a1.size(0)
        tp_token_size = (token_num + self.config.tp_size - 1) // self.config.tp_size
        slice_begin = min(tp_token_size * self.config.tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)
        if slice_size == 0 and token_num > 0:
            a1 = a1.new_empty((0, a1.shape[1]))
            topk_ids = topk_ids.new_empty((0, topk_ids.shape[1]))
            topk_weights = topk_weights.new_empty((0, topk_weights.shape[1]))
        return super().prepare(a1, a1_scale, a2_scale, topk_weights, topk_ids)


class PpuDeepepNormalRouterW8A8Int8(PpuDeepepNormalRouterNoQuant):
    """Quantize each local token once and dispatch INT8 data with its scale."""

    QUANT_METHOD = "W8A8_INT8_PER_CHANNEL_COMPRESSED"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.use_low_latency(config))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        checker.check(DeepEPWrapper.supported())

    def _prepare_dispatch_input(
        self,
        a1: torch.Tensor,
        slice_begin: int,
        slice_size: int,
        use_fp8: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_fp8:
            raise ValueError("W8A8 INT8 router cannot use FP8 dispatch")
        if a1.dtype != torch.bfloat16:
            raise ValueError(f"W8A8 DeepEP Normal expects BF16 input, got {a1.dtype}")
        tp_expert_a1 = torch.narrow(a1, 0, slice_begin, slice_size)
        return per_token_quant_int8(tp_expert_a1)


class PpuDeepepNormalRouterFp8PerBlock(DeepepNormalRouterFp8PerBlock):
    """FP8 per-block router without CUDA SM capability checks.

    Quantizes each local token once with FP32 group arithmetic and dispatches
    the FP8 data together with its ``(M, K // 128)`` per-group scale.
    """

    QUANT_METHOD = "FP8_PER_BLOCK"

    def __init__(self, config: MoEConfigAdapter, quant_config: FusedMoEQuantConfig):
        super().__init__(config, quant_config)
        # The nopad executor consumes exact per-expert counts. Inheriting the
        # CUDA router's 128-row alignment needlessly pads every small expert
        # and makes both GEMMs and activation quantization process those rows.
        nopad = (
            has_deep_gemm_fp8_grouped_nopad()
            and os.environ.get("MOE_FP8_NOPAD", "1") == "1"
        )
        if nopad:
            self.expert_alignment = 1
        parallel = config.parallelism_config
        # CP gives every EP rank the same padded number of input rows. This
        # provides a receive bound without a collective or a host readback.
        self._bounded_cp_dispatch = (
            nopad
            and config.dp_size == 1
            and parallel.tp_size == config.ep_size == config.world_size
            and parallel.prefill_cp_config.is_enabled()
            and not config.moe_config.use_deepep_internode
        )

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.use_low_latency(config))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        checker.check(DeepEPWrapper.supported())

    def _do_quant_fp8_per_block(
        self, a1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if a1.dtype != torch.bfloat16:
            raise ValueError(f"FP8 DeepEP Normal expects BF16 input, got {a1.dtype}")
        return fp8_quantize_128(a1)

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> "ExpertForwardPayload":
        if self._bounded_cp_dispatch and 0 < a1.shape[0] <= 128:
            if a1_scale is not None or a2_scale is not None:
                raise ValueError("DeepEPNormal a1_scale or a2_scale should be None")
            return self._prepare_bounded(a1, topk_weights, topk_ids)
        payload = super().prepare(a1, a1_scale, a2_scale, topk_weights, topk_ids)
        # DeepEP returns partition-local expert ids; the base class only remaps
        # them to global ids on the non-FP8 path, so redo it here for the
        # executor's `_to_local_expert_ids`.
        if payload.expert_topk_ids.numel() != 0:
            payload.expert_topk_ids = torch.where(
                payload.expert_topk_ids == -1,
                self.expert_num - 1 if self.rank_expert_offset == 0 else 0,
                payload.expert_topk_ids + self.rank_expert_offset,
            )
        return payload

    def _prepare_bounded(
        self, a1: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> ExpertForwardPayload:
        """Keep receive sizes and expert counts on the GPU for short CP input."""
        buffer = self.deepep_buffer_wrapper.buffer
        expert_input = self._do_quant_fp8_per_block(a1)
        ids = topk_ids.to(torch.int64)
        ranks, rdma_ranks, experts, membership, _ = buffer.get_dispatch_layout(
            ids, self.expert_num
        )
        output, recv_ids, recv_weights, _, self.handle, _ = buffer.dispatch(
            expert_input,
            num_tokens_per_rank=ranks,
            num_tokens_per_rdma_rank=rdma_ranks,
            is_token_in_rank=membership,
            num_tokens_per_expert=experts,
            topk_idx=ids,
            topk_weights=topk_weights,
            expert_alignment=1,
            num_worst_tokens=a1.shape[0] * self.ep_size,
        )
        # DeepEP fills unused receive rows with -1 expert ids. Count valid
        # assignments on device; no Python count list or H2D copy is needed.
        local_ids, counts = recompute_topk_ids_sum_expert_count(
            recv_ids, 0, self.expert_num_per_rank
        )
        expert_x, expert_x_scale = output
        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=a1.dtype,
            expert_topk_ids=local_ids,
            expert_topk_weights=recv_weights,
            expert_ids_are_local=True,
            expert_tokens_meta=ExpertTokensMetadata(expert_num_tokens=counts),
        )
