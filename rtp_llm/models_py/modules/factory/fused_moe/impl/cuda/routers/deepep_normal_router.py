from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPMode,
    DeepEPWrapper,
    DeepepWrapperConfig,
)
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
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128

from .util import calc_tp_slice, finalize_tp_gather, prepare_tp_slice


class DeepepNormalRouterBase(FusedMoeDataRouter):
    """Base class for DeepEP normal routers."""

    @classmethod
    def router_type(cls):
        return RouterType.DEEPEP_NORMAL

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepepNormalRouter can handle the configuration"""
        resolver = MoeConfigResolver()
        checker.check(get_sm()[0] >= 9)
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.use_low_latency(config))
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        expert_alignment: int,
    ):
        super().__init__(config, quant_config)

        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank
        self.dp_size = config.dp_size
        self.dp_rank = config.dp_rank
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.expert_num = config.expert_num
        self.expert_num_per_rank = self.expert_num // self.ep_size
        self.num_dispatchers = config.world_size // config.tp_size
        self.rank_expert_offset = self.ep_rank * self.expert_num_per_rank
        self.top_k = config.moe_topk_group
        deepep_config = DeepepWrapperConfig.from_config_adapter(self.config)
        self.deepep_buffer_wrapper = DeepEPWrapper.get_instance(deepep_config)
        assert (
            self.deepep_buffer_wrapper.mode == DeepEPMode.NORMAL
        ), "DeepEP mode should be NORMAL"
        self.async_mode = False
        self.expert_alignment = expert_alignment
        self.handle: Any = None

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("DeepEPNormal a1_scale or a2_scale should be None")
        act_dtype = a1.dtype

        # scatter (tp slice)
        tp_size = self.config.tp_size
        tp_rank = self.config.tp_rank
        slice_begin, slice_size, _ = calc_tp_slice(
            token_num=a1.size(0), tp_size=tp_size, tp_rank=tp_rank
        )

        # Apply quantization
        use_fp8 = (
            self.quant_config.is_quantized
            and self.quant_config.quant_dtype == torch.float8_e4m3fn
        )
        if use_fp8:
            a1_quant, a1_scale_quant = self._do_quant(a1)
            assert a1_scale_quant is not None
            tp_expert_a1, tp_expert_ids, tp_expert_scales, tp_expert_a1_scale = (
                prepare_tp_slice(
                    a1_quant,
                    topk_ids,
                    topk_weights,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    a1_scale=a1_scale_quant,
                )
            )
            assert tp_expert_a1_scale is not None
            tp_expert_input = (tp_expert_a1, tp_expert_a1_scale)
        else:
            tp_expert_a1, tp_expert_ids, tp_expert_scales, _ = prepare_tp_slice(
                a1,
                topk_ids,
                topk_weights,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
            tp_expert_input = tp_expert_a1

        # pre dispatch
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = self.deepep_buffer_wrapper.buffer.get_dispatch_layout(
            tp_expert_ids, self.expert_num
        )

        # dispatch
        (
            output,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            self.handle,
            _,
        ) = self.deepep_buffer_wrapper.buffer.dispatch(
            tp_expert_input,
            None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            tp_expert_ids,
            tp_expert_scales,
            expert_alignment=self.expert_alignment,
        )

        expert_x_scale: Optional[torch.Tensor] = None
        expert_x: torch.Tensor
        if use_fp8:
            assert isinstance(output, tuple), "output should be a tuple"
            expert_x, expert_x_scale = output
            # TODO: move it to the executor
            if self.quant_config.is_per_act_token:
                expert_x_scale = expert_x_scale[:, 0].contiguous()
        else:
            assert isinstance(output, torch.Tensor), "output should be a tensor"
            expert_x = output

        expert_num_tokens = torch.tensor(
            num_recv_tokens_per_expert_list, device=expert_x.device, dtype=torch.int32
        )

        if recv_topk_idx.numel() != 0 and (not use_fp8):
            expert_topk_ids = torch.where(
                recv_topk_idx == -1,
                self.expert_num - 1 if self.rank_expert_offset == 0 else 0,
                recv_topk_idx + self.rank_expert_offset,
            )
        else:
            expert_topk_ids = recv_topk_idx

        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=act_dtype,
            expert_topk_ids=expert_topk_ids,
            expert_topk_weights=recv_topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens,
                expert_num_tokens_cpu=num_recv_tokens_per_expert_list,
            ),
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        assert self.handle is not None, "handler is None"
        assert payload.fused_expert_output is not None, "fused_expert_output is None"
        out_token, _, _ = self.deepep_buffer_wrapper.buffer.combine(
            payload.fused_expert_output, self.handle
        )
        self.handle = None

        return finalize_tp_gather(
            out_token,
            tp_size=self.config.tp_size,
            extra_finalize_args=extra_finalize_args,
        )

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (
            self.quant_config.quant_dtype == torch.float8_e4m3fn
            and self.quant_config.is_per_act_token
        ):
            return self._do_quant_fp8_per_token(a1)
        elif (
            self.quant_config.quant_dtype == torch.float8_e4m3fn
            and self.quant_config.is_block_quantized
        ):
            return self._do_quant_fp8_per_block(a1)
        else:
            raise ValueError(f"Unsupported quant config: {self.quant_config}")

    def _do_quant_fp8_per_token(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        a1, a1_scale = scaled_fp8_per_token_quant(a1, None)
        assert a1.shape[1] % 128 == 0
        a1_scale = a1_scale.repeat(1, a1.shape[1] // 128)
        return a1, a1_scale

    def _do_quant_fp8_per_block(
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


class DeepepNormalRouterNoQuant(DeepepNormalRouterBase):
    """DeepEP normal router without quantization (for f16/bf16)."""

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, expert_alignment=1)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepepNormalRouterNoQuant can handle the configuration"""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)


class DeepepNormalRouterFp8PerTensor(DeepepNormalRouterBase):
    """DeepEP normal router with FP8 per-tensor quantization."""

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, expert_alignment=1)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepepNormalRouterFp8PerTensor can handle the configuration"""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method in ["FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR"]
        )


class DeepepNormalRouterFp8PerBlock(DeepepNormalRouterBase):
    """DeepEP normal router with FP8 per-block quantization."""

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, expert_alignment=128)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepepNormalRouterFp8PerBlock can handle the configuration"""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
