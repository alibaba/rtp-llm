from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.distributed.deepep_initializer import DeepEpInitializer
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
        checker.check(DeepEpInitializer.supported())

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
        self.deepep_buffer_wrapper = DeepEpInitializer.get_deepep_wrapper(self.config)
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

        # scatter
        tp_size = self.config.tp_size
        tp_rank = self.config.tp_rank
        token_num = a1.size(0)
        tp_token_size = (token_num + tp_size - 1) // tp_size

        slice_begin = min(tp_token_size * tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)

        # Apply quantization
        use_fp8 = (
            self.quant_config.is_quantized
            and self.quant_config.quant_dtype == torch.float8_e4m3fn
        )
        if use_fp8:
            a1_quant, a1_scale_quant = self._do_quant(a1)
            assert a1_scale_quant is not None
            tp_expert_a1 = torch.narrow(a1_quant, 0, slice_begin, slice_size)
            tp_expert_a1_scale = torch.narrow(
                a1_scale_quant, 0, slice_begin, slice_size
            )
            tp_expert_input = (tp_expert_a1, tp_expert_a1_scale)
        else:
            tp_expert_a1 = torch.narrow(a1, 0, slice_begin, slice_size)
            tp_expert_input = tp_expert_a1

        # pre dispatch
        tp_expert_ids = torch.narrow(topk_ids, 0, slice_begin, slice_size)
        tp_expert_scales = torch.narrow(topk_weights, 0, slice_begin, slice_size)

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

        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=act_dtype,
            expert_topk_ids=recv_topk_idx,
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

        # gather
        tp_size = self.config.tp_size
        assert extra_finalize_args is not None, "extra_finalize_args is None"
        original_num_tokens: int = extra_finalize_args["original_num_tokens"]
        tp_token_size = (original_num_tokens + tp_size - 1) // tp_size

        if tp_size > 1:
            # combine_x.size(0) might be 0
            if out_token.size(0) < tp_token_size:
                padding_out_token = torch.empty(
                    size=(tp_token_size - out_token.size(0), out_token.size(1)),
                    device=out_token.device,
                    dtype=out_token.dtype,
                )
                out_token = torch.cat([out_token, padding_out_token], dim=0)

            gatherd_output = all_gather(out_token, group=Group.TP).reshape(
                tp_size * tp_token_size, -1
            )
            gatherd_output = gatherd_output[:original_num_tokens, :]
            return gatherd_output

        # out_token should be a tensor with shape and dtype like a1
        return out_token

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
