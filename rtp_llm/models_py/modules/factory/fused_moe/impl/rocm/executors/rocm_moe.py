from typing import Any, Dict, Optional

import aiter
import torch
from aiter.fused_moe import fused_moe

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.utils.model_weight import W


def _moe_activation_type(activation: str) -> aiter.ActivationType:
    if activation in ("silu", "SiGLU"):
        return aiter.ActivationType.Silu
    return aiter.ActivationType.Gelu


def build_ep_expert_mask(
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    w1: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Mask for aiter fused_moe when EP>1: global topk_ids, local w1/w2 rows.
    Length ``num_experts``; entries for experts owned by this rank are 1, else 0.
    """
    if ep_size <= 1:
        return None
    local_e = w1.size(0)
    start = ep_rank * local_e
    end = start + local_e
    mask = torch.zeros(num_experts, dtype=torch.int32, device=w1.device)
    mask[start:end] = 1
    return mask


class RocmExpertsBf16(FusedMoeExpertExecutor):
    """ROCm BF16 (no quantization) MoE expert executor."""

    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        """Check if RocmExpertsBf16 can handle the configuration"""
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.num_experts = config.expert_num
        self.ep_rank = config.ep_rank
        self.ep_size = config.ep_size
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]

        self.expert_mask = build_ep_expert_mask(
            self.num_experts, self.ep_rank, self.ep_size, self.w1
        )

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "expert_x is None"
        assert payload.expert_x.size(-1) == self.w1.size(
            2
        ), f"Hidden size mismatch {payload.expert_x.size(-1)} != {self.w1.size(2)}"
        assert payload.expert_x.is_contiguous(), "Hidden_states must be contiguous"
        assert self.w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert self.w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert payload.expert_tokens_meta is not None

        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        assert topk_ids is not None
        assert topk_weights is not None

        assert self.w1.size(0) == self.local_num_experts
        assert self.w2.size(0) == self.local_num_experts

        hidden_states = payload.expert_x

        if apply_router_weight_on_input:
            assert (
                topk_weights.dim() == 2
            ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
            topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

        output = fused_moe(
            hidden_states,
            self.w1,
            self.w2,
            topk_weights,
            topk_ids,
            activation=_moe_activation_type(activation),
            expert_mask=expert_map if expert_map is not None else self.expert_mask,
        )

        return CombineForwardPayload(fused_expert_output=output)


class RocmExpertsFp8PerChannel(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        """Check if RocmExpertsFp8PerChannel can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method in ("FP8_PER_CHANNEL_COMPRESSED", "FP8_PER_CHANNEL_QUARK")
        )

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        # Update quant_config with FP8-specific settings
        self.quant_config.quant_dtype = torch.float8_e4m3fnuz
        self.quant_config.per_act_token_quant = True
        self.quant_config.per_out_ch_quant = True
        self.quant_config.block_shape = None

        self.num_experts = config.expert_num
        self.ep_rank = config.ep_rank
        self.ep_size = config.ep_size
        # Extract weights from dictionary
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.w1_scale = weights[W.moe_s1]
        self.w2_scale = weights[W.moe_s2]

        self.expert_mask = build_ep_expert_mask(
            self.num_experts, self.ep_rank, self.ep_size, self.w1
        )

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "expert_x is None"
        assert payload.expert_x.size(-1) == self.w1.size(
            2
        ), f"Hidden size mismatch {payload.expert_x.size(-1)} != {self.w1.size(2)}"

        assert payload.expert_x.is_contiguous(), "Hidden_states must be contiguous"
        assert self.w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert self.w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert payload.expert_tokens_meta is not None

        E = self.local_num_experts
        assert payload.expert_topk_ids is not None

        assert self.w1.size(0) == E
        assert self.w2.size(0) == E

        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        assert topk_ids is not None
        assert topk_weights is not None
        assert self.w1.size(0) == self.local_num_experts
        assert self.w2.size(0) == self.local_num_experts

        hidden_states = payload.expert_x
        if apply_router_weight_on_input:
            assert (
                topk_weights.dim() == 2
            ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
            topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

        output = fused_moe(
            hidden_states,
            self.w1,
            self.w2,
            topk_weights,
            topk_ids,
            quant_type=aiter.QuantType.per_Token,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            activation=_moe_activation_type(activation),
            expert_mask=expert_map if expert_map is not None else self.expert_mask,
        )

        return CombineForwardPayload(fused_expert_output=output)


class RocmExpertsFp8PerBlock(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        """Check if RocmExpertsFp8PerBlock can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method in ("FP8_PER_BLOCK",))

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        # Update quant_config with FP8-specific settings
        self.quant_config.quant_dtype = torch.float8_e4m3fnuz
        self.quant_config.per_act_token_quant = False
        self.quant_config.per_out_ch_quant = False
        self.quant_config.block_shape = None

        self.num_experts = config.expert_num
        self.ep_rank = config.ep_rank
        self.ep_size = config.ep_size
        # Extract weights from dictionary
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.w1_scale = weights[W.moe_s1]
        self.w2_scale = weights[W.moe_s2]

        self.expert_mask = build_ep_expert_mask(
            self.num_experts, self.ep_rank, self.ep_size, self.w1
        )

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "expert_x is None"
        assert payload.expert_x.size(-1) == self.w1.size(
            2
        ), f"Hidden size mismatch {payload.expert_x.size(-1)} != {self.w1.size(2)}"
        assert payload.expert_x.is_contiguous(), "Hidden_states must be contiguous"
        assert self.w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert self.w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert payload.expert_tokens_meta is not None

        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        assert topk_ids is not None
        assert topk_weights is not None
        assert self.w1.size(0) == self.local_num_experts
        assert self.w2.size(0) == self.local_num_experts

        hidden_states = payload.expert_x

        if apply_router_weight_on_input:
            assert (
                topk_weights.dim() == 2
            ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
            topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

        output = fused_moe(
            hidden_states,
            self.w1,
            self.w2,
            topk_weights,
            topk_ids,
            quant_type=aiter.QuantType.per_128x128,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            activation=_moe_activation_type(activation),
            expert_mask=expert_map if expert_map is not None else self.expert_mask,
        )

        return CombineForwardPayload(fused_expert_output=output)
