import copy
import logging
import os
from typing import Any, Dict, Optional

import aiter
import torch
from aiter.fused_moe import fused_moe

from rtp_llm.device.device_impl import is_gfx950
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
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm._utils import (
    get_rocm_fp8_dtype,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.flydsl.tuning import (
    get_qwen_ptpc_fp8_tuning,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


_FLYDSL_AVAILABLE = False
_FLYDSL_IMPORT_ATTEMPTED = False
_fused_moe_flydsl_fn = None
_FlyDSLActivationType = None
_FlyDSLQuantType = None


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _flydsl_fused_moe_enabled() -> bool:
    return _env_flag("USE_FLYDSL") and _env_flag("USE_FLYDSL_MOE")


def _load_flydsl_fused_moe() -> bool:
    global _FLYDSL_AVAILABLE
    global _FLYDSL_IMPORT_ATTEMPTED
    global _fused_moe_flydsl_fn
    global _FlyDSLActivationType
    global _FlyDSLQuantType

    if _FLYDSL_AVAILABLE:
        return True
    if _FLYDSL_IMPORT_ATTEMPTED:
        return False
    _FLYDSL_IMPORT_ATTEMPTED = True
    try:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.flydsl.fused_moe_flydsl import (
            fused_moe_flydsl as _fused_moe_flydsl_fn,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.flydsl.fused_moe_helper import (
            ActivationType as _FlyDSLActivationType,
            QuantType as _FlyDSLQuantType,
        )

        _FLYDSL_AVAILABLE = True
    except ImportError as e:
        logger.debug("FlyDSL fused_moe failed to import, falling back to AITER: %s", e)
    return _FLYDSL_AVAILABLE


def _moe_activation_type(activation: str) -> aiter.ActivationType:
    if activation in ("silu", "SiGLU"):
        return aiter.ActivationType.Silu
    return aiter.ActivationType.Gelu


def _flatten_per_channel_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 3 and scale.shape[-1] == 1:
        return scale.reshape(scale.shape[0], scale.shape[1]).contiguous()
    return scale


def _flydsl_fused_moe_unsupported_reason(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    effective_expert_mask: Optional[torch.Tensor],
    expert_ids_are_local: bool,
    num_experts: int,
    local_num_experts: int,
    ep_size: int,
) -> Optional[str]:
    if activation not in ("silu", "SiGLU"):
        return f"activation={activation!r}"
    if effective_expert_mask is not None:
        return "expert_mask is set"
    if expert_ids_are_local:
        return "expert ids are already local"
    if ep_size != 1 or num_experts != local_num_experts:
        return f"non-pure-TP experts ep_size={ep_size}, global={num_experts}, local={local_num_experts}"
    if hidden_states.dim() != 2 or topk_ids.dim() != 2:
        return f"unexpected dims hidden={tuple(hidden_states.shape)}, topk_ids={tuple(topk_ids.shape)}"
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        return f"hidden dtype={hidden_states.dtype}"
    if w1.dim() != 3 or w2.dim() != 3:
        return f"unexpected weight dims w1={tuple(w1.shape)}, w2={tuple(w2.shape)}"

    batch_m = hidden_states.shape[0]
    experts, w1_out, model_dim = w1.shape
    w2_model_dim, inter_dim = w2.shape[1], w2.shape[2]
    topk = topk_ids.shape[1]
    if (experts, topk, model_dim) != (512, 10, 4096):
        return f"shape E={experts}, topk={topk}, model_dim={model_dim}"
    if w1_out != 2 * inter_dim or w2_model_dim != model_dim:
        return f"inconsistent weights w1={tuple(w1.shape)}, w2={tuple(w2.shape)}"
    tuning = get_qwen_ptpc_fp8_tuning(inter_dim)
    if tuning is None:
        return f"inter_dim={inter_dim}"

    max_m = tuning.max_m
    if max_m > 0 and batch_m > max_m:
        return f"M={batch_m} exceeds FlyDSL tuned max M={max_m}"
    return None


def _log_flydsl_fallback(reason: str) -> None:
    logger.debug("Falling back to AITER fused_moe: %s", reason)


def _should_use_flydsl_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    effective_expert_mask: Optional[torch.Tensor],
    expert_ids_are_local: bool,
    num_experts: int,
    local_num_experts: int,
    ep_size: int,
) -> bool:
    if not _flydsl_fused_moe_enabled():
        return False

    reason = _flydsl_fused_moe_unsupported_reason(
        hidden_states,
        w1,
        w2,
        topk_ids,
        activation=activation,
        effective_expert_mask=effective_expert_mask,
        expert_ids_are_local=expert_ids_are_local,
        num_experts=num_experts,
        local_num_experts=local_num_experts,
        ep_size=ep_size,
    )
    if reason is not None:
        _log_flydsl_fallback(reason)
        return False
    if not _load_flydsl_fused_moe():
        return False
    return True


def _aiter_fp8_per_channel_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    activation: str,
    effective_expert_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        quant_type=aiter.QuantType.per_Token,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        activation=_moe_activation_type(activation),
        expert_mask=effective_expert_mask,
    )


def _flydsl_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    s1 = _flatten_per_channel_scale(w1_scale)
    s2 = _flatten_per_channel_scale(w2_scale)

    return _fused_moe_flydsl_fn(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=_FlyDSLActivationType.Silu,
        quant_type=_FlyDSLQuantType.per_Token,
        w1_scale=s1,
        w2_scale=s2,
    )


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

        self.expert_mask = build_ep_expert_mask(self.num_experts, self.ep_rank, self.ep_size, self.w1)

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

        # When router has already remapped IDs to local indices (e.g. MoriEpIntranodeRouter),
        # expert_mask (which is based on global IDs) must not be applied.
        effective_expert_mask = (
            None if payload.expert_ids_are_local else (expert_map if expert_map is not None else self.expert_mask)
        )

        hidden_states = payload.expert_x

        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
            topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

        output = fused_moe(
            hidden_states,
            self.w1,
            self.w2,
            topk_weights,
            topk_ids,
            activation=_moe_activation_type(activation),
            expert_mask=effective_expert_mask,
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
        checker.check(quant_method in ("FP8_PER_CHANNEL_COMPRESSED", "FP8_PER_CHANNEL_QUARK"))

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

        # Avoid mutating the strategy-shared FusedMoEQuantConfig instance.
        self.quant_config = copy.copy(self.quant_config)
        self.quant_config.quant_dtype = get_rocm_fp8_dtype()
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

        self.expert_mask = build_ep_expert_mask(self.num_experts, self.ep_rank, self.ep_size, self.w1)

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

        effective_expert_mask = (
            None if payload.expert_ids_are_local else (expert_map if expert_map is not None else self.expert_mask)
        )

        hidden_states = payload.expert_x
        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
            topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

        if _should_use_flydsl_fused_moe(
            hidden_states,
            self.w1,
            self.w2,
            topk_ids,
            activation=activation,
            effective_expert_mask=effective_expert_mask,
            expert_ids_are_local=payload.expert_ids_are_local,
            num_experts=self.num_experts,
            local_num_experts=self.local_num_experts,
            ep_size=self.ep_size,
        ):
            try:
                output = _flydsl_fused_moe(
                    hidden_states,
                    self.w1,
                    self.w2,
                    topk_weights,
                    topk_ids,
                    w1_scale=self.w1_scale,
                    w2_scale=self.w2_scale,
                )
            except (NotImplementedError, ValueError) as e:
                _log_flydsl_fallback(str(e))
                output = _aiter_fp8_per_channel_fused_moe(
                    hidden_states,
                    self.w1,
                    self.w2,
                    topk_weights,
                    topk_ids,
                    self.w1_scale,
                    self.w2_scale,
                    activation,
                    effective_expert_mask,
                )
        else:
            output = _aiter_fp8_per_channel_fused_moe(
                hidden_states,
                self.w1,
                self.w2,
                topk_weights,
                topk_ids,
                self.w1_scale,
                self.w2_scale,
                activation,
                effective_expert_mask,
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
        checker.check(quant_method in ("FP8_PER_BLOCK", "FP8_PER_BLOCK_QUARK"))

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

        # Avoid mutating the strategy-shared FusedMoEQuantConfig instance.
        # block_shape is left as set by the strategy ([128, 128]) so that
        # FusedMoEQuantConfig.is_block_quantized() stays consistent.
        self.quant_config = copy.copy(self.quant_config)
        self.quant_config.quant_dtype = get_rocm_fp8_dtype()
        self.quant_config.per_act_token_quant = False
        self.quant_config.per_out_ch_quant = False

        self.num_experts = config.expert_num
        self.ep_rank = config.ep_rank
        self.ep_size = config.ep_size
        # Extract weights from dictionary
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.w1_scale = weights[W.moe_s1]
        self.w2_scale = weights[W.moe_s2]

        self.expert_mask = build_ep_expert_mask(self.num_experts, self.ep_rank, self.ep_size, self.w1)

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

        effective_expert_mask = (
            None if payload.expert_ids_are_local else (expert_map if expert_map is not None else self.expert_mask)
        )

        hidden_states = payload.expert_x

        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
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
            expert_mask=effective_expert_mask,
        )

        return CombineForwardPayload(fused_expert_output=output)


class RocmExpertsFp4PerGroup(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        """Check if RocmExpertsFp4PerGroup can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method
            in (
                "FP4_PER_GROUP",
                "FP4_PER_GROUP_QUARK",
                "modelopt_fp4",
            )
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

        # Update quant_config with FP4-specific settings
        self.quant_config.quant_dtype = torch.float4_e2m1fn_x2
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

        self.hidden_size_raw = config.hidden_size
        self.intermediate_size_raw = config.model_config.moe_inter_size // config.tp_size
        packed_factor = 2 if self.w1.dtype == torch.uint8 else 1
        self.hidden_size_padded = self.w1.size(2) * packed_factor
        self.intermediate_size_padded = self.w2.size(1)
        self.hidden_pad = max(0, self.hidden_size_padded - self.hidden_size_raw)
        self.intermediate_pad = max(0, self.intermediate_size_padded - self.intermediate_size_raw)

        self.expert_mask = build_ep_expert_mask(self.num_experts, self.ep_rank, self.ep_size, self.w1)

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
        # assert hidden_size in (self.hidden_size_raw, self.hidden_size_padded), (
        #     f"Hidden size mismatch {hidden_size}, expected raw/padded size "
        #     f"{self.hidden_size_raw}/{self.hidden_size_padded}"
        # )
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
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
            topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

        # view w1 and w2 to float4_e2m1fn_x2 if they are uint8
        w1 = self.w1
        w2 = self.w2
        if w1.dtype == torch.uint8:
            w1 = w1.view(torch.float4_e2m1fn_x2)
            w1.is_shuffled = True
        if w2.dtype == torch.uint8:
            w2 = w2.view(torch.float4_e2m1fn_x2)
            w2.is_shuffled = True

        effective_expert_mask = (
            None if payload.expert_ids_are_local else (expert_map if expert_map is not None else self.expert_mask)
        )

        # CK moe_sorting kernel requires int32 topk_ids
        topk_ids = topk_ids.to(torch.int32)

        output = fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            quant_type=aiter.QuantType.per_1x32,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            activation=_moe_activation_type(activation),
            expert_mask=effective_expert_mask,
            doweight_stage1=apply_router_weight_on_input,
        )

        return CombineForwardPayload(fused_expert_output=output)


class RocmExpertsMXFp4(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        """Check if the ROCm MXFP4 executor can handle the configuration."""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "QuarkMXFP4")
        checker.check(is_gfx950())

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

        # Update quant_config with FP4-specific settings
        self.quant_config.quant_dtype = torch.float4_e2m1fn_x2
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

        self.hidden_size_raw = config.hidden_size
        self.intermediate_size_raw = config.model_config.moe_inter_size // config.tp_size
        packed_factor = 2 if self.w1.dtype == torch.uint8 else 1
        self.hidden_size_padded = self.w1.size(2) * packed_factor
        self.intermediate_size_padded = self.w2.size(1)

        self.expert_mask = build_ep_expert_mask(self.num_experts, self.ep_rank, self.ep_size, self.w1)

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
        # assert hidden_size in (self.hidden_size_raw, self.hidden_size_padded), (
        #     f"Hidden size mismatch {hidden_size}, expected raw/padded size "
        #     f"{self.hidden_size_raw}/{self.hidden_size_padded}"
        # )
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
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
            topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

        # view w1 and w2 to float4_e2m1fn_x2 if they are uint8
        w1 = self.w1
        w2 = self.w2
        if w1.dtype == torch.uint8:
            w1 = w1.view(torch.float4_e2m1fn_x2)
            w1.is_shuffled = True
        if w2.dtype == torch.uint8:
            w2 = w2.view(torch.float4_e2m1fn_x2)
            w2.is_shuffled = True

        effective_expert_mask = (
            None if payload.expert_ids_are_local else (expert_map if expert_map is not None else self.expert_mask)
        )

        output = fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            quant_type=aiter.QuantType.per_1x32,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            activation=_moe_activation_type(activation),
            expert_mask=effective_expert_mask,
            doweight_stage1=apply_router_weight_on_input,
        )

        return CombineForwardPayload(fused_expert_output=output)
