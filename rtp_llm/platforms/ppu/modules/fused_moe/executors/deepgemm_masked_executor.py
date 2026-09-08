"""PPU DeepGEMM masked executors for DeepEP low-latency mode."""

import logging
from typing import Any, Dict, Optional

import torch
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
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_mul_masked_bf16_no_post_quant_fwd,
)
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.platforms.ppu.kernels.int8.deepgemm_warmup import (
    get_deep_gemm_warmup_mode,
    warmup_grouped_bf16_gemm,
    warmup_grouped_int8_gemm,
)
from rtp_llm.platforms.ppu.kernels.int8.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    deep_gemm_default_num_sms,
    has_deep_gemm_bf16_grouped,
    has_deep_gemm_int8_grouped_masked,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_int8_gemm_nt_masked,
)
from rtp_llm.platforms.ppu.kernels.int8.int8_kernel.fused_silu_mul_int8_quant import (
    silu_and_mul_masked_per_token_quant_int8_fwd,
)
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


class DeepGemmBf16MaskedExecutor(FusedMoeExpertExecutor):
    """Consume DeepEP low-latency BF16 payload with PPU masked DeepGEMM."""

    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_MASKED

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) is None)
        checker.check(resolver.is_bf16(config))
        checker.check(has_deep_gemm_bf16_grouped())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__(config, quant_config, weights)
        self._w1 = weights[W.moe_w1]
        self._w2 = weights[W.moe_w2]
        self._w1_scale = weights.get(W.moe_s1, None)
        self._w2_scale = weights.get(W.moe_s2, None)

        self._num_experts, self._intermediate_size, self._hidden_size = self._w1.shape
        if self._intermediate_size % 2 != 0:
            raise ValueError("BF16 MoE gate/up dimension must be even")
        expected_w2_shape = (
            self._num_experts,
            self._hidden_size,
            self._intermediate_size // 2,
        )
        if self._w2.shape != expected_w2_shape:
            raise ValueError(
                f"BF16 MoE w2 shape {self._w2.shape} != {expected_w2_shape}"
            )
        if self._w1.dtype != torch.bfloat16 or self._w2.dtype != torch.bfloat16:
            raise ValueError(
                f"BF16 MoE expects BF16 weights, got {self._w1.dtype}/{self._w2.dtype}"
            )
        if self._w1_scale is not None or self._w2_scale is not None:
            raise ValueError("BF16 masked executor does not expect quant scales")

        self._num_gemm_sms = deep_gemm_default_num_sms()
        self._warmup_bf16_kernels()

    def _warmup_bf16_kernels(self) -> None:
        try:
            warmup_mode = get_deep_gemm_warmup_mode()
            if warmup_mode == "skip":
                return
            max_tp_tokens = max(
                1,
                (self.config.ll_num_max_token + self.config.tp_size - 1)
                // self.config.tp_size,
            )
            max_expected_m = max(
                1,
                max_tp_tokens
                * self.config.ep_size
                * self.config.moe_k
                // self.config.expert_num,
            )
            for weight in (self._w1, self._w2):
                warmup_grouped_bf16_gemm(
                    weight,
                    max_m=max_expected_m,
                    layout="masked",
                    mode=warmup_mode,
                    num_sms=self._num_gemm_sms,
                )
        except Exception:
            logger.exception(
                "DeepGEMM BF16 LL warmup failed; execution will use lazy JIT"
            )

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        if activation not in ("silu", "SiGLU"):
            raise ValueError(
                f"BF16 masked executor only supports gated SiLU, got {activation}"
            )
        if apply_router_weight_on_input:
            raise ValueError("DeepEP LL applies router weights during combine")
        if payload.expert_tokens_meta is None:
            raise ValueError("DeepEP LL payload is missing token metadata")
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        if masked_m is None:
            raise ValueError("DeepEP LL payload is missing per-expert token counts")

        expert_x = payload.expert_x
        expert_x_scale = payload.expert_x_scale
        if expert_x.dtype != torch.bfloat16:
            raise ValueError(f"DeepEP LL expected BF16 input, got {expert_x.dtype}")
        if expert_x_scale is not None:
            raise ValueError("BF16 DeepEP LL input must not carry quant scale")
        if expert_x.shape[0] != self._num_experts:
            raise ValueError(
                f"DeepEP returned {expert_x.shape[0]} experts, expected {self._num_experts}"
            )
        if expert_x.shape[2] != self._hidden_size:
            raise ValueError(
                f"DeepEP hidden size {expert_x.shape[2]} != {self._hidden_size}"
            )
        if masked_m.shape != (self._num_experts,):
            raise ValueError(
                f"masked_m shape {masked_m.shape} != ({self._num_experts},)"
            )

        max_tokens = expert_x.shape[1]
        expected_m = payload.expert_tokens_meta.expected_m
        expected_m = max_tokens if expected_m is None else min(max_tokens, expected_m)
        device = expert_x.device

        with configure_deep_gemm_num_sms(self._num_gemm_sms):
            gateup_output = torch.empty(
                (self._num_experts, max_tokens, self._intermediate_size),
                device=device,
                dtype=torch.bfloat16,
            )
            m_grouped_bf16_gemm_nt_masked(
                expert_x,
                self._w1,
                gateup_output,
                masked_m,
                expected_m,
            )
            dispose_tensor(expert_x)

            down_input = torch.empty(
                (self._num_experts, max_tokens, self._intermediate_size // 2),
                device=device,
                dtype=torch.bfloat16,
            )
            silu_mul_masked_bf16_no_post_quant_fwd(
                input=gateup_output,
                output=down_input,
                masked_m=masked_m,
                expected_m=expected_m,
                group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
            )
            dispose_tensor(gateup_output)

            down_output = torch.empty(
                (self._num_experts, max_tokens, self._hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
            m_grouped_bf16_gemm_nt_masked(
                down_input,
                self._w2,
                down_output,
                masked_m,
                expected_m,
            )
            dispose_tensor(down_input)

        return CombineForwardPayload(fused_expert_output=down_output)


class DeepGemmInt8MaskedExecutor(FusedMoeExpertExecutor):
    """Consume DeepEP's fixed-capacity INT8 payload without CPU synchronization."""

    QUANT_METHOD = "W8A8_INT8_PER_CHANNEL_COMPRESSED"

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_MASKED

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        checker.check(resolver.is_bf16(config))
        checker.check(has_deep_gemm_int8_grouped_masked())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self._w1 = weights[W.moe_w1]
        self._w2 = weights[W.moe_w2]
        self._w1_scale = weights[W.moe_s1]
        self._w2_scale = weights[W.moe_s2]

        self._num_experts, self._intermediate_size, self._hidden_size = self._w1.shape
        if self._intermediate_size % 2 != 0:
            raise ValueError("W8A8 MoE gate/up dimension must be even")
        expected_w2_shape = (
            self._num_experts,
            self._hidden_size,
            self._intermediate_size // 2,
        )
        if self._w2.shape != expected_w2_shape:
            raise ValueError(
                f"W8A8 MoE w2 shape {self._w2.shape} != {expected_w2_shape}"
            )
        if self._w1.dtype != torch.int8 or self._w2.dtype != torch.int8:
            raise ValueError(
                f"W8A8 MoE expects INT8 weights, got "
                f"{self._w1.dtype}/{self._w2.dtype}"
            )
        if (
            self._w1_scale.dtype != torch.float32
            or self._w2_scale.dtype != torch.float32
        ):
            raise ValueError(
                "W8A8 MoE expects FP32 weight scales, got "
                f"{self._w1_scale.dtype}/{self._w2_scale.dtype}"
            )
        if self._w1_scale.shape != (*self._w1.shape[:-1], 1):
            raise ValueError(
                f"w1 scale shape {self._w1_scale.shape} does not match "
                f"{self._w1.shape}"
            )
        if self._w2_scale.shape != (*self._w2.shape[:-1], 1):
            raise ValueError(
                f"w2 scale shape {self._w2_scale.shape} does not match "
                f"{self._w2.shape}"
            )

        self._num_gemm_sms = deep_gemm_default_num_sms()
        self._warmup_int8_kernels()

    def _warmup_int8_kernels(self) -> None:
        try:
            warmup_mode = get_deep_gemm_warmup_mode()
            if warmup_mode == "skip":
                return
            max_tp_tokens = max(
                1,
                (self.config.ll_num_max_token + self.config.tp_size - 1)
                // self.config.tp_size,
            )
            max_expected_m = max(
                1,
                max_tp_tokens
                * self.config.ep_size
                * self.config.moe_k
                // self.config.expert_num,
            )
            for weight in (
                (self._w1, self._w1_scale),
                (self._w2, self._w2_scale),
            ):
                warmup_grouped_int8_gemm(
                    weight,
                    max_m=max_expected_m,
                    layout="masked",
                    mode=warmup_mode,
                    num_sms=self._num_gemm_sms,
                )
        except Exception:
            logger.exception(
                "DeepGEMM INT8 LL warmup failed; execution will use lazy JIT"
            )

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        if activation not in ("silu", "SiGLU"):
            raise ValueError(
                f"W8A8 masked executor only supports gated SiLU, got {activation}"
            )
        if apply_router_weight_on_input:
            raise ValueError("DeepEP LL applies router weights during combine")
        if payload.expert_tokens_meta is None:
            raise ValueError("DeepEP LL payload is missing token metadata")
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        if masked_m is None:
            raise ValueError("DeepEP LL payload is missing per-expert token counts")

        expert_x = payload.expert_x
        expert_x_scale = payload.expert_x_scale
        if expert_x.dtype != torch.int8:
            raise ValueError(f"DeepEP LL expected INT8 input, got {expert_x.dtype}")
        if expert_x_scale is None or expert_x_scale.dtype != torch.float32:
            raise ValueError("DeepEP LL expected an FP32 per-token input scale")
        if expert_x.shape[0] != self._num_experts:
            raise ValueError(
                f"DeepEP returned {expert_x.shape[0]} experts, expected "
                f"{self._num_experts}"
            )
        if expert_x.shape[2] != self._hidden_size:
            raise ValueError(
                f"DeepEP hidden size {expert_x.shape[2]} != {self._hidden_size}"
            )
        if expert_x_scale.shape != (*expert_x.shape[:-1], 1):
            raise ValueError(
                f"input scale shape {expert_x_scale.shape} does not match "
                f"{expert_x.shape}"
            )
        if masked_m.shape != (self._num_experts,):
            raise ValueError(
                f"masked_m shape {masked_m.shape} != ({self._num_experts},)"
            )

        max_tokens = expert_x.shape[1]
        expected_m = payload.expert_tokens_meta.expected_m
        expected_m = max_tokens if expected_m is None else min(max_tokens, expected_m)
        device = expert_x.device

        with configure_deep_gemm_num_sms(self._num_gemm_sms):
            gateup_output = torch.empty(
                (self._num_experts, max_tokens, self._intermediate_size),
                device=device,
                dtype=torch.bfloat16,
            )
            m_grouped_int8_gemm_nt_masked(
                (expert_x, expert_x_scale),
                (self._w1, self._w1_scale),
                gateup_output,
                masked_m,
                expected_m,
            )
            dispose_tensor(expert_x)
            dispose_tensor(expert_x_scale)

            down_q = torch.empty(
                (
                    self._num_experts,
                    max_tokens,
                    self._intermediate_size // 2,
                ),
                device=device,
                dtype=torch.int8,
            )
            down_scale = torch.empty(
                (self._num_experts, max_tokens, 1),
                device=device,
                dtype=torch.float32,
            )
            silu_and_mul_masked_per_token_quant_int8_fwd(
                input=gateup_output,
                output=down_q,
                output_scale=down_scale,
                masked_m=masked_m,
            )
            dispose_tensor(gateup_output)

            down_output = torch.empty(
                (self._num_experts, max_tokens, self._hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
            m_grouped_int8_gemm_nt_masked(
                (down_q, down_scale),
                (self._w2, self._w2_scale),
                down_output,
                masked_m,
                expected_m,
            )
            dispose_tensor(down_q)
            dispose_tensor(down_scale)

        return CombineForwardPayload(fused_expert_output=down_output)
