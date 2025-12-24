from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    is_deep_gemm_e8m0_used,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0
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
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_mul_masked_bf16_no_post_quant_fwd,
    silu_mul_masked_fp8_post_quant_fwd,
)
from rtp_llm.models_py.utils.arch import get_num_device_sms
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.utils.model_weight import W


@dataclass
class MaskedGroupedFFNForwardPayload:
    expert_x: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int
    upgate_output: torch.Tensor
    down_input: torch.Tensor
    down_output: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    expert_x_scale: Optional[torch.Tensor] = (None,)
    down_input_scale: Optional[torch.Tensor] = (None,)
    w1_scale: Optional[torch.Tensor] = (None,)
    w2_scale: Optional[torch.Tensor] = (None,)


class DeepGemmMaskedExecutor(FusedMoeExpertExecutor):

    # The Deep Gemm kernels only support block size of 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls):
        return ExecutorType.DEEPGEMM_MASKED

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepGemmMaskedExecutor can handle the configuration"""
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(has_deep_gemm())
        checker.check(resolver.is_bf16(config))
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method in [None, "FP8_PER_BLOCK"])

    def __init__(
        self,
        config: MoEConfigAdapter,
        weights: Dict[str, torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ):
        """Initialize the DeepGemmMaskedExecutor.
        Args:
            config: Model configuration.
            weights: Dictionary containing model weights.
            quant_config: Quantization configuration.
        """
        super().__init__(quant_config=quant_config)
        self._config = config
        self._weights = weights
        # Initialize w1 and w2
        self._w1 = self._weights.get(W.moe_w1, None)
        self._w2 = self._weights.get(W.moe_w2, None)
        assert self._w1 is not None and self._w2 is not None
        # Check w1 and w2 shape
        self._E, self._N, self._K = self._w1.size()
        assert self._N % 2 == 0
        assert self._w2.size(0) == self._E
        assert self._w2.size(1) == self._K
        assert self._w2.size(2) == self._N // 2
        # Initialize w1 and w2 scale
        self._w1_scale = self._weights.get(W.moe_s1, None)
        self._w2_scale = self._weights.get(W.moe_s2, None)
        # Check fp8 block quantization
        self._use_fp8 = True
        if self.quant_config.is_quantized:
            assert self._w1_scale is not None and self._w2_scale is not None
            if (
                self.quant_config.quant_dtype == torch.float8_e4m3fn
                and self.quant_config.is_block_quantized
                and self.quant_config.block_shape == self.DEEPGEMM_BLOCK_SHAPE
            ):
                # Confirm to use fp8 block quantization
                self._use_fp8 = True
                self._num_squeeze_values = 1
                # Whether use fp8 block quantization with UE8M0 scale
                if is_deep_gemm_e8m0_used():
                    self._w1, self._w1_scale = requant_weight_ue8m0(
                        self._w1, self._w1_scale
                    )
                    self._w2, self._w2_scale = requant_weight_ue8m0(
                        self._w2, self._w2_scale
                    )
                    self._num_squeeze_values = 4
                # Check w1_scale and w2_scale shape
                assert (
                    self._w1_scale.size(0) == self._E
                    and self._w2_scale.size(0) == self._E
                )
                assert (
                    self._w1_scale.size(1)
                    == self._N
                    // self.DEEPGEMM_BLOCK_SHAPE[0]
                    // self._num_squeeze_values
                )
                assert (
                    self._w1_scale.size(2)
                    == self._K
                    // self.DEEPGEMM_BLOCK_SHAPE[1]
                    // self._num_squeeze_values
                )
                assert (
                    self._w2_scale.size(1)
                    == self._K
                    // self.DEEPGEMM_BLOCK_SHAPE[1]
                    // self._num_squeeze_values
                )
                assert (
                    self._w2_scale.size(2)
                    == self._N
                    // 2
                    // self.DEEPGEMM_BLOCK_SHAPE[0]
                    // self._num_squeeze_values
                )
            else:
                raise NotImplementedError(
                    "DeepGemmMaskedExecutor only supports fp8 block quantization with block shape 128x128"
                )
        else:
            # Confirm to use bf16
            self._use_fp8 = False
            assert self._w1_scale is None and self._w2_scale is None
        # Initialize number of SMs for DeepGEMM
        self._num_gemm_sms = get_num_device_sms()

    @property
    def local_num_experts(self) -> int:
        assert self._w1 is not None
        return self._w1.size(0)

    def _forward_masked_grouped_ffn(
        self,
        payload: MaskedGroupedFFNForwardPayload,
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Forward masked grouped FFN.
        Args:
            payload (MaskedGroupedFFNForwardPayload): Payload for masked grouped FFN forward.
            start_idx (int): Start index of the expert.
            end_idx (int): End index of the expert.
        """
        with configure_deep_gemm_num_sms(self._num_gemm_sms):
            if self._use_fp8:
                # Gate and Up GroupGEMM-0
                m_grouped_fp8_gemm_nt_masked(
                    (
                        payload.expert_x[start_idx:end_idx],
                        payload.expert_x_scale[start_idx:end_idx],
                    ),
                    (
                        payload.w1[start_idx:end_idx],
                        payload.w1_scale[start_idx:end_idx],
                    ),
                    payload.upgate_output[start_idx:end_idx],
                    payload.masked_m[start_idx:end_idx],
                    payload.expected_m,
                )
                # SiLU Activation
                silu_mul_masked_fp8_post_quant_fwd(
                    input=payload.upgate_output[start_idx:end_idx],
                    output=payload.down_input[start_idx:end_idx],
                    output_scale=payload.down_input_scale[start_idx:end_idx],
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    masked_m=payload.masked_m[start_idx:end_idx],
                    expected_m=payload.expected_m,
                    scale_ue8m0=is_deep_gemm_e8m0_used(),
                )
                # Down GroupGEMM-1
                m_grouped_fp8_gemm_nt_masked(
                    (
                        payload.down_input[start_idx:end_idx],
                        payload.down_input_scale[start_idx:end_idx],
                    ),
                    (
                        payload.w2[start_idx:end_idx],
                        payload.w2_scale[start_idx:end_idx],
                    ),
                    payload.down_output[start_idx:end_idx],
                    payload.masked_m[start_idx:end_idx],
                    payload.expected_m,
                )
            else:
                # Gate and Up GroupGEMM-0
                m_grouped_bf16_gemm_nt_masked(
                    payload.expert_x[start_idx:end_idx],
                    payload.w1[start_idx:end_idx],
                    payload.upgate_output[start_idx:end_idx],
                    payload.masked_m[start_idx:end_idx],
                    payload.expected_m,
                )
                # SiLU Activation
                silu_mul_masked_bf16_no_post_quant_fwd(
                    input=payload.upgate_output[start_idx:end_idx],
                    output=payload.down_input[start_idx:end_idx],
                    masked_m=payload.masked_m[start_idx:end_idx],
                    expected_m=payload.expected_m,
                    group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                )
                # Down GroupGEMM-1
                m_grouped_bf16_gemm_nt_masked(
                    payload.down_input[start_idx:end_idx],
                    payload.w2[start_idx:end_idx],
                    payload.down_output[start_idx:end_idx],
                    payload.masked_m[start_idx:end_idx],
                    payload.expected_m,
                )

    def _normal_execute(
        self,
        ffn_payload: MaskedGroupedFFNForwardPayload,
        combine_payload: CombineForwardPayload,
    ):
        """Execute normal masked grouped FFN.
        Args:
            ffn_payload (MaskedGroupedFFNForwardPayload): Payload for masked grouped FFN forward.
            combine_payload (CombineForwardPayload): Payload for combining expert outputs.
        """
        # Execute masked grouped FFN
        self._forward_masked_grouped_ffn(ffn_payload, 0, self._E)
        # Save expert output
        combine_payload.fused_expert_output = ffn_payload.down_output

    def _execute_fp8(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        """Execute FP8 experts computation.
        Args:
            payload (ExpertForwardPayload): Payload for expert computation.
            activation (str): Activation function.
            expert_map (Optional[torch.Tensor]): Expert map.
            a2_scale (Optional[torch.Tensor]): Scale for a2.
            apply_router_weight_on_input (bool): Whether to apply router weight on input.
            extra_expert_args (Optional[dict[str, Any]]): Extra expert arguments.
        """
        # Check and prepare payload data
        expert_x = payload.expert_x
        E, M, K = expert_x.size()
        assert E == self._E and K == self._K
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        assert len(masked_m) == self._E
        expected_m = (
            payload.expert_tokens_meta.expected_m
            if payload.expert_tokens_meta.expected_m is not None
            else M
        )
        expert_x_scale = payload.expert_x_scale
        assert expert_x_scale is not None
        assert expert_x_scale.size(0) == E
        assert expert_x_scale.size(1) == M
        assert expert_x_scale.size(2) == K // self.DEEPGEMM_BLOCK_SHAPE[1]

        # Initialize intermediate tensors
        current_device = expert_x.device
        upgate_output = torch.empty(
            (E, M, self._N), device=current_device, dtype=torch.bfloat16
        )
        down_input = torch.empty(
            (E, M, self._N // 2), device=current_device, dtype=torch.float8_e4m3fn
        )
        down_input_scale = torch.empty(
            (E, M, self._N // 2 // self.DEEPGEMM_BLOCK_SHAPE[0]),
            device=current_device,
            dtype=torch.float32,
        )
        down_output = torch.empty(
            (E, M, K), device=current_device, dtype=torch.bfloat16
        )

        # Initialize ffn payload
        ffn_payload = MaskedGroupedFFNForwardPayload(
            expert_x=expert_x,
            masked_m=masked_m,
            expected_m=expected_m,
            upgate_output=upgate_output,
            down_input=down_input,
            down_output=down_output,
            w1=self._w1,
            w2=self._w2,
            expert_x_scale=expert_x_scale,
            down_input_scale=down_input_scale,
            w1_scale=self._w1_scale,
            w2_scale=self._w2_scale,
        )
        # Initialize combine payload
        combine_payload = CombineForwardPayload()

        # Execute masked grouped FFN
        self._normal_execute(ffn_payload, combine_payload)

        # Free intermediate tensors
        dispose_tensor(expert_x)
        dispose_tensor(expert_x_scale)
        del upgate_output
        del down_input
        del down_input_scale

        return combine_payload

    def _execute_bf16(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        """Execute Bf16 experts computation.
        Args:
            payload (ExpertForwardPayload): Payload for expert computation.
            activation (str): Activation function.
            expert_map (Optional[torch.Tensor]): Expert map.
            a2_scale (Optional[torch.Tensor]): Scale for a2.
            apply_router_weight_on_input (bool): Whether to apply router weight on input.
            extra_expert_args (Optional[dict[str, Any]]): Extra expert arguments.
        """
        # Check and prepare payload data
        expert_x = payload.expert_x
        E, M, K = expert_x.size()
        assert E == self._E and K == self._K
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        assert len(masked_m) == self._E
        expected_m = (
            payload.expert_tokens_meta.expected_m
            if payload.expert_tokens_meta.expected_m is not None
            else M
        )

        # Initialize intermediate tensors
        current_device = expert_x.device
        upgate_output = torch.empty(
            (E, M, self._N), device=current_device, dtype=torch.bfloat16
        )
        down_input = torch.empty(
            (E, M, self._N // 2), device=current_device, dtype=torch.bfloat16
        )
        down_output = torch.empty(
            (E, M, K), device=current_device, dtype=torch.bfloat16
        )
        # Initialize ffn payload
        ffn_payload = MaskedGroupedFFNForwardPayload(
            expert_x=expert_x,
            masked_m=masked_m,
            expected_m=expected_m,
            upgate_output=upgate_output,
            down_input=down_input,
            down_output=down_output,
            w1=self._w1,
            w2=self._w2,
        )
        # Initialize combine payload
        combine_payload = CombineForwardPayload()

        # Execute masked grouped FFN
        self._normal_execute(ffn_payload, combine_payload)

        # Free intermediate tensors
        dispose_tensor(expert_x)
        del upgate_output
        del down_input

        return combine_payload

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        if self._use_fp8:
            return self._execute_fp8(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )
        else:
            return self._execute_bf16(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )
