from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    has_deep_gemm,
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
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.common.activation import (
    create_packed_scale_tensor,
    silu_and_mul_masked_post_quant_packed_fwd,
    silu_mul_masked_bf16_no_post_quant_fwd,
    silu_mul_masked_fp8_post_quant_fwd,
)
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.utils.model_weight import W


class DeepGemmMaskedExecutor(FusedMoeExpertExecutor):

    # The Deep Gemm kernels only support block size of 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls):
        return ExecutorType.DEEPGEMM_MASKED

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepGemmMaskedExecutor can handle the configuration"""
        resolver = MoeConfigResolver()
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 9)
        checker.check(resolver.is_bf16(config))
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method in [None, "FP8_PER_BLOCK"])
        checker.check(not resolver.enable_peo(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """Initialize the DeepGemmMaskedExecutor.
        Args:
            config: Model configuration.
            quant_config: Quantization configuration.
            weights: Dictionary containing model weights.
        """
        super().__init__(config, quant_config, weights)
        # w1 and w2
        self._w1 = weights[W.moe_w1]
        self._w2 = weights[W.moe_w2]
        self._num_local_experts, self._N, self._hidden_size = self._w1.size()
        assert self._N % 2 == 0
        assert self._w2.size(0) == self._num_local_experts
        assert self._w2.size(1) == self._hidden_size
        assert self._w2.size(2) == self._N // 2
        # w1_scale and w2_scale
        self._w1_scale = weights.get(W.moe_s1, None)
        self._w2_scale = weights.get(W.moe_s2, None)
        self._use_fp8 = self.quant_config.is_quantized
        if self._use_fp8:
            assert self._w1_scale is not None and self._w2_scale is not None
            if (
                self.quant_config.quant_dtype == torch.float8_e4m3fn
                and self.quant_config.is_block_quantized
                and self.quant_config.block_shape == self.DEEPGEMM_BLOCK_SHAPE
            ):
                # Confirm to use fp8 block quantization
                self._num_packed_scales = 1
                self._scale_dtype = torch.float32
                # Whether use fp8 block quantization with UE8M0 scale
                if is_deep_gemm_e8m0_used():
                    self._w1, self._w1_scale = requant_weight_ue8m0(
                        self._w1, self._w1_scale
                    )
                    self._w2, self._w2_scale = requant_weight_ue8m0(
                        self._w2, self._w2_scale
                    )
                    self._num_packed_scales = 4
                    self._scale_dtype = torch.int32
                # Check w1_scale and w2_scale dtype
                assert (
                    self._w1_scale.dtype == self._scale_dtype
                    and self._w2_scale.dtype == self._scale_dtype
                )
                # Check w1_scale and w2_scale shape
                assert (
                    self._w1_scale.size(0) == self._num_local_experts
                    and self._w2_scale.size(0) == self._num_local_experts
                )
                assert (
                    self._w1_scale.size(1) == self._N
                    if is_deep_gemm_e8m0_used()
                    else self._N // self.DEEPGEMM_BLOCK_SHAPE[0]
                )
                assert (
                    self._w1_scale.size(2)
                    == (
                        self._hidden_size // self.DEEPGEMM_BLOCK_SHAPE[1]
                        + self._num_packed_scales
                        - 1
                    )
                    // self._num_packed_scales
                )
                assert (
                    self._w2_scale.size(1) == self._hidden_size
                    if is_deep_gemm_e8m0_used()
                    else self._hidden_size // self.DEEPGEMM_BLOCK_SHAPE[1]
                )
                assert (
                    self._w2_scale.size(2)
                    == (
                        self._N // 2 // self.DEEPGEMM_BLOCK_SHAPE[0]
                        + self._num_packed_scales
                        - 1
                    )
                    // self._num_packed_scales
                )
            else:
                raise NotImplementedError(
                    "DeepGemmMaskedExecutor only supports fp8 block quantization with block shape 128x128"
                )
        else:
            assert self._w1_scale is None and self._w2_scale is None
        # Number of SMs for DeepGEMM
        self._num_gemm_sms = get_num_device_sms()

    def _check_and_prepare_execute(
        self,
        payload: ExpertForwardPayload,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        # expert_x
        expert_x = payload.expert_x
        num_local_experts, max_m, hidden_size = expert_x.size()
        assert (
            num_local_experts == self._num_local_experts
            and hidden_size == self._hidden_size
        ), "expert_x shape is not correct"
        # masked_m
        assert (
            payload.expert_tokens_meta is not None
            and payload.expert_tokens_meta.expert_num_tokens is not None
        )
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        assert len(masked_m) == num_local_experts
        # expected_m
        expected_m = (
            min(max_m, payload.expert_tokens_meta.expected_m)
            if payload.expert_tokens_meta.expected_m is not None
            else max_m
        )
        # expert_x_scale
        expert_x_scale = payload.expert_x_scale
        if self._use_fp8:
            # expert_x dtype is fp8
            assert expert_x.dtype == torch.float8_e4m3fn
            assert expert_x_scale is not None
            assert expert_x_scale.size(0) == num_local_experts
            assert expert_x_scale.size(1) == max_m
            assert (
                expert_x_scale.size(2)
                == (
                    hidden_size // self.DEEPGEMM_BLOCK_SHAPE[1]
                    + self._num_packed_scales
                    - 1
                )
                // self._num_packed_scales
            )
            assert expert_x_scale.dtype == self._scale_dtype
        else:
            # expert_x dtype is bf16
            assert expert_x.dtype == torch.bfloat16
            assert expert_x_scale is None

        return expert_x, masked_m, expected_m, expert_x_scale

    def _forward_masked_grouped_ffn(
        self,
        start_idx: int,
        end_idx: int,
        expert_x: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        down_output: Optional[torch.Tensor] = None,
        expert_x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward masked grouped FFN.
        Args:
            start_idx (int): Start index of the experts.
            end_idx (int): End index of the experts.
            expert_x (torch.Tensor): Expert input.
            masked_m (torch.Tensor): Masked input.
            expected_m (int): Expected output.
            down_output (Optional[torch.Tensor]): Down output.
            expert_x_scale (Optional[torch.Tensor]): Expert input scale.
        Returns:
            torch.Tensor: Down output.
        """
        # Get metadata
        device = expert_x.device
        num_local_experts, max_m, hidden_size = expert_x.size()
        num_slice_experts = end_idx - start_idx

        # Execute masked grouped FFN
        if self._use_fp8:
            # Allocate upgate output
            upgate_output = torch.empty(
                (num_slice_experts, max_m, self._N),
                device=device,
                dtype=torch.bfloat16,
            )

            # Gate and Up GroupGEMM-0
            m_grouped_fp8_gemm_nt_masked(
                a=(expert_x[start_idx:end_idx], expert_x_scale[start_idx:end_idx]),
                b=(self._w1[start_idx:end_idx], self._w1_scale[start_idx:end_idx]),
                output=upgate_output,
                masked_m=masked_m[start_idx:end_idx],
                expected_m=expected_m,
                disable_ue8m0_cast=(not is_deep_gemm_e8m0_used()),
            )

            # Free expert_x and expert_x_scale
            if end_idx == self._num_local_experts:
                dispose_tensor(expert_x)
                dispose_tensor(expert_x_scale)
            # Allocate down_input
            down_input = torch.empty(
                (num_slice_experts, max_m, self._N // 2),
                device=device,
                dtype=torch.float8_e4m3fn,
            )

            # SM100 (compute capability 10.x) uses fused packed kernel for better performance
            # when UE8M0 scale format is enabled
            sm_major = torch.cuda.get_device_capability()[0]
            if (
                sm_major == 10
                and is_deep_gemm_e8m0_used()
                and self._N % (self.DEEPGEMM_BLOCK_SHAPE[0] * 2 * 4) == 0
            ):
                # Create packed scale tensor with proper layout for deep_gemm
                # Shape: (E, T, G // 4) where G = hidden_dim // 2 // group_size
                down_input_scale = create_packed_scale_tensor(
                    expert_num=num_slice_experts,
                    token_num_padded=max_m,
                    hidden_dim=self._N,
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    device=device,
                )
                # Fused SiLU-and-mul + FP8 quantization with UE8M0 scale packing
                silu_and_mul_masked_post_quant_packed_fwd(
                    input=upgate_output,
                    output=down_input,
                    output_scale=down_input_scale,
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    masked_m=masked_m[start_idx:end_idx],
                )
            else:
                # Standard path for other SM versions
                down_input_scale = torch.empty(
                    (
                        num_slice_experts,
                        max_m,
                        self._N // 2 // self.DEEPGEMM_BLOCK_SHAPE[0],
                    ),
                    device=device,
                    dtype=torch.float32,
                )
                # SiLU Activation
                silu_mul_masked_fp8_post_quant_fwd(
                    input=upgate_output,
                    output=down_input,
                    output_scale=down_input_scale,
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    masked_m=masked_m[start_idx:end_idx],
                    expected_m=expected_m,
                    scale_ue8m0=is_deep_gemm_e8m0_used(),
                )

            # Free upgate output
            dispose_tensor(upgate_output)
            # Allocate down_output if it is None
            if down_output is None:
                down_output = torch.empty(
                    (num_local_experts, max_m, hidden_size),
                    device=device,
                    dtype=torch.bfloat16,
                )

            # Down GroupGEMM-1
            m_grouped_fp8_gemm_nt_masked(
                a=(down_input, down_input_scale),
                b=(self._w2[start_idx:end_idx], self._w2_scale[start_idx:end_idx]),
                output=down_output[start_idx:end_idx],
                masked_m=masked_m[start_idx:end_idx],
                expected_m=expected_m,
                disable_ue8m0_cast=(not is_deep_gemm_e8m0_used()),
            )

            # Free down_input and down_input_scale
            dispose_tensor(down_input)
            dispose_tensor(down_input_scale)

        else:
            # Allocate upgate output
            upgate_output = torch.empty(
                (num_slice_experts, max_m, self._N),
                device=device,
                dtype=torch.bfloat16,
            )

            # Gate and Up GroupGEMM-0
            m_grouped_bf16_gemm_nt_masked(
                a=expert_x[start_idx:end_idx],
                b=self._w1[start_idx:end_idx],
                output=upgate_output,
                masked_m=masked_m[start_idx:end_idx],
                expected_m=expected_m,
            )

            # Free expert_x
            if end_idx == self._num_local_experts:
                dispose_tensor(expert_x)
            # Allocate down_input
            down_input = torch.empty(
                (num_slice_experts, max_m, self._N // 2),
                device=device,
                dtype=torch.bfloat16,
            )

            # SiLU Activation
            silu_mul_masked_bf16_no_post_quant_fwd(
                input=upgate_output,
                output=down_input,
                masked_m=masked_m[start_idx:end_idx],
                expected_m=expected_m,
                group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
            )

            # Free upgate output
            dispose_tensor(upgate_output)
            # Allocate down_output if it is None
            if down_output is None:
                down_output = torch.empty(
                    (num_local_experts, max_m, hidden_size),
                    device=device,
                    dtype=torch.bfloat16,
                )

            # Down GroupGEMM-1
            m_grouped_bf16_gemm_nt_masked(
                a=down_input,
                b=self._w2[start_idx:end_idx],
                output=down_output[start_idx:end_idx],
                masked_m=masked_m[start_idx:end_idx],
                expected_m=expected_m,
            )

            # Free down_input
            dispose_tensor(down_input)

        return down_output

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        # Check and prepare execute data
        expert_x, masked_m, expected_m, expert_x_scale = (
            self._check_and_prepare_execute(payload)
        )
        # Forward masked grouped FFN
        with configure_deep_gemm_num_sms(self._num_gemm_sms):
            down_output = self._forward_masked_grouped_ffn(
                start_idx=0,
                end_idx=self._num_local_experts,
                expert_x=expert_x,
                masked_m=masked_m,
                expected_m=expected_m,
                down_output=None,
                expert_x_scale=expert_x_scale,
            )
        # Return combine forward payload
        return CombineForwardPayload(fused_expert_output=down_output)
