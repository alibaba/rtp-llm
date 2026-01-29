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
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(has_deep_gemm())
        checker.check(resolver.is_bf16(config))
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method in [None, "FP8_PER_BLOCK"])
        checker.check(get_sm()[0] >= 9)
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
        # Initialize w1 and w2
        self._w1 = weights[W.moe_w1]
        self._w2 = weights[W.moe_w2]
        # Check w1 and w2 shape
        self._E, self._N, self._K = self._w1.size()
        assert self._N % 2 == 0
        assert self._w2.size(0) == self._E
        assert self._w2.size(1) == self._K
        assert self._w2.size(2) == self._N // 2
        # Initialize w1 and w2 scale
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
                    self._w1_scale.size(0) == self._E
                    and self._w2_scale.size(0) == self._E
                )
                assert (
                    self._w1_scale.size(1) == self._N
                    if is_deep_gemm_e8m0_used()
                    else self._N // self.DEEPGEMM_BLOCK_SHAPE[0]
                )
                assert (
                    self._w1_scale.size(2)
                    == (
                        self._K // self.DEEPGEMM_BLOCK_SHAPE[1]
                        + self._num_packed_scales
                        - 1
                    )
                    // self._num_packed_scales
                )
                assert (
                    self._w2_scale.size(1) == self._K
                    if is_deep_gemm_e8m0_used()
                    else self._K // self.DEEPGEMM_BLOCK_SHAPE[1]
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
            # Confirm to use bf16
            assert self._w1_scale is None and self._w2_scale is None
        # Number of SMs for DeepGemm
        self._num_gemm_sms = get_num_device_sms()

    def _execute_fp8(
        self,
        expert_x: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        expert_payload: ExpertForwardPayload,
    ) -> CombineForwardPayload:
        """Execute FP8 experts computation.
        Args:
            expert_x (torch.Tensor): Expert input.
            masked_m (torch.Tensor): Masked input.
            expected_m (int): Expected output.
            expert_payload (ExpertForwardPayload): Expert payload.
        """
        expert_x_scale = expert_payload.expert_x_scale
        assert expert_x_scale is not None
        assert expert_x_scale.size(0) == expert_x.size(0)
        assert expert_x_scale.size(1) == expert_x.size(1)
        assert (
            expert_x_scale.size(2)
            == (
                expert_x.size(2) // self.DEEPGEMM_BLOCK_SHAPE[1]
                + self._num_packed_scales
                - 1
            )
            // self._num_packed_scales
        )
        assert expert_x_scale.dtype == self._scale_dtype

        device = expert_x.device
        num_local_experts, max_m, hidden_size = expert_x.shape
        assert self._w1_scale is not None and self._w2_scale is not None

        with configure_deep_gemm_num_sms(self._num_gemm_sms):
            # Check expert_x_scale is not None for fp8
            assert expert_x_scale is not None
            assert (
                self._w1_scale is not None and self._w2_scale is not None
            ), "w1 and w2 scales are not initialized"
            # Allocate upgate_output
            upgate_output = torch.empty(
                (num_local_experts, max_m, self._N),
                device=device,
                dtype=torch.bfloat16,
            )

            # Gate and Up GroupGEMM-0
            m_grouped_fp8_gemm_nt_masked(
                a=(expert_x, expert_x_scale),
                b=(self._w1, self._w1_scale),
                output=upgate_output,
                masked_m=masked_m,
                expected_m=expected_m,
                disable_ue8m0_cast=(not is_deep_gemm_e8m0_used()),
            )

            # Free expert_x and expert_x_scale
            dispose_tensor(expert_x)
            dispose_tensor(expert_x_scale)
            # Allocate down_input
            down_input = torch.empty(
                (num_local_experts, max_m, self._N // 2),
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
                    expert_num=num_local_experts,
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
                    masked_m=masked_m,
                )
            else:
                # Standard path for other SM versions
                down_input_scale = torch.empty(
                    (
                        num_local_experts,
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
                    masked_m=masked_m,
                    expected_m=expected_m,
                    scale_ue8m0=is_deep_gemm_e8m0_used(),
                )

            # Free upgate_output
            dispose_tensor(upgate_output)
            # Allocate down_output if it is None
            down_output = torch.empty(
                (num_local_experts, max_m, hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )

            # Down GroupGEMM-1
            m_grouped_fp8_gemm_nt_masked(
                a=(down_input, down_input_scale),
                b=(self._w2, self._w2_scale),
                output=down_output,
                masked_m=masked_m,
                expected_m=expected_m,
                disable_ue8m0_cast=(not is_deep_gemm_e8m0_used()),
            )

            # Free down_input and down_input_scale
            dispose_tensor(down_input)
            dispose_tensor(down_input_scale)

        return CombineForwardPayload(fused_expert_output=down_output)

    def _execute_bf16(
        self,
        expert_x: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        expert_payload: ExpertForwardPayload,
    ) -> CombineForwardPayload:
        """Execute Bf16 experts computation.
        Args:
            expert_x (torch.Tensor): Expert input.
            masked_m (torch.Tensor): Masked input.
            expected_m (int): Expected output.
            expert_payload (ExpertForwardPayload): Expert payload.
        Returns:
            CombineForwardPayload: Combine forward payload.
        """
        device = expert_x.device
        num_local_experts, max_m, hidden_size = expert_x.shape

        with configure_deep_gemm_num_sms(self._num_gemm_sms):
            # Check expert_x_scale is None for bf16
            assert expert_payload.expert_x_scale is None
            # Allocate upgate_output
            upgate_output = torch.empty(
                (num_local_experts, max_m, self._N),
                device=device,
                dtype=torch.bfloat16,
            )

            # Gate and Up GroupGEMM-0
            m_grouped_bf16_gemm_nt_masked(
                a=expert_x,
                b=self._w1,
                output=upgate_output,
                masked_m=masked_m,
                expected_m=expected_m,
            )
            # Free expert_x
            dispose_tensor(expert_x)
            # Allocate down_input
            down_input = torch.empty(
                (num_local_experts, max_m, self._N // 2),
                device=device,
                dtype=torch.bfloat16,
            )

            # SiLU Activation
            silu_mul_masked_bf16_no_post_quant_fwd(
                input=upgate_output,
                output=down_input,
                masked_m=masked_m,
                expected_m=expected_m,
                group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
            )

            # Free upgate_output
            dispose_tensor(upgate_output)
            # Allocate down_output if it is None
            down_output = torch.empty(
                (num_local_experts, max_m, hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )

            # Down GroupGEMM-1
            m_grouped_bf16_gemm_nt_masked(
                a=down_input,
                b=self._w2,
                output=down_output,
                masked_m=masked_m,
                expected_m=expected_m,
            )

            # Free down_input
            dispose_tensor(down_input)

        return CombineForwardPayload(fused_expert_output=down_output)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        # Check payload data
        assert (
            payload.expert_tokens_meta is not None
            and payload.expert_tokens_meta.expert_num_tokens is not None
        )
        expert_x = payload.expert_x
        E, M, K = expert_x.size()
        assert E == self._E and K == self._K
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        assert len(masked_m) == self._E
        expected_m = (
            min(M, payload.expert_tokens_meta.expected_m)
            if payload.expert_tokens_meta.expected_m is not None
            else M
        )

        if self._use_fp8:
            return self._execute_fp8(
                expert_x=expert_x,
                masked_m=masked_m,
                expected_m=expected_m,
                expert_payload=payload,
            )
        else:
            return self._execute_bf16(
                expert_x=expert_x,
                masked_m=masked_m,
                expected_m=expected_m,
                expert_payload=payload,
            )
