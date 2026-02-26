"""DeepGemm executor with contiguous-to-masked conversion.

This executor bridges PureTpRouter (contiguous output) with DeepGemmMasked
(masked input) by performing an efficient GPU-based layout conversion.
"""

from typing import Any, Dict, Optional, Tuple

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
from rtp_llm.ops.compute_ops import get_rtp_llm_ops
from rtp_llm.utils.model_weight import W


class DeepGemmContinuousToMaskedExecutor(FusedMoeExpertExecutor):
    """Executor that converts contiguous layout to masked and uses DeepGemmMasked.

    This executor is designed to work with PureTpRouter which outputs contiguous
    layout. It efficiently converts the data to masked layout on GPU and then
    leverages DeepGemmMasked's high-performance GEMM kernels.

    Flow:
        1. Receive contiguous input from PureTpRouter
        2. Convert to masked layout using GPU kernel
        3. Execute masked grouped GEMM using DeepGemmMasked logic
    """

    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls):
        return ExecutorType.DEEPGEMM_CONTINUOUS_TO_MASKED

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if this executor can handle the configuration."""
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

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """Initialize the executor.

        Args:
            config: Model configuration
            quant_config: Quantization configuration
            weights: Dictionary containing model weights
        """
        super().__init__(config, quant_config, weights)

        # Initialize weights
        self._w1 = weights[W.moe_w1]
        self._w2 = weights[W.moe_w2]

        # Validate shapes
        self._E, self._N, self._K = self._w1.size()
        assert self._N % 2 == 0
        assert self._w2.size(0) == self._E
        assert self._w2.size(1) == self._K
        assert self._w2.size(2) == self._N // 2

        # Initialize scales
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
                # FP8 block quantization setup
                self._num_packed_scales = 1
                self._scale_dtype = torch.float32

                # Convert to UE8M0 format if needed
                if is_deep_gemm_e8m0_used():
                    self._w1, self._w1_scale = requant_weight_ue8m0(
                        self._w1, self._w1_scale
                    )
                    self._w2, self._w2_scale = requant_weight_ue8m0(
                        self._w2, self._w2_scale
                    )
                    self._num_packed_scales = 4
                    self._scale_dtype = torch.int32

                assert (
                    self._w1_scale.dtype == self._scale_dtype
                    and self._w2_scale.dtype == self._scale_dtype
                )
            else:
                raise NotImplementedError(
                    "Only FP8 block quantization with block shape 128x128 is supported"
                )
        else:
            # BF16 mode
            assert self._w1_scale is None and self._w2_scale is None

        # DeepGEMM configuration
        self._num_gemm_sms = get_num_device_sms()

        # Maximum tokens per expert (for masked layout allocation)
        # This should be large enough to hold all tokens for any expert
        self._max_tokens_per_expert = config.ll_num_max_token

    def _convert_to_masked_layout(
        self,
        expert_x: torch.Tensor,
        expert_x_scale: Optional[torch.Tensor],
        topk_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Convert contiguous layout to masked layout.

        Args:
            expert_x: Input in contiguous layout [total_tokens, hidden_dim]
            expert_x_scale: Optional scale tensor [total_tokens, scale_dim]
            topk_ids: Expert IDs for each token [batch_size, top_k]

        Returns:
            Tuple of (masked_expert_x, masked_expert_x_scale, expert_num_tokens)
        """
        # Flatten topk_ids to get grouped_layout
        grouped_layout = topk_ids.view(-1).to(torch.int32)

        # Get rtp_llm_ops
        rtp_ops = get_rtp_llm_ops()

        # Convert data to masked layout
        masked_expert_x, expert_num_tokens = rtp_ops.convert_contiguous_to_masked(
            expert_x,
            grouped_layout,
            self._E,
            self._max_tokens_per_expert,
        )

        # Convert scale to masked layout if present
        if expert_x_scale is not None:
            masked_expert_x_scale, _ = rtp_ops.convert_contiguous_to_masked(
                expert_x_scale,
                grouped_layout,
                self._E,
                self._max_tokens_per_expert,
            )
        else:
            masked_expert_x_scale = None

        return masked_expert_x, masked_expert_x_scale, expert_num_tokens

    def _forward_masked_grouped_ffn(
        self,
        start_idx: int,
        end_idx: int,
        expert_x: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        expert_x_scale: Optional[torch.Tensor] = None,
        down_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward masked grouped FFN (copied from DeepGemmMaskedExecutor).

        Args:
            start_idx: Start expert index
            end_idx: End expert index
            expert_x: Expert input in masked layout
            masked_m: Number of valid tokens per expert
            expected_m: Expected max tokens
            expert_x_scale: Optional input scale
            down_output: Optional pre-allocated output tensor

        Returns:
            Output tensor
        """
        device = expert_x.device
        num_tokens, hidden_size = expert_x.size(1), expert_x.size(2)
        num_slice_experts = end_idx - start_idx

        with configure_deep_gemm_num_sms(self._num_gemm_sms):
            if self._use_fp8:
                # FP8 path
                assert expert_x_scale is not None
                assert self._w1_scale is not None and self._w2_scale is not None

                # Allocate upgate output
                upgate_output = torch.empty(
                    (num_slice_experts, num_tokens, self._N),
                    device=device,
                    dtype=torch.bfloat16,
                )

                # Gate and Up GEMM
                m_grouped_fp8_gemm_nt_masked(
                    (
                        expert_x[start_idx:end_idx],
                        expert_x_scale[start_idx:end_idx],
                    ),
                    (
                        self._w1[start_idx:end_idx],
                        self._w1_scale[start_idx:end_idx],
                    ),
                    upgate_output,
                    masked_m[start_idx:end_idx],
                    expected_m,
                    disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
                )

                # Free input tensors
                if end_idx == self._E:
                    dispose_tensor(expert_x)
                    dispose_tensor(expert_x_scale)

                # Allocate down input
                down_input = torch.empty(
                    (num_slice_experts, num_tokens, self._N // 2),
                    device=device,
                    dtype=torch.float8_e4m3fn,
                )

                # SM100 optimization with packed kernel
                sm_major = torch.cuda.get_device_capability()[0]
                if (
                    sm_major == 10
                    and is_deep_gemm_e8m0_used()
                    and self._N % (self.DEEPGEMM_BLOCK_SHAPE[0] * 2 * 4) == 0
                ):
                    down_input_scale = create_packed_scale_tensor(
                        expert_num=num_slice_experts,
                        token_num_padded=num_tokens,
                        hidden_dim=self._N,
                        quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                        device=device,
                    )
                    silu_and_mul_masked_post_quant_packed_fwd(
                        upgate_output,
                        down_input,
                        down_input_scale,
                        self.DEEPGEMM_BLOCK_SHAPE[0],
                        masked_m[start_idx:end_idx],
                    )
                else:
                    down_input_scale = torch.empty(
                        (
                            num_slice_experts,
                            num_tokens,
                            self._N // 2 // self.DEEPGEMM_BLOCK_SHAPE[0],
                        ),
                        device=device,
                        dtype=torch.float32,
                    )
                    silu_mul_masked_fp8_post_quant_fwd(
                        input=upgate_output,
                        output=down_input,
                        output_scale=down_input_scale,
                        quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                        masked_m=masked_m[start_idx:end_idx],
                        expected_m=expected_m,
                        scale_ue8m0=is_deep_gemm_e8m0_used(),
                    )

                dispose_tensor(upgate_output)

                # Allocate down output
                if down_output is None:
                    down_output = torch.empty(
                        (num_slice_experts, num_tokens, hidden_size),
                        device=device,
                        dtype=torch.bfloat16,
                    )

                # Down GEMM
                m_grouped_fp8_gemm_nt_masked(
                    (down_input, down_input_scale),
                    (
                        self._w2[start_idx:end_idx],
                        self._w2_scale[start_idx:end_idx],
                    ),
                    down_output[start_idx:end_idx],
                    masked_m[start_idx:end_idx],
                    expected_m,
                    disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
                )

                dispose_tensor(down_input)
                dispose_tensor(down_input_scale)

            else:
                # BF16 path
                assert expert_x_scale is None

                # Allocate upgate output
                upgate_output = torch.empty(
                    (num_slice_experts, num_tokens, self._N),
                    device=device,
                    dtype=torch.bfloat16,
                )

                # Gate and Up GEMM
                m_grouped_bf16_gemm_nt_masked(
                    expert_x[start_idx:end_idx],
                    self._w1[start_idx:end_idx],
                    upgate_output,
                    masked_m[start_idx:end_idx],
                    expected_m,
                )

                if end_idx == self._E:
                    dispose_tensor(expert_x)

                # Allocate down input
                down_input = torch.empty(
                    (num_slice_experts, num_tokens, self._N // 2),
                    device=device,
                    dtype=torch.bfloat16,
                )

                # SiLU activation
                silu_mul_masked_bf16_no_post_quant_fwd(
                    input=upgate_output,
                    output=down_input,
                    masked_m=masked_m[start_idx:end_idx],
                    expected_m=expected_m,
                    group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                )

                dispose_tensor(upgate_output)

                # Allocate down output
                if down_output is None:
                    down_output = torch.empty(
                        (num_slice_experts, num_tokens, hidden_size),
                        device=device,
                        dtype=torch.bfloat16,
                    )

                # Down GEMM
                m_grouped_bf16_gemm_nt_masked(
                    down_input,
                    self._w2[start_idx:end_idx],
                    down_output[start_idx:end_idx],
                    masked_m[start_idx:end_idx],
                    expected_m,
                )

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
        """Execute expert computation with contiguous-to-masked conversion.

        Args:
            payload: Forward payload from router (in contiguous layout)
            activation: Activation function name
            expert_map: Optional expert mapping
            a2_scale: Optional scale for second linear layer
            apply_router_weight_on_input: Whether to apply router weights on input
            extra_expert_args: Extra arguments for expert computation

        Returns:
            Combined forward payload with fused expert output
        """
        # Validate input
        assert payload.expert_x is not None, "expert_x is not initialized"
        assert payload.expert_topk_ids is not None, "topk_ids is not initialized"

        expert_x_contiguous = payload.expert_x
        expert_x_scale_contiguous = payload.expert_x_scale
        topk_ids = payload.expert_topk_ids

        # Get dimensions
        _, hidden_size = expert_x_contiguous.shape
        assert hidden_size == self._K

        # Prepare grouped_layout (needed for both conversions)
        grouped_layout = topk_ids.view(-1).to(torch.int32)

        # Get rtp_llm_ops
        rtp_ops = get_rtp_llm_ops()

        # Convert contiguous to masked layout
        masked_expert_x, expert_num_tokens = rtp_ops.convert_contiguous_to_masked(
            expert_x_contiguous,
            grouped_layout,
            self._E,
            self._max_tokens_per_expert,
        )

        # Convert scale if present
        if expert_x_scale_contiguous is not None:
            masked_expert_x_scale, _ = rtp_ops.convert_contiguous_to_masked(
                expert_x_scale_contiguous,
                grouped_layout,
                self._E,
                self._max_tokens_per_expert,
            )
        else:
            masked_expert_x_scale = None

        # Dispose original contiguous tensors
        dispose_tensor(expert_x_contiguous)
        if expert_x_scale_contiguous is not None:
            dispose_tensor(expert_x_scale_contiguous)

        # Determine expected_m (maximum tokens among all experts)
        expected_m = int(expert_num_tokens.max().item())

        # Execute masked grouped FFN
        down_output_masked = self._forward_masked_grouped_ffn(
            start_idx=0,
            end_idx=self._E,
            expert_x=masked_expert_x,
            masked_m=expert_num_tokens,
            expected_m=expected_m,
            expert_x_scale=masked_expert_x_scale,
            down_output=None,
        )

        # Convert masked output back to contiguous layout
        contiguous_output = rtp_ops.convert_masked_to_contiguous(
            down_output_masked,
            grouped_layout,
            expert_num_tokens,
        )

        # Dispose masked output
        dispose_tensor(down_output_masked)

        # Return output in contiguous layout (same shape as input)
        return CombineForwardPayload(fused_expert_output=contiguous_output)
