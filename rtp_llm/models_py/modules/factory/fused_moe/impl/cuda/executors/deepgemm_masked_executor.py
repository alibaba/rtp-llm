import logging
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
                    self._num_packed_scales = 1
                    self._scale_dtype = torch.float32
                # Check w1_scale and w2_scale
                assert (
                    self._w1_scale.dtype == self._scale_dtype
                    and self._w2_scale.dtype == self._scale_dtype
                )
                assert (
                    self._w1_scale.size(0) == self._E
                    and self._w2_scale.size(0) == self._E
                )
                assert (
                    self._w1_scale.size(1)
                    == self._N
                    // self.DEEPGEMM_BLOCK_SHAPE[0]
                    // self._num_packed_scales
                )
                assert (
                    self._w1_scale.size(2)
                    == self._K
                    // self.DEEPGEMM_BLOCK_SHAPE[1]
                    // self._num_packed_scales
                )
                assert (
                    self._w2_scale.size(1)
                    == self._K
                    // self.DEEPGEMM_BLOCK_SHAPE[1]
                    // self._num_packed_scales
                )
                assert (
                    self._w2_scale.size(2)
                    == self._N
                    // 2
                    // self.DEEPGEMM_BLOCK_SHAPE[0]
                    // self._num_packed_scales
                )
            else:
                raise NotImplementedError(
                    "DeepGemmMaskedExecutor only supports fp8 block quantization with block shape 128x128"
                )
        else:
            # Confirm to use bf16
            assert self._w1_scale is None and self._w2_scale is None
        # Initialize number of SMs for DeepGEMM
        self._num_gemm_sms = get_num_device_sms()

        # Log initialization info
        print("=" * 80)
        print("DeepGemmMaskedExecutor Initialization")
        print("=" * 80)
        print(f"Model Config:")
        print(f"  expert_num (E): {self._E}")
        print(f"  ffn_dim (N): {self._N}")
        print(f"  hidden_size (K): {self._K}")
        print(f"  tp_size: {config.tp_size}")
        print(f"  tp_rank: {config.tp_rank}")
        print(f"  ep_size: {config.ep_size}")
        print(f"  ep_rank: {config.ep_rank}")
        print(f"Weight Shapes:")
        print(f"  w1: {self._w1.shape}")
        print(f"  w2: {self._w2.shape}")
        print(
            f"  w1_scale: {self._w1_scale.shape if self._w1_scale is not None else None}"
        )
        print(
            f"  w2_scale: {self._w2_scale.shape if self._w2_scale is not None else None}"
        )
        print(f"Quantization Config:")
        print(f"  use_fp8: {self._use_fp8}")
        print(f"  is_quantized: {quant_config.is_quantized}")
        print(f"  quant_dtype: {quant_config.quant_dtype}")
        print(f"  is_block_quantized: {quant_config.is_block_quantized}")
        print(
            f"  block_shape: {quant_config.block_shape if quant_config.block_shape else None}"
        )
        if self._use_fp8:
            print(f"  num_packed_scales: {self._num_packed_scales}")
            print(f"  scale_dtype: {self._scale_dtype}")
            print(f"  use_ue8m0: {is_deep_gemm_e8m0_used()}")
        print(f"DeepGEMM Config:")
        print(f"  DEEPGEMM_BLOCK_SHAPE: {self.DEEPGEMM_BLOCK_SHAPE}")
        print(f"  num_gemm_sms: {self._num_gemm_sms}")
        print("=" * 80)

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
        """Forward masked grouped FFN.
        Args:
            start_idx (int): Start index of the expert.
            end_idx (int): End index of the expert.
            expert_x (torch.Tensor): Expert input.
            masked_m (torch.Tensor): Masked input.
            expected_m (int): Expected output.
            expert_x_scale (Optional[torch.Tensor]): Expert input scale.
            down_output (Optional[torch.Tensor]): Down output tensor. If None, a new tensor will be allocated.
        Returns:
            torch.Tensor: Down output.
        """
        print(f"_forward_masked_grouped_ffn called:")
        print(f"  start_idx: {start_idx}, end_idx: {end_idx}")
        print(f"  expert_x.shape: {expert_x.shape}, dtype: {expert_x.dtype}")
        print(f"expert_x: {expert_x}")
        print(f"  masked_m: {masked_m}")
        print(f"  expected_m: {expected_m}")
        if expert_x_scale is not None:
            print(f"  expert_x_scale: shape={expert_x_scale.shape}")
            print(f"expert_x_scale: {expert_x_scale}")
        else:
            print(f"  expert_x_scale: None")
        print(
            f"  down_output: {down_output.shape if down_output is not None else None}"
        )

        # Get metadata
        device = expert_x.device
        num_tokens, hidden_size = expert_x.size(1), expert_x.size(2)
        num_slice_experts = end_idx - start_idx

        print(
            f"  Computed: num_tokens={num_tokens}, hidden_size={hidden_size}, num_slice_experts={num_slice_experts}"
        )

        # Set number of SMs for DeepGEMM
        with configure_deep_gemm_num_sms(self._num_gemm_sms):
            if self._use_fp8:
                # Check expert_x_scale is not None for fp8
                assert expert_x_scale is not None
                assert (
                    self._w1_scale is not None and self._w2_scale is not None
                ), "w1 and w2 scales are not initialized"
                # Allocate upgate_output
                upgate_output = torch.empty(
                    (num_slice_experts, num_tokens, self._N),
                    device=device,
                    dtype=torch.bfloat16,
                )
                print(
                    f"  [FP8 Path] Allocated upgate_output: shape={upgate_output.shape}, dtype={upgate_output.dtype}"
                )

                # Gate and Up GroupGEMM-0
                print(
                    f"  [FP8 Path] Calling m_grouped_fp8_gemm_nt_masked (Gate and Up)"
                )
                print(
                    f"    Input expert_x slice: shape={expert_x[start_idx:end_idx].shape}"
                )
                print(
                    f"    Input expert_x_scale slice: shape={expert_x_scale[start_idx:end_idx].shape}"
                )
                print(f"    Weight w1 slice: shape={self._w1[start_idx:end_idx].shape}")
                print(
                    f"    Weight w1_scale slice: shape={self._w1_scale[start_idx:end_idx].shape}"
                )
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
                )
                print(
                    f"  [FP8 Path] After Gate and Up GroupGEMM-0: upgate_output stats:"
                )
                print(f"    shape={upgate_output.shape}, dtype={upgate_output.dtype}")
                print(
                    f"    min={upgate_output.min().item():.6f}, max={upgate_output.max().item():.6f}, mean={upgate_output.mean().item():.6f}"
                )
                print(f"upgate_output: {upgate_output}")

                # Free expert_x and expert_x_scale
                if end_idx == self._E:
                    dispose_tensor(expert_x)
                    dispose_tensor(expert_x_scale)
                # Allocate down_input and down_input_scale
                down_input = torch.empty(
                    (num_slice_experts, num_tokens, self._N // 2),
                    device=device,
                    dtype=torch.float8_e4m3fn,
                )
                down_input_scale = torch.empty(
                    (
                        num_slice_experts,
                        num_tokens,
                        (
                            self._N // 2 // self.DEEPGEMM_BLOCK_SHAPE[0]
                            + self._num_packed_scales
                            - 1
                        )
                        // self._num_packed_scales,
                    ),
                    device=device,
                    dtype=self._scale_dtype,
                )
                print(
                    f"  [FP8 Path] Allocated down_input: shape={down_input.shape}, dtype={down_input.dtype}"
                )
                print(
                    f"  [FP8 Path] Allocated down_input_scale: shape={down_input_scale.shape}, dtype={down_input_scale.dtype}"
                )

                # SiLU Activation
                print(
                    f"  [FP8 Path] Calling silu_mul_masked_fp8_post_quant_fwd (SiLU Activation)"
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
                print(f"  [FP8 Path] After SiLU Activation: down_input stats:")
                print(f"    shape={down_input.shape}, dtype={down_input.dtype}")
                print(f"    down_input_scale shape={down_input_scale.shape}")
                print(f"down_input: {down_input}")
                print(f"down_input_scale: {down_input_scale}")

                # Free upgate_output
                dispose_tensor(upgate_output)
                # Allocate down_output if it is None
                if down_output is None:
                    down_output = torch.empty(
                        (num_slice_experts, num_tokens, hidden_size),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    print(
                        f"  [FP8 Path] Allocated down_output: shape={down_output.shape}, dtype={down_output.dtype}"
                    )

                # Down GroupGEMM-1
                print(
                    f"  [FP8 Path] Calling m_grouped_fp8_gemm_nt_masked (Down projection)"
                )
                print(f"    Input down_input: shape={down_input.shape}")
                print(f"    Input down_input_scale: shape={down_input_scale.shape}")
                print(f"    Weight w2 slice: shape={self._w2[start_idx:end_idx].shape}")
                print(
                    f"    Weight w2_scale slice: shape={self._w2_scale[start_idx:end_idx].shape}"
                )
                m_grouped_fp8_gemm_nt_masked(
                    (
                        down_input,
                        down_input_scale,
                    ),
                    (
                        self._w2[start_idx:end_idx],
                        self._w2_scale[start_idx:end_idx],
                    ),
                    down_output[start_idx:end_idx],
                    masked_m[start_idx:end_idx],
                    expected_m,
                )
                print(
                    f"  [FP8 Path] After Down GroupGEMM-1: down_output[{start_idx}:{end_idx}] stats:"
                )
                print(
                    f"    shape={down_output[start_idx:end_idx].shape}, dtype={down_output[start_idx:end_idx].dtype}"
                )
                print(
                    f"    min={down_output[start_idx:end_idx].min().item():.6f}, max={down_output[start_idx:end_idx].max().item():.6f}, mean={down_output[start_idx:end_idx].mean().item():.6f}"
                )
                print(f"down_output: {down_output}")

                # Free down_input and down_input_scale
                dispose_tensor(down_input)
                dispose_tensor(down_input_scale)

            else:
                # Check expert_x_scale is None for bf16
                assert expert_x_scale is None
                # Allocate upgate_output
                upgate_output = torch.empty(
                    (num_slice_experts, num_tokens, self._N),
                    device=device,
                    dtype=torch.bfloat16,
                )
                print(
                    f"  [BF16 Path] Allocated upgate_output: shape={upgate_output.shape}, dtype={upgate_output.dtype}"
                )

                # Gate and Up GroupGEMM-0
                print(
                    f"  [BF16 Path] Calling m_grouped_bf16_gemm_nt_masked (Gate and Up)"
                )
                print(
                    f"    Input expert_x slice: shape={expert_x[start_idx:end_idx].shape}"
                )
                print(f"    Weight w1 slice: shape={self._w1[start_idx:end_idx].shape}")
                m_grouped_bf16_gemm_nt_masked(
                    expert_x[start_idx:end_idx],
                    self._w1[start_idx:end_idx],
                    upgate_output,
                    masked_m[start_idx:end_idx],
                    expected_m,
                )
                print(
                    f"  [BF16 Path] After Gate and Up GroupGEMM-0: upgate_output stats:"
                )
                print(f"    shape={upgate_output.shape}, dtype={upgate_output.dtype}")
                print(
                    f"    min={upgate_output.min().item():.6f}, max={upgate_output.max().item():.6f}, mean={upgate_output.mean().item():.6f}"
                )
                print(f"upgate_output: {upgate_output}")

                # Free expert_x
                if end_idx == self._E:
                    dispose_tensor(expert_x)
                # Allocate down_input
                down_input = torch.empty(
                    (num_slice_experts, num_tokens, self._N // 2),
                    device=device,
                    dtype=torch.bfloat16,
                )
                print(
                    f"  [BF16 Path] Allocated down_input: shape={down_input.shape}, dtype={down_input.dtype}"
                )

                # SiLU Activation
                print(
                    f"  [BF16 Path] Calling silu_mul_masked_bf16_no_post_quant_fwd (SiLU Activation)"
                )
                silu_mul_masked_bf16_no_post_quant_fwd(
                    input=upgate_output,
                    output=down_input,
                    masked_m=masked_m[start_idx:end_idx],
                    expected_m=expected_m,
                    group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                )
                print(f"  [BF16 Path] After SiLU Activation: down_input stats:")
                print(f"    shape={down_input.shape}, dtype={down_input.dtype}")
                print(
                    f"    min={down_input.min().item():.6f}, max={down_input.max().item():.6f}, mean={down_input.mean().item():.6f}"
                )
                print(f"down_input: {down_input}")

                # Free upgate_output
                dispose_tensor(upgate_output)
                # Allocate down_output if it is None
                if down_output is None:
                    down_output = torch.empty(
                        (num_slice_experts, num_tokens, hidden_size),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    print(
                        f"  [BF16 Path] Allocated down_output: shape={down_output.shape}, dtype={down_output.dtype}"
                    )

                # Down GroupGEMM-1
                print(
                    f"  [BF16 Path] Calling m_grouped_bf16_gemm_nt_masked (Down projection)"
                )
                print(f"    Input down_input: shape={down_input.shape}")
                print(f"    Weight w2 slice: shape={self._w2[start_idx:end_idx].shape}")
                m_grouped_bf16_gemm_nt_masked(
                    down_input,
                    self._w2[start_idx:end_idx],
                    down_output[start_idx:end_idx],
                    masked_m[start_idx:end_idx],
                    expected_m,
                )
                print(
                    f"  [BF16 Path] After Down GroupGEMM-1: down_output[{start_idx}:{end_idx}] stats:"
                )
                print(
                    f"    shape={down_output[start_idx:end_idx].shape}, dtype={down_output[start_idx:end_idx].dtype}"
                )
                print(
                    f"    min={down_output[start_idx:end_idx].min().item():.6f}, max={down_output[start_idx:end_idx].max().item():.6f}, mean={down_output[start_idx:end_idx].mean().item():.6f}"
                )
                print(f"down_output: {down_output}")

                # Free down_input
                dispose_tensor(down_input)

        print(
            f"_forward_masked_grouped_ffn returning: down_output.shape={down_output.shape}"
        )
        return down_output

    def _normal_execute(
        self,
        expert_x: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        expert_x_scale: Optional[torch.Tensor] = None,
    ) -> CombineForwardPayload:
        """Execute normal masked grouped FFN.
        Args:
            expert_x (torch.Tensor): Expert input.
            masked_m (torch.Tensor): Masked input.
            expected_m (int): Expected output.
            expert_x_scale (Optional[torch.Tensor]): Expert input scale.
        Returns:
            CombineForwardPayload: Combine forward payload.
        """
        print(f"_normal_execute called:")
        print(f"  expert_x.shape: {expert_x.shape}, dtype: {expert_x.dtype}")
        print(f"expert_x: {expert_x}")
        print(f"  masked_m: {masked_m}")
        print(f"  expected_m: {expected_m}")
        if expert_x_scale is not None:
            print(f"  expert_x_scale: shape={expert_x_scale.shape}")
            print(f"expert_x_scale: {expert_x_scale}")
        else:
            print(f"  expert_x_scale: None")

        # Execute masked grouped FFN
        down_output = self._forward_masked_grouped_ffn(
            0,
            self._E,
            expert_x,
            masked_m,
            expected_m,
            expert_x_scale=expert_x_scale,
            down_output=None,
        )

        print(f"_normal_execute returning: down_output.shape={down_output.shape}")
        # Return combine forward payload
        return CombineForwardPayload(fused_expert_output=down_output)

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
        print(f"_execute_fp8 called:")
        print(f"  activation: {activation}")
        print(f"  expert_map: {expert_map.shape if expert_map is not None else None}")
        print(f"  a2_scale: {a2_scale.shape if a2_scale is not None else None}")
        print(f"  apply_router_weight_on_input: {apply_router_weight_on_input}")
        print(f"  extra_expert_args: {extra_expert_args}")

        # Check payload data
        assert (
            payload.expert_tokens_meta is not None
            and payload.expert_tokens_meta.expert_num_tokens is not None
        )
        expert_x = payload.expert_x
        E, M, K = expert_x.size()
        print(
            f"  expert_x: shape={expert_x.shape}, dtype={expert_x.dtype}, E={E}, M={M}, K={K}"
        )
        print(f"expert_x: {expert_x}")

        assert E == self._E and K == self._K
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        assert len(masked_m) == self._E
        expected_m = (
            min(M, payload.expert_tokens_meta.expected_m)
            if payload.expert_tokens_meta.expected_m is not None
            else M
        )
        expert_x_scale = payload.expert_x_scale
        assert expert_x_scale is not None
        assert expert_x_scale.size(0) == E
        assert expert_x_scale.size(1) == M
        assert (
            expert_x_scale.size(2)
            == (K // self.DEEPGEMM_BLOCK_SHAPE[1] + self._num_packed_scales - 1)
            // self._num_packed_scales
        )
        assert expert_x_scale.dtype == self._scale_dtype
        print(f"  masked_m: {masked_m}")
        print(f"  expected_m: {expected_m}")
        print(
            f"  expert_x_scale: shape={expert_x_scale.shape}, dtype={expert_x_scale.dtype}"
        )
        print(f"expert_x_scale: {expert_x_scale}")
        # Normal execution
        combine_payload = self._normal_execute(
            expert_x, masked_m, expected_m, expert_x_scale
        )

        print(
            f"_execute_fp8 returning: combine_payload.fused_expert_output.shape={combine_payload.fused_expert_output.shape}"
        )
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

        # Normal execution
        combine_payload = self._normal_execute(expert_x, masked_m, expected_m)

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
