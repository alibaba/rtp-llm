from typing import Any, Callable, Dict, List, Optional

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

COMPUTE_STREAM = torch.cuda.Stream()


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
                    self._num_packed_scales = 4
                    self._scale_dtype = torch.int32
                # Check w1_scale and w2_scale
                assert (
                    self._w1_scale.dtype == torch.float32
                    and self._w2_scale.dtype == torch.float32
                )
                assert (
                    self._w1_scale.size(0) == self._E
                    and self._w2_scale.size(0) == self._E
                )
                assert self._w1_scale.size(1) == self._N // self.DEEPGEMM_BLOCK_SHAPE[0]
                assert self._w1_scale.size(2) == self._K // self.DEEPGEMM_BLOCK_SHAPE[1]
                assert self._w2_scale.size(1) == self._K // self.DEEPGEMM_BLOCK_SHAPE[1]
                assert (
                    self._w2_scale.size(2)
                    == self._N // 2 // self.DEEPGEMM_BLOCK_SHAPE[0]
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
        # Enable PEO FFN
        self._enable_peo_level = self.config.moe_config.enable_peo_level
        self._comm_stream: Optional[torch.cuda.Stream] = None
        if self._enable_peo_level > 0:
            # Initialize number of PEO rounds
            self._num_peo_rounds = self.config.moe_config.num_peo_rounds
            # Calculate the number of sms for DeepGemm
            self._num_comm_sms = self.config.moe_config.deep_ep_num_sm
            self._num_gemm_sms = get_num_device_sms() - self._num_comm_sms
            # Calculate number of experts per round
            self._num_local_experts = self.config.expert_num // self.config.ep_size
            assert self._num_local_experts % self._num_peo_rounds == 0
            self._num_experts_per_round = (
                self._num_local_experts // self._num_peo_rounds
            )

    def _require_comm_stream(self, extra_expert_args: Optional[dict]) -> None:
        """PEO level 3 requires a comm stream provided by the router (DeepEP) in extra_expert_args."""
        if self._comm_stream is not None:
            return
        if not extra_expert_args or "comm_stream" not in extra_expert_args:
            raise RuntimeError(
                "DeepGemmMaskedExecutor requires extra_expert_args['comm_stream'] to be provided "
                "when enable_peo_level == 3. Please pass router.comm_stream via FusedMoeExecutor.execute."
            )
        comm_stream = extra_expert_args["comm_stream"]
        if not isinstance(comm_stream, torch.cuda.Stream):
            raise TypeError(
                f"comm_stream must be torch.cuda.Stream, got {type(comm_stream)}"
            )
        self._comm_stream = comm_stream

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
        # Get metadata
        device = expert_x.device
        num_tokens, hidden_size = expert_x.size(1), expert_x.size(2)
        num_slice_experts = end_idx - start_idx

        # Execute masked grouped FFN
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

            # Gate and Up GroupGEMM-0
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

            # Free expert_x and expert_x_scale
            if end_idx == self._E:
                dispose_tensor(expert_x)
                dispose_tensor(expert_x_scale)
            # Allocate down_input
            down_input = torch.empty(
                (num_slice_experts, num_tokens, self._N // 2),
                device=device,
                dtype=torch.float8_e4m3fn,
            )

            # SM100 (compute capability 10.x) uses fused packed kernel for better performance
            # when UE8M0 scale format is enabled
            if is_deep_gemm_e8m0_used():
                # Create packed scale tensor with proper layout for deep_gemm
                # Shape: (E, T, G // 4) where G = hidden_dim // 2 // group_size
                down_input_scale = create_packed_scale_tensor(
                    expert_num=num_slice_experts,
                    token_num_padded=num_tokens,
                    hidden_dim=self._N,
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    device=device,
                )
                # Fused SiLU-and-mul + FP8 quantization with UE8M0 scale packing
                silu_and_mul_masked_post_quant_packed_fwd(
                    upgate_output,
                    down_input,
                    down_input_scale,
                    self.DEEPGEMM_BLOCK_SHAPE[0],
                    masked_m[start_idx:end_idx],
                )
            else:
                # Standard path for other SM versions
                down_input_scale = torch.empty(
                    (
                        num_slice_experts,
                        num_tokens,
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

            # Free upgate_output
            dispose_tensor(upgate_output)
            # Allocate down_output if it is None
            if down_output is None:
                down_output = torch.empty(
                    (num_slice_experts, num_tokens, hidden_size),
                    device=device,
                    dtype=torch.bfloat16,
                )

            # Down GroupGEMM-1
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

            # Gate and Up GroupGEMM-0
            m_grouped_bf16_gemm_nt_masked(
                expert_x[start_idx:end_idx],
                self._w1[start_idx:end_idx],
                upgate_output,
                masked_m[start_idx:end_idx],
                expected_m,
            )
            # Free expert_x
            if end_idx == self._E:
                dispose_tensor(expert_x)
            # Allocate down_input
            down_input = torch.empty(
                (num_slice_experts, num_tokens, self._N // 2),
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

            # Free upgate_output
            dispose_tensor(upgate_output)
            # Allocate down_output if it is None
            if down_output is None:
                down_output = torch.empty(
                    (num_slice_experts, num_tokens, hidden_size),
                    device=device,
                    dtype=torch.bfloat16,
                )

            # Down GroupGEMM-1
            m_grouped_bf16_gemm_nt_masked(
                down_input,
                self._w2[start_idx:end_idx],
                down_output[start_idx:end_idx],
                masked_m[start_idx:end_idx],
                expected_m,
            )

            # Free down_input
            dispose_tensor(down_input)
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
        # Set number of SMs for DeepGEMM
        with configure_deep_gemm_num_sms(self._num_gemm_sms):
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

        # Return combine forward payload
        return CombineForwardPayload(fused_expert_output=down_output)

    def _peo_execute(
        self,
        expert_x: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        dispatch_recv_events: List[torch.cuda.Event],
        expert_x_scale: Optional[torch.Tensor] = None,
    ) -> CombineForwardPayload:
        """Execute PEO masked grouped FFN.
        Args:
            expert_x (torch.Tensor): Expert input.
            masked_m (torch.Tensor): Masked input.
            expected_m (int): Expected output.
            dispatch_recv_events (List[torch.cuda.Event]): Dispatch recv events.
            expert_x_scale (Optional[torch.Tensor]): Expert input scale.
        Returns:
            CombineForwardPayload: Combine forward payload.
        """
        # Initialize down_output
        down_output = torch.empty(
            expert_x.shape,
            device=expert_x.device,
            dtype=torch.bfloat16,
        )
        # Initialize expert_executions
        expert_executions = []

        # Helper function to create closure for expert execution, solve CUDA Graph leads to a large number of CUDA streams problem, related to:
        # https://github.com/pytorch/pytorch/issues/155679
        # https://github.com/pytorch/pytorch/issues/152114
        def create_expert_execution(
            round_id: int, ignore_dispatch_recv_event: bool = False
        ) -> Callable[[], None]:
            """Create a closure function for expert execution.
            Args:
                round_id: The round ID for this execution.
                ignore_dispatch_recv_event: Whether to ignore the dispatch recv event.
            Returns:
                A function that executes the expert computation.
            """

            def execute_expert() -> None:
                # Determine the current stream and next stream
                current_gemm_stream = COMPUTE_STREAM
                next_stream = torch.cuda.current_stream()
                if self._enable_peo_level == 4 and round_id == 0:
                    current_gemm_stream, next_stream = next_stream, current_gemm_stream
                elif self._enable_peo_level == 3 and round_id >= (
                    self._num_peo_rounds - 2
                ):
                    current_gemm_stream = self._comm_stream
                    next_stream = torch.cuda.current_stream()
                # Determine the number of SMs for DeepGEMM
                num_gemm_sms = self._num_gemm_sms
                if self._enable_peo_level == 4 and round_id == 0:
                    num_gemm_sms = get_num_device_sms()
                elif self._enable_peo_level == 3 and round_id >= (
                    self._num_peo_rounds - 2
                ):
                    num_gemm_sms = self._num_gemm_sms + self._num_comm_sms // 2

                # Compute stream wait for dispatch recv event to finish
                if not ignore_dispatch_recv_event and (
                    self._enable_peo_level != 3
                    or round_id != (self._num_peo_rounds - 2)
                ):
                    COMPUTE_STREAM.wait_event(dispatch_recv_events[round_id])

                # Execute masked grouped FFN with compute stream
                with torch.cuda.stream(current_gemm_stream):
                    # Set number of SMs for DeepGEMM
                    with configure_deep_gemm_num_sms(num_gemm_sms):
                        self._forward_masked_grouped_ffn(
                            round_id * self._num_experts_per_round,
                            (round_id + 1) * self._num_experts_per_round,
                            expert_x,
                            masked_m,
                            expected_m,
                            expert_x_scale,
                            down_output,
                        )

                # Build dependency for Combine
                next_stream.wait_stream(current_gemm_stream)
                if self._enable_peo_level == 3 and round_id == (
                    self._num_peo_rounds - 3
                ):
                    self._comm_stream.wait_stream(current_gemm_stream)

            return execute_expert

        # Iterate over all rounds
        for round_id in range(self._num_peo_rounds):
            # Store closure function in expert_executions
            expert_executions.append(
                create_expert_execution(
                    round_id, ignore_dispatch_recv_event=(self._enable_peo_level == 4)
                )
            )

        return CombineForwardPayload(
            fused_expert_output=down_output,
            expert_executions=expert_executions,
        )

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

        if self._enable_peo_level > 0:
            # PEO execution
            combine_payload = self._peo_execute(
                expert_x=expert_x,
                masked_m=masked_m,
                expected_m=expected_m,
                expert_x_scale=expert_x_scale,
                dispatch_recv_events=expert_payload.dispatch_recv_events,
            )
        else:
            # Normal execution
            combine_payload = self._normal_execute(
                expert_x=expert_x,
                masked_m=masked_m,
                expected_m=expected_m,
                expert_x_scale=expert_x_scale,
            )

        return combine_payload

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
        if self._enable_peo_level > 0:
            # PEO execution
            combine_payload = self._peo_execute(
                expert_x=expert_x,
                masked_m=masked_m,
                expected_m=expected_m,
                dispatch_recv_events=expert_payload.dispatch_recv_events,
                expert_x_scale=None,
            )
        else:
            # Normal execution
            combine_payload = self._normal_execute(
                expert_x=expert_x,
                masked_m=masked_m,
                expected_m=expected_m,
                expert_x_scale=None,
            )

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

        # Require comm stream for PEO overlap (level 3)
        if self._enable_peo_level == 3:
            self._require_comm_stream(extra_expert_args)
            assert self._comm_stream is not None

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
