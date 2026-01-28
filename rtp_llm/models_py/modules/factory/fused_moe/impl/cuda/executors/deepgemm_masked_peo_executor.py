from typing import Any, Callable, Dict, List, Optional

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

COMPUTE_STREAM = torch.cuda.Stream()


class DeepGemmMaskedPeoExecutor(FusedMoeExpertExecutor):
    """
    PEO (Per-Expert Overlap) expert executor.

    It returns an `expert_executions` list in `CombineForwardPayload`, and the router will
    trigger per-round computation in its combine loop.
    """

    # The Deep Gemm kernels only support block size of 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls):
        return ExecutorType.DEEPGEMM_MASKED_PEO

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.enable_peo(config))
        checker.check(get_sm()[0] >= 9)
        checker.check(resolver.is_bf16(config))

        quant_method = resolver.get_quant_method(config)
        # PEO executor only supports no-quant / FP8 per-block (DeepGEMM masked kernels).
        checker.check(quant_method in [None, "FP8_PER_BLOCK"])
        checker.check(has_deep_gemm())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        # Initialize weights
        self._w1 = weights[W.moe_w1]
        self._w2 = weights[W.moe_w2]
        self._num_local_experts, self._N, self._K = self._w1.size()
        # Check weights shape
        assert self._N % 2 == 0
        assert self._w2.size(0) == self._num_local_experts
        assert self._w2.size(1) == self._K
        assert self._w2.size(2) == self._N // 2
        # Initialize weights scale
        self._w1_scale = weights.get(W.moe_s1, None)
        self._w2_scale = weights.get(W.moe_s2, None)
        # Check if use fp8
        self._use_fp8 = self.quant_config.is_quantized
        # If use fp8, check weights scale
        if self._use_fp8:
            assert self._w1_scale is not None and self._w2_scale is not None
            # Check if use fp8 block quantization
            if (
                self.quant_config.quant_dtype == torch.float8_e4m3fn
                and self.quant_config.is_block_quantized
                and self.quant_config.block_shape == self.DEEPGEMM_BLOCK_SHAPE
            ):
                # Initialize number of packed scales and scale dtype
                self._num_packed_scales = 1
                self._scale_dtype = torch.float32
                # If use UE8M0 scale, requantize weights
                if is_deep_gemm_e8m0_used():
                    self._w1, self._w1_scale = requant_weight_ue8m0(
                        self._w1, self._w1_scale
                    )
                    self._w2, self._w2_scale = requant_weight_ue8m0(
                        self._w2, self._w2_scale
                    )
                    # Update number of packed scales and scale dtype
                    self._num_packed_scales = 4
                    self._scale_dtype = torch.int32
            else:
                # Only support fp8 block quantization with block shape 128x128
                raise NotImplementedError(
                    "DeepGemmMaskedPeoExecutor only supports fp8 block quantization with block shape 128x128"
                )
        else:
            assert self._w1_scale is None and self._w2_scale is None

        # Get PEO overlap level
        self._enable_peo_level = self.config.moe_config.enable_peo_level
        if self._enable_peo_level <= 0:
            raise ValueError(
                "DeepGemmMaskedPeoExecutor requires enable_peo_level > 0, but got "
                f"{self._enable_peo_level}"
            )
        # Get number of PEO rounds
        self._num_peo_rounds = self.config.moe_config.num_peo_rounds
        if self._num_peo_rounds < 2:
            raise ValueError(
                f"num_peo_rounds must be greater than 1, but got {self._num_peo_rounds}"
            )
        # Get number of communication SMs
        self._num_comm_sms = self.config.moe_config.deep_ep_num_sm
        # Get total number of SMs
        self._num_device_total_sms = get_num_device_sms()
        # Get number of SMs for DeepGemm
        self._num_gemm_sms = self._num_device_total_sms - self._num_comm_sms * (
            2 if self._enable_peo_level == 3 else 1
        )
        if self._num_gemm_sms < 1 or self._num_gemm_sms >= self._num_device_total_sms:
            raise ValueError(
                f"num_gemm_sms must be > 0 and < num_device_total_sms ({self._num_device_total_sms}), "
                f"but got {self._num_gemm_sms}"
            )
        # Check if num_local_experts is divisible by num_peo_rounds
        assert self._num_local_experts % self._num_peo_rounds == 0, (
            f"num_local_experts ({self._num_local_experts}) must be divisible by "
            f"num_peo_rounds ({self._num_peo_rounds})"
        )
        self._num_experts_per_round = self._num_local_experts // self._num_peo_rounds
        # Initialize communication stream
        self._comm_stream: Optional[torch.cuda.Stream] = None

    def _require_comm_stream(self, extra_expert_args: Optional[dict]) -> None:
        """PEO level 3 requires a comm stream provided by the router (DeepEP) in extra_expert_args."""
        if self._comm_stream is not None:
            return
        if not extra_expert_args or "comm_stream" not in extra_expert_args:
            raise RuntimeError(
                "DeepGemmMaskedPeoExecutor requires extra_expert_args['comm_stream'] to be provided "
                "when enable_peo_level == 3. Please pass router.comm_stream via FusedMoe.forward."
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
        down_output: torch.Tensor,
        expert_x_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Forward masked grouped FFN.
        Args:
            start_idx (int): Start index of the experts.
            end_idx (int): End index of the experts.
            expert_x (torch.Tensor): Expert input.
            masked_m (torch.Tensor): Masked input.
            expected_m (int): Expected output.
            down_output (torch.Tensor): Down output.
            expert_x_scale (Optional[torch.Tensor]): Expert input scale.
        """
        # Get metadata
        device = expert_x.device
        num_tokens = expert_x.size(1)
        num_slice_experts = end_idx - start_idx

        # Execute masked grouped FFN
        if self._use_fp8:
            # Check expert_x_scale, w1_scale and w2_scale are not None for fp8
            assert expert_x_scale is not None
            assert self._w1_scale is not None and self._w2_scale is not None

            # Allocate upgate output
            upgate_output = torch.empty(
                (num_slice_experts, num_tokens, self._N),
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
            )
            # Free expert_x and expert_x_scale
            if end_idx == self._num_local_experts:
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
            # Free upgate output
            dispose_tensor(upgate_output)
            # Down GroupGEMM-1
            m_grouped_fp8_gemm_nt_masked(
                a=(down_input, down_input_scale),
                b=(self._w2[start_idx:end_idx], self._w2_scale[start_idx:end_idx]),
                output=down_output[start_idx:end_idx],
                masked_m=masked_m[start_idx:end_idx],
                expected_m=expected_m,
            )
            # Free down_input and down_input_scale
            dispose_tensor(down_input)
            dispose_tensor(down_input_scale)
        else:
            # Check expert_x_scale is None for bf16
            assert expert_x_scale is None
            # Allocate upgate output
            upgate_output = torch.empty(
                (num_slice_experts, num_tokens, self._N),
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
            # Free upgate output
            dispose_tensor(upgate_output)
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

    def _execute_peo(
        self,
        payload: ExpertForwardPayload,
        masked_m: torch.Tensor,
        expected_m: int,
    ) -> CombineForwardPayload:
        """Execute PEO expert FFN.
        Args:
            payload (ExpertForwardPayload): Expert forward payload.
            masked_m (torch.Tensor): Masked input.
            expected_m (int): Expected output.
        Returns:
            CombineForwardPayload: Combine forward payload.
        """
        # Get expert_x and expert_x_scale
        expert_x = payload.expert_x
        expert_x_scale = payload.expert_x_scale if self._use_fp8 else None
        # Get dispatch_recv_events
        dispatch_recv_events = payload.dispatch_recv_events
        # Check if ignore dispatch_recv_event
        ignore_dispatch_recv_event = (
            self._enable_peo_level == 4 and dispatch_recv_events is None
        )
        assert (
            dispatch_recv_events is not None and self._enable_peo_level != 4
        ) or ignore_dispatch_recv_event

        # Allocate down_output
        down_output = torch.empty(
            expert_x.shape,
            device=expert_x.device,
            dtype=torch.bfloat16,
        )

        # Helper function to create closure for expert execution, solve CUDA Graph leads to a large number of CUDA streams problem, related to:
        # https://github.com/pytorch/pytorch/issues/155679
        # https://github.com/pytorch/pytorch/issues/152114
        def create_expert_execution(round_id: int) -> Callable[[], None]:
            """Create a closure function for expert execution.
            Args:
                round_id: The round ID for this execution.
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
                    assert self._comm_stream is not None
                    current_gemm_stream = self._comm_stream
                    next_stream = torch.cuda.current_stream()
                # Determine the number of SMs for DeepGEMM
                num_gemm_sms = self._num_gemm_sms
                if self._enable_peo_level == 4 and round_id == 0:
                    num_gemm_sms = get_num_device_sms()
                elif self._enable_peo_level == 3 and round_id >= (
                    self._num_peo_rounds - 2
                ):
                    num_gemm_sms = self._num_gemm_sms + self._num_comm_sms

                # Compute stream wait for dispatch recv event to finish
                if (
                    (not ignore_dispatch_recv_event)
                    and dispatch_recv_events
                    and round_id < len(dispatch_recv_events)
                    and (
                        self._enable_peo_level != 3
                        or round_id != (self._num_peo_rounds - 2)
                    )
                ):
                    COMPUTE_STREAM.wait_event(dispatch_recv_events[round_id])

                # Execute masked grouped FFN with compute stream
                with torch.cuda.stream(current_gemm_stream):
                    with configure_deep_gemm_num_sms(num_gemm_sms):
                        self._forward_masked_grouped_ffn(
                            round_id * self._num_experts_per_round,
                            (round_id + 1) * self._num_experts_per_round,
                            expert_x,
                            masked_m,
                            expected_m,
                            down_output,
                            expert_x_scale,
                        )

                # Build dependency for Combine
                next_stream.wait_stream(current_gemm_stream)
                if self._enable_peo_level == 3 and round_id == (
                    self._num_peo_rounds - 3
                ):
                    assert self._comm_stream is not None
                    self._comm_stream.wait_stream(current_gemm_stream)

            return execute_expert

        # Create expert executions for each round
        expert_executions = [
            create_expert_execution(i) for i in range(self._num_peo_rounds)
        ]
        # Return combine forward payload
        return CombineForwardPayload(
            fused_expert_output=down_output,
            expert_executions=expert_executions,
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
        """Execute PEO expert executor.
        Args:
            payload (ExpertForwardPayload): Expert forward payload.
            activation (str): Activation function.
            expert_map (Optional[torch.Tensor]): Expert map.
            a2_scale (Optional[torch.Tensor]): A2 scale.
            apply_router_weight_on_input (bool): Whether to apply router weight on input.
            extra_expert_args (Optional[dict[str, Any]]): Extra expert arguments.
        Returns:
            CombineForwardPayload: Combine forward payload.
        """
        # Check expert_x and expert_x_scale
        expert_x = payload.expert_x
        num_local_experts, max_m, hidden_size = expert_x.shape
        assert num_local_experts == self._num_local_experts and hidden_size == self._K
        if self._use_fp8:
            assert payload.expert_x_scale is not None
            assert payload.expert_x_scale.size(0) == num_local_experts
            assert payload.expert_x_scale.size(1) == max_m
            assert (
                payload.expert_x_scale.size(2)
                == (
                    hidden_size // self.DEEPGEMM_BLOCK_SHAPE[1]
                    + self._num_packed_scales
                    - 1
                )
                // self._num_packed_scales
            )
            assert payload.expert_x_scale.dtype == self._scale_dtype
        else:
            assert payload.expert_x_scale is None
        # Check masked_m and expected_m
        assert (
            payload.expert_tokens_meta is not None
            and payload.expert_tokens_meta.expert_num_tokens is not None
        )
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        expected_m = payload.expert_tokens_meta.expected_m
        expected_m = min(max_m, expected_m) if expected_m is not None else max_m

        # Require comm stream for PEO level 3
        if self._enable_peo_level == 3:
            self._require_comm_stream(extra_expert_args)

        # Execute PEO expert FFN
        return self._execute_peo(payload, masked_m, expected_m)
