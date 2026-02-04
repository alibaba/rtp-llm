from typing import Any, Callable, Dict, List, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    has_deep_gemm,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
    DeepGemmMaskedExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm

COMPUTE_STREAM = torch.cuda.Stream()


class DeepGemmMaskedPeoExecutor(DeepGemmMaskedExecutor):

    @classmethod
    def executor_type(cls):
        return ExecutorType.DEEPGEMM_MASKED_PEO

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepGemmMaskedPeoExecutor can handle the configuration"""
        resolver = MoeConfigResolver()
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 9)
        checker.check(resolver.is_bf16(config))
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method in [None, "FP8_PER_BLOCK"])
        checker.check(resolver.enable_peo(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """Initialize the DeepGemmMaskedPeoExecutor.
        Args:
            config: Model configuration.
            quant_config: Quantization configuration.
            weights: Dictionary containing model weights.
        """
        super().__init__(config, quant_config, weights)
        # PEO overlap level
        self._enable_peo_level = self.config.moe_config.enable_peo_level
        if self._enable_peo_level <= 0:
            raise ValueError(
                "DeepGemmMaskedPeoExecutor requires enable_peo_level > 0, but got "
                f"{self._enable_peo_level}"
            )
        # Number of PEO rounds
        self._num_peo_rounds = self.config.moe_config.num_peo_rounds
        if self._num_peo_rounds < 2:
            raise ValueError(
                f"num_peo_rounds must be greater than 1, but got {self._num_peo_rounds}"
            )
        # Number of communication SMs
        self._num_comm_sms = self.config.moe_config.deep_ep_num_sm
        # Check if num_local_experts is divisible by num_peo_rounds
        assert self._num_local_experts % self._num_peo_rounds == 0, (
            f"num_local_experts ({self._num_local_experts}) must be divisible by "
            f"num_peo_rounds ({self._num_peo_rounds})"
        )
        # Total number of SMs
        self._num_device_total_sms = get_num_device_sms()
        # Number of SMs for DeepGEMM
        self._num_gemm_sms = self._num_device_total_sms - self._num_comm_sms * (
            2 if self._enable_peo_level == 3 else 1
        )
        # Check if num_gemm_sms is valid
        if self._num_gemm_sms < 1 or self._num_gemm_sms >= self._num_device_total_sms:
            raise ValueError(
                f"num_gemm_sms must be > 0 and < num_device_total_sms ({self._num_device_total_sms}), "
                f"but got {self._num_gemm_sms}"
            )
        # Number of experts per round
        self._num_experts_per_round = self._num_local_experts // self._num_peo_rounds
        # Communication stream for PEO level 3
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

    def _execute_peo(
        self,
        expert_x: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        expert_x_scale: Optional[torch.Tensor],
        dispatch_recv_events: Optional[List[torch.cuda.Event]],
    ) -> CombineForwardPayload:
        """Execute PEO expert FFN.
        Args:
            expert_x (torch.Tensor): Expert input.
            masked_m (torch.Tensor): Masked input.
            expected_m (int): Expected output.
            expert_x_scale (Optional[torch.Tensor]): Expert input scale.
            dispatch_recv_events (Optional[List[torch.cuda.Event]]): Dispatch receive events.
        Returns:
            CombineForwardPayload: Combine forward payload.
        """
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
        # Check and prepare execute data
        expert_x, masked_m, expected_m, expert_x_scale = (
            self._check_and_prepare_execute(payload)
        )
        # Require comm stream for PEO level 3
        if self._enable_peo_level == 3:
            self._require_comm_stream(extra_expert_args)
        # Execute PEO expert FFN
        return self._execute_peo(
            expert_x=expert_x,
            masked_m=masked_m,
            expected_m=expected_m,
            expert_x_scale=expert_x_scale,
            dispatch_recv_events=payload.dispatch_recv_events,
        )
