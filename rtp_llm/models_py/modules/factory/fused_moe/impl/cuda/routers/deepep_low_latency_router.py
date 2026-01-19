import os
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPMode,
    DeepEPWrapper,
    DeepepWrapperConfig,
)
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType
from rtp_llm.models_py.modules.factory.fused_moe.utils.deepep_configure import (
    calc_low_latency_max_token_per_rank,
)
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm

# DeepEP kernels quantize dispatch inputs in 128 element chunks.
DEEPEP_QUANT_BLOCK_SIZE = 128
# DeepEP Low-Latency supports hidden sizes
SUPPORTED_HIDDEN_SIZES = [1536, 2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]


class DeepEpLowLatencyRouter(FusedMoeDataRouter):
    """
    A data router for Mixture-of-Experts that utilizes deep_ep's low-latency communication primitives.

    This router dispatches tokens to experts and receives results from experts across all ep ranks.
    """

    @classmethod
    def router_type(cls):
        return RouterType.DEEPEP_LOW_LATENCY

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepEpLowLatencyRouter can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(get_sm()[0] >= 9)
        checker.check(resolver.is_ep_enabled(config))
        checker.check(resolver.use_low_latency(config))
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

        # Determine use_fp8_dispatch based on quant_config
        use_fp8_dispatch = (
            quant_config.is_quantized
            and quant_config.quant_dtype == torch.float8_e4m3fn
        )

        # DeepEpLowLatency-specific initialization
        self._num_experts = config.expert_num
        self._ll_num_max_token_per_rank = calc_low_latency_max_token_per_rank(
            config.max_generate_batch_size, config.tp_size, quant_config
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            self.config, self._ll_num_max_token_per_rank
        )
        self._wrapper = DeepEPWrapper.get_instance(deepep_config)
        assert (
            self._wrapper.mode == DeepEPMode.LOW_LATENCY
        ), "DeepEP mode should be LOW_LATENCY"
        self._buffer = self._wrapper.buffer
        self._num_topk = self._wrapper.num_topk
        self._num_max_dispatch_tokens_per_rank = self._wrapper.ll_num_max_token_per_rank
        self._comm_stream = self._wrapper.buffer.get_comm_stream()
        self._use_fp8_dispatch = use_fp8_dispatch
        self._zero_copy = False
        self._return_recv_hook = config.enable_comm_overlap
        self._async_finish = not self._return_recv_hook
        self._handle: Optional[Tuple[Any, ...]] = None
        self._use_accl_ep = self._wrapper.use_accl_ep
        # PEO level
        self._enable_peo_level = config.moe_config.enable_peo_level
        if self._enable_peo_level > 0:
            # Number of PEO rounds
            self._num_peo_rounds = config.moe_config.num_peo_rounds
            # Number of comm sms
            self._num_comm_sms = config.moe_config.deep_ep_num_sm
            self._num_device_total_sms = get_num_device_sms()
            # Check parameters
            if self._enable_peo_level not in [1, 2, 3, 4]:
                raise ValueError(
                    f"Invalid PEO level: {self._enable_peo_level} , only support 1, 2, 3, 4"
                )
            if self._num_peo_rounds < 2:
                raise ValueError(
                    f"num_peo_rounds must be greater than 1, but got {self._num_peo_rounds}"
                )
            if (
                self._num_comm_sms < 2
                or self._num_comm_sms >= self._num_device_total_sms
            ):
                raise ValueError(
                    f"num_comm_sms must be greater than 1 and less than num_device_total_sms: {self._num_device_total_sms}, but got {self._num_comm_sms}"
                )

    @property
    def deepep_wrapper(self) -> DeepEPWrapper:
        return self._wrapper

    @property
    def handle(self) -> Optional[Tuple[Any, ...]]:
        return self._handle

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        return self._comm_stream

    def _prepare_pre_tp_slice(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice the dispatch activation, topk_ids and topk_weights by tp.
        Args:
            a1 (torch.Tensor): The dispatch activation tensor.
            topk_ids (torch.Tensor): The topk ids tensor.
            topk_weights (torch.Tensor): The topk weights tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The sliced dispatch activation, topk_ids and topk_weights.
        """
        # Convert topk_ids to int64
        topk_ids = topk_ids.to(torch.int64)
        # Slice by tp
        tp_size = self.config.tp_size
        tp_rank = self.config.tp_rank
        token_num = a1.size(0)
        tp_token_size = (token_num + tp_size - 1) // tp_size
        slice_begin = min(tp_token_size * tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)
        tp_dispatch_input = torch.narrow(a1, 0, slice_begin, slice_size)
        tp_topk_ids = torch.narrow(topk_ids, 0, slice_begin, slice_size)
        tp_topk_weights = torch.narrow(topk_weights, 0, slice_begin, slice_size)

        return tp_dispatch_input, tp_topk_ids, tp_topk_weights

    def _calc_comm_send_recv_sms(
        self,
        num_comm_sms: int,
        num_device_total_sms: int,
        round_id: int,
        num_peo_rounds: int,
        is_dispatch: bool,
        peo_level: int,
    ) -> Tuple[int, int]:
        """
        Calculate the number of send and recv sms for the given round of PEO.
        """
        # Check if the parameters are valid
        if peo_level == 4 and is_dispatch:
            raise ValueError(
                f"PEO level 4 is not supported in _calc_comm_send_recv_sms for dispatch"
            )

        # Calculate number of recv sms
        if not is_dispatch:
            # For combine, recv always uses all device SMs
            num_comm_recv_sms = num_device_total_sms
        else:
            # For dispatch, recv depends on peo_level and round_id
            if peo_level in (1, 2):
                num_comm_recv_sms = (
                    num_device_total_sms if round_id == 0 else num_comm_sms
                )
            else:  # peo_level == 3
                num_comm_recv_sms = (
                    (num_device_total_sms - num_comm_sms // 2)
                    if round_id == 0
                    else num_comm_sms // 2
                )

        # Calculate number of send sms
        is_first_round = round_id == 0
        is_last_round = round_id == (num_peo_rounds - 1)

        if (
            (peo_level == 1 and is_dispatch)  # peo_level 1 dispatch always uses all SMs
            or (
                is_dispatch and is_first_round and peo_level in (2, 3)
            )  # peo_level 2/3 dispatch first round
            or (
                not is_dispatch and is_last_round
            )  # All combine modes use all SMs in last round
        ):
            num_comm_send_sms = num_device_total_sms
        else:
            # For non-special rounds, use num_comm_sms or num_comm_sms // 2 based on peo_level
            num_comm_send_sms = num_comm_sms // 2 if peo_level == 3 else num_comm_sms

        return num_comm_send_sms, num_comm_recv_sms

    def _peo_prepare(
        self, dispatch_args: Dict[str, Any], tp_topk_weights: torch.Tensor
    ) -> ExpertForwardPayload:
        """Dispatch tokens to experts with Per-Expert Overlap (PEO) level 1, 2, 3.

        Args:
            dispatch_args (Dict[str, Any]): Arguments for dispatching tokens to experts.
            tp_topk_weights (torch.Tensor): Topk weights tensor for this tp rank.
        Returns:
            ExpertForwardPayload: Expert forward payload.
        """
        # Calculate expected_m
        tp_num_tokens = dispatch_args["x"].size(0)
        expected_m = max(
            1,
            int(
                tp_num_tokens
                * self.config.ep_size
                * self._num_topk
                // self._num_experts
            ),
        )
        # Initialize expert forward payload
        expert_payload = ExpertForwardPayload(
            expert_x=None,
            expert_x_origin_dtype=dispatch_args["x"].dtype,
            expert_topk_ids=dispatch_args["topk_idx"],
            expert_topk_weights=tp_topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(expected_m=expected_m),
        )
        # Updata dispatch common arguments for all PEO levels
        dispatch_args.update(
            {
                "async_finish": False,
                "return_recv_hook": True,
                "use_expert_overlap": True,
                "num_rounds": self._num_peo_rounds,
                "hook_use_comm_stream": False,
            }
        )
        # Prepare _calc_comm_send_recv_sms partial function for all PEO levels
        _calc_comm_send_recv_sms_partial = partial(
            self._calc_comm_send_recv_sms,
            num_comm_sms=self._num_comm_sms,
            num_device_total_sms=self._num_device_total_sms,
            num_peo_rounds=self._num_peo_rounds,
            is_dispatch=True,
            peo_level=self._enable_peo_level,
        )
        # Dispatch for PEO level 1,2,3
        if self._enable_peo_level == 1:
            dispatch_recv_hooks = []
            dispatch_recv_events = []
            for round_id in range(self._num_peo_rounds):
                # Calculate number of send and recv sms
                send_num_sms, recv_num_sms = _calc_comm_send_recv_sms_partial(
                    round_id=round_id
                )
                # Update dispatch arguments
                dispatch_args.update(
                    {
                        "round_id": round_id,
                        "send_num_sms": send_num_sms,
                        "recv_num_sms": recv_num_sms,
                    }
                )
                # Dispatch send
                packed_recv_x, packed_recv_count, tmp_handle, _, hook = (
                    self._buffer.low_latency_dispatch(**dispatch_args)
                )
                # Save data for first round
                if round_id == 0:
                    expert_payload.expert_x = (
                        packed_recv_x[0] if self._use_fp8_dispatch else packed_recv_x
                    )
                    expert_payload.expert_x_scale = (
                        packed_recv_x[1] if self._use_fp8_dispatch else None
                    )
                    expert_payload.expert_tokens_meta.expert_num_tokens = (
                        packed_recv_count
                    )
                    self._handle = tmp_handle
                # Save hook for all rounds
                dispatch_recv_hooks.append(hook)
            for hook in dispatch_recv_hooks:
                # Execute dispatch recv hook
                hook()
                # Record dispatch recv event
                dispatch_recv_event = torch.cuda.Event()
                torch.cuda.current_stream().record_event(dispatch_recv_event)
                dispatch_recv_events.append(dispatch_recv_event)
            expert_payload.dispatch_recv_events = dispatch_recv_events
        elif self._enable_peo_level == 2:
            dispatch_recv_events = []
            for round_id in range(self._num_peo_rounds):
                # Calculate number of send and recv sms
                send_num_sms, recv_num_sms = _calc_comm_send_recv_sms_partial(
                    round_id=round_id
                )
                # Update dispatch arguments
                dispatch_args.update(
                    {
                        "round_id": round_id,
                        "send_num_sms": send_num_sms,
                        "recv_num_sms": recv_num_sms,
                    }
                )
                # Dispatch send
                packed_recv_x, packed_recv_count, tmp_handle, _, hook = (
                    self._buffer.low_latency_dispatch(**dispatch_args)
                )
                # Execute dispatch recv hook
                hook()
                # Record dispatch recv event
                dispatch_recv_event = torch.cuda.Event()
                torch.cuda.current_stream().record_event(dispatch_recv_event)
                dispatch_recv_events.append(dispatch_recv_event)
                # Save data for first round
                if round_id == 0:
                    expert_payload.expert_x = (
                        packed_recv_x[0] if self._use_fp8_dispatch else packed_recv_x
                    )
                    expert_payload.expert_x_scale = (
                        packed_recv_x[1] if self._use_fp8_dispatch else None
                    )
                    expert_payload.expert_tokens_meta.expert_num_tokens = (
                        packed_recv_count
                    )
                    self._handle = tmp_handle
            expert_payload.dispatch_recv_events = dispatch_recv_events
        elif self._enable_peo_level == 3:
            dispatch_send_hooks = []
            dispatch_send_events = []
            dispatch_recv_events = []
            for round_id in range(self._num_peo_rounds):
                # Calculate number of send and recv sms
                send_num_sms, recv_num_sms = _calc_comm_send_recv_sms_partial(
                    round_id=round_id
                )
                # Update dispatch arguments
                dispatch_args.update(
                    {
                        "round_id": round_id,
                        "send_num_sms": send_num_sms,
                        "recv_num_sms": recv_num_sms,
                        "hook_use_comm_stream": (
                            True if round_id < (self._num_peo_rounds - 1) else False
                        ),
                    }
                )
                # Dispatch send
                packed_recv_x, packed_recv_count, tmp_handle, _, hook = (
                    self._buffer.low_latency_dispatch(**dispatch_args)
                )
                # Save hook for all rounds
                dispatch_send_hooks.append(hook)
                # Record dispatch send event
                if round_id < (self._num_peo_rounds - 1):
                    dispatch_send_event = torch.cuda.Event()
                    torch.cuda.current_stream().record_event(dispatch_send_event)
                    dispatch_send_events.append(dispatch_send_event)
                # Save data for first round
                if round_id == 0:
                    expert_payload.expert_x = (
                        packed_recv_x[0] if self._use_fp8_dispatch else packed_recv_x
                    )
                    expert_payload.expert_x_scale = (
                        packed_recv_x[1] if self._use_fp8_dispatch else None
                    )
                    expert_payload.expert_tokens_meta.expert_num_tokens = (
                        packed_recv_count
                    )
                    self._handle = tmp_handle
            # Solve CUDA Graph leads to a large number of CUDA streams problem, related to:
            # https://github.com/pytorch/pytorch/issues/155679
            # https://github.com/pytorch/pytorch/issues/152114
            for round_id, hook in enumerate(dispatch_send_hooks):
                # Communication stream wait for dispatch send event
                if round_id < (self._num_peo_rounds - 1):
                    self._comm_stream.wait_event(dispatch_send_events[round_id])
                else:
                    torch.cuda.current_stream().wait_event(
                        dispatch_recv_events[self._num_peo_rounds - 2]
                    )
                # Execute dispatch recv hook
                hook()
                # Record dispatch recv event
                dispatch_recv_event = torch.cuda.Event()
                self._comm_stream.record_event(dispatch_recv_event)
                dispatch_recv_events.append(dispatch_recv_event)
            expert_payload.dispatch_recv_events = dispatch_recv_events
        else:
            raise ValueError(
                f"Invalid PEO level for dispatch: {self._enable_peo_level}"
            )

        return expert_payload

    def _normal_prepare(
        self, dispatch_args: Dict[str, Any], tp_topk_weights: torch.Tensor
    ) -> ExpertForwardPayload:
        """Normal prepare for DeepEP Low-Latency.
        Args:
            dispatch_args (Dict[str, Any]): Arguments for dispatching tokens to experts.
            tp_topk_weights (torch.Tensor): Topk weights tensor for this tp rank.
        Returns:
            ExpertForwardPayload: Expert forward payload.
        """
        # Calculate expected_m
        tp_num_tokens = dispatch_args["x"].size(0)
        expected_m = max(
            1,
            int(
                tp_num_tokens
                * self.config.ep_size
                * self._num_topk
                // self._num_experts
            ),
        )

        # Dispatch tokens
        expert_x, expert_num_tokens, self._handle, event, hook = (
            self._buffer.low_latency_dispatch(**dispatch_args)
        )
        hook() if self._return_recv_hook else event.current_stream_wait()

        if self._use_fp8_dispatch:
            assert isinstance(expert_x, tuple), "expert_x should be a tuple"
            expert_x, expert_x_scale = expert_x[0], expert_x[1]
        else:
            assert isinstance(expert_x, torch.Tensor), "expert_x should be a tensor"
            expert_x = expert_x
            expert_x_scale = None

        # Return expert forward payload
        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=dispatch_args["x"].dtype,
            expert_topk_ids=dispatch_args["topk_idx"],
            expert_topk_weights=tp_topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expected_m=expected_m,
                expert_num_tokens=expert_num_tokens,
            ),
        )

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        """
        Dispatches tokens to experts across all ep ranks.
        """
        # Check payload data
        num_tokens, hidden_size = a1.size()
        assert (
            hidden_size in SUPPORTED_HIDDEN_SIZES
            and hidden_size % DEEPEP_QUANT_BLOCK_SIZE == 0
        )
        tp_num_tokens = (num_tokens + self.config.tp_size - 1) // self.config.tp_size
        assert (
            tp_num_tokens <= self._num_max_dispatch_tokens_per_rank
        ), f"tp_num_tokens {tp_num_tokens} > self._num_max_dispatch_tokens_per_rank {self._num_max_dispatch_tokens_per_rank}"
        assert topk_ids.size(0) == num_tokens and topk_weights.size(0) == num_tokens
        assert (
            topk_ids.size(1) == self._num_topk
            and topk_weights.size(1) == self._num_topk
        )
        # Check quantization
        if self._use_fp8_dispatch:
            assert self.quant_config.is_block_quantized or (
                self.quant_config.is_per_act_token and self._use_accl_ep
            ), "DeepEP Low-Latency only supports fp8 block quantization or per_act_token quantization with ACCL-EP"
        else:
            assert not self.quant_config.is_quantized
        # Check handle
        assert self._handle is None

        # Slice dispatch activation, topk_ids and topk_weights by tp
        tp_dispatch_input, tp_topk_ids, tp_topk_weights = self._prepare_pre_tp_slice(
            a1, topk_ids, topk_weights
        )

        # Prepare dispatch basic arguments
        dispatch_args = {
            "x": tp_dispatch_input,
            "topk_idx": tp_topk_ids,
            "num_max_dispatch_tokens_per_rank": self._num_max_dispatch_tokens_per_rank,
            "num_experts": self._num_experts,
            "use_fp8": self._use_fp8_dispatch,
            "async_finish": self._async_finish,
            "return_recv_hook": self._return_recv_hook,
        }

        # Set quantization config for DeepEP low latency dispatch
        if self.quant_config.is_block_quantized and is_deep_gemm_e8m0_used():
            dispatch_args.update({"round_scale": True, "use_ue8m0": True})
        elif self.quant_config.is_per_act_token:
            dispatch_args.update({"pertoken_quant": True})

        if self._enable_peo_level > 0 and self._enable_peo_level < 4:
            # PEO prepare for level 1, 2, 3
            expert_payload = self._peo_prepare(dispatch_args, tp_topk_weights)
        else:
            # Normal prepare
            expert_payload = self._normal_prepare(dispatch_args, tp_topk_weights)

        return expert_payload

    def _peo_finalize(
        self, payload: CombineForwardPayload, combine_args: Dict[str, Any]
    ) -> torch.Tensor:
        """Combine expert outputs back to all original ranks with Per-Expert Overlap (PEO) level 1, 2, 3.
        Args:
            payload (CombineForwardPayload): Payload for combining expert outputs.
            combine_args (Dict[str, Any]): Arguments for combining expert outputs.
        Returns:
            torch.Tensor: Combined output tensor.
        """
        # Check payload
        assert (
            payload.expert_executions is not None
        ), "Expert executions are missing for PEO finalize()."
        # Updata combine common arguments for all PEO levels
        combine_args.update(
            {
                "async_finish": False,
                "return_recv_hook": True,
                "use_expert_overlap": True,
                "num_rounds": self._num_peo_rounds,
                "hook_use_comm_stream": False,
            }
        )
        # Prepare _calc_comm_send_recv_sms partial function for all PEO levels
        _calc_comm_send_recv_sms_partial = partial(
            self._calc_comm_send_recv_sms,
            num_comm_sms=self._num_comm_sms,
            num_device_total_sms=self._num_device_total_sms,
            num_peo_rounds=self._num_peo_rounds,
            is_dispatch=False,
            peo_level=self._enable_peo_level,
        )
        # Combine for PEO level 1, 2, 3, 4
        for round_id in range(self._num_peo_rounds):
            # Calculate number of send and recv sms
            send_num_sms, recv_num_sms = _calc_comm_send_recv_sms_partial(
                round_id=round_id
            )
            # Update combine arguments
            combine_args.update(
                {
                    "round_id": round_id,
                    "send_num_sms": send_num_sms,
                    "recv_num_sms": recv_num_sms,
                }
            )
            # Execute expert execution
            payload.expert_executions[round_id]()
            # Combine send
            combined_x, _, hook = self._buffer.low_latency_combine(**combine_args)

        # Execute combine recv hook
        hook()

        return combined_x

    def _normal_finalize(self, combine_args: Dict[str, Any]) -> torch.Tensor:
        """Normal finalize for DeepEP Low-Latency.
        Args:
            combine_args (Dict[str, Any]): Arguments for combining expert outputs.
        Returns:
            torch.Tensor: Combined output tensor.
        """
        # Normal finalize
        combined_x, event, hook = self._buffer.low_latency_combine(**combine_args)
        hook() if self._return_recv_hook else event.current_stream_wait()

        return combined_x

    def _finalize_post_tp_gather(
        self, combined_x: torch.Tensor, extra_finalize_args: Optional[Dict[str, Any]]
    ) -> torch.Tensor:
        """Finalize post tp gather for DeepEP Low-Latency.
        Args:
            combined_x (torch.Tensor): Combined output from all tp ranks.
            extra_finalize_args (Optional[Dict[str, Any]]): Extra finalize arguments.
        Returns:
            torch.Tensor: Post tp gathered combined output tensor.
        """
        # Check input data
        assert combined_x.dim() == 2
        assert extra_finalize_args is not None
        assert "original_num_tokens" in extra_finalize_args
        # Get original number of tokens
        tp_size = self.config.tp_size
        original_num_tokens = extra_finalize_args["original_num_tokens"]
        tp_token_size = (original_num_tokens + tp_size - 1) // tp_size
        if tp_size > 1:
            # combine_x.size(0) might be 0
            if combined_x.size(0) < tp_token_size:
                # Pad combined output if needed
                padding_combined_x = torch.empty(
                    size=(tp_token_size - combined_x.size(0), combined_x.size(1)),
                    device=combined_x.device,
                    dtype=combined_x.dtype,
                )
                combined_x = torch.cat([combined_x, padding_combined_x], dim=0)
            # Gather combined output from all tp ranks
            gatherd_output = all_gather(combined_x, group=Group.TP).reshape(
                tp_size * tp_token_size, -1
            )
            combined_x = gatherd_output[:original_num_tokens, :]

        return combined_x

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Combines expert outputs back to all original ranks.
        Weight application and reduction happens in the combine kernel.
        """
        # Check handle
        assert (
            self._handle is not None
        ), "DeepEP Low-latency combine handle is missing for finalize()."

        # Convert topk_ids to int64
        topk_ids = topk_ids.to(torch.int64)

        # Prepare combine basic arguments
        combine_args: Dict[str, Any] = {
            "x": payload.fused_expert_output,
            "topk_idx": topk_ids,
            "topk_weights": topk_weights,
            "handle": self._handle,
            "zero_copy": self._zero_copy,
            "async_finish": self._async_finish,
            "return_recv_hook": self._return_recv_hook,
        }

        if self._enable_peo_level > 0:
            # PEO finalize for level 1, 2, 3, 4
            combined_x = self._peo_finalize(payload, combine_args)
        else:
            # Normal finalize
            combined_x = self._normal_finalize(combine_args)

        # Finalize post tp gather
        combined_x = self._finalize_post_tp_gather(combined_x, extra_finalize_args)

        # Reset handle
        self._handle = None

        return combined_x
