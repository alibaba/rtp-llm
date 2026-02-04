from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm

from .deepep_low_latency_router import DeepEpLowLatencyRouter


class DeepEpLowLatencyPeoRouter(DeepEpLowLatencyRouter):
    """
    DeepEP low-latency router with PEO (Per-Expert Overlap) enabled.

    This router dispatches tokens to experts and combines expert outputs back with
    multi-round overlap between communication and computation.
    """

    @classmethod
    def router_type(cls):
        return RouterType.DEEPEP_LOW_LATENCY_PEO

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepEpLowLatencyPeoRouter can handle the configuration"""
        resolver = MoeConfigResolver()
        checker.check(get_sm()[0] >= 9)
        checker.check(resolver.is_ep_enabled(config))
        checker.check(resolver.use_low_latency(config))
        checker.check(DeepEPWrapper.supported())
        checker.check(resolver.enable_peo(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)
        # Initialize communication stream for PEO
        self._comm_stream = self._buffer.get_comm_stream()
        # Initialize PEO parameters
        self._enable_peo_level = config.moe_config.enable_peo_level
        if self._enable_peo_level <= 0:
            raise ValueError(
                "DeepEpLowLatencyPeoRouter requires enable_peo_level > 0, but got "
                f"{self._enable_peo_level}"
            )
        self._num_device_total_sms = get_num_device_sms()
        self._num_comm_sms = config.moe_config.deep_ep_num_sm
        self._num_peo_rounds = config.moe_config.num_peo_rounds
        # Check PEO level and parameters
        if self._enable_peo_level not in [1, 2, 3, 4]:
            raise ValueError(
                f"Invalid PEO level: {self._enable_peo_level} , only support 1, 2, 3, 4"
            )
        if self._num_peo_rounds < 2:
            raise ValueError(
                f"num_peo_rounds must be greater than 1, but got {self._num_peo_rounds}"
            )
        if self._num_comm_sms < 4 or self._num_comm_sms >= self._num_device_total_sms:
            raise ValueError(
                f"num_comm_sms must be not less than 4 and less than num_device_total_sms: "
                f"{self._num_device_total_sms}, but got {self._num_comm_sms}"
            )

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        return self._comm_stream

    def _calc_comm_send_recv_sms(
        self,
        num_comm_sms: int,
        num_device_total_sms: int,
        round_id: int,
        num_peo_rounds: int,
        is_dispatch: bool,
        peo_level: int,
    ) -> Tuple[int, int]:
        """Calculate the number of send and recv sms for the given round of PEO.

        Args:
            num_comm_sms (int): The number of communication SMs.
            num_device_total_sms (int): The total number of SMs.
            round_id (int): The round id.
            num_peo_rounds (int): The number of PEO rounds.
            is_dispatch (bool): Whether the current round is dispatch.
            peo_level (int): The PEO level.

        Returns:
            Tuple[int, int]: The number of send and recv SMs.
        """
        # Check if the parameters are valid
        if peo_level == 4 and is_dispatch:
            raise ValueError(
                "PEO level 4 is not supported in _calc_comm_send_recv_sms for dispatch"
            )
        # If the current round is combine, the number of recv SMs is the total number of SMs
        if not is_dispatch:
            num_comm_recv_sms = num_device_total_sms
        else:
            if peo_level in (1, 2):
                num_comm_recv_sms = (
                    num_device_total_sms if round_id == 0 else num_comm_sms
                )
            else:  # peo_level == 3
                num_comm_recv_sms = (
                    (num_device_total_sms - num_comm_sms)
                    if round_id == 0
                    else num_comm_sms
                )

        # send SMs:
        is_first_round = round_id == 0
        is_last_round = round_id == (num_peo_rounds - 1)
        if (
            (peo_level == 1 and is_dispatch)
            or (is_dispatch and is_first_round and peo_level in (2, 3))
            or ((not is_dispatch) and is_last_round)
        ):
            num_comm_send_sms = num_device_total_sms
        else:
            num_comm_send_sms = num_comm_sms

        return num_comm_send_sms, num_comm_recv_sms

    def _dispatch_peo(
        self,
        dispatch_args: Dict[str, Any],
        tp_topk_weights: torch.Tensor,
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
        # Update dispatch common arguments for all PEO levels
        dispatch_args.update(
            {
                "async_finish": False,
                "return_recv_hook": True,
                "use_expert_overlap": True,
                "num_rounds": self._num_peo_rounds,
                "hook_use_comm_stream": False,
            }
        )
        # Calculate the number of send and recv SMs for each round
        _calc_comm_send_recv_sms_partial = partial(
            self._calc_comm_send_recv_sms,
            num_comm_sms=self._num_comm_sms,
            num_device_total_sms=self._num_device_total_sms,
            num_peo_rounds=self._num_peo_rounds,
            is_dispatch=True,
            peo_level=self._enable_peo_level,
        )
        # Dispatch tokens to experts with PEO level 1
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
                # Dispatch tokens to experts
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
        # Dispatch tokens to experts with PEO level 2
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
                # Dispatch tokens to experts
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
            # Save dispatch recv events
            expert_payload.dispatch_recv_events = dispatch_recv_events
        # Dispatch tokens to experts with PEO level 3
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
                # Dispatch tokens to experts
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
                # Execute dispatch send hook
                hook()
                # Record dispatch recv event
                dispatch_recv_event = torch.cuda.Event()
                self._comm_stream.record_event(dispatch_recv_event)
                dispatch_recv_events.append(dispatch_recv_event)
            # Save dispatch recv events
            expert_payload.dispatch_recv_events = dispatch_recv_events
        else:
            raise ValueError(
                f"Invalid PEO level for dispatch: {self._enable_peo_level}"
            )

        return expert_payload

    def _combine_peo(
        self,
        payload: CombineForwardPayload,
        combine_args: Dict[str, Any],
    ) -> torch.Tensor:
        """Combine expert outputs back to all original ranks with Per-Expert Overlap (PEO) level 1, 2, 3.
        Args:
            payload (CombineForwardPayload): Payload for combining expert outputs.
            combine_args (Dict[str, Any]): Arguments for combining expert outputs.
        Returns:
            torch.Tensor: Combined output tensor.
        """
        # Check if expert executions are missing
        assert (
            payload.expert_executions is not None
        ), "Expert executions are missing for PEO finalize()."
        # Update combine arguments
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
        # Combine expert outputs for PEO level 1, 2, 3, 4
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
            # Combine expert outputs
            combined_x, _, hook = self._buffer.low_latency_combine(**combine_args)
        # Execute combine recv hook
        hook()
        return combined_x

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        """Dispatch tokens to experts with Per-Expert Overlap (PEO) level 1, 2, 3, 4.

        Args:
            a1 (torch.Tensor): The dispatch activation tensor.
            a1_scale (Optional[torch.Tensor]): The dispatch activation scale tensor.
            a2_scale (Optional[torch.Tensor]): The dispatch activation scale tensor.
            topk_weights (torch.Tensor): The topk weights tensor.
            topk_ids (torch.Tensor): The topk ids tensor.
        Returns:
            ExpertForwardPayload: Expert forward payload.
        """
        # Pre-process dispatch inputs
        _, _, tp_topk_weights, dispatch_args = self._pre_dispatch(
            a1=a1,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        # Dispatch tokens with PEO level 1, 2, 3 or 4
        if self._enable_peo_level > 0 and self._enable_peo_level < 4:
            # Dispatch tokens with PEO level 1, 2, 3
            return self._dispatch_peo(dispatch_args, tp_topk_weights)
        else:
            # PEO level 4: dispatch is the same as the normal router
            return self._dispatch_normal(dispatch_args, tp_topk_weights)

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Combine expert outputs back to original ranks (with PEO overlap).
        Args:
            payload (CombineForwardPayload): Payload for combining expert outputs.
            topk_weights (torch.Tensor): The topk weights tensor.
            topk_ids (torch.Tensor): The topk ids tensor.
            apply_router_weight_on_input (bool): Whether to apply router weight on input.
            extra_finalize_args (Optional[Dict[str, Any]]): Extra finalize arguments.
        Returns:
            torch.Tensor: Combined output tensor.
        """
        # Pre-process combine inputs
        combine_args = self._pre_combine(payload, topk_ids, topk_weights)
        # Combine expert outputs
        combined_x = self._combine_peo(payload, combine_args)
        # Post-process combined output
        return self._post_combine(combined_x, extra_finalize_args)
