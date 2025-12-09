import os
from typing import Any, Optional, Tuple

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, all_gather
from rtp_llm.models_py.distributed.deepep_wrapper import get_deepep_wrapper
from rtp_llm.models_py.modules.moe.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
    TopKWeightAndReduce,
)
from rtp_llm.models_py.modules.moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from rtp_llm.models_py.modules.moe.utils import FusedMoEQuantConfig

# DeepEP kernels quantize dispatch inputs in 128 element chunks.
DEEPEP_QUANT_BLOCK_SIZE = 128
# DeepEP Low-Latency supports hidden sizes
SUPPORTED_HIDDEN_SIZES = [1536, 2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]


class DeepEpLowLatencyRouter(FusedMoeDataRouter):
    """
    A data router for Mixture-of-Experts that utilizes deep_ep's low-latency communication primitives.

    This router dispatches tokens to experts and receives results from experts across all ep ranks.
    """

    def __init__(
        self,
        config: GptInitModelParameters,
        use_fp8_dispatch: bool = True,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self._config = config
        self._buffer = get_deepep_wrapper().buffer
        self._num_max_dispatch_tokens_per_rank = (
            get_deepep_wrapper().ll_num_max_token_per_rank
        )
        self._use_fp8_dispatch = use_fp8_dispatch
        self._zero_copy = zero_copy
        self._async_finish = async_finish
        self._return_recv_hook = return_recv_hook
        self._opt_level = int(os.environ.get("ACCL_LOW_LATENCY_OPTIMIZE", 1))
        self._handle: Optional[Tuple[Any, ...]] = None

    @property
    def handle(self) -> Optional[Tuple[Any, ...]]:
        return self._handle

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:
        """
        Dispatches tokens to experts across all ep ranks.
        """
        # assert
        assert a1.dim() == 2 and topk_ids.dim() == 2
        assert a1.size(0) == topk_ids.size(0)
        num_tokens = a1.size(0)
        num_dispatch_tokens_per_rank = (
            num_tokens + self._config.tp_size - 1
        ) // self._config.tp_size
        assert (
            num_dispatch_tokens_per_rank <= self._num_max_dispatch_tokens_per_rank
        ), f"Number of dispatch tokens {num_dispatch_tokens_per_rank} exceeds the maximum number of dispatch tokens per rank {self._num_max_dispatch_tokens_per_rank}."
        hidden_dim = a1.size(1)
        assert (
            hidden_dim in SUPPORTED_HIDDEN_SIZES
        ), f"Hidden Size {hidden_dim} not in supported list of hidden sizes: {SUPPORTED_HIDDEN_SIZES}."
        if self._use_fp8_dispatch:
            assert (
                hidden_dim % DEEPEP_QUANT_BLOCK_SIZE == 0
            ), f"DeepEP Low-Latency only supports hidden sizes that are divisible by {DEEPEP_QUANT_BLOCK_SIZE}."
        has_per_token_scales = (
            a1_scale.numel() != 1
            if a1_scale is not None
            else (a2_scale.numel() != 1 if a2_scale is not None else False)
        )
        assert (
            not has_per_token_scales
        ), "DeepEP Low-Latency kernels doesn't support dispatching per-token scales."
        assert (
            self._handle is None
        ), "DeepEP Low-latency dispatch handle should be clean before prepare()."

        # dispatch
        topk_ids = topk_ids.to(torch.int64)

        # scatter by tp
        tp_size = self._config.tp_size
        tp_rank = self._config.tp_rank
        token_num = a1.size(0)
        tp_token_size = (token_num + tp_size - 1) // tp_size

        slice_begin = min(tp_token_size * tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)

        tp_expert_input = torch.narrow(a1, 0, slice_begin, slice_size)
        tp_expert_ids = torch.narrow(topk_ids, 0, slice_begin, slice_size)
        tp_expert_scales = torch.narrow(topk_weights, 0, slice_begin, slice_size)

        expert_x, expert_num_tokens, self._handle, _, _ = (
            self._buffer.low_latency_dispatch(
                tp_expert_input,
                tp_expert_ids,
                self._num_max_dispatch_tokens_per_rank,
                num_experts,
                use_fp8=self._use_fp8_dispatch,
                async_finish=self._async_finish,
                return_recv_hook=self._return_recv_hook,
            )
        )

        if quant_config.is_per_act_token:
            assert expert_x[0].shape[1] == expert_x[1].shape[1]
            assert expert_x[1].shape[-1] == 1

        # return payload
        return ExpertForwardPayload(
            expert_x=expert_x[0] if self._use_fp8_dispatch else expert_x,
            expert_x_scale=expert_x[1] if self._use_fp8_dispatch else None,
            expert_x_origin_dtype=a1.dtype,
            expert_topk_weights=tp_expert_scales,
            expert_topk_ids=tp_expert_ids,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
            ),
        )

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        """
        Combines expert outputs back to all original ranks.
        """
        # assert
        assert isinstance(
            weight_and_reduce_impl, TopKWeightAndReduceDelegate
        ), "Weight application and reduction happens in the combine kernel."
        assert (
            self._handle is not None
        ), "DeepEP Low-latency combine handle is missing for finalize()."

        # combine
        topk_ids = topk_ids.to(torch.int64)
        combined_x, _, _ = self._buffer.low_latency_combine(
            fused_expert_output,
            topk_ids,
            topk_weights,
            self._handle,
            zero_copy=self._zero_copy,
            async_finish=self._async_finish,
            return_recv_hook=self._return_recv_hook,
        )
        # reset handle
        self._handle = None

        # gather
        tp_size = self._config.tp_size
        original_num_tokens = extra_finalize_args["original_num_tokens"]
        tp_token_size = (original_num_tokens + tp_size - 1) // tp_size

        if tp_size > 1:
            # combine_x.size(0) might be 0
            if combined_x.size(0) < tp_token_size:
                padding_combined_x = torch.empty(
                    size=(tp_token_size - combined_x.size(0), combined_x.size(1)),
                    device=combined_x.device,
                    dtype=combined_x.dtype,
                )
                combined_x = torch.cat([combined_x, padding_combined_x], dim=0)

            gatherd_output = all_gather(combined_x, group=Group.TP).reshape(
                tp_size * tp_token_size, -1
            )
            gatherd_output = gatherd_output[:original_num_tokens, :]
            return gatherd_output

        return combined_x
