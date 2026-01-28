from typing import Any, Dict, Optional, Tuple

import torch

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
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.util import (
    finalize_tp_gather,
    prepare_tp_slice,
)
from rtp_llm.models_py.utils.arch import get_sm

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
        checker.check(not resolver.enable_peo(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)
        # Initialize DeepEP low-latency wrapper and buffer
        self._num_experts = config.expert_num
        self._ll_num_max_token_per_rank = (
            DeepepWrapperConfig.calc_low_latency_max_token_per_rank(
                config.max_generate_batch_size, config.tp_size, quant_config
            )
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            self.config, self._ll_num_max_token_per_rank
        )
        self._wrapper = DeepEPWrapper.get_instance(deepep_config)
        assert (
            self._wrapper.mode == DeepEPMode.LOW_LATENCY
        ), "DeepEP mode should be LOW_LATENCY"
        self._buffer = self._wrapper.buffer
        # Initialize low-latency router parameters
        self._handle = None
        self._zero_copy = False
        self._async_finish = False
        self._num_topk = self._wrapper.num_topk
        self._use_accl_ep = self._wrapper.use_accl_ep
        self._return_recv_hook = config.enable_comm_overlap
        self._num_max_dispatch_tokens_per_rank = self._wrapper.ll_num_max_token_per_rank
        self._use_fp8_dispatch = (
            quant_config.is_quantized
            and quant_config.quant_dtype == torch.float8_e4m3fn
        )

    @property
    def deepep_wrapper(self) -> DeepEPWrapper:
        return self._wrapper

    @property
    def handle(self) -> Optional[Tuple[Any, ...]]:
        return self._handle

    def _pre_dispatch(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Pre-process dispatch inputs for DeepEP low-latency dispatch.

        Args:
            a1: [num_tokens, hidden_size]
            topk_ids: [num_tokens, num_topk]
            topk_weights: [num_tokens, num_topk]
        Returns:
            tp_dispatch_input: [tp_num_tokens, hidden_size]
            tp_topk_ids: [tp_num_tokens, num_topk]
            tp_topk_weights: [tp_num_tokens, num_topk]
            dispatch_args: Dict[str, Any]
        """
        # Check payload data
        num_tokens, hidden_size = a1.size()
        assert (
            hidden_size in SUPPORTED_HIDDEN_SIZES
            and hidden_size % DEEPEP_QUANT_BLOCK_SIZE == 0
        )
        tp_num_tokens = (num_tokens + self.config.tp_size - 1) // self.config.tp_size
        assert tp_num_tokens <= self._num_max_dispatch_tokens_per_rank, (
            f"tp_num_tokens {tp_num_tokens} > self._num_max_dispatch_tokens_per_rank "
            f"{self._num_max_dispatch_tokens_per_rank}"
        )
        assert topk_ids.size(0) == num_tokens and topk_weights.size(0) == num_tokens
        assert (
            topk_ids.size(1) == self._num_topk
            and topk_weights.size(1) == self._num_topk
        )

        # Check quantization
        if self._use_fp8_dispatch:
            assert self.quant_config.is_block_quantized or (
                self.quant_config.is_per_act_token and self._use_accl_ep
            ), (
                "DeepEP Low-Latency only supports fp8 block quantization or per_act_token "
                "quantization with ACCL-EP"
            )
        else:
            assert not self.quant_config.is_quantized

        # Check handle
        assert self._handle is None

        # Slice dispatch activation, topk_ids and topk_weights by tp
        tp_dispatch_input, tp_topk_ids, tp_topk_weights, _ = prepare_tp_slice(
            a1,
            topk_ids,
            topk_weights,
            tp_size=self.config.tp_size,
            tp_rank=self.config.tp_rank,
        )

        # Prepare dispatch basic arguments
        dispatch_args: Dict[str, Any] = {
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

        return tp_dispatch_input, tp_topk_ids, tp_topk_weights, dispatch_args

    def _dispatch_normal(
        self,
        dispatch_args: Dict[str, Any],
        tp_topk_weights: torch.Tensor,
    ) -> ExpertForwardPayload:
        """Normal low-latency dispatch path.
        Args:
            dispatch_args: Dict[str, Any]
            tp_topk_weights: [tp_num_tokens, num_topk]
        Returns:
            ExpertForwardPayload: Expert forward payload.
        """
        # Calculate the expected number of tokens for each expert
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
        expert_x, expert_num_tokens, self._handle, _, hook = (
            self._buffer.low_latency_dispatch(**dispatch_args)
        )
        if self._return_recv_hook:
            hook()

        # Check the type of the expert_x and get the expert_x_scale
        if self._use_fp8_dispatch:
            assert isinstance(expert_x, tuple), "expert_x should be a tuple"
            expert_x, expert_x_scale = expert_x[0], expert_x[1]
        else:
            assert isinstance(expert_x, torch.Tensor), "expert_x should be a tensor"
            expert_x_scale = None

        # Create and return the expert forward payload
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

    def _pre_combine(
        self,
        payload: CombineForwardPayload,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Dict[str, Any]:
        """Pre-process combine inputs for DeepEP low-latency combine.
        Args:
            payload: CombineForwardPayload
            topk_ids: [num_tokens, num_topk] (will be converted to int64 here)
            topk_weights: [num_tokens, num_topk]
        Returns:
            combine_args: Dict[str, Any]
        """
        # Check the handle
        assert (
            self._handle is not None
        ), "DeepEP Low-latency combine handle is missing for finalize()."
        # Convert topk_ids to int64
        topk_ids = topk_ids.to(torch.int64)
        # Create and return the combine arguments
        return {
            "x": payload.fused_expert_output,
            "topk_idx": topk_ids,
            "topk_weights": topk_weights,
            "handle": self._handle,
            "zero_copy": self._zero_copy,
            "async_finish": self._async_finish,
            "return_recv_hook": self._return_recv_hook,
        }

    def _post_combine(
        self,
        combined_x: torch.Tensor,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Post-process combined output for DeepEP low-latency combine.
        Args:
            combined_x: [num_tokens, hidden_size]
            extra_finalize_args: Optional[Dict[str, Any]]
        Returns:
            torch.Tensor: Combined output tensor.
        """
        # All-gather the combined_x across TP ranks
        combined_x = finalize_tp_gather(
            combined_x,
            tp_size=self.config.tp_size,
            extra_finalize_args=extra_finalize_args,
        )
        # Reset the handle
        self._handle = None
        return combined_x

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        """Dispatch tokens to experts across all ep ranks."""
        # Pre-process dispatch inputs
        _, _, tp_topk_weights, dispatch_args = self._pre_dispatch(
            a1=a1,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        # Dispatch tokens
        return self._dispatch_normal(dispatch_args, tp_topk_weights)

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Combine expert outputs back to all original ranks."""
        # Pre-process combine inputs
        combine_args = self._pre_combine(payload, topk_ids, topk_weights)
        # Combine expert outputs
        combined_x, _, hook = self._buffer.low_latency_combine(**combine_args)
        if self._return_recv_hook:
            hook()
        # Post-process combined output
        return self._post_combine(combined_x, extra_finalize_args)
