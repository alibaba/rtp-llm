import os
from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.distributed.deepep_initializer import DeepEpInitializer
from rtp_llm.models_py.distributed.deepep_wrapper import use_accl_ep
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
from rtp_llm.models_py.utils.arch import get_num_device_sms

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
        checker.check(resolver.is_ep_enabled(config))
        checker.check(resolver.use_low_latency(config))
        checker.check(DeepEpInitializer.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        use_fp8_dispatch: bool = True,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self._config = config
        self._num_experts = config.expert_num
        wrapper = DeepEpInitializer.get_deepep_wrapper(self._config)
        self._buffer = wrapper.buffer
        self._num_max_dispatch_tokens_per_rank = wrapper.ll_num_max_token_per_rank
        self._use_fp8_dispatch = use_fp8_dispatch
        self._zero_copy = zero_copy
        self._async_finish = async_finish
        self._return_recv_hook = return_recv_hook
        self._opt_level = int(os.environ.get("ACCL_LOW_LATENCY_OPTIMIZE", 1))
        self._handle: Optional[Tuple[Any, ...]] = None
        self._use_accl_ep = use_accl_ep()

    @property
    def handle(self) -> Optional[Tuple[Any, ...]]:
        return self._handle

    def _normal_prepare(
        self, dispatch_args: dict[str, Any], expert_payload: ExpertForwardPayload
    ):
        """Normal prepare for DeepEP Low-Latency.
        Args:
            dispatch_args (dict[str, Any]): Arguments for dispatching tokens to experts.
            expert_payload (ExpertForwardPayload): Payload for expert computation.
        """
        # Normal prepare
        expert_x, expert_num_tokens, self._handle, _, _ = (
            self._buffer.low_latency_dispatch(**dispatch_args)
        )
        # Save data
        expert_payload.expert_x = expert_x[0] if self._use_fp8_dispatch else expert_x
        expert_payload.expert_x_scale = expert_x[1] if self._use_fp8_dispatch else None
        expert_payload.expert_tokens_meta.expert_num_tokens = expert_num_tokens

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:
        """
        Dispatches tokens to experts across all ep ranks.
        """
        # Check input data
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

        # Convert topk_ids to int64
        topk_ids = topk_ids.to(torch.int64)
        # Scatter by tp
        tp_size = self._config.tp_size
        tp_rank = self._config.tp_rank
        token_num = a1.size(0)
        tp_token_size = (token_num + tp_size - 1) // tp_size
        slice_begin = min(tp_token_size * tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)
        tp_expert_input = torch.narrow(a1, 0, slice_begin, slice_size)
        tp_expert_ids = torch.narrow(topk_ids, 0, slice_begin, slice_size)
        tp_expert_scales = torch.narrow(topk_weights, 0, slice_begin, slice_size)

        # Prepare dispatch basic arguments
        dispatch_args = {
            "x": tp_expert_input,
            "topk_idx": tp_expert_ids,
            "num_max_dispatch_tokens_per_rank": self._num_max_dispatch_tokens_per_rank,
            "num_experts": self._num_experts,
            "use_fp8": self._use_fp8_dispatch,
            "async_finish": self._async_finish,
            "return_recv_hook": self._return_recv_hook,
        }
        # Select quantization of DeepEP low latency dispatch
        if self._use_fp8_dispatch:
            dispatch_args.update({"use_fp8": True})
            if quant_config.is_block_quantized:
                if is_deep_gemm_e8m0_used():
                    dispatch_args.update({"round_scale": True, "use_ue8m0": True})
            elif quant_config.is_per_act_token:
                if self._use_accl_ep:
                    dispatch_args.update({"pertoken_quant": True})
                else:
                    raise NotImplementedError(
                        "Only ACCL-EP supports fp8 per_act_token quantization"
                    )
            else:
                raise NotImplementedError(
                    "DeepEpLowLatencyRouter only supports fp8 block or per_act_token quantization"
                )
        else:
            dispatch_args.update({"use_fp8": False})

        # Initialize expert payload
        expert_payload = ExpertForwardPayload(
            expert_x_origin_dtype=a1.dtype,
            expert_topk_weights=tp_expert_scales,
            expert_topk_ids=tp_expert_ids,
            expert_tokens_meta=ExpertTokensMetadata(
                expected_m=max(
                    1,
                    int(
                        token_num
                        * self._config.dp_size
                        * self._config.moe_k
                        // self._num_experts
                    ),
                )
            ),
        )

        # Normal prepare
        self._normal_prepare(dispatch_args, expert_payload)

        # Check scale shape for DeepEP low latency dispatch
        E, M, K = expert_payload.expert_x.shape
        if self._use_fp8_dispatch:
            assert expert_payload.expert_x_scale is not None
            assert expert_payload.expert_x_scale.shape[0] == E
            assert expert_payload.expert_x_scale.shape[1] == M
            if quant_config.is_block_quantized:
                if is_deep_gemm_e8m0_used():
                    assert expert_payload.expert_x_scale.dtype == torch.int32
                    assert (
                        expert_payload.expert_x_scale.shape[2]
                        == (K // DEEPEP_QUANT_BLOCK_SIZE + 3) // 4
                    )
                else:
                    assert expert_payload.expert_x_scale.dtype == torch.float32
                    assert (
                        expert_payload.expert_x_scale.shape[2]
                        == K // DEEPEP_QUANT_BLOCK_SIZE
                    )
            elif quant_config.is_per_act_token:
                if self._use_accl_ep:
                    assert expert_payload.expert_x_scale.dtype == torch.float32
                    assert expert_payload.expert_x_scale.shape[2] == 1
                else:
                    raise RuntimeError(
                        "Only ACCL-EP supports fp8 per_act_token quantization"
                    )
            else:
                raise RuntimeError(
                    "DeepEpLowLatencyRouter only supports fp8 block or per_act_token quantization"
                )
        else:
            assert expert_payload.expert_x_scale is None

        return expert_payload

    def _normal_finalize(self, combine_args: dict[str, Any]) -> torch.Tensor:
        """Normal finalize for DeepEP Low-Latency.
        Args:
            combine_args (dict[str, Any]): Arguments for combining expert outputs.
        """
        # Normal finalize
        combined_x, _, _ = self._buffer.low_latency_combine(**combine_args)
        return combined_x

    def _finalize_post_tp_gather(
        self, combined_x: torch.Tensor, extra_finalize_args: Optional[Dict[str, Any]]
    ) -> torch.Tensor:
        """Finalize post tp gather for DeepEP Low-Latency.
        Args:
            combined_x (torch.Tensor): Combined output from all tp ranks.
            extra_finalize_args (Optional[Dict[str, Any]]): Extra finalize arguments.
        """
        # Check input data
        assert combined_x.dim() == 2
        assert extra_finalize_args is not None
        assert "original_num_tokens" in extra_finalize_args
        # Get original number of tokens
        tp_size = self._config.tp_size
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
        combine_args = {
            "x": payload.fused_expert_output,
            "topk_idx": topk_ids,
            "topk_weights": topk_weights,
            "handle": self._handle,
            "zero_copy": self._zero_copy,
            "async_finish": self._async_finish,
            "return_recv_hook": self._return_recv_hook,
        }
        if self._use_accl_ep:
            combine_args.update({"opt_level": self._opt_level})

        # Normal finalize
        combined_x = self._normal_finalize(combine_args)

        # Finalize post tp gather
        combined_x = self._finalize_post_tp_gather(combined_x, extra_finalize_args)
        # reset handle
        self._handle = None

        return combined_x
