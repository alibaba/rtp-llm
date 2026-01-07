import logging
import os
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
        self._ll_num_max_token_per_rank = self._calc_low_latency_max_token_per_rank(
            config.max_generate_batch_size, config.tp_size, quant_config
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            self.config, self._ll_num_max_token_per_rank
        )
        wrapper = DeepEPWrapper.get_instance(deepep_config)
        assert (
            wrapper.mode == DeepEPMode.LOW_LATENCY
        ), "DeepEP mode should be LOW_LATENCY"
        self._buffer = wrapper.buffer
        self._num_topk = wrapper.num_topk
        self._num_max_dispatch_tokens_per_rank = wrapper.ll_num_max_token_per_rank
        self._use_fp8_dispatch = use_fp8_dispatch
        self._zero_copy = False
        self._async_finish = False
        self._return_recv_hook = False
        self._opt_level = int(os.environ.get("ACCL_LOW_LATENCY_OPTIMIZE", 1))
        self._handle: Optional[Tuple[Any, ...]] = None
        self._use_accl_ep = wrapper.use_accl_ep

        # Log initialization info
        print("=" * 80)
        print("DeepEpLowLatencyRouter Initialization")
        print("=" * 80)
        print(f"Model Config:")
        print(f"  expert_num: {self._num_experts}")
        print(f"  num_topk: {self._num_topk}")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  tp_size: {config.tp_size}")
        print(f"  tp_rank: {config.tp_rank}")
        print(f"  ep_size: {config.ep_size}")
        print(f"  ep_rank: {config.ep_rank}")
        print(f"  dp_size: {config.dp_size}")
        print(f"  dp_rank: {config.dp_rank}")
        print(f"  max_generate_batch_size: {config.max_generate_batch_size}")
        print(f"Quantization Config:")
        print(f"  is_quantized: {quant_config.is_quantized}")
        print(f"  quant_dtype: {quant_config.quant_dtype}")
        print(f"  is_block_quantized: {quant_config.is_block_quantized}")
        print(f"  is_per_act_token: {quant_config.is_per_act_token}")
        print(
            f"  block_shape: {quant_config.block_shape if quant_config.block_shape else None}"
        )
        print(f"DeepEP Low Latency Config:")
        print(f"  use_fp8_dispatch: {self._use_fp8_dispatch}")
        print(f"  ll_num_max_token_per_rank: {self._ll_num_max_token_per_rank}")
        print(
            f"  num_max_dispatch_tokens_per_rank: {self._num_max_dispatch_tokens_per_rank}"
        )
        print(f"  zero_copy: {self._zero_copy}")
        print(f"  async_finish: {self._async_finish}")
        print(f"  return_recv_hook: {self._return_recv_hook}")
        print(f"  opt_level (ACCL_LOW_LATENCY_OPTIMIZE): {self._opt_level}")
        print(f"  use_accl_ep: {self._use_accl_ep}")
        print(f"DeepEP Wrapper:")
        print(f"  mode: {wrapper.mode}")
        print(f"  buffer: {type(self._buffer).__name__}")
        print("=" * 80)

    @property
    def handle(self) -> Optional[Tuple[Any, ...]]:
        return self._handle

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
        print("_prepare_pre_tp_slice called:")
        print(f"  a1.shape: {a1.shape}, dtype: {a1.dtype}")
        print(f"  topk_ids.shape: {topk_ids.shape}")
        print(f"  topk_weights.shape: {topk_weights.shape}")

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

        print(f"_prepare_pre_tp_slice returning:")
        print(f"  tp_dispatch_input.shape: {tp_dispatch_input.shape}")
        print(f"tp_dispatch_input: {tp_dispatch_input}")
        print(f"  tp_topk_ids.shape: {tp_topk_ids.shape}")
        print(f"tp_topk_ids: {tp_topk_ids}")
        print(f"  tp_topk_weights.shape: {tp_topk_weights.shape}")
        print(f"tp_topk_weights: {tp_topk_weights}")

        return tp_dispatch_input, tp_topk_ids, tp_topk_weights

    def _normal_prepare(
        self, dispatch_args: dict[str, Any], tp_topk_weights: torch.Tensor
    ):
        """Normal prepare for DeepEP Low-Latency.
        Args:
            dispatch_args (dict[str, Any]): Arguments for dispatching tokens to experts.
            tp_topk_weights (torch.Tensor): Topk weights tensor for this tp rank.
        """
        # Calculate expected_m
        tp_num_tokens = dispatch_args["x"].size(0)
        print(f"tp_num_tokens: {tp_num_tokens}")
        expected_m = max(
            1,
            int(
                tp_num_tokens
                * self.config.ep_size
                * self._num_topk
                // self._num_experts
            ),
        )
        print(
            f"expected_m: {expected_m}, tp_num_tokens: {tp_num_tokens}, ep_size: {self.config.ep_size}, _num_topk: {self._num_topk}, _num_experts:{self._num_experts}"
        )
        # Dispatch tokens
        print("Calling low_latency_dispatch...")
        expert_x, expert_num_tokens, self._handle, _, _ = (
            self._buffer.low_latency_dispatch(**dispatch_args)
        )
        print(f"After low_latency_dispatch:")
        print(f"  expert_num_tokens: {expert_num_tokens}")
        print(f"expert_num_tokens: {expert_num_tokens}")
        if self._use_fp8_dispatch:
            assert isinstance(expert_x, tuple), "expert_x should be a tuple"
            expert_x, expert_x_scale = expert_x[0], expert_x[1]
            print(f"  [FP8] expert_x.shape: {expert_x.shape}, dtype: {expert_x.dtype}")
            print(f"expert_x: {expert_x}")
            print(
                f"  [FP8] expert_x_scale.shape: {expert_x_scale.shape}, dtype: {expert_x_scale.dtype}"
            )
            print(f"expert_x_scale: {expert_x_scale}")
        else:
            assert isinstance(expert_x, torch.Tensor), "expert_x should be a tensor"
            expert_x = expert_x
            expert_x_scale = None
            print(f"  [BF16] expert_x.shape: {expert_x.shape}, dtype: {expert_x.dtype}")
            print(f"expert_x: {expert_x}")
            print(f"  expert_x_scale: None")
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
        print("prepare called:")
        print(f"  a1.shape: {a1.shape}, dtype: {a1.dtype}")
        print(f"a1: {a1}")
        print(f"  a1_scale: {a1_scale.shape if a1_scale is not None else None}")
        if a1_scale is not None:
            print(f"a1_scale: {a1_scale}")
        print(f"  a2_scale: {a2_scale.shape if a2_scale is not None else None}")
        if a2_scale is not None:
            print(f"a2_scale: {a2_scale}")
        print(
            f"  topk_weights.shape: {topk_weights.shape}, dtype: {topk_weights.dtype}"
        )
        print(f"topk_weights: {topk_weights}")
        print(f"  topk_ids.shape: {topk_ids.shape}, dtype: {topk_ids.dtype}")
        print(f"topk_ids: {topk_ids}")

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
        print(f"prepare dispatch_args before: {dispatch_args}")
        # Set quantization config for DeepEP low latency dispatch
        if self.quant_config.is_block_quantized and is_deep_gemm_e8m0_used():
            dispatch_args.update({"round_scale": True, "use_ue8m0": True})
        elif self.quant_config.is_per_act_token:
            dispatch_args.update({"pertoken_quant": True})
        print(f"prepare dispatch_args after: {dispatch_args}")
        # Normal prepare
        expert_payload = self._normal_prepare(dispatch_args, tp_topk_weights)

        print(
            f"prepare returning: expert_payload.expert_x.shape={expert_payload.expert_x.shape}"
        )
        return expert_payload

    def _normal_finalize(self, combine_args: dict[str, Any]) -> torch.Tensor:
        """Normal finalize for DeepEP Low-Latency.
        Args:
            combine_args (dict[str, Any]): Arguments for combining expert outputs.
        """
        print("_normal_finalize called:")
        print(f"  combine_args keys: {combine_args.keys()}")
        print(f"  combine_args['x'].shape: {combine_args['x'].shape}")

        # Normal finalize
        print("Calling low_latency_combine...")
        combined_x, _, _ = self._buffer.low_latency_combine(**combine_args)

        print(f"_normal_finalize after low_latency_combine:")
        print(f"  combined_x.shape: {combined_x.shape}, dtype: {combined_x.dtype}")
        print(f"combined_x (after combine): {combined_x}")

        return combined_x

    def _finalize_post_tp_gather(
        self, combined_x: torch.Tensor, extra_finalize_args: Optional[Dict[str, Any]]
    ) -> torch.Tensor:
        """Finalize post tp gather for DeepEP Low-Latency.
        Args:
            combined_x (torch.Tensor): Combined output from all tp ranks.
            extra_finalize_args (Optional[Dict[str, Any]]): Extra finalize arguments.
        """
        print("_finalize_post_tp_gather called:")
        print(f"  combined_x.shape: {combined_x.shape}, dtype: {combined_x.dtype}")
        print(f"  extra_finalize_args: {extra_finalize_args}")

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
            print(f"  Calling all_gather with tp_size={tp_size}")
            gatherd_output = all_gather(combined_x, group=Group.TP).reshape(
                tp_size * tp_token_size, -1
            )
            print(f"  After all_gather: gatherd_output.shape={gatherd_output.shape}")
            combined_x = gatherd_output[:original_num_tokens, :]
            print(
                f"  After slicing to original_num_tokens: combined_x.shape={combined_x.shape}"
            )

        print(
            f"_finalize_post_tp_gather returning: combined_x.shape={combined_x.shape}"
        )
        print(f"combined_x (after tp gather): {combined_x}")
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
        print("finalize called:")
        print(
            f"  payload.fused_expert_output.shape: {payload.fused_expert_output.shape}, dtype: {payload.fused_expert_output.dtype}"
        )
        print(f"fused_expert_output: {payload.fused_expert_output}")
        print(f"  topk_weights.shape: {topk_weights.shape}")
        print(f"topk_weights: {topk_weights}")
        print(f"  topk_ids.shape: {topk_ids.shape}")
        print(f"topk_ids: {topk_ids}")
        print(f"  apply_router_weight_on_input: {apply_router_weight_on_input}")

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
        if self._use_accl_ep:
            combine_args.update({"opt_level": self._opt_level})

        # Normal finalize
        combined_x = self._normal_finalize(combine_args)

        # Finalize post tp gather
        combined_x = self._finalize_post_tp_gather(combined_x, extra_finalize_args)
        # reset handle
        self._handle = None

        print(
            f"finalize returning: combined_x.shape={combined_x.shape}, dtype={combined_x.dtype}"
        )
        print(f"combined_x: {combined_x}")
        return combined_x

    def _calc_low_latency_max_token_per_rank(
        self,
        max_generate_batch_size: int,
        tp_size: int,
        quant_config: FusedMoEQuantConfig,
    ) -> int:
        ll_num_max_token_per_rank = (max_generate_batch_size + tp_size - 1) // tp_size
        # deepgemm masked with max_m < 64 get incorrect result, related: https://github.com/deepseek-ai/DeepGEMM/issues/268
        if not quant_config.is_quantized or quant_config.is_block_quantized:
            matched_tokens = [64, 128]
        elif quant_config.is_per_act_token:
            matched_tokens = [
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
            ]
        else:
            raise ValueError("Unsupported quantization config")
        if ll_num_max_token_per_rank > 128:
            ll_num_max_token_per_rank = ((ll_num_max_token_per_rank + 127) // 128) * 128
            return ll_num_max_token_per_rank
        for t in matched_tokens:
            if ll_num_max_token_per_rank <= t:
                ll_num_max_token_per_rank = t
                return ll_num_max_token_per_rank
        return 128
