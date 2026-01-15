from typing import Any, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPMode,
    DeepEPWrapper,
    DeepepWrapperConfig,
)
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


def _calc_low_latency_max_token_per_rank(
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


class AfdDataRouterAttn(FusedMoeDataRouter):
    """
    A data router for Mixture-of-Experts that utilizes deep_ep's m2n low-latency communication primitives.

    This router is used for attention service, it receives compute result from FFN ranks.
    """

    @classmethod
    def router_type(cls):
        return RouterType.AFD_ATTN_ROUTER

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if AfdDataRouterAttn can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.use_low_latency(config))
        checker.check(resolver.is_afd_enabled(config))
        checker.check(not resolver.is_afd_ffn_rank(config))
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)
        self.workspace: Optional[torch.Tensor] = None
        self.prefetch_output: Optional[torch.Tensor] = None
        self.cached_topk_ids: Optional[torch.Tensor] = None
        self.cached_topk_weights: Optional[torch.Tensor] = None
        self.config = config

        self._ll_num_max_token_per_rank = _calc_low_latency_max_token_per_rank(
            config.max_generate_batch_size, config.tp_size, quant_config
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            self.config, self._ll_num_max_token_per_rank
        )
        wrapper = DeepEPWrapper.get_instance(deepep_config)
        assert (
            wrapper.mode == DeepEPMode.LOW_LATENCY_M2N
        ), "DeepEP mode should be LOW_LATENCY_M2N"
        self.buffer = wrapper.buffer
        self.rank = config.parallelism_config.world_rank
        self.world_size = config.world_size
        self.num_attn_ranks = config.dp_size * config.tp_size
        self.num_experts = config.expert_num
        self.comm_handle = None
        self.num_max_dispatch_tokens_per_rank = wrapper.ll_num_max_token_per_rank
        self.last_send_event: Optional[DeepEPEventOverlap] = None

    def fetch(self) -> torch.Tensor:
        assert self.cached_topk_ids is not None
        assert self.cached_topk_weights is not None
        assert self.comm_handle is not None

        if self.last_send_event is not None:
            self.last_send_event.current_stream_wait()
            self.last_send_event = None

        num_tokens, num_topk = self.cached_topk_ids.size()

        _, event, _ = self.buffer.low_latency_combine_recv(
            self.cached_topk_ids,
            self.cached_topk_weights,
            self.comm_handle,
            self.num_attn_ranks,
            num_tokens,
            num_topk,
            zero_copy=False,
            async_finish=True,
            return_recv_hook=False,
            out=self.workspace,
        )
        event.current_stream_wait()
        self.comm_handle = None
        self.cached_topk_ids = None
        self.cached_topk_weights = None

        output = self.workspace

        assert isinstance(output, torch.Tensor)

        return output

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        assert a1.dim() == 2 and topk_ids.dim() == 2
        assert a1.size(0) == topk_ids.size(0)

        self.workspace = torch.zeros_like(a1)

        if self.comm_handle is not None:
            self.prefetch_output = self.fetch()

        num_tokens, hidden_dim = a1.size()
        num_topk = topk_ids.size(1)

        assert num_tokens <= self.num_max_dispatch_tokens_per_rank
        assert self.rank < self.num_attn_ranks, ""

        _, _, handle, event, _ = self.buffer.low_latency_dispatch_send(
            a1,
            topk_ids,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            self.num_attn_ranks,
            num_topk,
            use_fp8=False,
            async_finish=True,
            return_recv_hook=False,
        )
        self.last_send_event = event
        # event.current_stream_wait()
        self.comm_handle = handle
        self.cached_topk_ids = topk_ids
        self.cached_topk_weights = topk_weights

    def finalize(
        self,
        payload: CombineForwardPayload = None,
        topk_weights: torch.Tensor = None,
        topk_ids: torch.Tensor = None,
        apply_router_weight_on_input: bool = True,
        extra_finalize_args: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        if self.prefetch_output is not None:
            output = self.prefetch_output
            self.prefetch_output = None
            return output
        return self.fetch()


class AfdDataRouterFfn(FusedMoeDataRouter):
    """
    A data router for Mixture-of-Experts that utilizes deep_ep's m2n low-latency communication primitives.

    This router is used for FFN service, it receives tokens from attention ranks and computes the FFN output.
    """

    @classmethod
    def router_type(cls):
        return RouterType.AFD_FFN_ROUTER

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if AfdDataRouterFfn can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.use_low_latency(config))
        checker.check(resolver.is_afd_enabled(config))
        checker.check(resolver.is_afd_ffn_rank(config))
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig = None,
    ):
        super().__init__(config, quant_config)
        self.config = config
        self._ll_num_max_token_per_rank = _calc_low_latency_max_token_per_rank(
            config.max_generate_batch_size, config.tp_size, quant_config
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            self.config, self._ll_num_max_token_per_rank
        )
        wrapper = DeepEPWrapper.get_instance(deepep_config)
        assert (
            wrapper.mode == DeepEPMode.LOW_LATENCY_M2N
        ), "DeepEP mode should be LOW_LATENCY_M2N"

        self.buffer = wrapper.buffer
        self.rank = config.parallelism_config.world_rank
        self.world_size = config.world_size

        self.num_attn_ranks = config.dp_size * config.tp_size
        self.num_experts = config.expert_num
        self.comm_handle = None
        self.num_max_dispatch_tokens_per_rank = wrapper.ll_num_max_token_per_rank
        self.last_send_event: Optional[deep_ep.EventOverlap] = None

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        assert a1.dim() == 2 and topk_ids.dim() == 2
        assert a1.size(0) == topk_ids.size(0)
        assert (
            self.comm_handle is None
        ), "Communication handle should be clean before prepare()."

        num_tokens, hidden_dim = a1.size()
        num_topk = topk_ids.size(1)

        assert num_tokens <= self.num_max_dispatch_tokens_per_rank

        assert self.rank >= self.num_attn_ranks

        if self.last_send_event is not None:
            self.last_send_event.current_stream_wait()
            self.last_send_event = None

        packed_recv_x, packed_recv_count, handle, event, _ = (
            self.buffer.low_latency_dispatch_recv(
                hidden_dim,
                num_topk,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                self.num_attn_ranks,
                use_fp8=False,
                async_finish=True,
                return_recv_hook=False,
            )
        )
        event.current_stream_wait()
        assert isinstance(packed_recv_x, torch.Tensor)

        # N ranks wait for received data to build the payload for local experts.
        expert_tokens_meta = ExpertTokensMetadata(
            expert_num_tokens=packed_recv_count, expert_num_tokens_cpu=None
        )
        assert isinstance(packed_recv_x, torch.Tensor)
        payload = ExpertForwardPayload(
            expert_x=packed_recv_x,
            expert_x_scale=None,
            expert_tokens_meta=expert_tokens_meta,
        )

        self.comm_handle = handle

        return payload

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        assert (
            self.comm_handle is not None
        ), "Communication handle is missing for finalize()."

        handle = self.comm_handle

        _, num_topk = topk_ids.size()

        _, event, _ = self.buffer.low_latency_combine_send(
            payload.fused_expert_output,
            handle,
            num_topk,
            zero_copy=False,
            async_finish=True,
            return_recv_hook=False,
        )
        self.last_send_event = event
        self.comm_handle = None

        _, hidden_dim = extra_finalize_args["a1_shape"]
        return torch.empty(0, hidden_dim)
