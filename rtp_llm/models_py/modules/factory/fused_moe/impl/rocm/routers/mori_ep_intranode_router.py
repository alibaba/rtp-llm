import logging
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.distributed.moriep_wrapper import MoriEPWrapper
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


class MoriEpIntranodeRouter(FusedMoeDataRouter):
    @classmethod
    def router_type(cls) -> RouterType:
        return RouterType.MORI_EP_INTRANODE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.use_low_latency(config))
        checker.check(MoriEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.expert_num = config.expert_num
        self.expert_num_per_rank = self.expert_num // self.ep_size
        self.mori_buffer_wrapper = MoriEPWrapper.get_instance()

    @property
    def max_inp_tokens(self) -> int:
        return self.mori_buffer_wrapper.config.max_num_inp_token_per_rank

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        logging.debug(
            f"[MoriEpIntranodeRouter] prepare called, tokens={a1.shape[0]}, ep_rank={self.ep_rank}"
        )
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("MoriEpIntranode a1_scale or a2_scale should be None")

        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        if a1.shape[0] > self.max_inp_tokens:
            raise ValueError("Mori dispatch input exceeds configured capacity")
        return self._prepare_single(a1, topk_weights, topk_ids)

    def _prepare_single(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_buffer_wrapper.op.dispatch(a1, topk_weights, None, topk_ids)

        return ExpertForwardPayload(
            expert_x=dispatch_a1,
            expert_x_scale=dispatch_scale,
            expert_x_origin_dtype=None,
            expert_topk_ids=dispatch_ids,
            expert_topk_weights=dispatch_weights,
            expert_ids_are_local=False,
            valid_token_count=dispatch_recv_token_num,
            combine_indices=dispatch_ids,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=None,
                expert_num_tokens_cpu=None,
            ),
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        logging.debug(
            f"[MoriEpIntranodeRouter] finalize called, ep_rank={self.ep_rank}"
        )
        return self._finalize_single(
            payload.fused_expert_output,
            payload.combine_indices,
            extra_finalize_args,
        )

    def _finalize_single(
        self,
        fused_out: torch.Tensor,
        combine_indices: Optional[torch.Tensor],
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        # Combine uses the global dispatch expert ids carried with the payload; the
        # AIter expert kernel maps them to local experts via expert_mask internally.
        assert combine_indices is not None, "combine_indices missing for Mori finalize"
        global_dispatch_ids = combine_indices
        if global_dispatch_ids.dtype != torch.int32:
            global_dispatch_ids = global_dispatch_ids.to(torch.int32)

        recv_x = self.mori_buffer_wrapper.op.combine(
            fused_out, None, global_dispatch_ids
        )[0]

        if (
            extra_finalize_args is not None
            and "original_num_tokens" in extra_finalize_args
        ):
            original_num_tokens = extra_finalize_args["original_num_tokens"]
            if recv_x.shape[0] > original_num_tokens:
                recv_x = recv_x[:original_num_tokens]

        return recv_x
