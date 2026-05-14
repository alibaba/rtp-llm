import logging
from typing import Any, Dict, List, Optional

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
        self._dispatch_ids: Optional[torch.Tensor] = None
        self._is_chunked = False
        self._chunk_dispatch_ids: Optional[List[torch.Tensor]] = None
        self._chunk_recv_sizes: Optional[List[int]] = None
        self._chunk_input_sizes: Optional[List[int]] = None

    @property
    def max_inp_tokens(self) -> int:
        return self.mori_buffer_wrapper.config.max_num_inp_token_per_rank

    def _remap_to_local_ids(
        self,
        dispatch_ids: torch.Tensor,
        dispatch_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remap global expert IDs to local 0-based indices for this EP rank.

        Non-local experts are mapped to local index 0 (safe fallback) so the
        fused kernel computes valid output that is zeroed by weight=0.
        """
        from rtp_llm.models_py.triton_kernels.moe.remap_local_ids_kernel import (
            remap_to_local_ids,
        )

        local_start = self.ep_rank * self.expert_num_per_rank
        local_end = local_start + self.expert_num_per_rank
        return remap_to_local_ids(dispatch_ids, dispatch_weights, local_start, local_end)

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        logging.info(
            f"[MoriEpIntranodeRouter] prepare called, tokens={a1.shape[0]}, ep_rank={self.ep_rank}"
        )
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("MoriEpIntranode a1_scale or a2_scale should be None")

        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        max_tokens = self.max_inp_tokens
        num_tokens = a1.shape[0]

        if num_tokens <= max_tokens:
            return self._prepare_single(a1, topk_weights, topk_ids)
        return self._prepare_chunked(a1, topk_weights, topk_ids, max_tokens)

    def _prepare_single(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        self._is_chunked = False

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_buffer_wrapper.op.dispatch(a1, topk_weights, None, topk_ids)

        # Cache global dispatch_ids for use in finalize()'s combine call.
        self._dispatch_ids = dispatch_ids

        local_ids, local_weights = self._remap_to_local_ids(
            dispatch_ids, dispatch_weights
        )

        return ExpertForwardPayload(
            expert_x=dispatch_a1,
            expert_x_scale=dispatch_scale,
            expert_x_origin_dtype=None,
            expert_topk_ids=local_ids,
            expert_topk_weights=local_weights,
            expert_ids_are_local=True,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=None,
                expert_num_tokens_cpu=dispatch_recv_token_num,
            ),
        )

    def _prepare_chunked(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        chunk_size: int,
    ) -> ExpertForwardPayload:
        self._is_chunked = True
        self._chunk_dispatch_ids = []
        self._chunk_recv_sizes = []
        self._chunk_input_sizes = []

        all_dispatch_a1 = []
        all_dispatch_scale = []
        all_local_ids = []
        all_local_weights = []
        total_recv_token_nums = None

        num_tokens = a1.shape[0]

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)

            (
                dispatch_a1,
                dispatch_weights,
                dispatch_scale,
                dispatch_ids,
                dispatch_recv_token_num,
            ) = self.mori_buffer_wrapper.op.dispatch(
                a1[start:end], topk_weights[start:end], None, topk_ids[start:end]
            )

            self._chunk_dispatch_ids.append(dispatch_ids)
            self._chunk_recv_sizes.append(dispatch_a1.shape[0])
            self._chunk_input_sizes.append(end - start)

            all_dispatch_a1.append(dispatch_a1)
            if dispatch_scale is not None:
                all_dispatch_scale.append(dispatch_scale)

            local_ids, local_weights = self._remap_to_local_ids(
                dispatch_ids, dispatch_weights
            )

            all_local_ids.append(local_ids)
            all_local_weights.append(local_weights)

            if isinstance(dispatch_recv_token_num, torch.Tensor):
                recv_list = dispatch_recv_token_num.tolist()
            else:
                recv_list = list(dispatch_recv_token_num)
            if total_recv_token_nums is None:
                total_recv_token_nums = recv_list
            else:
                for i in range(len(total_recv_token_nums)):
                    total_recv_token_nums[i] += recv_list[i]

        return ExpertForwardPayload(
            expert_x=torch.cat(all_dispatch_a1, dim=0),
            expert_x_scale=(
                torch.cat(all_dispatch_scale, dim=0) if all_dispatch_scale else None
            ),
            expert_x_origin_dtype=None,
            expert_topk_ids=torch.cat(all_local_ids, dim=0),
            expert_topk_weights=torch.cat(all_local_weights, dim=0),
            expert_ids_are_local=True,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=None,
                expert_num_tokens_cpu=total_recv_token_nums,
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
        logging.info(
            f"[MoriEpIntranodeRouter] finalize called, ep_rank={self.ep_rank}, chunked={self._is_chunked}"
        )
        fused_out = payload.fused_expert_output

        if self._is_chunked:
            return self._finalize_chunked(fused_out, extra_finalize_args)
        return self._finalize_single(fused_out, extra_finalize_args)

    def _finalize_single(
        self,
        fused_out: torch.Tensor,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        # Use the cached global dispatch_ids for combine, not the local_ids
        # that were set as expert_topk_ids for the fused_moe kernel.
        global_dispatch_ids = self._dispatch_ids
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

    def _finalize_chunked(
        self,
        fused_out: torch.Tensor,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        results = []
        offset = 0

        for chunk_ids, recv_size, input_size in zip(
            self._chunk_dispatch_ids,
            self._chunk_recv_sizes,
            self._chunk_input_sizes,
        ):
            if chunk_ids.dtype != torch.int32:
                chunk_ids = chunk_ids.to(torch.int32)

            chunk_out = fused_out[offset : offset + recv_size]
            recv_x = self.mori_buffer_wrapper.op.combine(chunk_out, None, chunk_ids)[0]

            if recv_x.shape[0] > input_size:
                recv_x = recv_x[:input_size]

            results.append(recv_x)
            offset += recv_size

        self._chunk_dispatch_ids = None
        self._chunk_recv_sizes = None
        self._chunk_input_sizes = None

        return torch.cat(results, dim=0)
