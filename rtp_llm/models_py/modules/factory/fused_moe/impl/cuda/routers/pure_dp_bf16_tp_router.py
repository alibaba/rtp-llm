"""Eager BF16 TP-input extension of the existing all-gather PureDP router.

Keep PureDP's FP8/TP1 profile unchanged. This explicitly requested backend
scatters replicated attention-TP tokens once and retains global physical IDs
for TritonFusedMoeExecutor, which owns the global-to-local expert conversion.
"""
import os

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload, ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_dp_router import PureDpRouterBase
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import MoeConfigResolver


class PureDpRouterBf16Tp(PureDpRouterBase):
    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(config.moe_strategy == "nccl_bf16_reference")
        checker.check(not MoeConfigResolver().has_quantization(config))
        checker.check(config.parallelism_config.get_ffn_tp_size() == 1)
        checker.check(config.ep_size == config.world_size and config.ep_size > 1)
        checker.check(config.world_size == config.tp_size * config.dp_size)
        checker.check(config.ep_rank == config.world_rank)
        checker.check(config.tp_rank == config.world_rank % config.tp_size)
        checker.check(not config.enable_cuda_graph)

    def __init__(self, config, quant_config):
        super().__init__(config, quant_config)
        self.max_padded_tokens = int(os.environ.get("RTP_NCCL_EP_REFERENCE_MAX_PADDED_TOKENS", "8192"))
        if self.max_padded_tokens < 1 or config.expert_num % config.ep_size:
            raise ValueError("Invalid reference EP token bound or physical expert layout")

    def _do_quant(self, hidden_states):
        return hidden_states, None

    def prepare(self, a1, a1_scale, a2_scale, topk_weights, topk_ids):
        if a1.dtype != torch.bfloat16 or a1_scale is not None or a2_scale is not None:
            raise ValueError("Reference EP requires BF16 unquantized hidden states")
        if a1.is_cuda and torch.cuda.is_current_stream_capturing():
            raise ValueError("Reference EP requires eager execution")
        original_tokens = a1.size(0)
        span = (original_tokens + self.config.tp_size - 1) // self.config.tp_size
        begin = min(span * self.config.tp_rank, original_tokens)
        size = min(span, original_tokens - begin)
        # Attention input is replicated over TP. Avoid dispatching each token
        # TP times when using whole-world expert ownership.
        a1 = a1.narrow(0, begin, size)
        topk_weights = topk_weights.narrow(0, begin, size)
        topk_ids = topk_ids.narrow(0, begin, size)
        a1, topk_weights, topk_ids, max_n = self._pad_to_max(a1, topk_weights, topk_ids)
        total = max_n * self.config.world_size
        if total > self.max_padded_tokens:
            # All ranks saw identical lengths, so this rejects together before
            # the payload collectives and bounds each gather's live allocation.
            raise ValueError(f"Reference EP padded token bound exceeded: {total} > {self.max_padded_tokens}")
        if max_n:
            a1 = all_gather(a1, group=Group.DP_AND_TP)
            topk_weights = all_gather(topk_weights, group=Group.DP_AND_TP)
            topk_ids = all_gather(topk_ids, group=Group.DP_AND_TP)
        # Do not call the FP8 base.prepare() local-ID recomputation. The BF16
        # executor masks and subtracts its physical expert start ID itself.
        return ExpertForwardPayload(
            expert_x=a1, expert_x_origin_dtype=a1.dtype,
            expert_topk_ids=topk_ids, expert_topk_weights=topk_weights,
        )

    def finalize(self, payload, topk_weights, topk_ids, apply_router_weight_on_input, extra_finalize_args):
        if apply_router_weight_on_input or extra_finalize_args is None:
            raise ValueError("Reference EP requires output-side route weights and original token metadata")
        original = int(extra_finalize_args["original_num_tokens"])
        span = (original + self.config.tp_size - 1) // self.config.tp_size
        begin = min(span * self.config.tp_rank, original)
        size = min(span, original - begin)
        dtype = payload.fused_expert_output.dtype
        if payload.fused_expert_output.size(0):
            # The base's existing whole-world reduce_scatter delivers the
            # appropriate source-rank slice, now accumulating in FP32.
            work = CombineForwardPayload(fused_expert_output=payload.fused_expert_output.float())
            output = super().finalize(work, topk_weights, topk_ids, False, {"original_num_tokens": size}).to(dtype)
        else:
            output = payload.fused_expert_output
        if self.config.tp_size == 1 or span == 0:
            return output
        if output.size(0) < span:
            output = torch.nn.functional.pad(output, (0, 0, 0, span - output.size(0)))
        return all_gather(output, group=Group.TP)[:original]
