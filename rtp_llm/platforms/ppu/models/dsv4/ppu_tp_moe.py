"""PPU TP4 MoE: local shared/routed compute followed by one BF16 reduction."""

import logging
from functools import partial

import torch
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from .ppu_gate import PpuGate as Gate
from .ppu_moe_config import PpuMoeConfig as MoeCfg
from rtp_llm.utils.model_weight import W
from torch import nn

from ...kernels.ppu_moe_combine import combine_tp_partials
from .ppu_grouped_fp4 import PpuGroupedFP4Strategy
from .ppu_tp_shared_expert import PpuTPSharedExpert


class PpuTPMoE(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        moe_inter_dim,
        n_routed_experts,
        n_activated_experts,
        n_shared_experts,
        score_func,
        route_scale,
        swiglu_limit,
        n_hash_layers,
        vocab_size,
        layer_weights,
        tp_size,
        ep_size,
        ep_rank,
        max_tokens_per_rank,
        is_decode_role,
        *,
        tp_rank,
        platform_provider,
        world_size=None,
        world_rank=None,
        strategy=None,
        n_physical_experts=None,
        observer_factory=None,
        record_function_scope=None,
    ):
        super().__init__()
        if tp_size != 4 or ep_size != 1 or ep_rank != 0 or is_decode_role:
            raise ValueError("PPU TP MoE requires eager Prefill TP4/EP1")
        if n_shared_experts != 1 or int(max_tokens_per_rank) <= 0:
            raise ValueError(
                "PPU TP MoE requires one shared expert and positive capacity"
            )
        if platform_provider is None:
            raise ValueError("PPU TP MoE requires an instance-owned operator adapter")
        if n_physical_experts not in (None, n_routed_experts):
            raise ValueError("PPU TP MoE does not support redundant experts")
        self.layer_id, self.dim = layer_id, dim
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.route_scale = float(route_scale)
        self._platform_provider = platform_provider
        self.gate = Gate(
            layer_id,
            dim,
            n_routed_experts,
            n_activated_experts,
            score_func,
            route_scale,
            n_hash_layers,
            vocab_size,
            layer_weights=layer_weights,
            platform_provider=platform_provider,
        )
        cfg = MoeCfg(
            layer_id=layer_id,
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            swiglu_limit=swiglu_limit,
            ep_size=ep_size,
            ep_rank=ep_rank,
            n_local_experts=n_routed_experts,
            local_expert_start=0,
            local_expert_end=n_routed_experts,
            max_tokens_per_rank=self.max_tokens_per_rank,
            tp_size=tp_size,
        )
        self._strategy = PpuGroupedFP4Strategy(
            cfg,
            sglang_moe=True,
            fused_gather=True,
            fused_scale_gather=platform_provider._bool(
                "DSV4_MOE_SCALE_GATHER_FUSED", True
            ),
        )
        self._strategy.setup_weights(layer_weights)
        self.strategy_name = self._strategy.name
        if self._strategy.routed_tp_size != tp_size:
            raise ValueError(
                "Shared/routed BF16 reduction requires TP-sharded routed weights"
            )
        self.shared_experts = PpuTPSharedExpert(
            dim,
            moe_inter_dim,
            {
                "w13_w": layer_weights[W.v4_shared_w13_w],
                "w13_s": layer_weights[W.v4_shared_w13_s],
                "w2_w": layer_weights[W.v4_shared_w2_w],
                "w2_s": layer_weights[W.v4_shared_w2_s],
            },
            tp_size=tp_size,
            tp_rank=tp_rank,
            swiglu_limit=swiglu_limit,
            platform_provider=platform_provider,
        )
        self._reduce = partial(all_reduce, group=Group.TP, inplace=True)
        logging.info(
            "DSV4_PPU_TP_MOE layer=%d rank=%d shared_w13=%s shared_w2=%s "
            "collective=bf16_shared_plus_routed.v2",
            layer_id,
            tp_rank,
            tuple(self.shared_experts.w13.weight.shape),
            tuple(self.shared_experts.w2.weight.shape),
        )

    def _forward_local_chunk(self, x, input_ids):
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as recorder

        debug = recorder.should_record_layer(self.layer_id)
        prefix = f"L{self.layer_id:02d}_moe"
        if debug:
            recorder.record_if_level(2, prefix + "_x_in", x)
        with record_function_range("dsv4.moe.gate"):
            weights, indices = self.gate(x, input_ids, include_route_scale=False)
        if debug:
            recorder.record_if_level(2, prefix + "_topk_weights", weights)
            recorder.record_if_level(2, prefix + "_topk_indices", indices)
        with record_function_range("dsv4.moe.shared_expert_start"):
            shared = self.shared_experts(x)
        with record_function_range("dsv4.moe.routed_experts"):
            routed = self._strategy(x, weights, indices)
        with record_function_range("dsv4.moe.add_shared"):
            local = combine_tp_partials(routed, shared, self.route_scale)
        if debug:
            recorder.record_if_level(2, prefix + "_tp_local_bf16", local)
        with record_function_range("dsv4.moe.tp_all_reduce_bf16"):
            result = self._reduce(local)
        if debug:
            recorder.record_if_level(2, prefix + "_tp_reduced_bf16", result)
        return result

    def forward(self, x, input_ids, *, is_decode_forward=False, positions=None):
        if is_decode_forward:
            raise RuntimeError("PPU TP MoE v2 supports eager Prefill only")
        if x.dtype != torch.bfloat16 or x.shape[-1] != self.dim:
            raise ValueError(
                "PPU TP MoE requires BF16 input with the configured hidden size"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PPU TP MoE v2 supports eager Prefill only")
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        ids = input_ids.flatten()
        if ids.numel() != flat.shape[0]:
            raise ValueError("MoE input IDs and hidden token counts differ")
        if not flat.shape[0]:
            return x
        if flat.shape[0] <= self.max_tokens_per_rank:
            return self._forward_local_chunk(flat, ids).view(shape)
        # One per-call result; never reuse another layer/model's global buffer.
        out = torch.empty_like(flat)
        for start in range(0, flat.shape[0], self.max_tokens_per_rank):
            end = min(start + self.max_tokens_per_rank, flat.shape[0])
            out[start:end].copy_(
                self._forward_local_chunk(flat[start:end], ids[start:end])
            )
        return out.view(shape)
