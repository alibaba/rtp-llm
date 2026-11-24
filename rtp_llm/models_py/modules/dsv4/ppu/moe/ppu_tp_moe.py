"""PPU TP MoE: local shared/routed compute followed by one BF16 reduction."""

import logging
from dataclasses import replace
from functools import partial

import torch
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.kernels.ppu.moe.ppu_moe_combine import (
    add_pair_to_bf16,
    combine_tp_partials,
)
from rtp_llm.models_py.kernels.ppu.moe.ppu_shared_swiglu import silu_mul_merged_bf16
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.ppu.moe.gate import PpuGate as Gate
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_grouped_fp4 import (
    PpuGroupedW4A8Strategy,
)
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_mega_moe import (
    PpuTPMegaMoEW4A8Resources,
    PpuTPMegaMoEW4A8Strategy,
)
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_moe_config import PpuMoeConfig as MoeCfg
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_shared_expert import (
    PpuTPMegaMoEFP8SharedExpert,
    PpuTPMegaMoEFP8SharedResources,
    PpuTPSharedExpert,
)
from rtp_llm.utils.model_weight import W
from torch import nn


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
        if is_decode_role:
            raise ValueError("PPU TP MoE requires eager Prefill resources")
        if tp_size <= 0 or ep_size <= 0 or not 0 <= ep_rank < ep_size:
            raise ValueError("Invalid PPU TP/EP size or expert rank")
        if n_shared_experts != 1 or int(max_tokens_per_rank) <= 0:
            raise ValueError(
                "PPU TP MoE requires one shared expert and positive capacity"
            )
        if platform_provider is None:
            raise ValueError("PPU TP MoE requires an instance-owned operator adapter")
        if n_physical_experts not in (None, n_routed_experts):
            raise ValueError("PPU TP MoE does not support redundant experts")
        self.layer_id, self.dim = layer_id, dim
        self._tp_rank, self._tp_size = int(tp_rank), int(tp_size)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.route_scale = float(route_scale)
        self._fused_gather_shared = platform_provider._bool(
            "DSV4_MOE_GATHER_SHARED_FUSED", True
        )
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
        if platform_provider._bool("DSV4_PPU_PREFILL_MEGAMOE_W4A8", False):
            resources = getattr(
                platform_provider, "_prefill_megamoe_w4a8_resources", None
            )
            if resources is None:
                resources = PpuTPMegaMoEW4A8Resources()
                platform_provider._prefill_megamoe_w4a8_resources = resources
            self._strategy = PpuTPMegaMoEW4A8Strategy(cfg, resources=resources)
        else:
            self._strategy = PpuGroupedW4A8Strategy(
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
        shared_weights = {
            "w13_w": layer_weights[W.v4_shared_w13_w],
            "w13_s": layer_weights[W.v4_shared_w13_s],
            "w2_w": layer_weights[W.v4_shared_w2_w],
            "w2_s": layer_weights[W.v4_shared_w2_s],
        }
        if platform_provider._bool("DSV4_PPU_PREFILL_MEGAMOE_W4A8", False):
            shared_resources = getattr(
                platform_provider, "_prefill_megamoe_fp8_shared_resources", None
            )
            if shared_resources is None:
                shared_resources = PpuTPMegaMoEFP8SharedResources()
                platform_provider._prefill_megamoe_fp8_shared_resources = (
                    shared_resources
                )
            self.shared_experts = PpuTPMegaMoEFP8SharedExpert(
                replace(cfg, moe_inter_dim=moe_inter_dim // tp_size),
                shared_weights,
                swiglu_limit=swiglu_limit,
                module=self._strategy._mega,
                resources=shared_resources,
                scale_is_prepared=True,
            )
        else:
            self.shared_experts = PpuTPSharedExpert(
                dim,
                moe_inter_dim,
                shared_weights,
                tp_size=tp_size,
                tp_rank=tp_rank,
                swiglu_limit=swiglu_limit,
                platform_provider=platform_provider,
            )
        if isinstance(self._strategy, PpuTPMegaMoEW4A8Strategy):
            for key in (
                W.v4_routed_w1_w,
                W.v4_routed_w1_s,
                W.v4_routed_w2_w,
                W.v4_routed_w2_s,
                W.v4_routed_w3_w,
                W.v4_routed_w3_s,
            ):
                layer_weights.pop(key)
        self._fp32_reduce = isinstance(self.shared_experts, PpuTPMegaMoEFP8SharedExpert)
        self._reduce = (
            partial(all_reduce, group=Group.TP, inplace=True) if tp_size > 1 else None
        )
        self._shared_phases = (
            self.strategy_name == PpuGroupedW4A8Strategy.name
            and platform_provider._bool("DSV4_PPU_PREFILL_MOE_SHARED_PHASES", True)
            and tp_size == 4
            and ep_size == 1
            and dim == 4096
            and moe_inter_dim == 2048
            and n_routed_experts == 256
            and n_activated_experts == 8
            and self._fused_gather_shared
        )
        self._shared_stream = None
        logging.info(
            "DSV4_PPU_TP_MOE layer=%d rank=%d shared_w13=%s shared_w2=%s "
            "shared_operator=%s collective=%s",
            layer_id,
            tp_rank,
            tuple(shared_weights["w13_w"].shape),
            tuple(shared_weights["w2_w"].shape),
            type(self.shared_experts).__name__,
            (
                "fp32_shared_plus_routed_then_bf16.v1"
                if self._fp32_reduce
                else "bf16_shared_plus_routed.v2"
            ),
        )

    def _forward_local_chunk(self, x, input_ids):
        token_count = x.shape[0]
        with record_function_range("dsv4.moe.gate"):
            weights, indices = self.gate(x, input_ids, include_route_scale=False)
        if (
            self._shared_phases
            and x.is_cuda
            and 1024 <= x.shape[0] <= 65536
            and not getattr(self.shared_experts, "fuse_activation_quant", False)
        ):
            local = self._forward_phased(x, weights, indices)
        else:
            local = self._forward_serial(x, weights, indices)
        with record_function_range("dsv4.moe.tp_all_reduce"):
            result = self._reduce(local) if self._reduce is not None else local
            if self._fp32_reduce:
                result = add_pair_to_bf16(result[0], result[1])
        return result

    def _forward_serial(self, x, weights, indices):
        with record_function_range("dsv4.moe.shared_expert_start"):
            shared = self.shared_experts(x)
        with record_function_range("dsv4.moe.routed_experts"):
            if self._fp32_reduce:
                routed = self._strategy(
                    x, weights, indices, route_scale=self.route_scale
                )
                return torch.stack((routed, shared))
            if self._fused_gather_shared:
                return self._strategy(
                    x, weights, indices, shared=shared, route_scale=self.route_scale
                )
            if getattr(self._strategy, "preweights_route_scale", False):
                routed = self._strategy(
                    x, weights, indices, route_scale=self.route_scale
                )
                combine_scale = 1.0
            else:
                routed = self._strategy(x, weights, indices)
                combine_scale = self.route_scale
        with record_function_range("dsv4.moe.add_shared"):
            return combine_tp_partials(routed, shared, combine_scale)

    def _forward_phased(self, x, weights, indices):
        from rtp_llm.models_py.kernels.ppu.runtime import PpuStreamPool

        if self._shared_stream is None:
            provider = self._platform_provider
            pool = getattr(provider, "_prefill_moe_stream_pool", None)
            if pool is None:
                pool = PpuStreamPool()
                provider._prefill_moe_stream_pool = pool
            self._shared_stream = pool.get("prefill_moe_shared", x.device)
        current = torch.cuda.current_stream(x.device)
        try:
            phases = SharedPhases(self.shared_experts, x, self._shared_stream)
            with record_function_range("dsv4.moe.routed_experts"):
                return self._strategy(
                    x,
                    weights,
                    indices,
                    shared=phases.output,
                    route_scale=self.route_scale,
                    shared_schedule=phases,
                )
        finally:
            # Drain auxiliary work even when a routed operator raises.
            current.wait_stream(self._shared_stream)

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


class SharedPhases:
    def __init__(self, shared, x, stream):
        assert not getattr(shared, "fuse_activation_quant", False)
        self.shared = shared
        self.main = torch.cuda.current_stream(x.device)
        self.stream = stream
        self.output = torch.empty_like(x)
        self.output.record_stream(stream)
        x.record_stream(stream)
        stream.wait_stream(self.main)
        with torch.cuda.stream(stream):
            self.gate_up = shared.w13(x)
            self.up_done = stream.record_event()

    def before_up(self):
        self.main.wait_event(self.up_done)
        with torch.cuda.stream(self.stream):
            hidden = silu_mul_merged_bf16(self.gate_up, self.shared.swiglu_limit)
            self.quantized = self.shared.w2.quantize_input(hidden)

    def after_up(self):
        self.stream.wait_stream(self.main)
        with torch.cuda.stream(self.stream):
            self.shared.w2.forward_quantized(*self.quantized, out=self.output)
            self.done = self.stream.record_event()

    def before_down(self):
        self.main.wait_event(self.done)
