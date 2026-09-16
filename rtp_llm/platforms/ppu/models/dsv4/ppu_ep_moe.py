"""PPU EP MoE composition with instance-owned shared-expert scheduling.

The engine owns DeepEP communication. This module binds the selected PPU
routed implementation and preserves BF16 routed/shared outputs through the
FP32 add epilogue. It does not participate in CUDA's FP8/FP4 strategy registry.
"""

from contextlib import nullcontext

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv4.chunk_env import chunked_moe_enabled
from rtp_llm.models_py.triton_kernels.moe.shared_expert import (
    fused_moe_epilogue as combine_routed_and_shared,
)
from rtp_llm.utils.model_weight import W

from .ppu_gate import PpuGate
from .ppu_moe_config import PpuMoeConfig


class PpuEPMoE(nn.Module):
    def __init__(
        self,
        *,
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
        tp_size=1,
        tp_rank=0,
        ep_size=8,
        ep_rank=0,
        world_size=None,
        world_rank=None,
        max_tokens_per_rank=8192,
        is_decode_role=False,
        strategy=None,
        n_physical_experts=None,
        platform_provider,
        strategy_type,
        strategy_kwargs=None,
        observer_factory=None,
        record_function_scope=nullcontext,
    ):
        super().__init__()
        if tp_size != 1 or tp_rank != 0 or ep_size != 8 or n_shared_experts != 1:
            raise ValueError("PPU EP MoE requires TP1/EP8 and one shared expert")
        if n_physical_experts not in (None, n_routed_experts):
            raise ValueError("PPU EP MoE does not support redundant experts")
        if (
            max_tokens_per_rank <= 0
            or n_routed_experts % ep_size
            or not 0 <= ep_rank < ep_size
        ):
            raise ValueError("Invalid PPU EP capacity or expert ownership")
        self.layer_id, self.dim = int(layer_id), int(dim)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self._is_decode_role = bool(is_decode_role)
        self.chunking_enabled = chunked_moe_enabled(platform_provider.execution_options)
        self._observer_factory = observer_factory
        self._record_function_scope = record_function_scope
        self.gate = PpuGate(
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
        local_experts = n_routed_experts // ep_size
        cfg = PpuMoeConfig(
            layer_id,
            dim,
            moe_inter_dim,
            n_routed_experts,
            n_activated_experts,
            swiglu_limit,
            ep_size,
            ep_rank,
            local_experts,
            ep_rank * local_experts,
            (ep_rank + 1) * local_experts,
            self.max_tokens_per_rank,
            tp_size,
        )
        self._strategy = strategy_type(cfg, **(strategy_kwargs or {}))
        self._strategy.setup_weights(layer_weights)
        self.strategy_name = self._strategy.name
        self.shared_experts = platform_provider.build_shared_expert(
            dim,
            moe_inter_dim,
            expert_weights={
                "w13_w": layer_weights[W.v4_shared_w13_w],
                "w13_s": layer_weights[W.v4_shared_w13_s],
                "w2_w": layer_weights[W.v4_shared_w2_w],
                "w2_s": layer_weights[W.v4_shared_w2_s],
            },
            swiglu_limit=swiglu_limit,
        )
        self._shared_executor = platform_provider.build_shared_expert_executor()
        self._shared_executor.prepare(self.shared_experts)

    def _route_and_start_shared(self, x, input_ids):
        early = self._shared_executor.start_before_routing
        if early:
            self._shared_executor.start(self.shared_experts, x)
        try:
            routing = self.gate(x, input_ids)
        except Exception:
            if early:
                self._shared_executor.finish()
            raise
        if not early:
            self._shared_executor.start(self.shared_experts, x)
        return routing

    def _run_chunk(self, x, input_ids, out, *, observer=None):
        weights, indices = self._route_and_start_shared(x, input_ids)
        try:
            if observer is not None:
                observer("input", x)
                observer("topk_weights", weights)
                observer("topk_indices", indices)
            routed = self._strategy(x, weights, indices)
        except Exception:
            self._shared_executor.finish()
            raise
        shared = self._shared_executor.finish()
        if observer is not None:
            observer("routed_y", routed)
            observer("shared_y", shared)
        combine_routed_and_shared(routed, shared, x.dtype, out=out)
        if observer is not None:
            observer("final_y", out)

    def forward(self, x, input_ids, *, is_decode_forward=False, positions=None):
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        ids = input_ids.reshape(-1)
        if ids.numel() != flat.size(0):
            raise ValueError("MoE input_ids/token mismatch")
        capacity = self.max_tokens_per_rank
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if flat.size(0) > capacity and (
            self._is_decode_role or is_decode_forward or capturing
        ):
            raise ValueError("Decode or graph MoE input exceeds max_tokens_per_rank")
        if flat.size(0) > capacity and not self.chunking_enabled:
            raise ValueError("PPU EP MoE input exceeds capacity with chunking disabled")
        # Each call owns its output, including graph-pool allocations. Retaining
        # a result from one model/layer cannot alias another model's workspace.
        out = torch.empty_like(flat)
        for start in range(0, max(flat.size(0), 1), capacity):
            end = min(start + capacity, flat.size(0))
            observer = (
                self._observer_factory(
                    positions[start:end] if positions is not None else None
                )
                if self._observer_factory
                else None
            )
            with self._record_function_scope():
                self._run_chunk(
                    flat[start:end], ids[start:end], out[start:end], observer=observer
                )
        return out.view(shape)
