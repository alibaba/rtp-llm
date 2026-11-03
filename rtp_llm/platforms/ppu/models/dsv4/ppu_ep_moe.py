"""PPU EP MoE composition with instance-owned shared-expert scheduling.

The engine owns DeepEP communication. This module binds the selected PPU
routed implementation and preserves BF16 routed/shared outputs through the
FP32 add epilogue. It does not participate in CUDA's FP8/FP4 strategy registry.

At TP>1 the hidden tokens entering MoE are replicated across the TP group.
Each rank dispatches only its contiguous token slice and all-gathers the
combined result, so every (token, expert) pair is computed exactly once and
the shared expert runs on one slice instead of every rank.
"""

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn

from rtp_llm.platforms.ppu.models.dsv4.impl.chunk_env import chunked_moe_enabled
from rtp_llm.models_py.triton_kernels.moe.shared_expert import (
    fused_moe_epilogue as combine_routed_and_shared,
)
from rtp_llm.utils.model_weight import W

from rtp_llm.platforms.ppu.models.dsv4.ppu_gate import PpuGate
from rtp_llm.platforms.ppu.models.dsv4.ppu_moe_config import PpuMoeConfig


def _gather_tp_slices(out: torch.Tensor, original_tokens: int, tp_size: int):
    """All-gather per-rank token slices back into the replicated layout."""

    from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

    tp_tokens = (original_tokens + tp_size - 1) // tp_size
    if out.size(0) < tp_tokens:
        padding = torch.zeros(
            (tp_tokens - out.size(0), out.size(1)),
            dtype=out.dtype,
            device=out.device,
        )
        out = torch.cat((out, padding), dim=0)
    gathered = all_gather(out, group=Group.TP)
    return gathered.reshape(tp_size * tp_tokens, out.size(1))[:original_tokens]


def _slice_tokens_for_tp(flat, ids, positions, active_token_mask, tp_size, tp_rank):
    """Narrow the replicated token batch to this rank's dispatch slice."""

    original_tokens = flat.size(0)
    tp_tokens = (original_tokens + tp_size - 1) // tp_size
    slice_begin = min(tp_tokens * tp_rank, original_tokens)
    slice_size = min(original_tokens - slice_begin, tp_tokens)
    flat = flat.narrow(0, slice_begin, slice_size)
    ids = ids.narrow(0, slice_begin, slice_size)
    if positions is not None:
        positions = positions.narrow(0, slice_begin, slice_size)
    if active_token_mask is not None:
        active_token_mask = active_token_mask.narrow(0, slice_begin, slice_size)
    return flat, ids, positions, active_token_mask, original_tokens


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
        if n_shared_experts != 1:
            raise ValueError("PPU EP MoE requires one shared expert")
        if n_physical_experts not in (None, n_routed_experts):
            raise ValueError("PPU EP MoE does not support redundant experts")
        if (
            tp_size <= 0
            or not 0 <= tp_rank < tp_size
            or ep_size <= 0
            or max_tokens_per_rank <= 0
            or n_routed_experts % ep_size
            or not 0 <= ep_rank < ep_size
        ):
            raise ValueError("Invalid PPU EP capacity or expert ownership")
        self.layer_id, self.dim = int(layer_id), int(dim)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self._tp_size = int(tp_size)
        self._tp_rank = int(tp_rank)
        self._ep_size = int(ep_size)
        self._is_decode_role = bool(is_decode_role)
        self._fuse_hc_sum = getattr(platform_provider, "_decode_fusions", False)
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
            sglang_moe=True,
        )
        self._shared_executor = platform_provider.build_shared_expert_executor()
        self._shared_executor.prepare(self.shared_experts)

    def _route_and_start_shared(
        self, x, input_ids, active_token_mask=None, input_quantized=None
    ):
        shared_kwargs = (
            {} if input_quantized is None else {"input_quantized": input_quantized}
        )
        early = self._shared_executor.start_before_routing
        if early:
            self._shared_executor.start(self.shared_experts, x, **shared_kwargs)
        try:
            masked_gate = getattr(self.gate, "forward_with_mask", None)
            if callable(masked_gate):
                routing = masked_gate(x, input_ids, active_token_mask)
            else:
                weights, indices = self.gate(x, input_ids)
                routing = weights, indices, active_token_mask
        except Exception:
            if early:
                self._shared_executor.finish()
            raise
        if not early:
            self._shared_executor.start(self.shared_experts, x, **shared_kwargs)
        return routing

    def _run_experts(
        self,
        x,
        input_ids,
        *,
        observer=None,
        active_token_mask=None,
        input_quantized=None,
    ):
        weights, indices, remaining_mask = self._route_and_start_shared(
            x,
            input_ids,
            active_token_mask if observer is None else None,
            input_quantized,
        )
        if observer is not None:
            remaining_mask = active_token_mask
        try:
            if observer is not None:
                observer("input", x)
                observer("topk_weights", weights)
                observer("topk_indices", indices)
            if remaining_mask is None:
                routed = self._strategy(x, weights, indices)
            else:
                routed = self._strategy(
                    x, weights, indices, active_token_mask=remaining_mask
                )
        except Exception:
            self._shared_executor.finish()
            raise
        shared = self._shared_executor.finish()
        if observer is not None:
            observer("routed_y", routed)
            observer("shared_y", shared)
        return routed, shared

    def _run_chunk(self, x, input_ids, out, *, observer=None, active_token_mask=None):
        routed, shared = self._run_experts(
            x, input_ids, observer=observer, active_token_mask=active_token_mask
        )
        combine_routed_and_shared(routed, shared, x.dtype, out=out)
        if observer is not None:
            observer("final_y", out)

    def forward_with_hc(
        self,
        x,
        input_ids,
        hc,
        residual,
        post,
        comb,
        *,
        active_token_mask=None,
        input_quantized=None,
    ):
        """Return None before execution when the ordinary FFN/HC path is needed."""
        can_fuse = getattr(hc, "can_fuse_moe_sum", None)
        if (
            not self._fuse_hc_sum
            or not self._is_decode_role
            or self._tp_size != 1
            or self._strategy.output_dtype != torch.bfloat16
            or not callable(can_fuse)
            or not can_fuse(x, residual, post, comb)
            or x.numel() // self.dim > self.max_tokens_per_rank
        ):
            return None
        # Tensor recording requires the materialized MoE output boundary.
        if self._observer_factory and self._observer_factory(None) is not None:
            return None
        flat, ids = x.reshape(-1, self.dim), input_ids.reshape(-1)
        if ids.numel() != flat.size(0):
            raise ValueError("MoE input_ids/token mismatch")
        if active_token_mask is not None and (
            active_token_mask.shape != (flat.size(0),)
            or active_token_mask.dtype != torch.bool
            or active_token_mask.device != flat.device
            or not active_token_mask.is_contiguous()
        ):
            raise ValueError("MoE active_token_mask must be one device bool per token")
        with self._record_function_scope():
            routed, shared = self._run_experts(
                flat,
                ids,
                active_token_mask=active_token_mask,
                input_quantized=input_quantized,
            )
        # finish() above joins the shared stream before either buffer is read.
        return hc.post_moe_sum(
            routed.view_as(x), shared.view_as(x), residual, post, comb
        )

    def forward_with_hc_pre(
        self, residual, input_ids, hc, norm, *, active_token_mask=None
    ):
        from ...modules.linear.fp8_linear import PpuFp8Linear
        from .ppu_shared_expert import PpuSharedExpertExecutor

        if (
            not self._fuse_hc_sum
            or not self._is_decode_role
            or self._tp_size != 1
            or self._strategy.output_dtype != torch.bfloat16
            or not isinstance(self._shared_executor, PpuSharedExpertExecutor)
            or type(self.shared_experts.w13) is not PpuFp8Linear
            or self.shared_experts.w13.quantization != "v2_column"
            or not callable(getattr(hc, "can_fuse_pre_quant", None))
            or not hc.can_fuse_pre_quant(residual, norm)
            or not getattr(hc, "_fuse_moe_sum", False)
            or residual.numel() // (4 * self.dim) > self.max_tokens_per_rank
            or (self._observer_factory and self._observer_factory(None) is not None)
        ):
            return None
        x, post, comb, quantized = hc.pre_norm_quantized(residual, norm)
        result = self.forward_with_hc(
            x,
            input_ids,
            hc,
            residual,
            post,
            comb,
            active_token_mask=active_token_mask,
            input_quantized=quantized,
        )
        if result is None:
            # The inputs have been prepared, but no expert work has run yet.
            return hc.post(
                self.forward(
                    x,
                    input_ids,
                    is_decode_forward=True,
                    active_token_mask=active_token_mask,
                ),
                residual,
                post,
                comb,
            )
        return result

    def forward(
        self,
        x,
        input_ids,
        *,
        is_decode_forward=False,
        positions=None,
        active_token_mask=None,
    ):
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        ids = input_ids.reshape(-1)
        if ids.numel() != flat.size(0):
            raise ValueError("MoE input_ids/token mismatch")
        if active_token_mask is not None and (
            active_token_mask.shape != (flat.size(0),)
            or active_token_mask.dtype != torch.bool
            or active_token_mask.device != flat.device
            or not active_token_mask.is_contiguous()
        ):
            raise ValueError("MoE active_token_mask must be one device bool per token")
        capacity = self.max_tokens_per_rank
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if flat.size(0) > capacity and (
            self._is_decode_role or is_decode_forward or capturing
        ):
            raise ValueError("Decode or graph MoE input exceeds max_tokens_per_rank")
        original_tokens = flat.size(0)
        if self._tp_size > 1:
            # DeepEP dispatch consumes one rank's token slice, not the
            # TP-replicated batch. Slices are deterministic and identical on
            # every rank, so chunk counts and collective order stay in sync.
            flat, ids, positions, active_token_mask, original_tokens = (
                _slice_tokens_for_tp(
                    flat, ids, positions, active_token_mask,
                    self._tp_size, self._tp_rank,
                )
            )
        rounds = max((flat.size(0) + capacity - 1) // capacity, 1)
        if not self._is_decode_role and not is_decode_forward and not capturing:
            if not dist.is_initialized() or dist.get_world_size() != self._ep_size:
                raise RuntimeError("PPU Prefill requires the engine's WORLD-sized EP group")
            # The engine creates DeepEP on WORLD; every rank must dispatch the
            # same number of rounds even after its local token slice is empty.
            common_rounds = torch.tensor(rounds, dtype=torch.int64, device=flat.device)
            dist.all_reduce(common_rounds, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
            rounds = int(common_rounds.item())
        if rounds > 1 and not self.chunking_enabled:
            raise ValueError("PPU EP MoE input exceeds capacity with chunking disabled")
        # Each call owns its output, including graph-pool allocations. Retaining
        # a result from one model/layer cannot alias another model's workspace.
        out = torch.empty_like(flat)
        for chunk in range(rounds):
            start = min(chunk * capacity, flat.size(0))
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
                    flat[start:end],
                    ids[start:end],
                    out[start:end],
                    observer=observer,
                    active_token_mask=(
                        active_token_mask[start:end]
                        if active_token_mask is not None
                        else None
                    ),
                )
        if self._tp_size > 1:
            out = _gather_tp_slices(out, original_tokens, self._tp_size)
        return out.view(shape)
