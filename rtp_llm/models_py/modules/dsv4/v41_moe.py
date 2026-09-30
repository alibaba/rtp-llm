"""Bounded local routing and FP32 combination for V4.1 FP4 expert shards."""

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout
from rtp_llm.models_py.modules.dsv4.v41_expert import V41_EXPERT_RECIPE
from rtp_llm.models_py.modules.dsv4.v41_routed import (
    GROUPED_ALIGNMENT,
    V41RoutedExperts,
    _validate_input_scales,
)


def _assert_values(condition, message):
    if condition.device.type == "cpu":
        if not bool(condition):
            raise ValueError(message)
    else:
        torch._assert_async(condition, message)


class V41LocalMoE(nn.Module):
    """Return a local shard's routed contribution in FP32, without collectives.

    Masked execution uses fixed capacity and device-only route planning. Eager
    contiguous execution reads expert counts once to bound prefill padding.
    A caller must collectively schedule chunks before entering this component.
    Workspaces belong to the invocation; no class-level caches are shared.
    """

    recipe = V41_EXPERT_RECIPE

    def __init__(
        self,
        experts,
        *,
        expert_start_id=0,
        max_tokens=4096,
        masked_token_limit=128,
        max_topk=6
    ):
        super().__init__()
        if not isinstance(experts, V41RoutedExperts):
            raise ValueError("V4.1 local routing requires V41RoutedExperts")
        if (
            type(expert_start_id) is not int
            or not 0
            <= expert_start_id
            <= torch.iinfo(torch.int32).max - experts.experts
        ):
            raise ValueError("V4.1 local expert start must be a nonnegative integer")
        if (
            type(max_tokens) is not int
            or max_tokens <= 0
            or type(masked_token_limit) is not int
            or not 0 < masked_token_limit <= max_tokens
        ):
            raise ValueError("V4.1 local token bounds must be positive and ordered")
        if type(max_topk) is not int or max_topk <= 0:
            raise ValueError("V4.1 local top-k bound must be positive")
        self.experts = experts
        self.expert_start_id = expert_start_id
        self.max_tokens, self.masked_token_limit = max_tokens, masked_token_limit
        self.max_topk = max_topk

    def forward_received(self, payload, dispatch_layout, *, layout="contiguous"):
        """Return FP32 local contributions from an already-admitted EP payload."""
        if (
            type(dispatch_layout) is not V41DispatchLayout
            or dispatch_layout.recipe != self.recipe
            or dispatch_layout.hidden != self.experts.dim
            or dispatch_layout.topk > self.max_topk
            or self.expert_start_id + self.experts.experts > dispatch_layout.experts
        ):
            raise ValueError(
                "V4.1 received dispatch layout disagrees with expert shard"
            )
        dispatch_layout.views(payload)
        if (
            payload.device != self.experts.device
            or payload.shape[0] > self.max_tokens
            or layout not in ("contiguous", "masked")
            or (layout == "masked" and payload.shape[0] > self.masked_token_limit)
        ):
            raise ValueError(
                "V4.1 received payload exceeds device, layout or token bounds"
            )
        values, scales, weights, indices = dispatch_layout.unpack(payload)
        return self(values, weights, indices, layout=layout, input_scales=scales)

    def forward(
        self,
        x,
        weights,
        indices,
        *,
        valid_rows=None,
        layout="contiguous",
        input_scales=None
    ):
        """Route BF16 or explicit received E4M3/linear UE8M0 rows without reweighting."""
        dtype = torch.bfloat16 if input_scales is None else torch.float8_e4m3fn
        if (
            x.ndim != 2
            or x.shape[1] != self.experts.dim
            or x.dtype != dtype
            or x.device != self.experts.device
            or not x.is_contiguous()
            or (torch.is_grad_enabled() and x.requires_grad)
        ):
            raise ValueError(
                "V4.1 MoE input requires contiguous inference BF16 or scaled E4M3 [N,D]"
            )
        if input_scales is not None:
            _validate_input_scales(x, input_scales)
        source = x if input_scales is None else x.view(torch.uint8)
        rows, dim = x.shape
        if rows > self.max_tokens or layout not in ("contiguous", "masked"):
            raise ValueError("V4.1 routing layout or token bound is invalid")
        if layout == "masked" and rows > self.masked_token_limit:
            raise ValueError("V4.1 masked routing exceeds the admitted token bound")
        if x.is_cuda and layout == "contiguous":
            with torch.cuda.device(x.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "V4.1 contiguous routing uses eager host counts; capture requires masked routing"
                    )
        if (
            weights.ndim != 2
            or weights.shape[0] != rows
            or not 0 < weights.shape[1] <= self.max_topk
            or weights.dtype != torch.float32
            or weights.device != x.device
            or not weights.is_contiguous()
            or (torch.is_grad_enabled() and weights.requires_grad)
            or indices.shape != weights.shape
            or indices.dtype not in (torch.int32, torch.int64)
            or indices.device != x.device
            or not indices.is_contiguous()
        ):
            raise ValueError(
                "V4.1 routes require contiguous device FP32 weights and integer [N,K] ids"
            )
        if valid_rows is None:
            valid_rows = torch.ones(rows, dtype=torch.bool, device=x.device)
        if (
            valid_rows.shape != (rows,)
            or valid_rows.dtype != torch.bool
            or valid_rows.device != x.device
            or not valid_rows.is_contiguous()
        ):
            raise ValueError("V4.1 routing requires a contiguous device bool row mask")
        experts, topk = self.experts.experts, weights.shape[1]
        local = indices.long() - self.expert_start_id
        owned = valid_rows[:, None] & (local >= 0) & (local < experts)
        _assert_values(
            (~owned | (torch.isfinite(weights) & (weights >= 0))).all(),
            "V4.1 owned route weights must be finite and nonnegative",
        )
        active = owned & (weights != 0)
        keys = torch.where(active, local, experts)
        ordered_ids = keys.sort(dim=-1).values
        _assert_values(
            (
                ~(
                    (ordered_ids[:, 1:] == ordered_ids[:, :-1])
                    & (ordered_ids[:, 1:] < experts)
                )
            ).all(),
            "V4.1 active expert ids must be unique per token",
        )
        if not rows:
            return torch.empty((0, dim), device=x.device, dtype=torch.float32)
        route_count = rows * topk
        # Stable ordering fixes each expert's row order without floating atomics.
        order = torch.argsort(keys.flatten(), stable=True)
        sorted_experts = keys.flatten().index_select(0, order)
        counts = torch.zeros(experts + 1, dtype=torch.int64, device=x.device)
        counts.scatter_add_(0, keys.flatten(), torch.ones_like(keys.flatten()))
        starts = counts.cumsum(0) - counts
        offsets = torch.arange(route_count, device=x.device) - starts.index_select(
            0, sorted_experts
        )
        sorted_active = sorted_experts < experts
        route_map = torch.full((route_count,), -1, device=x.device, dtype=torch.int64)
        if layout == "contiguous":
            actual_counts = tuple(counts[:experts].cpu().tolist())
            segments = tuple(
                (n + GROUPED_ALIGNMENT - 1) // GROUPED_ALIGNMENT * GROUPED_ALIGNMENT
                for n in actual_counts
            )
            allocated, active_count = sum(segments), sum(actual_counts)
            if not allocated:
                return torch.zeros((rows, dim), device=x.device, dtype=torch.float32)
            segment_starts, cursor = [], 0
            for size in segments:
                segment_starts.append(cursor)
                cursor += size
            base = torch.tensor(segment_starts, dtype=torch.int64, device=x.device)
            order = order[:active_count]
            destinations = (
                base.index_select(0, sorted_experts[:active_count])
                + offsets[:active_count]
            )
            routed_x = torch.zeros(
                (allocated, dim), device=x.device, dtype=source.dtype
            )
            routed_weights = torch.zeros(
                allocated, device=x.device, dtype=torch.float32
            )
            valid = torch.zeros(allocated, device=x.device, dtype=torch.bool)
            valid[destinations] = True
            routed_x.index_copy_(0, destinations, source.index_select(0, order // topk))
            routed_scales = None
            if input_scales is not None:
                routed_scales = torch.zeros(
                    (allocated, dim // 32), device=x.device, dtype=torch.uint8
                )
                routed_scales.index_copy_(
                    0, destinations, input_scales.index_select(0, order // topk)
                )
            routed_weights.index_copy_(
                0, destinations, weights.flatten().index_select(0, order)
            )
            route_map.index_copy_(0, order, destinations)
            down = self.experts.forward_contiguous(
                routed_x.view(x.dtype),
                segments,
                valid,
                routed_weights,
                input_scales=routed_scales,
            )
        else:
            capacity = (
                (rows + GROUPED_ALIGNMENT - 1) // GROUPED_ALIGNMENT * GROUPED_ALIGNMENT
            )
            allocated = experts * capacity
            # Give every inactive route a distinct disposable slot after storage.
            # This avoids races or overwriting a live row during device scatter.
            destinations = torch.where(
                sorted_active,
                sorted_experts * capacity + offsets,
                allocated + torch.arange(route_count, device=x.device),
            )
            routed_x = torch.zeros(
                (allocated + route_count, dim), device=x.device, dtype=source.dtype
            )
            routed_weights = torch.zeros(
                allocated + route_count, device=x.device, dtype=torch.float32
            )
            safe_x = torch.where(active.any(-1)[:, None], source, 0)
            routed_x.index_copy_(0, destinations, safe_x.index_select(0, order // topk))
            routed_scales = None
            if input_scales is not None:
                routed_scales = torch.zeros(
                    (allocated + route_count, dim // 32),
                    device=x.device,
                    dtype=torch.uint8,
                )
                routed_scales.index_copy_(
                    0, destinations, input_scales.index_select(0, order // topk)
                )
                routed_scales = routed_scales[:allocated].view(
                    experts, capacity, dim // 32
                )
            sorted_weights = (
                torch.where(active, weights, 0).flatten().index_select(0, order)
            )
            routed_weights.index_copy_(0, destinations, sorted_weights)
            route_map.index_copy_(
                0, order, torch.where(sorted_active, destinations, -1)
            )
            down = self.experts.forward_masked(
                routed_x[:allocated].view(x.dtype).view(experts, capacity, dim),
                counts[:experts].to(torch.int32),
                routed_weights[:allocated].view(experts, capacity),
                input_scales=routed_scales,
            ).reshape(allocated, dim)
        # The expert already applied weights before its BF16/down-input rounding.
        # Accumulate in expert-ID order and FP32, as the checkpoint loop does.
        output = torch.zeros((rows, dim), device=x.device, dtype=torch.float32)
        down = torch.cat(
            (down, torch.zeros((1, dim), device=x.device, dtype=down.dtype))
        )
        route_map = route_map.reshape(rows, topk)
        route_map = route_map.gather(1, keys.argsort(dim=-1, stable=True))
        for slot in range(topk):
            destination = torch.where(
                route_map[:, slot] >= 0, route_map[:, slot], allocated
            )
            output.add_(down.index_select(0, destination).float())
        return output
