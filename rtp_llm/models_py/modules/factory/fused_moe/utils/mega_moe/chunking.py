"""A forward-scoped token plan for eager MegaMoE prefill."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.distributed as dist

MAX_MEGA_MOE_RANKS = 72
_PREFILL_PLAN: ContextVar[tuple[dist.ProcessGroup, int, int] | None] = ContextVar(
    "mega_moe_prefill_plan", default=None
)


def max_ep_tokens(tokens: int, device: torch.device, group: dist.ProcessGroup) -> int:
    count = torch.tensor(tokens, dtype=torch.int64, device=device)
    dist.all_reduce(count, op=dist.ReduceOp.MAX, group=group)
    return int(count.item())


class MegaMoePrefillChunker:
    def __init__(self, chunk_tokens: int, group: dist.ProcessGroup):
        if chunk_tokens <= 0:
            raise ValueError("MegaMoE prefill chunk size must be positive")
        self.chunk_tokens = chunk_tokens
        self.group = group

    @contextmanager
    def plan(self, tokens: int, device: torch.device) -> Iterator[None]:
        """Exchange token counts once on the same group as the expert kernel."""
        maximum = max_ep_tokens(tokens, device, self.group)
        plan = _PREFILL_PLAN.set((self.group, tokens, maximum))
        try:
            yield
        finally:
            _PREFILL_PLAN.reset(plan)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_once: Callable[[torch.Tensor], torch.Tensor],
        prepare_chunks: Callable[[torch.Tensor], Callable[[int, int], torch.Tensor]],
    ) -> torch.Tensor:
        plan = _PREFILL_PLAN.get()
        if plan is None:
            # Decode, graph capture and eager forwards without a plan keep
            # the original path. There is no layer-local collective fallback.
            return forward_once(hidden_states)

        group, tokens, maximum = plan
        if group is not self.group or tokens != hidden_states.size(0):
            raise RuntimeError("MegaMoE prefill plan does not match the layer input")
        if maximum <= self.chunk_tokens and tokens:
            return forward_once(hidden_states)

        forward_chunk = prepare_chunks(hidden_states)
        output = None
        for start in range(0, max(maximum, 1), self.chunk_tokens):
            begin, end = min(start, tokens), min(start + self.chunk_tokens, tokens)
            # Empty local slices still host experts for the other ranks.
            chunk_output = forward_chunk(begin, end)
            if output is None:
                output = torch.empty_like(hidden_states, dtype=chunk_output.dtype)
            # The executor's output buffer is reused by the next launch.
            output[begin:end].copy_(chunk_output)
        assert output is not None
        return output
