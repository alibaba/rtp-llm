"""Bound MegaMoE workspace while keeping every EP rank in every round."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterable, Iterator, Optional

import torch
import torch.distributed as dist

# (group, local token count, maximum token count), valid for one layer stack.
_CHUNK_PLAN = ContextVar("mega_moe_fp8_chunk_plan", default=None)


def _capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


class MegaMoeChunker:
    def __init__(self, chunk_tokens: int, capacity: int, group: dist.ProcessGroup):
        if chunk_tokens <= 0 or capacity <= 0:
            raise ValueError("MegaMoE chunk size and capacity must be positive")
        self.chunk_tokens = min(chunk_tokens, capacity)
        self.capacity = capacity
        self.group = group

    def max_tokens(self, tokens: int, device: torch.device) -> int:
        plan = _CHUNK_PLAN.get()
        if plan is not None and plan[0] is self.group:
            if plan[1] != tokens:
                raise RuntimeError("MegaMoE token count changed inside the layer stack")
            return plan[2]
        count = torch.tensor(tokens, dtype=torch.int64, device=device)
        dist.all_reduce(count, op=dist.ReduceOp.MAX, group=self.group)
        return int(count.item())

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_chunk: Callable[[torch.Tensor], torch.Tensor],
        prepare_chunks: Optional[
            Callable[[torch.Tensor], Callable[[int, int], torch.Tensor]]
        ] = None,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        if _capturing(hidden_states.device):
            # Existing decode graphs are single-launch, with no host-side sync.
            if tokens > self.capacity:
                raise ValueError(
                    f"MegaMoE graph tokens={tokens} exceed capacity={self.capacity}; "
                    "run this prefill eagerly"
                )
            return forward_chunk(hidden_states)
        maximum = self.max_tokens(tokens, hidden_states.device)
        if maximum <= self.chunk_tokens:
            return forward_chunk(hidden_states)
        prepared_forward = (
            prepare_chunks(hidden_states) if prepare_chunks is not None else None
        )
        output = None
        for start in range(0, maximum, self.chunk_tokens):
            begin = min(start, tokens)
            end = min(start + self.chunk_tokens, tokens)
            # Empty ranks still host experts for tokens sent by their peers.
            chunk_output = (
                prepared_forward(begin, end)
                if prepared_forward is not None
                else forward_chunk(hidden_states[begin:end])
            )
            if output is None:
                output = torch.empty_like(hidden_states, dtype=chunk_output.dtype)
            # MegaMoE returns a view of reusable workspace. Consume it before
            # packing/launching the next chunk on the same CUDA stream.
            output[begin:end].copy_(chunk_output)
        return output


@contextmanager
def mega_moe_chunk_plan(
    layers: Iterable[torch.nn.Module], hidden_states: torch.Tensor
) -> Iterator[None]:
    """Amortize the EP token-count collective over one Qwen layer stack."""
    chunker = next(
        (
            layer.mlp._mega_moe_chunker
            for layer in layers
            if getattr(getattr(layer, "mlp", None), "_mega_moe_chunker", None)
            is not None
        ),
        None,
    )
    if chunker is None or _capturing(hidden_states.device):
        yield
        return
    tokens = hidden_states.shape[0]
    maximum = chunker.max_tokens(tokens, hidden_states.device)
    state = _CHUNK_PLAN.set((chunker.group, tokens, maximum))
    try:
        yield
    finally:
        _CHUNK_PLAN.reset(state)
