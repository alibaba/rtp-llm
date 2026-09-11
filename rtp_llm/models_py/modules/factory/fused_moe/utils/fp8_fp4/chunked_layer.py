"""Token chunking and synchronized EP execution for FP8/FP4 MoE layers."""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Callable, ContextManager, Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeLayer,
)

DEFAULT_MOE_CHUNK_TOKENS = 16384
Observer = Callable[[str, torch.Tensor], None]
ObserverFactory = Callable[[Optional[torch.Tensor]], Optional[Observer]]
_FINAL_OUT_CACHE: dict[tuple, torch.Tensor] = {}
_CHUNKED_MOE_LOGGED = False
_SYNCHRONIZED_CHUNK_TOKENS: ContextVar[Optional[int]] = ContextVar(
    "moe_synchronized_chunk_tokens", default=None
)


def cp_padded_tokens_per_rank_bound(max_seq_len: int, cp_size: int) -> int:
    cp_size = max(int(cp_size), 1)
    max_seq_len = max(int(max_seq_len), 0)
    if cp_size <= 1 or max_seq_len == 0:
        return max_seq_len
    global_alignment = cp_size * 2
    padded_seq_len = (
        (max_seq_len + global_alignment - 1) // global_alignment
    ) * global_alignment
    return padded_seq_len // cp_size


def resolve_moe_max_tokens_per_rank(
    max_seq_len: int,
    current_max_tokens_per_rank: int,
    cp_size: int,
    max_generate_batch_size: int,
    *,
    max_context_batch_size: int = 1,
    is_decode_role: bool = False,
    is_speculative: bool = False,
    gen_num_per_cycle: int = 0,
    chunking_enabled: bool = True,
    chunk_tokens: int = DEFAULT_MOE_CHUNK_TOKENS,
) -> int:
    max_generate_batch_size = int(max_generate_batch_size)
    if max_generate_batch_size <= 0:
        raise ValueError(
            f"max_generate_batch_size must be positive, got {max_generate_batch_size}"
        )
    if is_decode_role:
        tokens_per_batch = max(int(gen_num_per_cycle) + 1, 1) if is_speculative else 1
        return max_generate_batch_size * tokens_per_batch

    budget = int(current_max_tokens_per_rank)
    cp_size = max(int(cp_size), 1)
    if cp_size > 1:
        # The incoming scheduler budget is global, before per-request zigzag
        # padding. Each of B requests can add at most (2 * CP - 1) tokens;
        # the padded total is a multiple of 2 * CP. Bound both that total
        # and B maximum-length requests, then apply the MoE chunk limit.
        batch_size = min(max(int(max_context_batch_size), 1), max(budget, 0))
        alignment = 2 * cp_size
        budget = min(
            cp_padded_tokens_per_rank_bound(max_seq_len, cp_size) * batch_size,
            2 * ((budget + batch_size * (alignment - 1)) // alignment),
        )
    # A disabled/zero prefill budget must not create zero-sized runtime buffers.
    return max(min(budget, chunk_tokens) if chunking_enabled else budget, 1)


def _get_or_create_final_out(
    capacity: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (device, dim, dtype)
    cached = _FINAL_OUT_CACHE.get(key)
    if cached is None or cached.size(0) < capacity:
        cached = torch.empty((max(capacity, 1), dim), dtype=dtype, device=device)
        _FINAL_OUT_CACHE[key] = cached
    return cached


@contextmanager
def synchronized_moe_chunk_plan(
    layers: Iterable[nn.Module], tokens: int, device: torch.device
):
    """Synchronize the largest EP token count once for a prefill layer stack."""
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    ):
        yield
        return

    needs_sync = any(
        getattr(getattr(layer, "ffn", layer), "strategy_name", None)
        in {"mega_moe", "mega_moe_se"}
        and not getattr(getattr(layer, "ffn", layer), "_is_decode_role", False)
        and getattr(getattr(layer, "ffn", layer), "chunking_enabled", False)
        for layer in layers
    )
    if not needs_sync:
        yield
        return

    token_count = torch.tensor(int(tokens), dtype=torch.int64, device=device)
    dist.all_reduce(token_count, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
    token = _SYNCHRONIZED_CHUNK_TOKENS.set(int(token_count.item()))
    try:
        yield
    finally:
        _SYNCHRONIZED_CHUNK_TOKENS.reset(token)


class ChunkedFp8Fp4MoeLayer(nn.Module):
    """Run a canonical FP8/FP4 MoE layer within its per-rank token capacity."""

    def __init__(
        self,
        layer_id: int,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int,
        score_func: str,
        route_scale: float,
        swiglu_limit: float,
        n_hash_layers: int,
        vocab_size: int,
        layer_weights: Optional[Dict] = None,
        ep_size: int = 1,
        ep_rank: int = 0,
        world_size: Optional[int] = None,
        world_rank: Optional[int] = None,
        max_tokens_per_rank: int = 8192,
        is_decode_role: bool = False,
        strategy: Optional[str] = None,
        n_physical_experts: Optional[int] = None,
        *,
        chunking_enabled: bool = True,
        model_type: str = "generic_fp8_fp4_moe",
        moe_w1_layout: str = "up_gate",
        has_shared_expert_gate: bool = False,
        observer_factory: Optional[ObserverFactory] = None,
        record_function_scope: Callable[[], ContextManager] = nullcontext,
    ) -> None:
        super().__init__()
        if layer_weights is None:
            raise ValueError("ChunkedFp8Fp4MoeLayer requires per-layer weights")
        self.layer_id = int(layer_id)
        self.dim = int(dim)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self._is_decode_role = bool(is_decode_role)
        self.chunking_enabled = chunking_enabled
        self._observer_factory = observer_factory
        self._record_function_scope = record_function_scope
        self._moe = Fp8Fp4MoeLayer(
            layer_id=layer_id,
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_shared_experts=n_shared_experts,
            score_func=score_func,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            n_hash_layers=n_hash_layers,
            vocab_size=vocab_size,
            layer_weights=layer_weights,
            ep_size=ep_size,
            ep_rank=ep_rank,
            world_size=world_size,
            world_rank=world_rank,
            max_tokens_per_rank=max_tokens_per_rank,
            strategy=strategy or "auto",
            model_type=model_type,
            warmup_include_capacity=chunking_enabled and not is_decode_role,
            moe_w1_layout=moe_w1_layout,
            has_shared_expert_gate=has_shared_expert_gate,
            physical_expert_num=n_physical_experts,
        )
        self.strategy_name = self._moe.strategy_name

    @property
    def fused_moe(self):
        return self._moe.fused_moe

    @property
    def gate(self):
        return self._moe.gate

    @property
    def shared_experts(self):
        return self._moe.shared_experts

    def _should_chunk(self, tokens: int, *, is_decode_forward: bool = False) -> bool:
        max_tokens = self.max_tokens_per_rank
        if max_tokens <= 0:
            raise ValueError(f"max_tokens_per_rank must be positive, got {max_tokens}")
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        is_decode = self._is_decode_role or is_decode_forward
        if is_decode or capturing:
            if tokens > max_tokens:
                mode = "decode" if is_decode else "CUDA graph capture"
                raise ValueError(
                    f"{mode} MoE input tokens={tokens} exceeds "
                    f"max_tokens_per_rank={max_tokens}"
                )
            return False
        return self.chunking_enabled and tokens > max_tokens

    def _synchronized_chunk_tokens(
        self,
        tokens: int,
        device: torch.device,
        *,
        is_decode_forward: bool = False,
    ) -> int:
        if (
            self.strategy_name not in {"mega_moe", "mega_moe_se"}
            or self._is_decode_role
            or is_decode_forward
            or not self.chunking_enabled
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return tokens
        planned_tokens = _SYNCHRONIZED_CHUNK_TOKENS.get()
        if planned_tokens is not None:
            return planned_tokens
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return tokens
        token_count = torch.tensor(tokens, dtype=torch.int64, device=device)
        dist.all_reduce(token_count, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
        return int(token_count.item())

    def _observer(self, positions: Optional[torch.Tensor]):
        return self._observer_factory(positions) if self._observer_factory else None

    def _call_moe(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        *,
        observer: Optional[Callable[[str, torch.Tensor], None]] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with self._record_function_scope():
            return self._moe(
                x,
                input_ids,
                observer=observer,
                out=out,
            )

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        *,
        is_decode_forward: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        global _CHUNKED_MOE_LOGGED
        shape = x.shape
        flat_x = x.reshape(-1, self.dim)
        flat_ids = None if input_ids is None else input_ids.reshape(-1)
        if flat_ids is not None and flat_ids.numel() != flat_x.size(0):
            raise ValueError(
                f"input_ids has {flat_ids.numel()} tokens, expected {flat_x.size(0)}"
            )
        observer = self._observer(positions)
        synchronized_tokens = self._synchronized_chunk_tokens(
            flat_x.size(0),
            flat_x.device,
            is_decode_forward=is_decode_forward,
        )
        output = _get_or_create_final_out(
            max(flat_x.size(0), self.max_tokens_per_rank),
            self.dim,
            flat_x.dtype,
            flat_x.device,
        )[: flat_x.size(0)]
        if not self._should_chunk(
            synchronized_tokens, is_decode_forward=is_decode_forward
        ):
            if observer is None:
                self._call_moe(flat_x, flat_ids, out=output)
            else:
                self._call_moe(flat_x, flat_ids, observer=observer, out=output)
            return output.view(shape)

        if not _CHUNKED_MOE_LOGGED:
            _CHUNKED_MOE_LOGGED = True
            logging.info(
                "[MoE] chunked forward: synchronized_tokens=%d chunk_tokens=%d",
                synchronized_tokens,
                self.max_tokens_per_rank,
            )
        for start in range(0, synchronized_tokens, self.max_tokens_per_rank):
            local_start = min(start, flat_x.size(0))
            end = min(start + self.max_tokens_per_rank, flat_x.size(0))
            chunk_ids = None if flat_ids is None else flat_ids[local_start:end]
            chunk_observer = None
            if observer is not None:
                chunk_positions = (
                    positions[local_start:end] if positions is not None else None
                )
                chunk_observer = self._observer(chunk_positions)
            if chunk_observer is None:
                self._call_moe(
                    flat_x[local_start:end],
                    chunk_ids,
                    out=output[local_start:end],
                )
            else:
                self._call_moe(
                    flat_x[local_start:end],
                    chunk_ids,
                    observer=chunk_observer,
                    out=output[local_start:end],
                )
        return output.view(shape)
