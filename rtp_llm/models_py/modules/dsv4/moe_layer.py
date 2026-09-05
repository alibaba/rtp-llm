"""DeepSeek-V4 boundary wrapper for the generic FP8/FP4 MoE layer."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4.chunk_env import (
    DEFAULT_DSV4_CHUNK_TOKENS,
    dsv4_chunk_tokens_from_env,
    dsv4_global_chunk_tokens_configured,
)
from rtp_llm.models_py.modules.dsv4.moe_weight_adapter import adapt_dsv4_moe_weights
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeLayer,
)

DEFAULT_MOE_CHUNK_TOKENS = DEFAULT_DSV4_CHUNK_TOKENS
_FINAL_OUT_CACHE: dict[tuple, torch.Tensor] = {}
_CHUNKED_MOE_LOGGED = False
_SYNCHRONIZED_CHUNK_TOKENS: ContextVar[Optional[int]] = ContextVar(
    "dsv4_synchronized_chunk_tokens", default=None
)


def chunked_moe_enabled() -> bool:
    if dsv4_global_chunk_tokens_configured():
        return moe_chunk_tokens_from_env() > 0
    return os.environ.get("DSV4_MOE_CHUNK_PREFILL", "1") != "0"


def moe_chunk_tokens_from_env(default: int = DEFAULT_MOE_CHUNK_TOKENS) -> int:
    min_value = 0 if dsv4_global_chunk_tokens_configured() else 1
    return dsv4_chunk_tokens_from_env(
        "DSV4_MOE_CHUNK_TOKENS",
        default,
        min_value=min_value,
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
    is_decode_role: bool = False,
    is_speculative: bool = False,
    gen_num_per_cycle: int = 0,
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
        budget = min(
            budget,
            max(cp_padded_tokens_per_rank_bound(max_seq_len, cp_size), 4096),
        )
    return min(budget, moe_chunk_tokens_from_env()) if chunked_moe_enabled() else budget


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
        not chunked_moe_enabled()
        or not dist.is_available()
        or not dist.is_initialized()
        or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    ):
        yield
        return

    needs_sync = any(
        getattr(getattr(layer, "ffn", layer), "strategy_name", None)
        in {"mega_moe", "mega_moe_se"}
        and not getattr(getattr(layer, "ffn", layer), "_is_decode_role", False)
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


class Dsv4MoeLayer(nn.Module):
    """Adapt DSV4 weights and token chunking to the generic fused-MoE layer."""

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
    ) -> None:
        super().__init__()
        if layer_weights is None:
            raise ValueError("Dsv4MoeLayer requires per-layer weights")
        self.layer_id = int(layer_id)
        self.dim = int(dim)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self._is_decode_role = bool(is_decode_role)
        adapt_dsv4_moe_weights(layer_weights, moe_inter_dim, n_shared_experts)
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
            model_type="deepseek_v4",
            warmup_include_capacity=chunked_moe_enabled() and not is_decode_role,
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

    def _should_chunk(self, tokens: int) -> bool:
        max_tokens = self.max_tokens_per_rank
        if max_tokens <= 0:
            raise ValueError(f"max_tokens_per_rank must be positive, got {max_tokens}")
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if self._is_decode_role or capturing:
            if tokens > max_tokens:
                mode = "decode" if self._is_decode_role else "CUDA graph capture"
                raise ValueError(
                    f"{mode} MoE input tokens={tokens} exceeds "
                    f"max_tokens_per_rank={max_tokens}"
                )
            return False
        return chunked_moe_enabled() and tokens > max_tokens

    def _synchronized_chunk_tokens(self, tokens: int, device: torch.device) -> int:
        planned_tokens = _SYNCHRONIZED_CHUNK_TOKENS.get()
        if planned_tokens is not None:
            return planned_tokens
        if (
            self.strategy_name not in {"mega_moe", "mega_moe_se"}
            or self._is_decode_role
            or not chunked_moe_enabled()
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return tokens
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return tokens
        token_count = torch.tensor(tokens, dtype=torch.int64, device=device)
        dist.all_reduce(token_count, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
        return int(token_count.item())

    def _debug_observer(self, positions: Optional[torch.Tensor]):
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        if not _rt.should_record_layer(self.layer_id):
            return None
        names = {
            "input": "moe_x_in",
            "topk_weights": "moe_topk_weights",
            "topk_indices": "moe_topk_indices",
            "routed_y": "moe_routed_y",
            "shared_y": "moe_shared_y",
            "final_y": "moe_y",
        }
        global_pos = int(getattr(_rt, "_DBG_GLOBAL_POS", -1))

        def observe(kind: str, tensor: torch.Tensor) -> None:
            suffix = names.get(kind)
            if suffix is None:
                return
            name = f"L{self.layer_id:02d}_{suffix}"
            _rt.record_if_level(2, name, tensor)
            if (
                global_pos < 0
                or positions is None
                or positions.numel() != tensor.size(0)
            ):
                return
            mask = positions.to(device=tensor.device, dtype=torch.long).reshape(-1)
            mask = mask == global_pos
            _rt.record_if_level(
                2,
                f"{name}_pos{global_pos}",
                tensor[mask].contiguous(),
            )

        return observe

    def _call_moe(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        *,
        observer: Optional[Callable[[str, torch.Tensor], None]] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with _profiler.moe_record_function_scope():
            return self._moe(
                x,
                input_ids,
                observer=observer,
                out=out,
            )

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        global _CHUNKED_MOE_LOGGED
        shape = x.shape
        flat_x = x.reshape(-1, self.dim)
        flat_ids = None if input_ids is None else input_ids.reshape(-1)
        if flat_ids is not None and flat_ids.numel() != flat_x.size(0):
            raise ValueError(
                f"input_ids has {flat_ids.numel()} tokens, expected {flat_x.size(0)}"
            )
        positions = getattr(self, "_dbg_positions", None)
        observer = self._debug_observer(positions)
        synchronized_tokens = self._synchronized_chunk_tokens(
            flat_x.size(0), flat_x.device
        )
        if not self._should_chunk(synchronized_tokens):
            if observer is None:
                return self._call_moe(flat_x, flat_ids).view(shape)
            return self._call_moe(flat_x, flat_ids, observer=observer).view(shape)

        if not _CHUNKED_MOE_LOGGED:
            _CHUNKED_MOE_LOGGED = True
            logging.info(
                "[DeepSeekV4 MoE] chunked forward: synchronized_tokens=%d chunk_tokens=%d",
                synchronized_tokens,
                self.max_tokens_per_rank,
            )
        output = _get_or_create_final_out(
            flat_x.size(0), self.dim, flat_x.dtype, flat_x.device
        )[: flat_x.size(0)]
        for start in range(0, synchronized_tokens, self.max_tokens_per_rank):
            local_start = min(start, flat_x.size(0))
            end = min(start + self.max_tokens_per_rank, flat_x.size(0))
            chunk_ids = None if flat_ids is None else flat_ids[local_start:end]
            chunk_observer = None
            if observer is not None:
                chunk_positions = (
                    positions[local_start:end] if positions is not None else None
                )
                chunk_observer = self._debug_observer(chunk_positions)
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
