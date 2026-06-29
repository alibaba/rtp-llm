"""Small non-MegaMoE CUDA graph islands for DSV4 prefill.

This module is intentionally narrow and env-gated.  It captures only HC-pre +
RMSNorm islands, leaving attention, CP/NCCL, compressor history, and MegaMoE
eager.  The default model path never imports or uses it.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Callable

import torch

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class _IslandState:
    static_input: torch.Tensor
    graph: torch.cuda.CUDAGraph
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    replay_count: int = 0


@dataclasses.dataclass
class _BridgeState:
    static_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    static_freqs_cis: torch.Tensor | None
    graph: torch.cuda.CUDAGraph
    outputs: tuple[torch.Tensor, ...]
    replay_count: int = 0


class PrefillIslandGraphManager:
    """Per-layer graph cache for graph-safe pre islands."""

    def __init__(self) -> None:
        self._states: dict[tuple[int, str, tuple[int, ...], str, str], _IslandState] = {}
        self._disabled: set[tuple[int, str, tuple[int, ...], str, str]] = set()
        self._bridge_states: dict[tuple[object, ...], _BridgeState] = {}
        self._bridge_disabled: set[tuple[object, ...]] = set()
        self._disabled_prefixes: set[tuple[int, str]] = set()
        self._bridge_disabled_prefixes: set[tuple[int, int, str]] = set()

    @staticmethod
    def _key(layer: object, kind: str, x: torch.Tensor):
        return (
            id(layer),
            kind,
            tuple(int(v) for v in x.shape),
            str(x.dtype),
            str(x.device),
        )

    @staticmethod
    def _tensor_sig(x: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
        return (tuple(int(v) for v in x.shape), str(x.dtype), str(x.device))

    def _bridge_key(
        self,
        current_layer: object,
        next_layer: object,
        ffn_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        common: object | None = None,
        include_qkv: bool = False,
        include_q: bool = False,
    ) -> tuple[object, ...]:
        bridge_kind = "ffn_post_attn_pre"
        if include_qkv:
            bridge_kind = "ffn_post_attn_pre_qkv_q" if include_q else "ffn_post_attn_pre_qkv"
        return (
            id(current_layer),
            id(next_layer),
            bridge_kind,
            self._tensor_sig(ffn_out),
            self._tensor_sig(residual),
            self._tensor_sig(post),
            self._tensor_sig(comb),
            self._tensor_sig(common.freqs_cis) if include_qkv and common is not None else None,
        )

    @staticmethod
    def _max_shapes_per_layer_kind() -> int:
        try:
            return max(int(os.environ.get("DSV4_PREFILL_ISLAND_GRAPH_MAX_SHAPES", "1")), 1)
        except ValueError:
            return 1

    @staticmethod
    def _attn_ffn_bridge_max_tokens() -> int:
        try:
            return int(os.environ.get("DSV4_PREFILL_ISLAND_ATTN_FFN_BRIDGE_MAX_TOKENS", "512"))
        except ValueError:
            return 0

    @staticmethod
    def _attn_ffn_bridge_max_estimated_bytes() -> int:
        try:
            return int(
                os.environ.get(
                    "DSV4_PREFILL_ISLAND_ATTN_FFN_BRIDGE_MAX_ESTIMATED_BYTES",
                    str(256 * 1024 * 1024),
                )
            )
        except ValueError:
            return 0

    @staticmethod
    def _tensor_nbytes(x: torch.Tensor) -> int:
        return int(x.numel()) * int(x.element_size())

    def _attn_ffn_bridge_estimated_bytes(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> int:
        # The graph pins both static inputs and graph-owned outputs with the same
        # shapes for this island. Use a conservative per-layer estimate.
        return 2 * sum(self._tensor_nbytes(t) for t in inputs)

    def _attn_ffn_bridge_within_limits(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> bool:
        max_tokens = self._attn_ffn_bridge_max_tokens()
        if max_tokens <= 0:
            return False
        tokens = int(inputs[0].shape[0]) if inputs[0].dim() > 0 else 0
        if tokens > max_tokens:
            return False
        max_bytes = self._attn_ffn_bridge_max_estimated_bytes()
        if max_bytes <= 0:
            return False
        return self._attn_ffn_bridge_estimated_bytes(inputs) <= max_bytes

    def _disable_bridge_prefix(self, prefix: tuple[int, int, str]) -> None:
        self._bridge_disabled_prefixes.add(prefix)
        for key in list(self._bridge_states):
            if key[:3] == prefix:
                del self._bridge_states[key]
        for key in list(self._bridge_disabled):
            if key[:3] == prefix:
                self._bridge_disabled.remove(key)

    def _shape_count_for_layer_kind(self, layer: object, kind: str) -> int:
        prefix = (id(layer), kind)
        return sum(1 for key in self._states if key[:2] == prefix)

    def _bridge_count_for_pair(
        self, current_layer: object, next_layer: object, kind: str
    ) -> int:
        prefix = (id(current_layer), id(next_layer), kind)
        return sum(1 for key in self._bridge_states if key[:3] == prefix)

    def _attn_ffn_bridge_key(
        self,
        layer: object,
        attn_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> tuple[object, ...]:
        return (
            id(layer),
            id(layer),
            "attn_post_ffn_pre",
            self._tensor_sig(attn_out),
            self._tensor_sig(residual),
            self._tensor_sig(post),
            self._tensor_sig(comb),
        )

    def run_pre(
        self,
        *,
        layer: object,
        kind: str,
        x: torch.Tensor,
        eager_fn: Callable[[torch.Tensor], tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not (x.is_cuda and torch.cuda.is_available()):
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        key = self._key(layer, kind, x)
        prefix = (id(layer), kind)
        if prefix in self._disabled_prefixes:
            return None
        if key in self._disabled:
            return None
        state = self._states.get(key)
        if state is None:
            if self._shape_count_for_layer_kind(
                layer, kind
            ) >= self._max_shapes_per_layer_kind():
                self._disabled_prefixes.add(prefix)
                return None
            try:
                state = self._capture(key, x, eager_fn)
                self._states[key] = state
            except Exception:
                self._disabled_prefixes.add(prefix)
                _LOG.exception("DSV4 prefill island graph capture failed kind=%s", kind)
                return None
        state.static_input.copy_(x, non_blocking=True)
        state.graph.replay()
        state.replay_count += 1
        return state.outputs

    def run_attn_post_ffn_pre_bridge(
        self,
        *,
        layer: object,
        attn_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        inputs = (attn_out, residual, post, comb)
        if not all(t.is_cuda for t in inputs) or not torch.cuda.is_available():
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        # Keep the CP/MegaMoE readiness barrier outside graph capture. The first
        # forward for each layer runs eager and flips this flag.
        if getattr(layer, "_cp_sync_after_attn_done", True) is False:
            return None
        if not self._attn_ffn_bridge_within_limits(inputs):
            return None
        key = self._attn_ffn_bridge_key(layer, attn_out, residual, post, comb)
        kind = "attn_post_ffn_pre"
        prefix = (id(layer), id(layer), kind)
        if prefix in self._bridge_disabled_prefixes:
            return None
        if key in self._bridge_disabled:
            return None
        state = self._bridge_states.get(key)
        if state is None:
            if self._bridge_count_for_pair(
                layer, layer, kind
            ) >= self._max_shapes_per_layer_kind():
                self._disable_bridge_prefix(prefix)
                return None
            try:
                state = self._capture_attn_ffn_bridge(
                    layer=layer,
                    attn_out=attn_out,
                    residual=residual,
                    post=post,
                    comb=comb,
                )
                self._bridge_states[key] = state
            except Exception:
                self._disable_bridge_prefix(prefix)
                _LOG.exception(
                    "DSV4 prefill attn->ffn bridge graph capture failed layer=%s",
                    getattr(layer, "layer_id", "?"),
                )
                return None
        for dst, src in zip(state.static_inputs, inputs):
            dst.copy_(src, non_blocking=True)
        state.graph.replay()
        state.replay_count += 1
        return state.outputs

    def run_ffn_post_attn_pre_bridge(
        self,
        *,
        current_layer: object,
        next_layer: object,
        ffn_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        common: object | None = None,
        include_qkv: bool = False,
        include_q: bool = False,
    ) -> tuple[torch.Tensor, ...] | None:
        inputs = (ffn_out, residual, post, comb)
        if not all(t.is_cuda for t in inputs) or not torch.cuda.is_available():
            return None
        if include_qkv and (common is None or not common.freqs_cis.is_cuda):
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        key = self._bridge_key(
            current_layer,
            next_layer,
            ffn_out,
            residual,
            post,
            comb,
            common=common,
            include_qkv=include_qkv,
            include_q=include_q,
        )
        kind = "ffn_post_attn_pre"
        if include_qkv:
            kind = "ffn_post_attn_pre_qkv_q" if include_q else "ffn_post_attn_pre_qkv"
        prefix = (id(current_layer), id(next_layer), kind)
        if prefix in self._bridge_disabled_prefixes:
            return None
        if key in self._bridge_disabled:
            return None
        state = self._bridge_states.get(key)
        if state is None:
            if self._bridge_count_for_pair(
                current_layer, next_layer, kind
            ) >= self._max_shapes_per_layer_kind():
                self._disable_bridge_prefix(prefix)
                return None
            try:
                state = self._capture_bridge(
                    current_layer=current_layer,
                    next_layer=next_layer,
                    ffn_out=ffn_out,
                    residual=residual,
                    post=post,
                    comb=comb,
                    common=common,
                    include_qkv=include_qkv,
                    include_q=include_q,
                )
                self._bridge_states[key] = state
            except Exception:
                self._disable_bridge_prefix(prefix)
                _LOG.exception(
                    "DSV4 prefill bridge graph capture failed layer=%s next=%s",
                    getattr(current_layer, "layer_id", "?"),
                    getattr(next_layer, "layer_id", "?"),
                )
                return None
        for dst, src in zip(state.static_inputs, inputs):
            dst.copy_(src, non_blocking=True)
        if include_qkv and state.static_freqs_cis is not None:
            state.static_freqs_cis.copy_(common.freqs_cis, non_blocking=True)
        state.graph.replay()
        state.replay_count += 1
        return state.outputs

    def _capture(
        self,
        key,
        x: torch.Tensor,
        eager_fn: Callable[[torch.Tensor], tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> _IslandState:
        static_input = torch.empty_like(x)
        static_input.copy_(x, non_blocking=True)
        torch.cuda.synchronize(static_input.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, x_pre, post, comb = eager_fn(static_input)
        torch.cuda.synchronize(static_input.device)
        return _IslandState(
            static_input=static_input, graph=graph, outputs=(x_pre, post, comb)
        )

    def _capture_bridge(
        self,
        *,
        current_layer: object,
        next_layer: object,
        ffn_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        common: object | None = None,
        include_qkv: bool = False,
        include_q: bool = False,
    ) -> _BridgeState:
        static_inputs = (
            torch.empty_like(ffn_out),
            torch.empty_like(residual),
            torch.empty_like(post),
            torch.empty_like(comb),
        )
        for dst, src in zip(static_inputs, (ffn_out, residual, post, comb)):
            dst.copy_(src, non_blocking=True)
        static_freqs_cis = None
        if include_qkv:
            assert common is not None
            static_freqs_cis = torch.empty_like(common.freqs_cis)
            static_freqs_cis.copy_(common.freqs_cis, non_blocking=True)
        torch.cuda.synchronize(ffn_out.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bridge_h = current_layer.prefill_fast_ffn_post(*static_inputs)
            _, x_pre, next_post, next_comb = next_layer.prefill_fast_attn_pre(bridge_h)
            if include_qkv:
                local_qkv = next_layer.attn.prefill_fast_compute_local_qkv(
                    x_pre,
                    common,
                    freqs_cis=static_freqs_cis,
                    include_q=include_q,
                )
        torch.cuda.synchronize(ffn_out.device)
        outputs = (bridge_h, x_pre, next_post, next_comb)
        if include_qkv:
            outputs = outputs + tuple(local_qkv)
        return _BridgeState(
            static_inputs=static_inputs,
            static_freqs_cis=static_freqs_cis,
            graph=graph,
            outputs=outputs,
        )

    def _capture_attn_ffn_bridge(
        self,
        *,
        layer: object,
        attn_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> _BridgeState:
        static_inputs = (
            torch.empty_like(attn_out),
            torch.empty_like(residual),
            torch.empty_like(post),
            torch.empty_like(comb),
        )
        for dst, src in zip(static_inputs, (attn_out, residual, post, comb)):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(attn_out.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, _, attn_hc_post, _ = layer._prefill_fast_hc_impls()
            ffn_residual = attn_hc_post(*static_inputs)
            _, x_pre, ffn_post, ffn_comb = layer.prefill_fast_ffn_pre(ffn_residual)
        torch.cuda.synchronize(attn_out.device)
        return _BridgeState(
            static_inputs=static_inputs,
            static_freqs_cis=None,
            graph=graph,
            outputs=(ffn_residual, x_pre, ffn_post, ffn_comb),
        )
