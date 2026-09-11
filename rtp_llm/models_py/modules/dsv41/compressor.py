"""Explicit V4.1 owner compression with request-local ratio2 carry.

This eager CUDA component preserves the official projection/pooling/norm
boundaries. CP row assembly, speculative rollback, scheduling and Graph capture
are separate integration work. It never publishes a cache checkpoint itself.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv41.cache_layout import (
    CacheIdentity,
    CacheLayout,
    CacheRegion,
    layer_sources,
)
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages
from rtp_llm.models_py.modules.dsv41.compact_writer import (
    CompactWriteResult,
    encode_compact,
    write_compact,
)
from rtp_llm.models_py.modules.dsv41.math import hc_pre, rms_norm


def is_supported(values: torch.Tensor) -> bool:
    return (
        values.is_cuda
        and values.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(values.device)[0] == 10
    )


def _require_execution(values: torch.Tensor) -> None:
    if os.environ.get("DSV41_OWNER_COMPRESSOR", "0") != "1":
        raise RuntimeError("owner compressor requires DSV41_OWNER_COMPRESSOR=1")
    if not is_supported(values):
        raise RuntimeError("owner compressor requires BF16 on a Blackwell CUDA GPU")
    with torch.cuda.device(values.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "this eager compressor component does not support Graph capture"
            )
    if torch.is_autocast_enabled() or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(
            "official compressor FP32 boundaries require autocast and TF32 disabled"
        )


def _owner_ratio(layer: int) -> int:
    if type(layer) is not int:
        raise ValueError("owner layer must be an integer")
    source = layer_sources(layer)
    if not source.writes_global or not source.writes_index_k:
        raise ValueError("only L2/L8/L14/L20 may construct an owner compressor")
    return source.ratio


def _norm_weight(weight: torch.Tensor, dimension: int, device: torch.device) -> None:
    if (
        weight.shape != (dimension,)
        or weight.dtype not in (torch.bfloat16, torch.float32)
        or weight.device != device
    ):
        raise ValueError(
            "norm weight must match its complete latent dimension and device"
        )


@dataclass(frozen=True)
class PairCarry:
    owner_layer: int
    request_id: str
    identity: CacheIdentity
    next_position: int
    partial_kv: Optional[torch.Tensor] = None
    partial_score: Optional[torch.Tensor] = None

    @classmethod
    def empty(
        cls,
        owner_layer: int,
        request_id: str,
        identity: CacheIdentity,
        next_position: int = 0,
    ) -> PairCarry:
        if _owner_ratio(owner_layer) != 2 or not request_id:
            raise ValueError(
                "ratio2 carry needs an owner and an explicit request identity"
            )
        if (
            type(next_position) is not int
            or not 0 <= next_position <= 1048576
            or next_position % 2
        ):
            raise ValueError(
                "an empty pair is valid only at an even materialized boundary"
            )
        return cls(owner_layer, request_id, identity, next_position)

    def validate(
        self,
        owner: int,
        request_id: str,
        identity: CacheIdentity,
        position: int,
        device: torch.device,
    ) -> None:
        if (
            self.owner_layer != owner
            or self.request_id != request_id
            or self.identity != identity
            or self.next_position != position
        ):
            raise ValueError(
                "ratio2 carry has a different owner/request/cache identity or position"
            )
        if position % 2:
            for tensor in (self.partial_kv, self.partial_score):
                if (
                    tensor is None
                    or tensor.shape != (512,)
                    or tensor.dtype != torch.float32
                    or tensor.device != device
                ):
                    raise ValueError(
                        "odd continuation needs its real FP32 partial KV and score"
                    )
        elif self.partial_kv is not None or self.partial_score is not None:
            raise ValueError("even continuation cannot contain a pending ratio2 pair")


@dataclass(frozen=True)
class CompressedLatents:
    owner_layer: int
    request_id: str
    identity: CacheIdentity
    query_start: int
    query_end: int
    unrotated: torch.Tensor
    group_positions: torch.Tensor
    visible_lengths: torch.Tensor
    next_pair: Optional[PairCarry]
    logical_entries_per_page: int


@dataclass(frozen=True)
class CompressorRoPE:
    dimension: int = 64
    theta: float = 160000.0
    original_context: int = 65536
    factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1

    def __post_init__(self) -> None:
        if (
            self.dimension != 64
            or self.theta != 160000.0
            or self.original_context != 65536
            or self.factor != 16.0
            or self.beta_fast != 32
            or self.beta_slow != 1
        ):
            raise ValueError(
                "compressor RoPE must match the fixed V4.1 global/index configuration"
            )

    def frequencies(self, positions: torch.Tensor) -> torch.Tensor:
        if (
            positions.ndim != 1
            or positions.dtype != torch.int64
            or not positions.is_cuda
        ):
            raise ValueError("RoPE positions must be a CUDA int64 vector")
        if torch.any((positions < 0) | (positions >= 1048576)).item():
            raise ValueError("RoPE position exceeds the actual model context limit")
        dimensions = torch.arange(
            0, self.dimension, 2, dtype=torch.float32, device=positions.device
        )
        frequencies = 1.0 / (self.theta ** (dimensions / self.dimension))

        def corrected_dimension(rotations: int) -> float:
            return (
                self.dimension
                * math.log(self.original_context / (rotations * 2 * math.pi))
                / (2 * math.log(self.theta))
            )

        low = max(math.floor(corrected_dimension(self.beta_fast)), 0)
        high = min(math.ceil(corrected_dimension(self.beta_slow)), self.dimension - 1)
        ramp = (
            (
                torch.arange(
                    self.dimension // 2, dtype=torch.float32, device=positions.device
                )
                - low
            )
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        frequencies = frequencies / self.factor * (1 - smooth) + frequencies * smooth
        phases = torch.outer(positions.float(), frequencies)
        return torch.polar(torch.ones_like(phases), phases)


def _rotate(values: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    result = values.clone()
    tail = torch.view_as_complex(values[:, -64:].float().unflatten(-1, (-1, 2)))
    result[:, -64:] = torch.view_as_real(tail * frequencies).flatten(-2)
    return result


@torch.inference_mode()
def prepare_owner_hidden(
    owner_layer: int,
    hidden_hc: torch.Tensor,
    pre_mix: torch.Tensor,
    attn_norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply incoming pre_mix, then attn_norm; L20 must not use an HC mean."""
    _owner_ratio(owner_layer)
    _require_execution(hidden_hc)
    if hidden_hc.ndim != 3 or hidden_hc.shape[1:] != (4, 5120):
        raise ValueError("owner input must contain four 5120-wide HC streams per row")
    if pre_mix.device != hidden_hc.device:
        raise ValueError("pre_mix and HC input must share a device")
    _norm_weight(attn_norm_weight, 5120, hidden_hc.device)
    return rms_norm(hc_pre(hidden_hc, pre_mix), attn_norm_weight, eps=1e-20)


class OwnerCompressor(nn.Module):
    def __init__(
        self,
        owner_layer: int,
        wkv: torch.Tensor,
        norm_weight: torch.Tensor,
        wgate: Optional[torch.Tensor] = None,
        *,
        layout: Optional[CacheLayout] = None,
    ):
        super().__init__()
        self.owner_layer = owner_layer
        self.ratio = _owner_ratio(owner_layer)
        self.layout = layout or CacheLayout()
        required_dtype = torch.float32 if self.ratio == 2 else torch.bfloat16
        if wkv.shape != (512, 5120) or wkv.dtype != required_dtype or not wkv.is_cuda:
            raise ValueError(
                "owner wkv requires the official 512x5120 geometry and ratio-specific dtype"
            )
        _norm_weight(norm_weight, 512, wkv.device)
        if self.ratio == 2:
            if (
                wgate is None
                or wgate.shape != wkv.shape
                or wgate.dtype != torch.float32
                or wgate.device != wkv.device
            ):
                raise ValueError(
                    "ratio2 requires FP32 wgate with the same geometry as wkv"
                )
        elif wgate is not None:
            raise ValueError("ratio1 has no gate or pair state")
        self.register_buffer("wkv", wkv)
        self.register_buffer("norm_weight", norm_weight)
        self.register_buffer("wgate", wgate)

    @torch.inference_mode()
    def forward(
        self,
        normalized_hidden: torch.Tensor,
        *,
        start_pos: int,
        request_id: str,
        identity: CacheIdentity,
        pair: Optional[PairCarry] = None,
    ) -> CompressedLatents:
        _require_execution(normalized_hidden)
        if (
            normalized_hidden.ndim != 2
            or normalized_hidden.shape[1] != 5120
            or normalized_hidden.device != self.wkv.device
            or not request_id
            or identity.layout_fingerprint != self.layout.fingerprint
        ):
            raise ValueError(
                "owner input, request identity or typed cache layout does not match"
            )
        rows = normalized_hidden.shape[0]
        if type(start_pos) is not int:
            raise ValueError("start_pos must be an absolute integer token position")
        end = start_pos + rows
        if start_pos < 0 or end > 1048576:
            raise ValueError("owner rows must remain within the model context limit")
        if self.ratio == 1 and pair is not None:
            raise ValueError("ratio1 cannot consume ratio2 carry")
        if self.ratio == 2:
            if pair is None:
                if start_pos != 0:
                    raise ValueError(
                        "continuation requires explicit prior carry or an even restored boundary"
                    )
                pair = PairCarry.empty(self.owner_layer, request_id, identity)
            pair.validate(
                self.owner_layer,
                request_id,
                identity,
                start_pos,
                normalized_hidden.device,
            )

        positions = torch.arange(
            start_pos, end, dtype=torch.int64, device=normalized_hidden.device
        )
        visible = ((positions + 1) // self.ratio).to(torch.int32)
        if rows == 0:
            return CompressedLatents(
                self.owner_layer,
                request_id,
                identity,
                start_pos,
                end,
                normalized_hidden.new_empty((0, 512)),
                positions,
                visible,
                pair,
                self.layout.token_block_size // self.ratio,
            )
        if self.ratio == 1:
            latent = rms_norm(
                F.linear(normalized_hidden, self.wkv), self.norm_weight, eps=1e-20
            )
            next_pair = None
        else:
            assert self.wgate is not None and pair is not None
            inputs = normalized_hidden.float()
            kv, score = F.linear(inputs, self.wkv), F.linear(inputs, self.wgate)
            if start_pos % 2:
                assert pair.partial_kv is not None and pair.partial_score is not None
                kv = torch.cat((pair.partial_kv.unsqueeze(0), kv), dim=0)
                score = torch.cat((pair.partial_score.unsqueeze(0), score), dim=0)
            cutoff = kv.shape[0] - end % 2
            # Clone only the pending row. Old request/checkpoint carry remains
            # independent from this forward's temporary projection storage.
            next_pair = PairCarry(
                self.owner_layer,
                request_id,
                identity,
                end,
                kv[-1].clone() if end % 2 else None,
                score[-1].clone() if end % 2 else None,
            )
            grouped_kv = kv[:cutoff].reshape(-1, 2, 512)
            grouped_score = score[:cutoff].reshape(-1, 2, 512)
            pooled = (grouped_kv * grouped_score.softmax(dim=1)).sum(dim=1)
            # The BF16 round-trip precedes norm, as in the original Compressor.
            latent = rms_norm(
                pooled.to(normalized_hidden.dtype), self.norm_weight, eps=1e-20
            )
        first_group = start_pos // self.ratio
        group_positions = (
            torch.arange(
                first_group,
                first_group + latent.shape[0],
                dtype=torch.int64,
                device=latent.device,
            )
            * self.ratio
        )
        return CompressedLatents(
            self.owner_layer,
            request_id,
            identity,
            start_pos,
            end,
            latent,
            group_positions,
            visible,
            next_pair,
            self.layout.token_block_size // self.ratio,
        )


@dataclass(frozen=True)
class OwnerPageBinding:
    owner_layer: int
    identity: CacheIdentity
    pages: CompactPages


@dataclass(frozen=True)
class OwnerKVRows:
    source: CompressedLatents
    global_values: torch.Tensor
    index_values: torch.Tensor

    def encode(self) -> tuple[CompactWriteResult, CompactWriteResult]:
        return (
            encode_compact(self.global_values, CacheRegion.GLOBAL),
            encode_compact(self.index_values, CacheRegion.INDEX_K),
        )

    def store(
        self,
        global_binding: OwnerPageBinding,
        index_binding: OwnerPageBinding,
        global_slots: torch.Tensor,
        index_slots: torch.Tensor,
    ) -> tuple[CompactWriteResult, CompactWriteResult]:
        for binding, region in (
            (global_binding, CacheRegion.GLOBAL),
            (index_binding, CacheRegion.INDEX_K),
        ):
            if (
                binding.owner_layer != self.source.owner_layer
                or binding.identity != self.source.identity
                or binding.pages.region != region
                or binding.pages.entries_per_page
                != self.source.logical_entries_per_page
            ):
                raise ValueError(
                    "compact destination has a different physical owner, region or cache identity"
                )
            binding.pages.validate(self.source.unrotated.device)
        for slots in (global_slots, index_slots):
            if (
                slots.device != self.source.unrotated.device
                or slots.dtype not in (torch.int32, torch.int64)
                or slots.shape != (self.source.unrotated.shape[0],)
                or not slots.is_contiguous()
            ):
                raise ValueError(
                    "each physical pool requires its own contiguous CUDA slot map"
                )
        # Both statuses must succeed before the caller publishes the joint state.
        return (
            write_compact(self.global_values, global_binding.pages, global_slots),
            write_compact(self.index_values, index_binding.pages, index_slots),
        )


@torch.inference_mode()
def prepare_owner_kv(
    source: CompressedLatents,
    index_weight: torch.Tensor,
    index_norm: torch.Tensor,
    rope: Optional[CompressorRoPE] = None,
) -> OwnerKVRows:
    """Produce owner index K from the untouched normed main latent, then RoPE each."""
    _owner_ratio(source.owner_layer)
    _require_execution(source.unrotated)
    if (
        index_weight.shape != (128, 512)
        or index_weight.dtype != torch.bfloat16
        or index_weight.device != source.unrotated.device
    ):
        raise ValueError("owner index projection requires BF16 128x512 weights")
    _norm_weight(index_norm, 128, source.unrotated.device)
    frequencies = (rope or CompressorRoPE()).frequencies(source.group_positions)
    index_k = rms_norm(F.linear(source.unrotated, index_weight), index_norm, eps=1e-20)
    return OwnerKVRows(
        source, _rotate(source.unrotated, frequencies), _rotate(index_k, frequencies)
    )
