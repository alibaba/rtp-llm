"""Cross-layer DSV4 decode post+pre+norm deferral using ``deep_gemm.mega_mhc``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from rtp_llm.models_py.modules.dsv4.hc import mega_mhc as _mega_mhc


@dataclass(frozen=True)
class PendingDecodePost:
    """Deferred FFN ``hc.post`` awaiting fusion into the next layer's attention ``pre`` + norm."""

    sublayer_out: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    hc_unit: Any

    def materialize(self) -> torch.Tensor:
        """Apply the deferred ``hc_unit.post`` when cross-layer fusion is skipped."""
        return self.hc_unit.post(
            self.sublayer_out,
            self.residual,
            self.post,
            self.comb,
        )


def _is_dsv4_block(layer: Any) -> bool:
    from rtp_llm.models_py.modules.dsv4.block import Block

    return type(layer) is Block


def _has_mega_attention_adapter(layer: Any) -> bool:
    return (
        getattr(layer, "_mega_csa_adapter", None) is not None
        or getattr(layer, "_mega_hca_adapter", None) is not None
    )


def can_defer_decode_post(
    layer: Any,
    next_layer: Optional[Any],
    *,
    device: Optional[torch.device] = None,
    preserve_output: bool = False,
) -> bool:
    """Return True when ``layer`` FFN post can defer into non-Mega ``next_layer``.

    Deferral is disabled whenever ``layer``'s output must be materialized
    immediately (for auxiliary hidden capture or after the final layer), or
    when ``next_layer`` uses a Mega attention adapter (whose tiled entry consumes
    a materialized residual tensor directly).
    """
    if preserve_output or next_layer is None:
        return False
    if not (_is_dsv4_block(layer) and _is_dsv4_block(next_layer)):
        return False
    if _has_mega_attention_adapter(next_layer):
        return False
    if device is None:
        norm_weight = getattr(getattr(next_layer, "attn_norm", None), "weight", None)
        device = getattr(norm_weight, "device", None)
    if device is None:
        return False
    return _mega_mhc.is_supported(
        layer.ffn_hc,
        next_layer.attn_hc,
        next_layer.attn_norm,
        device,
    )


def forward_decode_layer(
    layer: Any,
    hidden: Union[torch.Tensor, PendingDecodePost],
    attn_metadata: Any,
    input_ids: torch.Tensor,
    *,
    kv_cache: Any = None,
    defer_post: bool = False,
) -> Union[torch.Tensor, PendingDecodePost]:
    """Execute one decode layer, consuming and/or deferring ``PendingDecodePost``."""
    prepared_attn = None
    layer_input: torch.Tensor
    is_block = _is_dsv4_block(layer)

    if isinstance(hidden, PendingDecodePost):
        fused = None
        if is_block and not _has_mega_attention_adapter(layer):
            fused = _mega_mhc.try_fused_post_pre(
                hidden.sublayer_out,
                hidden.residual,
                hidden.post,
                hidden.comb,
                previous=hidden.hc_unit,
                next_hc=layer.attn_hc,
                norm=layer.attn_norm,
            )
        if fused is not None:
            layer_input, x_pre, post, comb = fused
            prepared_attn = (x_pre, post, comb)
        else:
            layer_input = hidden.materialize()
    else:
        layer_input = hidden

    if is_block and (prepared_attn is not None or defer_post):
        return layer.forward_decode(
            layer_input,
            attn_metadata,
            input_ids,
            kv_cache=kv_cache,
            _prepared_attn=prepared_attn,
            _defer_ffn_post=bool(defer_post),
        )
    return layer.forward_decode(
        layer_input,
        attn_metadata,
        input_ids,
        kv_cache=kv_cache,
    )


__all__ = [
    "PendingDecodePost",
    "can_defer_decode_post",
    "forward_decode_layer",
]
