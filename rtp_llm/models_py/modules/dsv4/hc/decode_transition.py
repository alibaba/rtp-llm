"""Forward-local delayed-HC transitions between decode blocks."""

from __future__ import annotations

import os
from typing import NamedTuple

import torch

from rtp_llm.models_py.modules.dsv4 import _record_tensor
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv4.hc.v41_mega_mhc import try_fused_post_pre


class PendingDecodePost(NamedTuple):
    output: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    previous: DelayedHCUnit

    def materialize(self) -> torch.Tensor:
        return self.previous.post(self.output, self.residual, self.post, self.comb)


def _enabled() -> bool:
    return (
        os.environ.get("DSV41_FUSED_CROSS_LAYER_MHC", "1") == "1"
        and os.environ.get("DSV41_MEGA_MHC", "1") != "0"
    )


def _plain_decode_block(layer) -> bool:
    from rtp_llm.models_py.modules.dsv4.block import Block

    return type(layer) is Block and not _record_tensor.should_record_layer(
        layer.layer_id
    )


def can_defer_decode_post(layer, next_layer, preserve_output: bool) -> bool:
    if (
        preserve_output
        or not _enabled()
        or not _plain_decode_block(layer)
        or not _plain_decode_block(next_layer)
        or next_layer.engram is not None
        or type(layer.ffn_hc) is not DelayedHCUnit
        or type(next_layer.attn_hc) is not DelayedHCUnit
    ):
        return False
    previous = next_layer.attn_hc._previous_ref
    return previous is not None and previous() is layer.ffn_hc


def forward_decode_layer(
    layer,
    hidden: torch.Tensor | PendingDecodePost,
    attn_metadata,
    input_ids: torch.Tensor,
    *,
    next_layer=None,
    preserve_output: bool = False,
    kv_cache=None,
    attn_fn=None,
) -> torch.Tensor | PendingDecodePost:
    """Fuse only an unobserved layer boundary; retain the old call otherwise.

    Pending tensors live in this invocation's layer loop. Graph capture records
    their producer and consumer on the current stream; replay does not reuse a
    Python cache of values computed by a preceding forward.
    """
    prepared = None
    if isinstance(hidden, PendingDecodePost):
        if _enabled() and _plain_decode_block(layer) and layer.engram is None:
            prepared = try_fused_post_pre(
                hidden.output,
                hidden.residual,
                hidden.post,
                hidden.comb,
                hidden.previous,
                layer.attn_hc,
                layer.attn_norm,
            )
        hidden = hidden.materialize() if prepared is None else prepared[0]
    defer_post = can_defer_decode_post(layer, next_layer, preserve_output)
    kwargs = {"kv_cache": kv_cache}
    if attn_fn is not None:
        kwargs["attn_fn"] = attn_fn
    if prepared is not None or defer_post:
        kwargs["_prepared_attn"] = prepared
        kwargs["_defer_ffn_post"] = defer_post
    return layer.forward_decode(hidden, attn_metadata, input_ids, **kwargs)
