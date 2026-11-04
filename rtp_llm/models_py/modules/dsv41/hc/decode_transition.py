"""Forward-local delayed-HC transitions between decode blocks."""

from __future__ import annotations

import os
from typing import NamedTuple

import torch

from rtp_llm.device.device_type import is_sm90
from rtp_llm.models_py.modules.dsv4 import _record_tensor
from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv41.hc.v41_mega_mhc import try_fused_post_pre
from rtp_llm.models_py.modules.dsv41.moe.shared_expert import PendingCombine

# ``combine`` is only passed when a deferred pair exists, so the fused-seam
# signature stays backward compatible for every other caller.
_NO_COMBINE_KW: dict[str, object] = {}


class PendingDecodePost(NamedTuple):
    # Either the materialized sublayer output or the deferred routed/shared
    # pair the fused boundary rebuilds as bf16(routed + shared).
    output: "torch.Tensor | PendingCombine"
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    previous: DelayedHCUnit

    def materialize(self) -> torch.Tensor:
        output = self.output
        if isinstance(output, PendingCombine):
            output = output.materialize()
        return self.previous.post(output, self.residual, self.post, self.comb)


def _enabled() -> bool:
    return (
        os.environ.get("DSV41_FUSED_CROSS_LAYER_MHC", "1") == "1"
        and os.environ.get("DSV41_MEGA_MHC", "1") != "0"
    )


def _plain_decode_block(layer) -> bool:
    from rtp_llm.models_py.modules.dsv41.block import Block

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


def _can_quantize_attention_input(layer, x, attn_fn) -> bool:
    if attn_fn is not None:
        return False
    from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import is_supported
    from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8

    attn = layer.attn
    return type(attn) is AttentionV41FP8 and is_supported(
        getattr(attn, "wq_a_wkv", None), x, attn.q_norm, attn.q_lora_rank
    )


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
    routing_metadata=None,
    capture_aux_hidden=None,
) -> torch.Tensor | PendingDecodePost:
    """Fuse only an unobserved layer boundary; retain the old call otherwise.

    Pending tensors live in this invocation's layer loop. Graph capture records
    their producer and consumer on the current stream; replay does not reuse a
    Python cache of values computed by a preceding forward.
    """
    prepared = None
    if isinstance(hidden, PendingDecodePost):
        if _enabled() and _plain_decode_block(layer) and layer.engram is None:
            output = hidden.output
            combine = None
            if isinstance(output, PendingCombine):
                # Hand the two BF16 operands over instead of the materialized
                # bf16(routed + shared); the SM90 seam rebuilds it inline.
                combine = (output.routed, output.shared)
                output = output.routed
            # The SM90 seam returns a BF16 normalized input. Let the existing
            # QKV linear choose its native BF16 small-M path or its established
            # group-32 FP8 quantizer. MegaMHC's optional FP8 output uses a
            # different minimum scale and would change tiny-input results.
            output_fp8 = _can_quantize_attention_input(
                layer, output, attn_fn
            ) and not is_sm90(output.device)
            prepared = try_fused_post_pre(
                output,
                hidden.residual,
                hidden.post,
                hidden.comb,
                hidden.previous,
                layer.attn_hc,
                layer.attn_norm,
                output_fp8=output_fp8,
                **(_NO_COMBINE_KW if combine is None else {"combine": combine}),
            )
        hidden = hidden.materialize() if prepared is None else prepared[0]
    if capture_aux_hidden is not None:
        # The aux-hidden hook needs a materialized residual. Capturing it here
        # lets the boundary stay deferred: the fused seam already produces the
        # same BF16 residual (mhc_sm90_vllm_two_kernel.py:99-101), so the hook
        # no longer has to force this layer's post out of line.
        capture_aux_hidden(hidden)
    defer_post = can_defer_decode_post(layer, next_layer, preserve_output)
    kwargs = {"kv_cache": kv_cache}
    if routing_metadata is not None:
        kwargs["routing_metadata"] = routing_metadata
    if attn_fn is not None:
        kwargs["attn_fn"] = attn_fn
    if prepared is not None or defer_post:
        kwargs["_prepared_attn"] = prepared
        kwargs["_defer_ffn_post"] = defer_post
    return layer.forward_decode(hidden, attn_metadata, input_ids, **kwargs)
