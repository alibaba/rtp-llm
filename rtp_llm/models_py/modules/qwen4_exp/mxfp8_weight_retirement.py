"""Initialization-only ownership transfer for Qwen4 online MXFP8 projections.

Only known consumers that successfully installed Qwen4Mxfp8Linear are retired.
No shared storage resize, tensor migration, environment reads or forward-time
mutation occurs here. ModelWeights retains name/shape metadata for update errors.
"""

from rtp_llm.models_py.modules.qwen4_exp.mxfp8_linear import Qwen4Mxfp8Linear
from rtp_llm.utils.model_weight import W


def _retire(owner, layer_id, name, projection):
    if not isinstance(projection, Qwen4Mxfp8Linear):
        return
    entries = owner.global_weights if layer_id is None else owner.weights[layer_id]
    if name not in entries:
        return
    seen = set()
    while owner is not None and id(owner) not in seen:
        seen.add(id(owner))
        entries = owner.global_weights if layer_id is None else owner.weights[layer_id]
        if name in entries:
            owner.retire_weight(layer_id, name)
        owner = getattr(owner, "_fp8_storage_owner", None)


def retire_layer_sources(owner, layer_id, layer):
    attention = layer.self_attn
    if isinstance(getattr(attention, "in_proj_fused", None), Qwen4Mxfp8Linear):
        # Both dictionary views keep the same merged BF16 allocation alive.
        for key in (W.linear_attn_qkvz_w, W.linear_attn_ba_w):
            _retire(owner, layer_id, key, attention.in_proj_fused)
    for attr, key in (
        ("in_proj_qkvz", W.linear_attn_qkvz_w),
        ("in_proj_ba", W.linear_attn_ba_w),
        ("out_proj", W.linear_attn_out_w),
        ("qkv_proj", W.attn_qkv_w),
        ("o_proj", W.attn_o_w),
        ("gate", W.attn_gate_w),
    ):
        _retire(owner, layer_id, key, getattr(attention, attr, None))
