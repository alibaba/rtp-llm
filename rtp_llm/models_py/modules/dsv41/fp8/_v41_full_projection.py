"""One BF16 projection for FULL decode's same-input KV and index head weights."""

import torch

from rtp_llm.resident_weights import WeightDict


@torch.no_grad()
def prepare(attn):
    if not (
        attn.is_kv_source
        and attn.is_index_source
        and attn.tp_size == 1
        and attn.compress_ratio in (1, 2)
        and (attn.head_dim, attn.index_n_heads) == (512, 32)
    ):
        return
    weights = [attn.global_wkv]
    if attn.compress_ratio == 2:
        weights.append(attn.global_wgate)
    weights.append(attn.index_weights)
    device = weights[0].device
    if device.type != "cuda" or any(
        w.device != device or w.dtype != torch.bfloat16 or not w.is_contiguous()
        for w in weights
    ):
        return
    if any(w.shape != (512, 5120) for w in weights[:-1]) or weights[-1].shape != (
        32,
        5120,
    ):
        return
    width = sum(w.shape[0] for w in weights)
    combined = getattr(attn, "_full_projection_weight", None)
    if combined is None or combined.device != device or combined.shape != (width, 5120):
        attn._full_projection_weight = torch.cat(weights)
    else:
        # Keep graph-captured weight addresses stable across weight reloads.
        for target, source in zip(
            combined.split([w.shape[0] for w in weights]), weights
        ):
            target.copy_(source)

    # Publish views into the graph-stable storage back to the authoritative
    # weight dictionary, so in-place weight updates also update this projection.
    from rtp_llm.utils.model_weight import W

    names = [("global_wkv", W.v4_compressor_wkv)]
    if attn.compress_ratio == 2:
        names.append(("global_wgate", W.v4_compressor_wgate))
    names.append(("index_weights", W.v4_indexer_weights_proj_w))
    owner = getattr(attn, "_qb_weight_owner", None)
    if isinstance(owner, WeightDict) and owner.manager is not None:
        attn._full_projection_preserve_sources = True
    if getattr(attn, "_full_projection_preserve_sources", False):
        # Keep registered tensors authoritative. prepare() refreshes the derived
        # graph allocation from these unchanged attributes after weight reload.
        return
    for (attribute, key), view in zip(
        names, attn._full_projection_weight.split([w.shape[0] for w in weights])
    ):
        setattr(attn, attribute, view)
        if owner is not None:
            owner[key] = view


def project(attn, x, *, defer_head_cast=False):
    weight = getattr(attn, "_full_projection_weight", None)
    if weight is None or weight.device != x.device:
        return None
    from .compressor import _linear_bf16_bf16_fp32

    projected = _linear_bf16_bf16_fp32(x.view(-1, 5120), weight)
    values = projected[:, :512]
    scores = projected[:, 512:1024] if attn.compress_ratio == 2 else values
    # The overlap path casts after issuing its independent KV writes.
    heads = projected[:, -32:]
    if not defer_head_cast:
        heads = heads.to(torch.bfloat16)
    return values, scores, heads
