"""One BF16 projection for FULL decode's same-input KV and index head weights."""

import torch


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


def project(attn, x):
    weight = getattr(attn, "_full_projection_weight", None)
    if weight is None or weight.device != x.device:
        return None
    from .compressor import _linear_bf16_bf16_fp32

    projected = _linear_bf16_bf16_fp32(x.view(-1, 5120), weight)
    values = projected[:, :512]
    scores = projected[:, 512:1024] if attn.compress_ratio == 2 else values
    # Preserve F.linear's BF16 head-weight boundary before indexer scaling.
    heads = projected[:, -32:].to(torch.bfloat16)
    return values, scores, heads
