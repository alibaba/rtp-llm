"""Decode vocabulary conversion, shared ownership and restart-only publication."""

import logging
import weakref
from typing import NamedTuple

import torch
import torch.distributed as dist

from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX
from rtp_llm.utils.model_weight import W


class _VocabLayout(NamedTuple):
    group: object
    rank: int
    vocab_size: int
    scales: object


class PreparedVocabShard:
    def __init__(self, weight, layout):
        self.weight = weight
        self.scales = layout.scales
        self.vocab_size = layout.vocab_size
        self.start = layout.rank * weight.shape[0]
        # The converted owner carries enough metadata for a later draft even
        # if its previous Python consumer has been destroyed. Neither the
        # metadata nor this weak cache retains the hidden-sharded source.
        weight._rtp_decode_vocab_layout = layout
        weight._rtp_embedding_vocab_size = layout.vocab_size
        weight._rtp_decode_vocab_shard = weakref.ref(self)


@torch.no_grad()
def _repartition_vocab(weight, group, weight_scale=None):
    """Exchange hidden shards once at startup; never requantize FP8 weights.

    A source FP8 hidden shard has its own scalar scale. The vocabulary shard
    retains those scales per hidden slice, reproducing the original lookup's
    BF16 conversions. Shared target/draft weights share the converted storage.
    Converted owners are shared with the draft. All ranks agree on the
    layout before any rank skips the exchange.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("repartition vocabulary before Graph capture")
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    fp8 = weight.dtype == torch.float8_e4m3fn
    layout = getattr(weight, "_rtp_decode_vocab_layout", None)
    valid = (
        size == 2
        and weight.is_cuda
        and weight.ndim == 2
        and weight.is_contiguous()
        and min(weight.shape) > 0
        and weight.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and (
            (
                layout is not None
                and layout.group is group
                and layout.rank == rank
                and weight.shape[0] == (layout.vocab_size + size - 1) // size
            )
            or (
                layout is None
                and (
                    not fp8 or (weight_scale is not None and weight_scale.numel() == 1)
                )
            )
        )
    )
    peers = [None] * size
    policy = (
        valid,
        tuple(weight.shape),
        str(weight.dtype),
        layout is not None,
        None if layout is None else layout.vocab_size,
    )
    dist.all_gather_object(peers, policy, group=group)
    if not all(p[0] and p[1:] == policy[1:] for p in peers):
        raise ValueError(
            "decode vocabulary requires matching contiguous BF16/FP8 TP2 layouts"
        )
    if layout is not None:
        logging.info(
            "Reusing Qwen4 decode vocabulary shard for shared target/draft weights"
        )
        cached = getattr(weight, "_rtp_decode_vocab_shard", lambda: None)()
        return cached if cached is not None else PreparedVocabShard(weight, layout)

    vocab_size, local_width = weight.shape
    rows = (vocab_size + size - 1) // size
    raw = weight.view(torch.uint8) if fp8 else weight
    if rows * size != vocab_size:
        raw = torch.nn.functional.pad(raw, (0, 0, 0, rows * size - vocab_size))
    received = torch.empty_like(raw)
    dist.all_to_all_single(received, raw, group=group)
    converted = received.reshape(size, rows, local_width).transpose(0, 1).contiguous()
    converted = converted.reshape(rows, size * local_width)
    if fp8:
        converted = converted.view(torch.float8_e4m3fn)
        scale = weight_scale.to(device=weight.device, dtype=torch.float32).reshape(1)
        scales = torch.empty(size, device=weight.device, dtype=torch.float32)
        dist.all_gather_into_tensor(scales, scale, group=group)
    else:
        scales = None
    resource = PreparedVocabShard(
        converted, _VocabLayout(group, rank, vocab_size, scales)
    )
    weight._rtp_requires_restart_for_update = True
    converted._rtp_requires_restart_for_update = True
    if weight_scale is not None:
        weight_scale._rtp_requires_restart_for_update = True
    logging.info(
        "Qwen4 decode vocabulary repartitioned: hidden_shard=%s vocab_shard=%s dtype=%s",
        tuple(weight.shape),
        tuple(converted.shape),
        weight.dtype,
    )
    return resource


@torch.no_grad()
def prepare_decode_vocab(
    *, source_weights, compute_weights, group
) -> PreparedVocabShard:
    """Prepare one shared vocabulary owner before constructing its consumers.

    Both loaded and compute dictionaries are required, including when they are
    the same object. Publication replaces only embedding entries, preserving
    other consumers such as a tied output head. Source storage is not retained.
    """
    shard = _repartition_vocab(
        source_weights[W.embedding],
        group,
        source_weights.get(W.embedding + FP8_STORAGE_SCALE_SUFFIX),
    )
    source_weights[W.embedding] = shard.weight
    compute_weights[W.embedding] = shard.weight
    # Protect the loader's scalar FP8 scale owner as well as the converted
    # weight; wrappers may otherwise only mark a reshaped view of the scale.
    for name in (W.embedding, W.embedding + FP8_STORAGE_SCALE_SUFFIX):
        owner = source_weights.get(name)
        if owner is not None:
            owner._rtp_requires_restart_for_update = True
    return shard
