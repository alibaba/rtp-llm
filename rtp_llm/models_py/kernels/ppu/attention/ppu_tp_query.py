"""Shard qualified 16K Prefill Indexer queries across the TP8 group."""

import torch

from rtp_llm.models_py.distributed import collective_torch as collective


def tp8_identity():
    # Do not trigger lazy communicator initialization or infer TP from WORLD.
    config = getattr(collective, "_parallelism_config", None)
    if (
        config is None
        or int(config.tp_size) != 8
        or not getattr(collective, "_initialized", False)
        or not torch.distributed.is_initialized()
    ):
        return None
    group = collective._get_group(collective.Group.TP)
    if torch.distributed.get_world_size(group) != 8:
        return None
    rank = torch.distributed.get_rank(group)
    if not 0 <= rank < 8:
        return None
    return group, rank


def eligible(q, kv, topk, recording=False):
    return (
        not recording
        and q[0].ndim == 3
        and kv[0].ndim == 2
        and tuple(q[0].shape[:2]) == (16384, 64)
        and kv[0].shape[0] == 8192
        and topk == 512
        and q[0].is_cuda
        and torch.cuda.get_device_name(q[0].device) == "ZW-M890P"
        and not torch.cuda.is_current_stream_capturing()
    )


def try_tp_query_topk(
    q, kv, weights, starts, ends, *, score_fn, topk_fn, topk=512, recording=False
):
    if not eligible(q, kv, topk, recording):
        return None
    identity = tp8_identity()
    if identity is None:
        return None
    _, rank = identity
    begin = rank * 2048
    end = begin + 2048
    scores = score_fn(
        (q[0][begin:end], q[1][begin:end]),
        kv,
        weights[begin:end],
        starts[begin:end],
        ends[begin:end],
        clean_logits=False,
        logits_dtype=torch.bfloat16,
    )
    local = topk_fn(scores, starts[begin:end], ends[begin:end], topk=topk)
    result = collective.all_gather(local.contiguous(), collective.Group.TP)
    assert (
        result.shape == (16384, 512)
        and result.dtype == torch.int32
        and result.is_contiguous()
    )
    return result
