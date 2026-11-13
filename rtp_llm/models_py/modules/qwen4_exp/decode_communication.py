"""Model-owned TP2 decode reductions and vocabulary partitioning.

The opt-in changes draft layer reductions, PLE and token embedding together.
It does not change the generic collective policy or model Graph capture sizes.
"""

import logging
import os

import torch
import torch.distributed as dist
from torch import nn

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.all_reduce import get_all_reduce
from rtp_llm.models_py.modules.qwen4_exp.decode_vocab_weights import PreparedVocabShard
from rtp_llm.models_py.modules.qwen4_exp.tp_reduce import Qwen4ReduceResource


class Qwen4DecodeReduce(Qwen4ReduceResource):
    """One ordered workspace per target/draft model, for every N in [1, 64]."""

    def __init__(self, communicator):
        super().__init__(communicator, max_bytes=64 * 2560 * 2)

    def __call__(self, tensor, group):
        if (
            not self.closed
            and group == collective_torch.Group.TP
            and tensor.is_cuda
            and tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
            and 1 <= tensor.shape[0] <= 64
            and tensor.shape[1] == 2560
            and tensor.is_contiguous()
            and tensor.data_ptr() % 16 == 0
        ):
            return self.workspace.all_reduce(tensor)
        return collective_torch.all_reduce(tensor, group, inplace=False)


def prepare_decode_reduce(config, parallelism):
    role = str(getattr(parallelism, "role_type", "")).rsplit(".", 1)[-1]
    if not (
        role == "DECODE"
        and parallelism.tp_size == 2
        and parallelism.dp_size == 1
        and parallelism.local_world_size == 2
    ):
        return None
    communicator = get_all_reduce()
    if communicator is None:
        return None
    mode = os.getenv("RTP_QWEN4_DECODE_CUSTOM_AR", "0")
    eligible = (
        config.hidden_size == 2560
        and not communicator.disabled
        and communicator.cpp_small_enabled
    )
    peers = [None] * 2
    dist.all_gather_object(peers, (mode, eligible), group=communicator.group)
    if any(p[0] not in ("0", "1") or p[0] != mode for p in peers):
        raise ValueError(
            "RTP_QWEN4_DECODE_CUSTOM_AR must be the same 0 or 1 on every TP rank"
        )
    if mode == "0" or not all(p[1] for p in peers):
        return None
    reducer = Qwen4DecodeReduce(communicator)
    logging.info(
        "Qwen4 decode custom AllReduce enabled: model=%s rows=1..64 hidden=2560",
        "draft" if getattr(config, "is_mtp", False) else "target",
    )
    return reducer


class DecodeVocabEmbedding(nn.Module):
    """Vocabulary-local lookup followed by a full-hidden-width AllReduce."""

    def __init__(self, shard: PreparedVocabShard, reduce):
        super().__init__()
        self.shard = shard
        self.weight = shard.weight
        self.tp_size = 2
        self.reduce = reduce

    def forward(
        self, input, position_ids=None, token_types=None, text_tokens_mask=None
    ):
        if (
            input.device != self.weight.device
            or input.ndim != 1
            or input.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("decode vocabulary requires packed CUDA token ids")
        if (
            text_tokens_mask is not None
            and text_tokens_mask.numel()
            and (
                text_tokens_mask.device != input.device
                or text_tokens_mask.shape != input.shape
            )
        ):
            raise ValueError("vocabulary text mask must match packed token ids")
        from rtp_llm.models_py.triton_kernels.common.decode_vocab_lookup import (
            vocab_rows,
        )

        local = vocab_rows(
            input,
            self.shard.weight,
            self.shard.start,
            self.shard.vocab_size,
            self.shard.scales,
            text_tokens_mask,
        )
        return self.reduce(local, collective_torch.Group.TP)
