"""Model-owned TP2 decode reductions and vocabulary partitioning.

Eligible decode models use draft layer reductions, PLE and token embedding together.
It does not change the generic collective policy or model Graph capture sizes.
"""

import logging

import torch
import torch.distributed as dist
from torch import nn

from rtp_llm.config.qwen4_exp_config import resolve_decode_communication_mode
from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.all_reduce import get_all_reduce
from rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce import (
    IpcInitializationError,
)
from rtp_llm.models_py.modules.qwen4_exp.decode_vocab_weights import PreparedVocabShard
from rtp_llm.models_py.modules.qwen4_exp.tp_reduce import Qwen4ReduceResource


class Qwen4DecodeReduce(Qwen4ReduceResource):
    """An ordered target/draft workspace for every N in [1, 64].

    Target and draft execute serially on the engine's communication stream.
    They share IPC storage, while each model has its own close state.
    """

    def __init__(self, communicator):
        super().__init__(communicator, max_bytes=64 * 2560 * 2)

    def share(self):
        if self.closed:
            raise ValueError("cannot share a closed decode IPC workspace")
        reducer = object.__new__(type(self))
        reducer.workspace = self.workspace
        reducer.closed = False
        return reducer

    def __call__(self, tensor, group):
        if (
            not self.closed
            and group == collective_torch.Group.TP
            and tensor.is_cuda
            and tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
            and 1 <= tensor.shape[0] <= 64
            and tensor.shape[1] == 2560
        ):
            # Rank-local storage offsets must not change the agreed backend.
            # The workspace stages unaligned addresses; normalize strides here.
            source = tensor if tensor.is_contiguous() else tensor.contiguous()
            return self.workspace.all_reduce(source)
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
    mode = resolve_decode_communication_mode(config)
    eligible = (
        config.hidden_size == 2560
        and not communicator.disabled
        and communicator.cpp_small_enabled
    )
    inherited = hasattr(config, "_qwen4_target_decode_reduce")
    source = getattr(config, "_qwen4_target_decode_reduce", None)
    state = "new" if not inherited else "fallback" if source is None else "reuse"
    if source is not None and (
        source.closed
        or source.workspace.group is not communicator.group
        or source.workspace.device != communicator.device
    ):
        state = "invalid"
    peers = [None] * 2
    dist.all_gather_object(peers, (mode, eligible, state), group=communicator.group)
    if any(
        p[0] not in ("default", "vocab_parallel_ipc") or p[0] != mode for p in peers
    ):
        raise ValueError("qwen4_decode_communication_mode must match on every TP rank")
    if any(p[2] != state for p in peers):
        raise ValueError("Qwen4 target IPC state must match on every TP rank")
    if state == "invalid":
        raise ValueError(
            "Qwen4 draft has a closed or incompatible target IPC workspace"
        )
    if mode == "default" or not all(p[1] for p in peers):
        if state == "reuse":
            raise ValueError(
                "Qwen4 draft cannot change initialized target IPC strategy"
            )
        return None
    if state == "fallback":
        return None
    if state == "reuse":
        # Target has already published its vocabulary layout. Reuse its
        # completed allocation instead of risking a second setup failure after
        # the shared vocabulary has been repartitioned.
        reducer = source.share()
    else:
        try:
            reducer = Qwen4DecodeReduce(communicator)
        except IpcInitializationError:
            # This exception is raised only after every rank agrees setup
            # failed. No vocabulary has been changed yet; draft inherits None.
            logging.warning(
                "Qwen4 decode IPC setup failed; retaining default collectives"
            )
            return None
    logging.info(
        "Qwen4 decode custom AllReduce enabled: model=%s rows=1..64 hidden=2560 "
        "mode=vocab_parallel_ipc embedding=vocabulary PLE/draft=custom_all_reduce; "
        "embedding weight updates require restart",
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
