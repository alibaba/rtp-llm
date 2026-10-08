"""BF16 Native MTP projection collectives, independent of target FP8 storage."""

import logging

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, _get_group, all_gather
from rtp_llm.models_py.distributed.custom_all_gather import create_custom_all_gather
from rtp_llm.models_py.distributed.push_reduce_scatter import create_push_reduce_scatter
from rtp_llm.models_py.modules.kimi_k3.collectives import reduce_scatter


class KimiK3MtpBf16Collectives:
    """Initialize private staging and Push RS workspaces before Graph capture.

    Native MTP runs on the model compute stream. Its Q1 proposal and fixed-Q
    update share these workspaces serially, as in feat/k3_dev. Real long prompt
    initialization outside the configured Decode capacity keeps the existing
    collective path. Target projections never use this instance.
    """

    def __init__(self, device, *, max_tokens, hidden_size):
        group = _get_group(Group.TP)
        self.tp_size = group.size()
        self.max_tokens = int(max_tokens)
        self.hidden_size = int(hidden_size)
        if self.max_tokens <= 0 or self.max_tokens % self.tp_size:
            raise ValueError("MTP BF16 collective capacity must be divisible by TP")
        self.gather = create_custom_all_gather(
            group,
            device,
            max_m=self.max_tokens,
            k=self.hidden_size,
            fp8=False,
            staging_only=True,
        )
        self.scatter = create_push_reduce_scatter(
            group,
            device,
            max_m=self.max_tokens,
            n=self.hidden_size,
        )
        self.output = (
            torch.empty(
                (self.max_tokens // self.tp_size, self.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            if self.scatter is not None
            else None
        )
        logging.info(
            "[K3_MTP_BF16_TP] TP%d max_tokens=%d hidden=%d AG=%s RS=%s",
            self.tp_size,
            self.max_tokens,
            self.hidden_size,
            "custom_staging" if self.gather is not None else "nccl",
            "push" if self.scatter is not None else "nccl",
        )

    def all_gather(self, local_input):
        physical_tokens = local_input.shape[0] * self.tp_size
        if self.gather is None or physical_tokens > self.max_tokens:
            return all_gather(local_input, Group.TP)
        if (
            local_input.dtype != torch.bfloat16
            or local_input.shape[1] != self.hidden_size
        ):
            raise ValueError("MTP staging AG requires BF16 hidden states")
        return self.gather.all_gather(local_input, staging=True)

    def reduce_scatter(self, partial):
        if self.scatter is None or partial.shape[0] > self.max_tokens:
            return reduce_scatter(partial, Group.TP)
        if (
            partial.dtype != torch.bfloat16
            or partial.shape[1] != self.hidden_size
            or partial.shape[0] % self.tp_size
        ):
            raise ValueError("MTP Push RS requires request-padded BF16 projections")
        output = self.output[: partial.shape[0] // self.tp_size]
        self.scatter.reduce_scatter(partial.contiguous(), output)
        return output
