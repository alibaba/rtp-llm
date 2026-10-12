"""Exact pinned-host vocabulary lookup using the existing PLE UVA gather."""

import torch
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.qwen4_exp.ple_gather_triton import (
    gather_local,
    prepare_cpu_uva_shards,
)
from torch import nn


class Qwen4CpuVocabEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor, tp_size: int, device: torch.device):
        super().__init__()
        if (
            weight.device.type != "cpu"
            or not weight.is_pinned()
            or weight.ndim != 2
            or weight.dtype not in (torch.bfloat16, torch.float32)
            or not weight.is_contiguous()
            or min(weight.shape) <= 0
            or tp_size < 1
        ):
            raise ValueError(
                "CPU vocabulary lookup requires pinned contiguous BF16/F32 rows"
            )
        self.tp_size = tp_size
        self.hidden_size = weight.shape[1]
        self.vocab_size = weight.shape[0]
        self.device = device
        self.tile_width = next(
            width
            for width in (256, 128, 64, 32, 16, 8, 4, 2, 1)
            if self.hidden_size % width == 0
        )
        self.column_tiles = self.hidden_size // self.tile_width
        flat = weight.view(-1, self.tile_width)
        self.shards, self.pointer_table = prepare_cpu_uva_shards([flat], [0], 1, device)
        self.device = self.pointer_table.device
        self.weight = self.shards[0].view_as(weight)
        self.columns = torch.arange(
            self.column_tiles, dtype=torch.int64, device=self.device
        )

    def forward(
        self,
        input: torch.Tensor,
        position_ids=None,
        token_types=None,
        text_tokens_mask=None,
    ) -> torch.Tensor:
        if (
            input.device != self.device
            or input.ndim != 1
            or input.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("CPU vocabulary lookup requires packed CUDA token ids")
        tokens = input.to(torch.int64)
        tokens = torch.where((tokens >= 0) & (tokens < self.vocab_size), tokens, -1)
        ids = tokens.unsqueeze(-1) * self.column_tiles + self.columns
        output = gather_local(ids, self.shards, (0,), 1, self.pointer_table).view(
            input.numel(), self.hidden_size
        )
        if text_tokens_mask is not None and text_tokens_mask.numel():
            if (
                text_tokens_mask.device != input.device
                or text_tokens_mask.shape != input.shape
            ):
                raise ValueError("vocabulary text mask must match packed token ids")
            output = torch.where(text_tokens_mask.bool().unsqueeze(-1), output, 0)
        if self.tp_size > 1:
            rows, width = output.shape
            output = all_gather(output, group=Group.TP)
            output = (
                output.reshape(self.tp_size, rows, width)
                .transpose(0, 1)
                .contiguous()
                .reshape(rows, -1)
            )
        return output
