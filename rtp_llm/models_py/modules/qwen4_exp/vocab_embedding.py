"""GPU-resident FP8 vocabulary lookup with BF16 output and TP ordering."""

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.qwen4_exp.ple.ple_gather_triton import gather_local


class Qwen4Fp8VocabEmbedding(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        tp_size: int,
        device: torch.device,
        weight_scale=None,
    ):
        super().__init__()
        if (
            not weight.is_cuda
            or weight.dtype != torch.float8_e4m3fn
            or weight.ndim != 2
            or not weight.is_contiguous()
            or min(weight.shape) <= 0
            or tp_size < 1
            or weight.device != device
        ):
            raise ValueError(
                "FP8 vocabulary requires contiguous CUDA E4M3 rows on its execution device"
            )
        if weight_scale is None or weight_scale.numel() != 1:
            raise ValueError("FP8 vocabulary requires one scale")
        self.weight_scale = weight_scale.to(device=device, dtype=torch.float32).reshape(
            1
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
        self.shards = [flat]
        self.pointer_table = torch.tensor(
            [flat.data_ptr()], dtype=torch.int64, device=weight.device
        )
        self.device = weight.device
        self.weight = self.shards[0].view_as(weight)
        self.weight._rtp_requires_restart_for_update = True
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
            raise ValueError("FP8 vocabulary lookup requires packed CUDA token ids")
        tokens = input.to(torch.int64)
        tokens = torch.where((tokens >= 0) & (tokens < self.vocab_size), tokens, -1)
        ids = tokens.unsqueeze(-1) * self.column_tiles + self.columns
        output = gather_local(
            ids, self.shards, (0,), 1, self.pointer_table, scale=self.weight_scale
        ).view(input.numel(), self.hidden_size)
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
