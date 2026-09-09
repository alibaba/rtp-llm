"""GLM-5.3 router projection in FP32, with bounded activation storage."""

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.distributed.sequence_parallel import TokenShardLayout


class Glm53FP32Router(nn.Module):
    def __init__(self, weight: torch.Tensor, chunk_rows: int = 8192):
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.float32:
            raise ValueError(
                "GLM-5.3 router requires a loaded FP32 [hidden, experts] weight"
            )
        if chunk_rows <= 0:
            raise ValueError("router chunk_rows must be positive")
        self.register_buffer("weight", weight.T.contiguous())
        self.chunk_rows = chunk_rows

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("GLM-5.3 FP32 router requires matmul.allow_tf32=False")
        if hidden_states.shape[0] <= self.chunk_rows:
            return F.linear(hidden_states.float(), self.weight)
        return torch.cat(
            [
                F.linear(x.float(), self.weight)
                for x in hidden_states.split(self.chunk_rows)
            ],
            dim=0,
        )

    def forward_shard(
        self, hidden_states: torch.Tensor, layout: TokenShardLayout
    ) -> torch.Tensor:
        """Project local rows with the unsharded router's GEMM shapes.

        Only a boundary chunk needs zero rows in place of another rank's
        tokens. Keeping its original M and row offsets avoids changing the
        FP32 reduction algorithm at a top-k near tie. No remote token data is
        needed. Aligned large shards take the ordinary bounded fast path.
        """
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("GLM-5.3 FP32 router requires matmul.allow_tf32=False")
        if hidden_states.ndim != 2 or hidden_states.shape[0] != layout.local_tokens:
            raise ValueError("router input must match the local token layout")
        begin = layout.local_start
        valid = layout.local_valid_tokens
        end = begin + valid
        if (
            begin < 0
            or valid < 0
            or valid > layout.local_tokens
            or (valid and end > layout.logical_tokens)
        ):
            raise ValueError("invalid router token layout")
        if (
            valid == layout.local_tokens
            and begin % self.chunk_rows == 0
            and (end % self.chunk_rows == 0 or end == layout.logical_tokens)
        ):
            return self(hidden_states)
        output = hidden_states.new_empty(
            (layout.local_tokens, self.weight.shape[0]), dtype=torch.float32
        )
        output[valid:].zero_()
        for chunk_begin in range(
            begin // self.chunk_rows * self.chunk_rows, end, self.chunk_rows
        ):
            chunk_end = min(chunk_begin + self.chunk_rows, layout.logical_tokens)
            first, last = max(begin, chunk_begin), min(end, chunk_end)
            if first >= last:
                continue
            local_first, chunk_first = first - begin, first - chunk_begin
            count = last - first
            local = hidden_states.narrow(0, local_first, count)
            if count == chunk_end - chunk_begin:
                chunk = local
            else:
                chunk = hidden_states.new_zeros(
                    (chunk_end - chunk_begin, hidden_states.shape[1])
                )
                chunk.narrow(0, chunk_first, count).copy_(local)
            logits = F.linear(chunk.float(), self.weight)
            output.narrow(0, local_first, count).copy_(
                logits.narrow(0, chunk_first, count)
            )
        return output
