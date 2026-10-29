"""GLM-5.3 router projection in FP32, with bounded activation storage."""

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.distributed.sequence_parallel import TokenShardLayout
from rtp_llm.models_py.modules.dsv4.forward_metadata import (
    metadata_cache,
    tokenwise_router_enabled,
)

_PREFILL_PROJECTION_ROWS = 256


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
        if metadata_cache() is not None and not tokenwise_router_enabled():
            return self._fixed_projection(hidden_states, _PREFILL_PROJECTION_ROWS)
        # Match the one-token Decode projection throughout Decode and target
        # verification, including batches of multiple requests. cuBLAS may
        # otherwise change its FP32 reduction with M; the last-bit differences
        # can cross a later FP8 KV rounding boundary. Keep the small direct-call
        # path consistent when no model-forward scope is installed.
        if hidden_states.shape[0] > 1 and (
            hidden_states.shape[0] <= 4 or tokenwise_router_enabled()
        ):
            return torch.cat(
                [F.linear(row.float(), self.weight) for row in hidden_states.split(1)],
                dim=0,
            )
        if hidden_states.shape[0] <= self.chunk_rows:
            return F.linear(hidden_states.float(), self.weight)
        return torch.cat(
            [
                F.linear(x.float(), self.weight)
                for x in hidden_states.split(self.chunk_rows)
            ],
            dim=0,
        )

    def _fixed_projection(self, hidden_states: torch.Tensor, rows: int) -> torch.Tensor:
        # The final cuBLAS chunk otherwise changes M with the request batch.
        # Fixed 256-row GEMMs preserve the FP32 reduction at every token,
        # including shard boundaries, and avoid the less accurate large-M path.
        from rtp_llm.models_py.modules.glm53_fixed_projection_graph import (
            fixed_projection_graph,
        )

        projected = fixed_projection_graph(hidden_states, self.weight, rows)
        if projected is not None:
            return projected
        tokens = hidden_states.shape[0]
        output = hidden_states.new_empty(
            (tokens, self.weight.shape[0]), dtype=torch.float32
        )
        for begin in range(0, tokens, rows):
            count = min(rows, tokens - begin)
            chunk = hidden_states.narrow(0, begin, count)
            if count < rows:
                padded = hidden_states.new_zeros((rows, hidden_states.shape[1]))
                padded[:count].copy_(chunk)
                chunk = padded
            output[begin : begin + count].copy_(
                F.linear(chunk.float(), self.weight)[:count]
            )
        return output

    def forward_fixed_rows(self, hidden_states: torch.Tensor, rows: int):
        if rows > self.chunk_rows or metadata_cache() is None:
            return None
        # Preserve explicit fixed-M overrides independently of the native
        # Prefill policy, including when projection graph replay is disabled.
        return self._fixed_projection(hidden_states, rows)

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
        if metadata_cache() is not None and not tokenwise_router_enabled():
            # Fixed-M Prefill projections can consume local rows directly;
            # no padding to the global router's variable final chunk is needed.
            output = self._fixed_projection(hidden_states, _PREFILL_PROJECTION_ROWS)
            output[valid:].zero_()
            return output
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
            logits = self(chunk)
            output.narrow(0, local_first, count).copy_(
                logits.narrow(0, chunk_first, count)
            )
        return output
