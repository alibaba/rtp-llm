"""GLM-5.3 router projection in FP32, with bounded activation storage."""

from functools import lru_cache
import os
from threading import get_ident

import torch
import torch.nn.functional as F
from rtp_llm.models_py.distributed.sequence_parallel import TokenShardLayout
from torch import nn


@lru_cache(maxsize=128)
def _zigzag_batches(fragments, chunk_rows, logical_tokens):
    chunks = {}
    for local, canonical, length in fragments:
        while length:
            begin = canonical // chunk_rows * chunk_rows
            count = min(length, begin + chunk_rows - canonical)
            chunks.setdefault(begin, []).append((local, canonical - begin, count))
            local += count
            canonical += count
            length -= count
    batches = []
    for begin, pieces in sorted(chunks.items()):
        rows = min(chunk_rows, logical_tokens - begin)
        for batch_rows, batch in batches:
            if rows == batch_rows and all(
                start + count <= other or other + size <= start
                for _, start, count in pieces for _, other, size in batch
            ):
                batch.extend(pieces)
                break
        else:
            batches.append((rows, list(pieces)))
    return tuple((rows, tuple(pieces)) for rows, pieces in batches)


@lru_cache(maxsize=128)
def _zigzag_packed_batches(fragments, chunk_rows, logical_tokens):
    """Compact complete SIMT row tiles, retaining canonical GEMM M.

    A row stays at the same position inside a 64-row tile. Only complete
    tiles are moved between GEMMs with identical M. Unsupported fragments
    retain the canonical placement rather than changing the reduction shape.
    """
    if chunk_rows % 64 or any(c % 64 or n % 64 for _, c, n in fragments):
        return _zigzag_batches(fragments, chunk_rows, logical_tokens)
    buckets = {}
    for local, canonical, length in fragments:
        while length:
            begin = canonical // chunk_rows * chunk_rows
            rows = min(chunk_rows, logical_tokens - begin)
            count = min(length, begin + rows - canonical)
            if rows % 64:
                return _zigzag_batches(fragments, chunk_rows, logical_tokens)
            buckets.setdefault(rows, []).append((local, count))
            local += count
            canonical += count
            length -= count
    batches = []
    for rows, pieces in buckets.items():
        offset, batch = 0, []
        for local, length in pieces:
            while length:
                count = min(length, rows - offset)
                batch.append((local, offset, count))
                local += count
                offset += count
                length -= count
                if offset == rows:
                    batches.append((rows, tuple(batch)))
                    offset, batch = 0, []
        if batch:
            batches.append((rows, tuple(batch)))
    return tuple(batches)


@lru_cache(maxsize=32)
def _zigzag_workspace(device, stream, thread, rows, width):
    # Shared across layers on this stream. Every consumer is enqueued before
    # the next zero/copy producer; different streams never alias the buffer.
    return torch.empty((rows, width), dtype=torch.float32, device=device)


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

    def forward_fixed_rows(self, hidden_states: torch.Tensor, rows: int):
        from rtp_llm.models_py.modules.glm53_fixed_projection_graph import (
            fixed_projection_graph,
        )

        if rows > self.chunk_rows:
            return None
        return fixed_projection_graph(hidden_states, self.weight, rows)

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
        if hasattr(layout, "canonical_fragments"):
            return self._forward_zigzag(hidden_states, layout)
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

    def _forward_zigzag(self, hidden_states, layout):
        """Coalesce disjoint canonical rows without changing GEMM M/offsets.

        FP32 reduction/top-k behavior depends on the original GEMM shape and
        row offsets. Distinct canonical chunks may share one GEMM only when
        their occupied row intervals do not overlap and their M is identical.
        """
        if hidden_states.is_cuda and os.environ.get('GLM53_ROUTER_ZIGZAG_PACKED', '0') == '1':
            return self._forward_zigzag_packed(hidden_states, layout)
        batches = _zigzag_batches(tuple(layout.canonical_fragments), self.chunk_rows,
                                 layout.logical_tokens)
        output = hidden_states.new_empty(
            (layout.local_tokens, self.weight.shape[0]), dtype=torch.float32
        )
        stream = (int(torch.cuda.current_stream(hidden_states.device).cuda_stream)
                  if hidden_states.is_cuda else None)
        for rows, fragments in batches:
            chunk = _zigzag_workspace(hidden_states.device, stream, get_ident(), rows, hidden_states.shape[1])
            chunk.zero_()
            for local, offset, count in fragments:
                chunk.narrow(0, offset, count).copy_(
                    hidden_states.narrow(0, local, count)
                )
            logits = F.linear(chunk, self.weight)
            for local, offset, count in fragments:
                output.narrow(0, local, count).copy_(
                    logits.narrow(0, offset, count)
                )
        return output

    def _forward_zigzag_packed(self, hidden_states, layout):
        from .glm53_router_packing import pack_rows, scatter_rows, row_maps

        batches = _zigzag_packed_batches(tuple(layout.canonical_fragments),
                                         self.chunk_rows, layout.logical_tokens)
        output = hidden_states.new_empty(
            (layout.local_tokens, self.weight.shape[0]), dtype=torch.float32)
        stream = int(torch.cuda.current_stream(hidden_states.device).cuda_stream)
        for rows, fragments in batches:
            chunk = _zigzag_workspace(hidden_states.device, stream, get_ident(),
                                      rows, hidden_states.shape[1])
            mapping = row_maps(hidden_states.device, stream, get_ident(), rows, fragments)
            pack_rows(hidden_states, mapping, chunk)
            logits = F.linear(chunk, self.weight)
            scatter_rows(logits, mapping, output)
        return output
