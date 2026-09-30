"""Bounded V4.1 FP4 index scoring and forward-local key selection.

Projected queries have already received RoPE. Head weights are the BF16
weights-projection output, before the reference's 1/64 scaling. The caller
owns page-table/request identity and carries L20 candidates with the same rows.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rtp_llm.config.dsv41_config import V41LayerSources
from rtp_llm.models_py.modules.dsv4.v41_cache import (
    V41PagedCache,
    dequantize_rows_eager,
    quantize_rows_eager,
)


@dataclass(frozen=True)
class V41IndexResult:
    topk: torch.Tensor
    candidate_blocks: torch.Tensor | None
    candidate_counts: torch.Tensor | None


def _ordered_positions(scores: torch.Tensor, count: int) -> torch.Tensor:
    rows, width = scores.shape
    values, positions = scores.topk(min(count, width), dim=-1, sorted=False)
    positions = torch.where(values > -torch.inf, positions, width).sort(-1).values
    positions = torch.where(positions < width, positions, -1).to(torch.int32)
    return F.pad(positions, (0, count - positions.shape[1]), value=-1).contiguous()


def _select(scores, visible, *, candidate_source, candidates):
    rows, width = scores.shape
    blocks = (width + 7) // 8
    produced, counts = None, None
    if candidate_source:
        block_scores = F.pad(scores, (0, -width % 8), value=-torch.inf)
        block_scores = block_scores.reshape(rows, blocks, 8).amax(-1)
        latest = (visible - 1) // 8
        block_scores = block_scores.masked_fill(
            torch.arange(blocks, device=scores.device)[None, :] == latest[:, None],
            torch.inf,
        )
        produced = _ordered_positions(block_scores, 2048)
        counts = (produced >= 0).sum(-1, dtype=torch.int32)
    elif candidates is not None:
        # A separate sentinel column prevents -1 padding from clearing block 0.
        valid = (candidates >= 0) & (candidates < blocks)
        safe = torch.where(valid, candidates, blocks).long()
        keep = torch.zeros((rows, blocks + 1), dtype=torch.bool, device=scores.device)
        keep.scatter_(1, safe, valid)
        mask = keep[:, :blocks].repeat_interleave(8, -1)[:, :width]
        scores = scores.masked_fill(~mask, -torch.inf)
    return V41IndexResult(_ordered_positions(scores, 512), produced, counts)


class V41Indexer:
    def __init__(
        self,
        sources: V41LayerSources,
        *,
        max_positions: int,
        max_query_rows: int = 128,
        backend: str = "triton",
    ):
        layer = sources.layer_id
        ratio = 2 if layer < 20 else 1
        owner = layer if layer < 20 else 20
        expected = V41LayerSources(
            layer, ratio, owner, layer, None if layer < 20 else 20
        )
        if layer not in (2, 8, 14, 20, 24, 28, 32, 36) or sources != expected:
            raise ValueError(
                "V4.1 indexer requires an exact global selection-source binding"
            )
        if (
            type(max_positions) is not int
            or not 1 <= max_positions <= 32768 // ratio
            or type(max_query_rows) is not int
            or not 1 <= max_query_rows <= 128
        ):
            raise ValueError(
                "V4.1 indexer exceeds the bounded 32K context / 128 row profile"
            )
        if backend not in ("triton", "eager"):
            raise ValueError("V4.1 indexer backend must be triton or eager")
        self.sources = sources
        self.max_positions = max_positions
        self.max_query_rows = max_query_rows
        self.backend = backend
        self._warmed = set()

    def score(self, query, weights, cache, page_table, positions, valid_rows):
        if query.ndim != 3:
            raise ValueError("V4.1 indexer queries must be [R,32,128]")
        rows = query.shape[0]
        columns = (self.max_positions + 256 // self.sources.compress_ratio - 1) // (
            256 // self.sources.compress_ratio
        )
        if (
            query.shape != (rows, 32, 128)
            or query.dtype != torch.bfloat16
            or cache.tag != "indexer_kv"
            or cache.raw.device != query.device
            or rows > self.max_query_rows
        ):
            raise ValueError(
                "V4.1 indexer requires bounded BF16 [R,32,128] queries and an index cache"
            )
        specs = (
            (query, (rows, 32, 128), (torch.bfloat16,)),
            (weights, (rows, 32), (torch.bfloat16,)),
            (page_table, (rows, columns), (torch.int32, torch.int64)),
            (positions, (rows,), (torch.int32, torch.int64)),
            (valid_rows, (rows,), (torch.bool,)),
        )
        for value, shape, dtypes in specs:
            if (
                value.shape != shape
                or value.dtype not in dtypes
                or value.device != query.device
                or not value.is_contiguous()
            ):
                raise ValueError(
                    "V4.1 indexer input shape/dtype/device/stride mismatch"
                )
        visible = torch.where(
            valid_rows, (positions.long() + 1) // self.sources.compress_ratio, 0
        )
        if self.backend == "eager":
            if query.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("V4.1 eager indexer cannot be captured")
            self._validate_values(
                query, weights, cache, page_table, positions, valid_rows
            )
            return (
                self._eager_score(
                    query, weights, cache, page_table, visible, valid_rows
                ),
                visible,
            )
        if not query.is_cuda or torch.cuda.get_device_capability(query.device) != (
            12,
            0,
        ):
            raise RuntimeError("V4.1 indexer requires an exact SM120 device")
        key = (
            query.device,
            page_table.dtype,
            positions.dtype,
            self.max_positions,
            self.sources.compress_ratio,
        )
        if torch.cuda.is_current_stream_capturing() and key not in self._warmed:
            raise RuntimeError("Warm V4.1 indexer layout before graph capture")
        out = torch.empty(
            (rows, self.max_positions), dtype=torch.float32, device=query.device
        )
        if rows:
            from rtp_llm.models_py.modules.dsv4.fp8._v41_indexer_triton import score

            score(
                query,
                weights,
                cache,
                page_table,
                positions,
                valid_rows,
                self.sources.compress_ratio,
                out,
            )
            self._warmed.add(key)
        return out, visible

    def __call__(
        self,
        query,
        weights,
        cache,
        page_table,
        positions,
        *,
        valid_rows,
        candidates=None,
    ):
        uses_candidates = self.sources.layer_id > 20
        if uses_candidates != (candidates is not None):
            raise ValueError("V4.1 candidate input must match the selection source")
        if candidates is not None and (
            candidates.shape != (query.shape[0], 2048)
            or candidates.dtype != torch.int32
            or candidates.device != query.device
            or not candidates.is_contiguous()
        ):
            raise ValueError("V4.1 candidates must be contiguous device int32 [R,2048]")
        scores, visible = self.score(
            query, weights, cache, page_table, positions, valid_rows
        )
        if candidates is not None and self.backend == "eager":
            for row in range(query.shape[0]):
                active = candidates[row][candidates[row] >= 0]
                expected = min(2048, (int(visible[row]) + 7) // 8)
                if (
                    bool((candidates[row] < -1).any())
                    or active.numel() != expected
                    or active.unique().numel() != expected
                    or bool((active >= (visible[row] + 7) // 8).any())
                    or (
                        expected and not bool((active == (visible[row] - 1) // 8).any())
                    )
                ):
                    raise ValueError(
                        "V4.1 candidates violate causal coverage/newest-block identity"
                    )
        return _select(
            scores,
            visible,
            candidate_source=self.sources.layer_id == 20,
            candidates=candidates,
        )

    def _validate_values(self, query, weights, cache, pages, positions, valid):
        if (
            not bool(torch.isfinite(query[valid]).all())
            or not bool(torch.isfinite(weights[valid]).all())
            or bool(
                (
                    (positions[valid] < 0)
                    | (
                        positions[valid]
                        >= self.max_positions * self.sources.compress_ratio
                    )
                ).any()
            )
        ):
            raise ValueError(
                "V4.1 live index queries/weights must be finite and positions in range"
            )
        for row in range(query.shape[0]):
            if not valid[row]:
                continue
            count = (int(positions[row]) + 1) // self.sources.compress_ratio
            used = (count + 256 // self.sources.compress_ratio - 1) // (
                256 // self.sources.compress_ratio
            )
            if bool(
                (
                    (pages[row, :used] < 0) | (pages[row, :used] >= cache.raw.shape[0])
                ).any()
            ):
                raise ValueError(
                    "V4.1 visible index keys require allocated cache pages"
                )

    def _eager_score(self, query, weights, cache, pages, visible, valid):
        rows = query.shape[0]
        out = torch.full(
            (rows, self.max_positions),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        )
        for row in range(rows):
            if not valid[row]:
                continue
            q = dequantize_rows_eager(
                quantize_rows_eager(query[row], "indexer_kv"), "indexer_kv"
            )
            logical = torch.arange(int(visible[row]), device=query.device)
            used = 256 // self.sources.compress_ratio
            slots = (
                pages[row, logical // used].long() * 256 + logical % used
            ).contiguous()
            k = cache.read(slots, backend="eager")
            dots = (q.float() @ k.float().T).bfloat16()
            head_weights = (weights[row] * (1.0 / 64)).bfloat16()
            weighted = (dots.relu().float() * head_weights.float()[:, None]).bfloat16()
            out[row, : logical.numel()] = weighted.float().sum(0).bfloat16().float()
        return out
