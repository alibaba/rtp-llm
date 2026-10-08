"""Source-aware attention execution over caller-admitted rank-local cache plans.

Plans must already provide complete CP predecessors and visible cache rows.
SWA writes must target scratch or an overwrite-safe ring. These are execution
contracts, not scheduler admission or a CP exchange implementation.
"""

from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv4.v41_attention import V41SparseAttention
from rtp_llm.models_py.modules.dsv4.v41_cache import (
    V41PagedCache,
    logical_to_physical_slots,
)
from rtp_llm.models_py.modules.dsv4.v41_compressor import (
    V41PairMetadata,
    V41PairState,
    _validate_pair_values,
)
from rtp_llm.models_py.modules.dsv4.v41_indexer import V41Indexer, V41IndexResult


def _integers(value, shape, device, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or value.dtype not in (torch.int32, torch.int64)
        or value.device != device
        or not value.is_contiguous()
    ):
        raise ValueError(f"V4.1 {name} requires contiguous device integers {shape}")


@dataclass(frozen=True)
class V41ForwardRows:
    invocation_id: str
    request_ids: torch.Tensor
    positions: torch.Tensor
    valid_rows: torch.Tensor

    def snapshot(self):
        valid = self.valid_rows
        if (
            type(self.invocation_id) is not str
            or not self.invocation_id
            or not isinstance(valid, torch.Tensor)
            or valid.ndim != 1
            or valid.numel() > 128
            or valid.dtype != torch.bool
            or not valid.is_contiguous()
        ):
            raise ValueError(
                "V4.1 forward requires invocation identity and at most 128 row masks"
            )
        for name in ("request_ids", "positions"):
            _integers(getattr(self, name), valid.shape, valid.device, name)
        if bool(
            (
                (self.request_ids[valid] < 0)
                | (self.positions[valid] < 0)
                | (self.positions[valid] >= 32768)
            ).any()
        ):
            raise ValueError(
                "V4.1 live request/position identity exceeds the 32K profile"
            )
        pairs = torch.stack((self.request_ids[valid], self.positions[valid]), -1)
        if pairs.unique(dim=0).shape[0] != pairs.shape[0]:
            raise ValueError("V4.1 forward repeats a request token")
        return V41ForwardRows(
            self.invocation_id,
            torch.where(valid, self.request_ids, -1),
            torch.where(valid, self.positions, -1),
            valid.clone(),
        )

    def require_same(self, other):
        if (
            not isinstance(other, V41ForwardRows)
            or self.invocation_id != other.invocation_id
            or any(
                a.device != b.device or not torch.equal(a, b)
                for a, b in (
                    (self.request_ids, other.request_ids),
                    (self.positions, other.positions),
                    (self.valid_rows, other.valid_rows),
                )
            )
        ):
            raise ValueError("V4.1 selections belong to different forward rows")


@dataclass(frozen=True)
class V41Selections:
    rows: V41ForwardRows
    source_layer_id: int
    kv_source_layer_id: int
    compress_ratio: int
    result: V41IndexResult

    def require(self, sources, rows):
        rows.require_same(self.rows)
        if (self.source_layer_id, self.kv_source_layer_id, self.compress_ratio) != (
            sources.index_source_layer_id,
            sources.kv_source_layer_id,
            sources.compress_ratio,
        ):
            raise ValueError("V4.1 selections disagree with the global source binding")

    def validate(self):
        """Validate externally handed-off selections before any stage cache writes."""
        rows, result = self.rows, self.result
        rows.snapshot()
        count, device = rows.valid_rows.numel(), rows.valid_rows.device
        _integers(result.topk, (count, 512), device, "top-k")
        if result.topk.dtype != torch.int32 or self.compress_ratio not in (1, 2):
            raise ValueError("V4.1 selections require int32 top-k and ratio 1 or 2")
        visible = torch.where(
            rows.valid_rows, (rows.positions + 1) // self.compress_ratio, 0
        )

        def positions(value, visible, width):
            active = value >= 0
            ordered = value.sort(-1).values
            if (
                bool(((value < -1) | (active & (value >= visible[:, None]))).any())
                or bool(
                    ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()
                )
                or not torch.equal(active.sum(-1), visible.clamp(max=width))
            ):
                raise ValueError(
                    "V4.1 selections have invalid, duplicate or missing causal entries"
                )

        positions(result.topk, visible, 512)
        if self.kv_source_layer_id == 20:
            _integers(result.candidate_blocks, (count, 2048), device, "candidates")
            _integers(result.candidate_counts, (count,), device, "candidate counts")
            if (
                result.candidate_blocks.dtype != torch.int32
                or result.candidate_counts.dtype != torch.int32
            ):
                raise ValueError("V4.1 candidates require int32")
            blocks = (visible + 7) // 8
            positions(result.candidate_blocks, blocks, 2048)
            mask = (
                torch.arange(2048, device=device)[None]
                < result.candidate_counts[:, None]
            )
            if (
                not torch.equal(result.candidate_counts, blocks.clamp(max=2048).int())
                or not torch.equal(result.candidate_blocks >= 0, mask)
                or bool(
                    (
                        rows.valid_rows
                        & ~(result.candidate_blocks == (blocks - 1)[:, None]).any(-1)
                    ).any()
                )
            ):
                raise ValueError("V4.1 candidate count/padding/newest block mismatch")
        elif result.candidate_blocks is not None or result.candidate_counts is not None:
            raise ValueError("V4.1 ratio-2 selections cannot carry L20 candidates")


@dataclass(frozen=True)
class V41AttentionPlan:
    swa: V41PagedCache
    swa_writes: torch.Tensor
    swa_reads: torch.Tensor
    kv_source_layer_id: int | None = None
    global_cache: V41PagedCache | None = None
    index_cache: V41PagedCache | None = None
    global_pages: torch.Tensor | None = None
    index_pages: torch.Tensor | None = None
    global_writes: torch.Tensor | None = None
    index_writes: torch.Tensor | None = None
    pair_state: V41PairState | None = None
    pair_metadata: V41PairMetadata | None = None

    def validate(self, sources, rows, max_positions):
        n, device = rows.valid_rows.numel(), rows.valid_rows.device
        if bool((rows.positions[rows.valid_rows] >= max_positions).any()):
            raise ValueError("V4.1 forward exceeds the admitted attention context")
        if (
            not isinstance(self.swa, V41PagedCache)
            or self.swa.tag != "swa_kv"
            or self.swa.raw.device != device
            or self.kv_source_layer_id != sources.kv_source_layer_id
        ):
            raise ValueError("V4.1 attention plan cache/source mismatch")
        _integers(self.swa_writes, (n,), device, "SWA writes")
        if (
            not isinstance(self.swa_reads, torch.Tensor)
            or self.swa_reads.ndim != 2
            or self.swa_reads.shape[1] > 128
        ):
            raise ValueError("V4.1 SWA reads exceed the window")
        _integers(self.swa_reads, (n, self.swa_reads.shape[1]), device, "SWA reads")
        self.swa._check_slots(
            torch.where(rows.valid_rows, self.swa_writes, -1), "eager", writing=True
        )
        self.swa._check_slots(
            torch.where(rows.valid_rows[:, None], self.swa_reads, -1).flatten(), "eager"
        )
        ratio = sources.compress_ratio
        if not ratio:
            if any(
                value is not None
                for value in (
                    self.global_cache,
                    self.index_cache,
                    self.global_pages,
                    self.index_pages,
                    self.global_writes,
                    self.index_writes,
                    self.pair_state,
                    self.pair_metadata,
                )
            ):
                raise ValueError("V4.1 SWA-only plan cannot contain GLOBAL/index state")
            return
        width = (max_positions + 255) // 256
        for cache, pages, tag in (
            (
                self.global_cache,
                self.global_pages,
                "csa_kv" if ratio == 2 else "hca_kv",
            ),
            (self.index_cache, self.index_pages, "indexer_kv"),
        ):
            if (
                not isinstance(cache, V41PagedCache)
                or cache.tag != tag
                or cache.raw.device != device
            ):
                raise ValueError("V4.1 GLOBAL/index plan cache mismatch")
            _integers(pages, (n, width), device, "page table")
            required = (
                torch.arange(width, device=device)[None] * 256
                <= rows.positions[:, None]
            )
            required &= rows.valid_rows[:, None]
            if bool((required & ((pages < 0) | (pages >= cache.raw.shape[0]))).any()):
                raise ValueError(
                    "V4.1 visible page table refers to an unallocated page"
                )
        if sources.owns_kv:
            complete = rows.valid_rows & ((rows.positions + 1) % ratio == 0)
            for cache, pages, writes in (
                (self.global_cache, self.global_pages, self.global_writes),
                (self.index_cache, self.index_pages, self.index_writes),
            ):
                _integers(writes, (n,), device, "source writes")
                cache._check_slots(
                    torch.where(complete, writes, -1), "eager", writing=True
                )
                for row in range(n):
                    if not bool(complete[row]):
                        continue
                    expected = logical_to_physical_slots(
                        pages[row],
                        ((rows.positions[row : row + 1] + 1) // ratio - 1),
                        cache.tag,
                        ratio,
                    )
                    if not torch.equal(writes[row : row + 1].long(), expected):
                        raise ValueError(
                            "V4.1 source writes disagree with their logical page mapping"
                        )
        elif self.global_writes is not None or self.index_writes is not None:
            raise ValueError("V4.1 consumers cannot publish source caches")
        if sources.owns_kv and ratio == 2:
            if (
                not isinstance(self.pair_state, V41PairState)
                or not isinstance(self.pair_metadata, V41PairMetadata)
                or self.pair_state.raw.device != device
            ):
                raise ValueError("V4.1 ratio-2 producer requires admitted pair state")
            meta = self.pair_metadata
            if any(
                a.device != b.device or not torch.equal(a, b)
                for a, b in ((meta.valid_rows, rows.valid_rows),)
            ) or any(
                a.device != b.device
                or a.shape != b.shape
                or not torch.equal(a[rows.valid_rows], b[rows.valid_rows])
                for a, b in (
                    (meta.positions, rows.positions),
                    (meta.request_ids, rows.request_ids),
                )
            ):
                raise ValueError("V4.1 pair identity differs from forward rows")
            for name in ("previous_rows", "previous_slots", "write_slots"):
                _integers(getattr(meta, name), (n,), device, name)
            dummy = torch.zeros(n, 512, dtype=torch.float32, device=device)
            _validate_pair_values(
                dummy,
                dummy,
                torch.ones(512, device=device),
                self.pair_state,
                meta,
                complete,
                torch.where(rows.valid_rows, meta.write_slots, -1),
            )
        elif self.pair_state is not None or self.pair_metadata is not None:
            raise ValueError("V4.1 pair state belongs only to ratio-2 producers")


class V41StageAttention:
    """Publish, select/reuse by global source ID, then attend on one rank.

    validate() is an eager preflight. The stage validates every layer before
    beginning; device failures after publication require aborting the forward.
    """

    def __init__(self, sources, *, max_positions=32768, backend="triton"):
        if type(max_positions) is not int or not 1 <= max_positions <= 32768:
            raise ValueError("V4.1 stage attention requires a bounded 32K context")
        self.sources, self.max_positions = sources, max_positions
        self.attention = V41SparseAttention(backend=backend)
        self.indexer = (
            V41Indexer(
                sources,
                max_positions=(max_positions + sources.compress_ratio - 1)
                // sources.compress_ratio,
                backend=backend,
            )
            if sources.owns_index
            else None
        )

    def __call__(self, frame, sinks, plan, rows, selections):
        sources = self.sources
        if sources.compress_ratio and not sources.owns_index:
            if selections is None:
                raise ValueError("V4.1 GLOBAL attention requires source selections")
            selections.require(sources, rows)
        candidates, counts = None, None
        if sources.owns_index and sources.layer_id > 20:
            if selections is None or selections.kv_source_layer_id != 20:
                raise ValueError(
                    "V4.1 later index sources require this forward's L20 candidates"
                )
            rows.require_same(selections.rows)
            candidates, counts = (
                selections.result.candidate_blocks,
                selections.result.candidate_counts,
            )
            if candidates is None or counts is None:
                raise ValueError(
                    "V4.1 later index sources require L20 candidates and counts"
                )
        kwargs = {}
        if sources.owns_kv:
            kwargs = dict(
                global_cache=plan.global_cache,
                global_slots=plan.global_writes,
                index_cache=plan.index_cache,
                index_slots=plan.index_writes,
            )
        frame.kv.publish(plan.swa, plan.swa_writes, **kwargs)
        if sources.owns_index:
            result = self.indexer(
                frame.query.index_query,
                frame.query.index_weights,
                plan.index_cache,
                plan.index_pages,
                rows.positions,
                valid_rows=rows.valid_rows,
                candidates=candidates,
            )
            if candidates is not None:
                result = V41IndexResult(result.topk, candidates, counts)
            selections = V41Selections(
                rows,
                sources.layer_id,
                sources.kv_source_layer_id,
                sources.compress_ratio,
                result,
            )
        kwargs = {}
        if sources.compress_ratio:
            if selections is None:
                raise ValueError("V4.1 GLOBAL attention requires source selections")
            selections.require(sources, rows)
            slots = torch.full_like(selections.result.topk, -1, dtype=torch.int64)
            for row in range(rows.valid_rows.numel()):
                if bool(rows.valid_rows[row]):
                    slots[row] = logical_to_physical_slots(
                        plan.global_pages[row],
                        selections.result.topk[row],
                        plan.global_cache.tag,
                        sources.compress_ratio,
                    )
            kwargs = dict(global_cache=plan.global_cache, global_slots=slots)
        attended = self.attention(
            frame.query.query,
            sinks,
            plan.swa,
            plan.swa_reads,
            valid_rows=rows.valid_rows,
            **kwargs,
        )
        return attended, selections
