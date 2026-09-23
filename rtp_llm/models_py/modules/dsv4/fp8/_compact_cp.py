"""CPU planning and byte-oracle contracts for compact CP; not a CUDA implementation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

CSA_INDEXER = "INDEXER_KV"
CSA_MAIN = "CSA_KV"
HCA_MAIN = "HCA_KV"
_ALLOWED_STATUS = frozenset((1,))


class CompactCPIneligible(ValueError):
    """Raised only for malformed caller inputs; supported fallbacks are plans."""


def _get(record: Any, name: str, default: Any = None) -> Any:
    return (
        record.get(name, default)
        if isinstance(record, Mapping)
        else getattr(record, name, default)
    )


def _host_tuple(values: Any, name: str) -> Tuple[Any, ...]:
    """Accept only host iterables (torch CPU tensors work via ``tolist``)."""
    if values is None:
        raise CompactCPIneligible(f"missing {name}")
    if hasattr(values, "device") and str(values.device) != "cpu":
        raise CompactCPIneligible(f"{name} must be host-visible")
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        return tuple(values)
    except TypeError as exc:
        raise CompactCPIneligible(f"{name} must be a host sequence") from exc


def _immutable_mapping(
    values: Mapping[Any, int], name: str
) -> Tuple[Tuple[int, int], ...]:
    out = []
    for key, value in values.items():
        if type(key) is not int or type(value) is not int:
            raise CompactCPIneligible(
                f"{name} must map integer logical boundaries to slots"
            )
        out.append((key, value))
    return tuple(sorted(out))


@dataclass(frozen=True)
class CompactCPInterval:
    absolute_start: int
    length: int
    raw_offset: int

    @property
    def end(self) -> int:
        return self.absolute_start + self.length


@dataclass(frozen=True)
class LogicalCompressedRow:
    request_id: int
    absolute_boundary: int
    pool_tag: str
    ratio: int

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not int
            or type(self.absolute_boundary) is not int
            or self.absolute_boundary < 0
            or type(self.ratio) is not int
            or self.ratio not in (4, 128)
            or self.pool_tag not in (CSA_INDEXER, CSA_MAIN, HCA_MAIN)
        ):
            raise CompactCPIneligible(
                "logical row identity has invalid field types/domain"
            )


@dataclass(frozen=True)
class HaloRow:
    absolute_position: int
    source_rank: int
    source_segment: int
    is_local: bool
    source_kind: str  # current_raw or same_request_history


@dataclass(frozen=True)
class CompactPoolSpec:
    """Receiver-local split-page pool description captured at plan construction."""

    tag: str
    capacity: int
    data_row_bytes: int
    scale_row_bytes: int
    # logical compressed boundary -> physical slot; a value of -1 is explicitly unallocated.
    receiver_block_table: Mapping[int, int]
    table_generation: str
    storage_identity: str
    page_rows: int = 1
    allocator_unallocated: int = -1
    sharded: bool = False

    def __post_init__(self) -> None:
        if self.tag not in (CSA_INDEXER, CSA_MAIN, HCA_MAIN):
            raise CompactCPIneligible(f"unknown pool tag {self.tag}")
        if (
            min(
                self.capacity, self.data_row_bytes, self.scale_row_bytes, self.page_rows
            )
            <= 0
        ):
            raise CompactCPIneligible(
                "pool capacity, row widths, and page rows must be positive"
            )
        if not self.table_generation or not self.storage_identity:
            raise CompactCPIneligible("pool table/storage identity is required")


@dataclass(frozen=True)
class CollectiveStep:
    ordinal: int
    pool_tag: str
    event_chain: Tuple[str, ...] = ("producer", "pack", "gather", "scatter", "consumer")


@dataclass(frozen=True)
class CompactCPPlan:
    """Immutable, forward/stage/rank-local decision.  No runtime activation bit exists."""

    forward_id: str
    stage_id: int
    stage_group: Tuple[int, ...]
    layer_id: int
    cp_rank: int
    cp_group: Tuple[int, ...]
    request_id: int
    request_start: int
    chunk_start: int
    chunk_length: int
    padded_length: int
    intervals: Tuple[CompactCPInterval, ...]
    ratio: int
    attention_type: str
    eligible: bool
    fallback_reason: Optional[str]
    same_request_history: Tuple[Tuple[int, int], ...]
    csa_halos: Tuple[HaloRow, ...]
    published_tails: Tuple[Tuple[int, ...], ...]
    source_rows: Tuple[LogicalCompressedRow, ...]
    receive_rows: Tuple[LogicalCompressedRow, ...]
    receiver_slots: Tuple[Tuple[LogicalCompressedRow, int], ...]
    pools: Tuple[Tuple[str, CompactPoolSpec], ...]
    schedule: Tuple[CollectiveStep, ...]
    workspace_region: str
    workspace_bytes: int
    lease_id: str
    required_release_event: str
    state_read_still_required: bool

    def pool(self, tag: str) -> CompactPoolSpec:
        return dict(self.pools)[tag]

    def receiver_slot(self, identity: LogicalCompressedRow) -> int:
        # Identity includes request and ratio: boundary/tag alone are not portable.
        for expected, slot in self.receiver_slots:
            if expected == identity:
                return slot
        raise CompactCPIneligible("logical row has no receiver-local resolved slot")

    def expected_rows(self, source: bool) -> Tuple[LogicalCompressedRow, ...]:
        return self.source_rows if source else self.receive_rows


@dataclass(frozen=True)
class PackedRow:
    identity: LogicalCompressedRow
    data: bytes
    scales: bytes

    @property
    def wire_bytes(self) -> bytes:
        return self.data + self.scales


@dataclass
class SplitPoolBytes:
    """CPU split-page oracle; it models data/scale separation rather than flattening pages."""

    spec: CompactPoolSpec
    data: bytearray
    scales: bytearray

    @classmethod
    def poison(cls, spec: CompactPoolSpec, byte: int = 0xA5) -> "SplitPoolBytes":
        return cls(
            spec,
            bytearray([byte]) * (spec.capacity * spec.data_row_bytes),
            bytearray([byte]) * (spec.capacity * spec.scale_row_bytes),
        )

    def _range(self, slot: int, width: int) -> slice:
        if type(slot) is not int or not 0 <= slot < self.spec.capacity:
            raise CompactCPIneligible("receiver slot is outside pool capacity")
        return slice(slot * width, (slot + 1) * width)

    def read(self, slot: int) -> Tuple[bytes, bytes]:
        return (
            bytes(self.data[self._range(slot, self.spec.data_row_bytes)]),
            bytes(self.scales[self._range(slot, self.spec.scale_row_bytes)]),
        )

    def write(self, slot: int, data: bytes, scales: bytes) -> None:
        if (
            len(data) != self.spec.data_row_bytes
            or len(scales) != self.spec.scale_row_bytes
        ):
            raise CompactCPIneligible(
                "packed row width disagrees with split pool layout"
            )
        self.data[self._range(slot, self.spec.data_row_bytes)] = data
        self.scales[self._range(slot, self.spec.scale_row_bytes)] = scales


@dataclass
class CompactCPLease:
    lease_id: str
    workspace_region: str
    state: str = "new"
    pending_handles: int = 0

    def producer_done(self) -> None:
        if self.state != "new":
            raise CompactCPIneligible("producer event is not first")
        self.state = "producer"

    def packed(self) -> None:
        if self.state != "producer":
            raise CompactCPIneligible("pack requires producer event")
        self.state = "packed"

    def gathered(self) -> None:
        if self.state != "packed":
            raise CompactCPIneligible("gather requires packed payload")
        self.state, self.pending_handles = "gathered", 1

    def scattered(self) -> None:
        if self.state != "gathered" or self.pending_handles != 1:
            raise CompactCPIneligible(
                "scatter requires exactly one pending gather handle"
            )
        self.state, self.pending_handles = "scattered", 0

    def consumed_and_release(self) -> None:
        if self.state != "scattered" or self.pending_handles:
            raise CompactCPIneligible("cannot release a pending compact CP lease")
        self.state = "released"

    def abort(self) -> None:
        self.pending_handles = 0
        self.state = "released"


class CompactCPTransportAdapter:
    """Candidate-only byte adapter. It performs no collective and cannot enable a model path."""

    def __init__(self, plan: CompactCPPlan, pool_bytes: Mapping[str, SplitPoolBytes]):
        if not plan.eligible:
            raise CompactCPIneligible("fallback plan has no compact transport")
        self.plan = plan
        self.pool_bytes = dict(pool_bytes)
        self.lease = CompactCPLease(plan.lease_id, plan.workspace_region)

    @staticmethod
    def runtime_activation_allowed() -> bool:
        # Deliberately constant: only a later reviewed caller may bind a GPU operation.
        return False

    def _validate_pool_bindings(self) -> None:
        if set(self.pool_bytes) != {tag for tag, _ in self.plan.pools}:
            raise CompactCPIneligible("pool binding set differs from immutable plan")
        for tag, spec in self.plan.pools:
            pool = self.pool_bytes.get(tag)
            if pool is None or pool.spec != spec:
                raise CompactCPIneligible("pool binding differs from immutable plan")
            if (
                type(pool.data) is not bytearray
                or type(pool.scales) is not bytearray
                or len(pool.data) != spec.capacity * spec.data_row_bytes
                or len(pool.scales) != spec.capacity * spec.scale_row_bytes
            ):
                raise CompactCPIneligible(
                    "pool backing storage differs from declared capacity"
                )

    @staticmethod
    def _exact_rows(
        rows: Iterable[LogicalCompressedRow],
        expected: Tuple[LogicalCompressedRow, ...],
        what: str,
    ) -> Tuple[LogicalCompressedRow, ...]:
        rows = tuple(rows)
        if (
            len(rows) != len(set(rows))
            or set(rows) != set(expected)
            or len(rows) != len(expected)
        ):
            raise CompactCPIneligible(
                f"{what} identities are not the exact unique planned set"
            )
        return rows

    def pack(
        self,
        identities: Iterable[LogicalCompressedRow],
        statuses: Mapping[LogicalCompressedRow, int],
    ) -> Tuple[PackedRow, ...]:
        # Validate the whole publication batch before exposing any row/handle.
        try:
            rows = self._exact_rows(identities, self.plan.source_rows, "source")
            self._validate_pool_bindings()
            if set(statuses) != set(rows) or any(
                type(statuses[row]) is not int or statuses[row] not in _ALLOWED_STATUS
                for row in rows
            ):
                raise CompactCPIneligible(
                    "writer statuses are not exactly all planned source rows"
                )
            staged = []
            for identity in rows:
                spec = self.plan.pool(identity.pool_tag)
                slot = self.plan.receiver_slot(identity)
                if slot == spec.allocator_unallocated:
                    raise CompactCPIneligible("sender table lacks a valid logical row")
                data, scales = self.pool_bytes[identity.pool_tag].read(slot)
                staged.append(PackedRow(identity, data, scales))
        except Exception:
            self.lease.abort()
            raise
        self.lease.producer_done()
        self.lease.packed()
        return tuple(staged)

    def accept_gathered(self) -> None:
        """Mark a receiver-side completed gather; sender packing is on another rank."""
        if self.lease.state != "new":
            raise CompactCPIneligible("receiver gather can begin only once")
        self.lease.state, self.lease.pending_handles = "gathered", 1

    def scatter(self, gathered: Iterable[PackedRow]) -> None:
        if self.lease.state == "packed":
            self.lease.gathered()
        elif self.lease.state != "gathered" or self.lease.pending_handles != 1:
            raise CompactCPIneligible(
                "scatter requires a local or accepted gathered payload"
            )
        try:
            rows = tuple(gathered)
            identities = self._exact_rows(
                (row.identity for row in rows), self.plan.receive_rows, "receive"
            )
            self._validate_pool_bindings()
            staged = []
            slots = set()
            for identity, row in zip(identities, rows):
                spec = self.plan.pool(identity.pool_tag)
                if not isinstance(row.data, bytes) or not isinstance(row.scales, bytes):
                    raise CompactCPIneligible(
                        "packed rows must be immutable byte payloads"
                    )
                if (
                    len(row.data) != spec.data_row_bytes
                    or len(row.scales) != spec.scale_row_bytes
                ):
                    raise CompactCPIneligible(
                        "packed row width disagrees with immutable pool layout"
                    )
                slot = self.plan.receiver_slot(identity)
                slot_key = (identity.pool_tag, slot)
                if slot_key in slots:
                    raise CompactCPIneligible(
                        "two logical rows resolve to one receiver slot in a pool"
                    )
                slots.add(slot_key)
                staged.append((identity, slot, row.data, row.scales))
            # All identity, width, pool, and slot checks above precede every write.
            for identity, slot, data, scales in staged:
                self.pool_bytes[identity.pool_tag].write(slot, data, scales)
        except Exception:
            self.lease.abort()
            raise
        self.lease.scattered()

    def release_after_consumer(self) -> None:
        self.lease.consumed_and_release()


@dataclass(frozen=True)
class CompactCPProposalSummary:
    cp_rank: int
    cp_group: Tuple[int, ...]
    eligible: bool
    schedule: Tuple[str, ...]
    forward_identity: tuple
    wire_layouts: tuple
    receive_rows: Tuple[LogicalCompressedRow, ...]


@dataclass(frozen=True)
class CompactCPReconciliation:
    agreed: bool
    eligible: bool
    schedule: Tuple[str, ...]
    reason: str


def proposal_summary(plan: CompactCPPlan) -> CompactCPProposalSummary:
    return CompactCPProposalSummary(
        plan.cp_rank,
        plan.cp_group,
        plan.eligible,
        tuple(step.pool_tag for step in plan.schedule),
        (
            plan.forward_id,
            plan.stage_id,
            plan.layer_id,
            plan.request_id,
            plan.request_start,
            plan.chunk_start,
            plan.chunk_length,
            plan.padded_length,
            plan.ratio,
            plan.attention_type,
        ),
        tuple(
            (tag, spec.data_row_bytes, spec.scale_row_bytes) for tag, spec in plan.pools
        ),
        plan.receive_rows,
    )


def reconcile_compact_cp_proposals(
    proposals: Iterable[CompactCPProposalSummary], *, cp_group: Sequence[int]
) -> CompactCPReconciliation:
    """Pure host reconciliation; callers must obtain all peer summaries before a collective."""
    expected = tuple(cp_group)
    items = tuple(proposals)
    if len(expected) != 4 or len(set(expected)) != 4:
        return CompactCPReconciliation(False, False, (), "INVALID_CP_GROUP")
    if len(items) != 4 or {item.cp_rank for item in items} != set(range(4)):
        return CompactCPReconciliation(False, False, (), "MISSING_PEER_PROPOSAL")
    if any(item.cp_group != expected for item in items):
        return CompactCPReconciliation(False, False, (), "CP_GROUP_MISMATCH")
    if not all(item.eligible for item in items):
        return CompactCPReconciliation(True, False, (), "PEER_INELIGIBLE")
    if len({item.forward_identity for item in items}) != 1:
        return CompactCPReconciliation(False, False, (), "FORWARD_IDENTITY_MISMATCH")
    if (
        len({item.wire_layouts for item in items}) != 1
        or len({item.receive_rows for item in items}) != 1
    ):
        return CompactCPReconciliation(False, False, (), "WIRE_LAYOUT_OR_ROWS_MISMATCH")
    schedules = {item.schedule for item in items}
    if len(schedules) != 1:
        return CompactCPReconciliation(False, False, (), "SCHEDULE_DISAGREEMENT")
    return CompactCPReconciliation(True, True, items[0].schedule, "AGREED")


def _fallback(**kwargs: Any) -> CompactCPPlan:
    reason = kwargs.pop("reason")
    return CompactCPPlan(
        eligible=False,
        fallback_reason=reason,
        intervals=(),
        same_request_history=(),
        csa_halos=(),
        published_tails=(),
        source_rows=(),
        receive_rows=(),
        receiver_slots=(),
        schedule=(),
        workspace_bytes=0,
        **kwargs,
    )


def _runs(positions: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    if not positions:
        return ()
    values = sorted(positions)
    if len(values) != len(set(values)):
        raise CompactCPIneligible("real local positions contain duplicates")
    out, start, prev = [], values[0], values[0]
    for pos in values[1:]:
        if pos != prev + 1:
            out.append((start, prev - start + 1))
            start = pos
        prev = pos
    out.append((start, prev - start + 1))
    return tuple(out)


def _zigzag_intervals(positions: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """Preserve the two CP ownership halves even where their global ranges touch."""
    if len(positions) == 0 or len(positions) % 2:
        return ()
    halves = (
        tuple(positions[: len(positions) // 2]),
        tuple(positions[len(positions) // 2 :]),
    )
    if any(not half or len(_runs(half)) != 1 for half in halves):
        return ()
    if len(set(positions)) != len(positions):
        raise CompactCPIneligible("real local positions contain duplicates")
    return tuple(_runs(half)[0] for half in halves)


def _owner_map(
    rank_positions: Mapping[int, Sequence[int]]
) -> Dict[int, Tuple[int, int]]:
    owner: Dict[int, Tuple[int, int]] = {}
    for rank, values in rank_positions.items():
        intervals = _zigzag_intervals(values)
        if len(intervals) != 2:
            raise CompactCPIneligible(
                "rank does not expose two zigzag ownership intervals"
            )
        for segment, (start, length) in enumerate(intervals):
            for pos in range(start, start + length):
                if pos in owner:
                    raise CompactCPIneligible(
                        "global CP geometry overlaps between ranks"
                    )
                owner[pos] = (rank, segment)
    return owner


def build_compact_cp_plan(
    ctx: Any,
    *,
    forward_id: str,
    stage_id: int,
    layer_id: int,
    attention_type: str,
    pools: Mapping[str, CompactPoolSpec],
    cp_group: Sequence[int],
    request_id: int = 0,
    request_start: Optional[int] = None,
    rank_global_positions: Optional[Mapping[int, Sequence[int]]] = None,
    history_rank_global_positions: Optional[Mapping[int, Sequence[int]]] = None,
    feature_requested: bool = False,
    cuda_graph: bool = False,
    warmup: bool = False,
    external_prefix: bool = False,
    workspace_bytes: int = 0,
    pending_lease: bool = False,
) -> CompactCPPlan:
    """Build a local proposal; reconcile all stage proposals before choosing a schedule."""
    cp_size, cp_rank = _get(ctx, "cp_size"), _get(ctx, "cp_rank")
    # Snapshot table contents now. Parent-owned storage/table mutation after this
    # point cannot alter a forward plan's receiver resolution.
    pools = {
        tag: CompactPoolSpec(
            tag=spec.tag,
            capacity=spec.capacity,
            data_row_bytes=spec.data_row_bytes,
            scale_row_bytes=spec.scale_row_bytes,
            receiver_block_table=MappingProxyType(dict(spec.receiver_block_table)),
            table_generation=spec.table_generation,
            storage_identity=spec.storage_identity,
            page_rows=spec.page_rows,
            allocator_unallocated=spec.allocator_unallocated,
            sharded=spec.sharded,
        )
        for tag, spec in pools.items()
    }
    base = dict(
        forward_id=str(forward_id),
        stage_id=stage_id,
        stage_group=tuple(range(0, 4) if stage_id == 0 else range(4, 8)),
        layer_id=layer_id,
        cp_rank=cp_rank,
        cp_group=tuple(cp_group),
        request_id=request_id,
        request_start=0 if request_start is None else request_start,
        chunk_start=0,
        chunk_length=0,
        padded_length=0,
        ratio=4 if attention_type == "CSA" else 128,
        attention_type=attention_type,
        pools=tuple(sorted(pools.items())),
        workspace_region="compact_cp_packed",
        lease_id=f"{forward_id}:s{stage_id}:r{cp_rank}:l{layer_id}",
        required_release_event="consumer",
    )
    if cp_size == 1:
        return _fallback(reason="CP1_BYPASS", state_read_still_required=True, **base)
    expected_stage_group = tuple(range(0, 4) if stage_id == 0 else range(4, 8))
    if (
        type(stage_id) is not int
        or stage_id not in (0, 1)
        or cp_size != 4
        or cp_rank not in range(4)
        or tuple(cp_group) != expected_stage_group
        or len(set(cp_group)) != 4
    ):
        return _fallback(
            reason="UNSUPPORTED_CP_TOPOLOGY", state_read_still_required=True, **base
        )
    if not feature_requested:
        return _fallback(
            reason="FEATURE_DEFAULT_OFF", state_read_still_required=True, **base
        )
    if any(
        (
            _get(ctx, "kv_cache_sharded", False),
            cuda_graph,
            warmup,
            external_prefix,
            pending_lease,
            _get(ctx, "batch_size", 1) != 1,
            _get(ctx, "varlen", False),
            _get(ctx, "decode", False),
        )
    ):
        return _fallback(
            reason="UNSUPPORTED_UNIFORM_CONDITION",
            state_read_still_required=True,
            **base,
        )
    if attention_type not in ("CSA", "HCA"):
        return _fallback(
            reason="UNSUPPORTED_ATTENTION_TYPE", state_read_still_required=True, **base
        )
    if any(pool.sharded for pool in pools.values()):
        return _fallback(reason="SHARDED_POOL", state_read_still_required=True, **base)
    positions = _host_tuple(_get(ctx, "global_positions"), "global_positions")
    real_mask = _host_tuple(_get(ctx, "local_is_real"), "local_is_real")
    if len(positions) != len(real_mask) or not all(type(x) is int for x in positions):
        raise CompactCPIneligible("CPContext positions/mask are malformed")
    local = tuple(p for p, is_real in zip(positions, real_mask) if bool(is_real))
    if not local:
        return _fallback(reason="NO_REAL_ROWS", state_read_still_required=True, **base)
    runs = _zigzag_intervals(local)
    if len(runs) != 2:
        return _fallback(
            reason="NOT_TWO_ZIGZAG_INTERVALS", state_read_still_required=True, **base
        )
    # CPContext has prefix_length, not current_chunk_start.  Never guess from
    # rank-local zigzag positions: ranks 1..3 do not begin at the chunk start.
    chunk_start = _get(ctx, "current_chunk_start", _get(ctx, "prefix_length"))
    chunk_length = _get(
        ctx, "current_chunk_length", _get(ctx, "seq_len_full", len(local))
    )
    padded = _get(ctx, "padded_seq_len", cp_size * len(positions))
    if (
        type(chunk_start) is not int
        or type(chunk_length) is not int
        or chunk_length <= 0
        or type(request_start) is not int
        or request_start < 0
        or request_start > chunk_start
    ):
        return _fallback(
            reason="MISSING_STABLE_HOST_REQUEST_GEOMETRY",
            state_read_still_required=True,
            **base,
        )
    base.update(
        chunk_start=chunk_start,
        chunk_length=chunk_length,
        padded_length=padded,
        request_start=request_start,
    )
    if padded != cp_size * len(positions) or any(
        p < chunk_start or p >= chunk_start + chunk_length for p in local
    ):
        return _fallback(
            reason="INVALID_PADDED_OR_CHUNK_GEOMETRY",
            state_read_still_required=True,
            **base,
        )
    if len(local) * cp_size != chunk_length:
        return _fallback(
            reason="NONUNIFORM_REAL_GEOMETRY", state_read_still_required=True, **base
        )
    all_positions = rank_global_positions or {cp_rank: local}
    if set(all_positions) != set(range(4)):
        return _fallback(
            reason="MISSING_RANK_UNIFORM_GEOMETRY",
            state_read_still_required=True,
            **base,
        )
    if tuple(all_positions.get(cp_rank, ())) != local:
        return _fallback(
            reason="LOCAL_CONTEXT_PEER_GEOMETRY_MISMATCH",
            state_read_still_required=True,
            **base,
        )
    try:
        owner = _owner_map(
            {rank: tuple(values) for rank, values in all_positions.items()}
        )
    except CompactCPIneligible:
        return _fallback(
            reason="RANK_GEOMETRY_GAP_OR_OUTSIDE_CHUNK",
            state_read_still_required=True,
            **base,
        )
    expected = set(range(chunk_start, chunk_start + chunk_length))
    if set(owner) != expected:
        return _fallback(
            reason="RANK_GEOMETRY_GAP_OR_OUTSIDE_CHUNK",
            state_read_still_required=True,
            **base,
        )
    ratio = 4 if attention_type == "CSA" else 128
    if attention_type == "HCA" and any(
        (start - base["request_start"]) % ratio or length % ratio
        for start, length in runs
    ):
        return _fallback(
            reason="HCA_TAIL_OR_UNALIGNED_WINDOW",
            state_read_still_required=True,
            **base,
        )
    if attention_type == "CSA" and any(
        (start - base["request_start"]) % ratio or length % ratio
        for start, length in runs
    ):
        return _fallback(
            reason="CSA_UNALIGNED_WINDOW", state_read_still_required=True, **base
        )
    intervals = tuple(
        CompactCPInterval(start, length, offset)
        for offset, (start, length) in zip((0, runs[0][1]), runs)
    )
    history = (
        ()
        if chunk_start == base["request_start"]
        else ((base["request_start"], chunk_start - base["request_start"]),)
    )
    history_owner = {}
    if history:
        if history_rank_global_positions is None or set(
            history_rank_global_positions
        ) != set(range(4)):
            return _fallback(
                reason="MISSING_SAME_REQUEST_HISTORY_GEOMETRY",
                state_read_still_required=True,
                **base,
            )
        try:
            history_owner = _owner_map(
                {
                    rank: tuple(values)
                    for rank, values in history_rank_global_positions.items()
                }
            )
        except CompactCPIneligible:
            return _fallback(
                reason="INVALID_SAME_REQUEST_HISTORY_GEOMETRY",
                state_read_still_required=True,
                **base,
            )
    halos = []
    tails = []
    if attention_type == "CSA":
        for seg, interval in enumerate(intervals):
            tails.append(tuple(range(interval.end - 4, interval.end)))
            for pos in range(interval.absolute_start - 4, interval.absolute_start):
                if pos < base["request_start"]:
                    continue
                source = owner.get(pos)
                source_kind = (
                    "current_raw" if pos >= chunk_start else "same_request_history"
                )
                if source is None and pos < chunk_start:
                    source = history_owner.get(pos)
                if source is None:
                    return _fallback(
                        reason="MISSING_CSA_EXACT_HALO",
                        state_read_still_required=True,
                        **base,
                    )
                halos.append(
                    HaloRow(
                        pos, source[0], source[1], source[0] == cp_rank, source_kind
                    )
                )
    tags = (CSA_INDEXER, CSA_MAIN) if attention_type == "CSA" else (HCA_MAIN,)
    if set(pools) != set(tags):
        return _fallback(
            reason="POOL_ROLE_MISMATCH", state_read_still_required=True, **base
        )
    boundaries = tuple(
        pos
        for pos in range(chunk_start, chunk_start + chunk_length)
        if (pos - base["request_start"] + 1) % ratio == 0
    )
    source_boundaries = tuple(pos for pos in boundaries if owner[pos][0] == cp_rank)
    recv_rows, src_rows, receiver_slots = [], [], []
    for tag in tags:
        spec = pools[tag]
        table = dict(
            _immutable_mapping(spec.receiver_block_table, f"{tag} block table")
        )
        for boundary in boundaries:
            slot = table.get(boundary, spec.allocator_unallocated)
            if slot == spec.allocator_unallocated or not 0 <= slot < spec.capacity:
                return _fallback(
                    reason="UNRESOLVED_RECEIVER_SLOT",
                    state_read_still_required=True,
                    **base,
                )
            row = LogicalCompressedRow(request_id, boundary, tag, ratio)
            recv_rows.append(row)
            receiver_slots.append((row, slot))
            if boundary in source_boundaries:
                src_rows.append(row)
    schedule = tuple(CollectiveStep(i, tag) for i, tag in enumerate(tags))
    needed_workspace = sum(
        (pools[tag].data_row_bytes + pools[tag].scale_row_bytes)
        * len(source_boundaries)
        for tag in tags
    )
    if workspace_bytes < needed_workspace:
        return _fallback(
            reason="INSUFFICIENT_PACK_WORKSPACE", state_read_still_required=True, **base
        )
    return CompactCPPlan(
        intervals=intervals,
        eligible=True,
        fallback_reason=None,
        same_request_history=history,
        csa_halos=tuple(halos),
        published_tails=tuple(tails),
        source_rows=tuple(src_rows),
        receive_rows=tuple(recv_rows),
        receiver_slots=tuple(receiver_slots),
        schedule=schedule,
        workspace_bytes=needed_workspace,
        state_read_still_required=chunk_start > base["request_start"],
        **base,
    )
