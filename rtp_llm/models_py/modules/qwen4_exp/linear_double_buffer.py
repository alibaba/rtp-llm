"""Pool-owned FP32 GDN state with indexed replay records.

Only the cache manager owns or publishes state. This module creates views and
small index/pointer metadata. Double-buffer verification writes scratch and
partial acceptance reconstructs it before C++ publishes a bank swap. Single
SSM verification is read-only; every nonempty accepted prefix replays in place.
An in-place commit is irreversible once submitted. Native code must wait for
the caller's stream and reach cross-rank consensus before calling finalize.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

# Match kDummyRequestStateBlockId in cpp/cache/CacheGroupType.h. The graph
# initializer marks all state/replay slots together; allocator-owned live rows are
# positive. Neither the null ID (-1) nor a partially marked row is a dummy.
DUMMY_REQUEST_STATE_BLOCK_ID = -2


def _pool_rows(base: torch.Tensor, dtype: torch.dtype, width: int, tag: str):
    if base.ndim < 2 or not base.is_contiguous() or base.dtype != dtype:
        raise ValueError(f"{tag} requires contiguous {dtype} cache-manager storage")
    rows = base.view(base.shape[0], -1)
    if rows.shape[0] < 1 or rows.shape[1] != width:
        raise ValueError(f"{tag} row width differs from the declared state layout")
    return rows


def split_pool_views(
    conv_base,
    state_base,
    replay_base,
    *,
    heads: int,
    key_dim: int,
    value_dim: int,
    conv_dim: int,
    conv_length: int,
    query: int,
    combined_linear: bool = False,
):
    """Return logical conv/KV views and physical FP32 replay rows, without copies."""
    if key_dim != 128 or value_dim != 128 or query not in (2, 3, 4):
        raise ValueError("Split-state hybrid requires FP32 128x128 chain MTP")
    if min(heads, conv_dim, conv_length) <= 0:
        raise ValueError("Split-state pool dimensions must be positive")
    if not (conv_base.device == state_base.device == replay_base.device):
        raise ValueError("Split-state pools must share one device")
    if combined_linear:
        if (
            conv_base.ndim != 2
            or conv_base.dtype != torch.bfloat16
            or conv_base.shape[1] != conv_length * conv_dim
            or conv_base.stride(1) != 1
        ):
            raise ValueError("Combined linear pool has an invalid convolution view")
        conv = conv_base
    else:
        conv = _pool_rows(conv_base, torch.bfloat16, conv_length * conv_dim, "linear")
    state = _pool_rows(
        state_base, torch.float32, heads * value_dim * key_dim, "linear_ssm"
    )
    replay = _pool_rows(
        replay_base, torch.float32, query * heads * 3 * key_dim, "linear_replay"
    )
    # The physical SSM layout is V,K. ATOM exposes a logical K,V view; the
    # FlyDSL wrapper transposes it back without allocating a temporary state.
    return (
        conv.view(-1, conv_length, conv_dim).transpose(-1, -2),
        state.view(-1, heads, value_dim, key_dim).transpose(-1, -2),
        replay.view(-1, query, heads, 3, key_dim),
    )


@triton.jit
def _resolve_pool_indices(
    SSM,
    RECORD,
    PREFIX,
    READ,
    WRITE,
    RECORD_IDS,
    INVALID,
    ROWS: tl.constexpr,
    SSM_STRIDE: tl.constexpr,
    RECORD_STRIDE: tl.constexpr,
    SSM_COL_STRIDE: tl.constexpr,
    SSM_ROWS: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    QUERY: tl.constexpr,
    MINIMUM_LENGTH: tl.constexpr,
    DUMMY_STATE_ID: tl.constexpr,
    TILE: tl.constexpr,
    STATE_SLOTS: tl.constexpr = 2,
):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    exists = row < ROWS
    real = exists & (tl.load(PREFIX + row, exists, 0) > MINIMUM_LENGTH)
    src = tl.load(SSM + row * SSM_STRIDE, exists, 0)
    if STATE_SLOTS == 1:
        dst = src
    else:
        dst = tl.load(SSM + row * SSM_STRIDE + SSM_COL_STRIDE, exists, 0)
    record = tl.load(RECORD + row * RECORD_STRIDE, exists, 0)
    valid = (src > 0) & (src < SSM_ROWS) & (dst > 0) & (dst < SSM_ROWS)
    valid &= (record > 0) & (record < RECORD_ROWS)
    if STATE_SLOTS == 2:
        valid &= src != dst
    dummy = (src == DUMMY_STATE_ID) & (dst == DUMMY_STATE_ID)
    dummy &= record == DUMMY_STATE_ID
    live = real & valid
    # Graph-initialization dummies retain normal attention length geometry.
    # They, invalid rows, and padding all resolve to reserved row zero. A
    # partial dummy marker still fails validation before any real pool write.
    tl.store(READ + row, tl.where(live, src, 0), exists)
    tl.store(RECORD_IDS + row, tl.where(live, record, 0), exists)
    for token in tl.static_range(QUERY):
        tl.store(WRITE + row * QUERY + token, tl.where(live, dst, 0), exists)
    tl.store(
        INVALID + tl.program_id(0), tl.sum((real & ~valid & ~dummy).to(tl.int32), 0)
    )


def resolve_pool_indices(
    ssm_table,
    replay_table,
    prefix_lengths,
    buffers,
    state_rows,
    replay_rows,
    *,
    minimum_length=0,
    state_slots=2,
):
    """Refresh capture-stable small index buffers from the current device tables."""
    reads, writes, record_ids, invalid = buffers
    rows, query = writes.shape
    if (
        state_slots not in (1, 2)
        or ssm_table.ndim != 2
        or replay_table.ndim != 2
        or ssm_table.shape[0] != rows
        or replay_table.shape[0] != rows
        or ssm_table.shape[1] != state_slots
        or replay_table.shape[1] != 1
        or prefix_lengths.ndim != 1
        or prefix_lengths.numel() != rows
    ):
        raise ValueError(
            f"Split-state tables require [B,{state_slots}], [B,1], and prefix[B]"
        )
    for tensor in (
        ssm_table,
        replay_table,
        prefix_lengths,
        reads,
        writes,
        record_ids,
        invalid,
    ):
        if tensor.dtype != torch.int32 or tensor.device != reads.device:
            raise ValueError("Split-state index metadata must be same-device int32")
    if not prefix_lengths.is_contiguous():
        raise ValueError("Split-state prefix lengths must be contiguous")
    _resolve_pool_indices[(triton.cdiv(rows, 128),)](
        ssm_table,
        replay_table,
        prefix_lengths,
        reads,
        writes,
        record_ids,
        invalid,
        rows,
        ssm_table.stride(0),
        replay_table.stride(0),
        ssm_table.stride(1),
        state_rows,
        replay_rows,
        query,
        minimum_length,
        DUMMY_REQUEST_STATE_BLOCK_ID,
        128,
        STATE_SLOTS=state_slots,
    )
    torch._assert_async(torch.all(invalid == 0), "Invalid live split-state pool row")


@dataclass(frozen=True)
class LinearDoubleBufferStage:
    states: tuple[torch.Tensor, ...]
    records: tuple[torch.Tensor, ...]
    state_ptrs: torch.Tensor
    record_ptrs: torch.Tensor
    reads: torch.Tensor
    writes: torch.Tensor
    record_ids: torch.Tensor
    in_place: bool = False
    checkpoint: Optional[tuple] = None

    @property
    def query(self):
        return self.writes.shape[1]


@triton.jit
def _partial_replay(
    STATE_PTRS,
    RECORD_PTRS,
    READ,
    WRITE,
    RECORD_IDS,
    ACCEPT,
    CHECKPOINT_PTRS,
    PAGED_WRITES,
    SEQUENCE_LENGTHS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUERY: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    RECORD_STRIDE: tl.constexpr,
    TILE_V: tl.constexpr,
    ACCEPT_STRIDE: tl.constexpr,
    IN_PLACE: tl.constexpr = False,
    CHECKPOINT_PERIOD: tl.constexpr = 0,
    CHECKPOINT_STRIDE: tl.constexpr = 0,
    PAGED_WRITE_STRIDE: tl.constexpr = 0,
    HEAD_FIRST: tl.constexpr = False,
):
    row = tl.program_id(1) if HEAD_FIRST else tl.program_id(0)
    layer_head = tl.program_id(0) if HEAD_FIRST else tl.program_id(1)
    layer = layer_head // HEADS
    head = layer_head % HEADS
    count = tl.load(ACCEPT + row * ACCEPT_STRIDE)
    src = tl.load(READ + row)
    dst = src if IN_PLACE else tl.load(WRITE + row * QUERY)
    record = tl.load(RECORD_IDS + row)
    valid = (src > 0) & (src < STATE_ROWS) & (dst > 0) & (dst < STATE_ROWS)
    valid &= (record > 0) & (record < RECORD_ROWS)
    if not IN_PLACE:
        valid &= src != dst
    # Single SSM has no full-accept checkpoint: every accepted prefix must
    # replay. Each program owns disjoint V rows and loads them before storing;
    # k/u/decay are immutable records, so src == dst has no cross-tile hazard.
    replay = IN_PLACE | (count < QUERY)
    checkpoint_step = 0
    checkpoint_slot = -1
    if CHECKPOINT_PERIOD:
        # RTP target-verify sequence lengths include the first uncached token.
        cached = tl.load(SEQUENCE_LENGTHS + row) - 1
        checkpoint_step = CHECKPOINT_PERIOD - cached % CHECKPOINT_PERIOD
        if (cached > 0) & (checkpoint_step <= count) & (checkpoint_step <= QUERY):
            checkpoint_slot = tl.load(
                PAGED_WRITES + row * PAGED_WRITE_STRIDE + checkpoint_step - 1
            )
    if valid & (count > 0) & (count <= QUERY) & (replay | (checkpoint_slot > 0)):
        state = tl.load(STATE_PTRS + layer).to(tl.pointer_type(tl.float32))
        records = tl.load(RECORD_PTRS + layer).to(tl.pointer_type(tl.float32))
        vv = tl.program_id(2) * TILE_V + tl.arange(0, TILE_V)
        kk = tl.arange(0, DIM)
        offset = head * DIM * DIM + vv[:, None] * DIM + kk[None, :]
        value = tl.load(
            state + src.to(tl.int64) * STATE_STRIDE + offset, vv[:, None] < DIM, 0
        )
        for token in tl.static_range(QUERY):
            base = (
                record.to(tl.int64) * RECORD_STRIDE + (token * HEADS + head) * 3 * DIM
            )
            key = tl.load(records + base + kk)
            update = tl.load(records + base + DIM + vv, vv < DIM, 0)
            decay = tl.load(records + base + 2 * DIM)
            value = tl.where(
                token < count, value * decay + update[:, None] * key[None, :], value
            )
            if CHECKPOINT_PERIOD:
                if (checkpoint_slot > 0) & (checkpoint_step == token + 1):
                    checkpoint = tl.load(CHECKPOINT_PTRS + layer).to(
                        tl.pointer_type(tl.float32)
                    )
                    tl.store(
                        checkpoint + checkpoint_slot.to(tl.int64) * CHECKPOINT_STRIDE + offset,
                        value, vv[:, None] < DIM,
                    )
        if replay:
            tl.store(
                state + dst.to(tl.int64) * STATE_STRIDE + offset, value, vv[:, None] < DIM
            )


def replay_partial_acceptance(stage: LinearDoubleBufferStage, accept: torch.Tensor):
    state, records = stage.states[0], stage.records[0]
    ptrs, paged_writes, lengths, period, stride = (
        stage.checkpoint if stage.checkpoint is not None
        else (stage.state_ptrs, stage.writes, stage.reads, 0, 0)
    )
    rows = accept.numel()
    layer_heads = len(stage.states) * state.shape[1]
    # The gfx942 non-PD MTP2 batch visits 36 layers of 24 heads per rank.
    head_first = (
        rows == 16
        and stage.in_place
        and stage.query == 3
        and period > 0
        and len(stage.states) == 36
        and state.shape[1] == 24
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(state.device).gcnArchName.split(":")[0] == "gfx942"
    )
    tile_v = 8 if head_first else 32
    grid = (layer_heads, rows, 128 // tile_v) if head_first else (rows, layer_heads, 128 // tile_v)
    _partial_replay[grid](
        stage.state_ptrs,
        stage.record_ptrs,
        stage.reads,
        stage.writes,
        stage.record_ids,
        accept,
        ptrs,
        paged_writes,
        lengths,
        state.shape[1],
        128,
        stage.query,
        state.shape[0],
        records.shape[0],
        state.stride(0),
        records.stride(0),
        tile_v,
        ACCEPT_STRIDE=accept.stride(0),
        IN_PLACE=stage.in_place,
        CHECKPOINT_PERIOD=period,
        CHECKPOINT_STRIDE=stride,
        PAGED_WRITE_STRIDE=paged_writes.stride(0),
        HEAD_FIRST=head_first,
        num_warps=1 if head_first else 4,
    )


@triton.jit
def _validate_acceptance(ACCEPT, READ, VALID, ROWS: tl.constexpr,
                         QUERY: tl.constexpr, TILE: tl.constexpr,
                         ACCEPT_STRIDE: tl.constexpr, READ_STRIDE: tl.constexpr):
    row = tl.arange(0, TILE)
    live = row < ROWS
    count = tl.load(ACCEPT + row * ACCEPT_STRIDE, live, 0)
    state = tl.load(READ + row * READ_STRIDE, live, 0)
    valid = (count >= 0) & (count <= QUERY) & ((count == 0) | (state > 0))
    tl.store(VALID, tl.sum((live & ~valid).to(tl.int32), 0) == 0)


class LinearDoubleBufferTransaction:
    """Shared transaction; native owns completion consensus and publication."""

    def __init__(self, commit: Callable = replay_partial_acceptance, *, in_place=False):
        if type(in_place) is not bool:
            raise TypeError("The in-place commit mode must be a bool")
        self.in_place = in_place
        self.active: Optional[LinearDoubleBufferStage] = None
        self.graphs: dict[int, Optional[LinearDoubleBufferStage]] = {}
        self.pending: Optional[torch.Tensor] = None
        self.submitted = False
        self.poisoned = False
        self._commit = commit
        self._valid_acceptance = None

    def bind(self, stage):
        if self.poisoned or self.pending is not None:
            raise RuntimeError(
                "Cannot replace split-state indices during a transaction or after failure"
            )
        if stage is not None and stage.in_place is not self.in_place:
            raise ValueError("Stage and transaction state layouts disagree")
        self.active = stage

    def clear(self):
        self.bind(None)

    def save(self, key):
        if self.poisoned or self.pending is not None:
            raise RuntimeError("Cannot save a graph during a double-buffer transaction")
        # None denotes a non-speculative single-token warmup/decode graph.
        self.graphs[int(key)] = self.active

    def activate(self, key):
        self.bind(self.graphs[int(key)])

    def prepare(self, accept):
        stage = self.active
        if self.poisoned or stage is None or self.pending is not None:
            raise RuntimeError("Invalid double-buffer transaction prepare")
        if (
            accept.ndim != 1
            or accept.dtype != torch.int32
            or accept.device != stage.reads.device
            or not 0 < accept.numel() <= stage.reads.numel()
        ):
            raise ValueError("Invalid double-buffer accept vector")
        if accept.is_cuda:
            # One fused range/ownership reduction replaces the eager chain of
            # comparisons, boolean operations and all(). Keep the async device
            # assertion and native fail-stop completion contract intact.
            if self._valid_acceptance is None or self._valid_acceptance.device != accept.device:
                self._valid_acceptance = torch.empty((), dtype=torch.bool, device=accept.device)
            _validate_acceptance[(1,)](
                accept, stage.reads, self._valid_acceptance, accept.numel(),
                stage.query, triton.next_power_of_2(accept.numel()),
                accept.stride(0), stage.reads.stride(0), num_warps=4,
            )
            valid = self._valid_acceptance
        else:
            valid = torch.all((accept >= 0) & (accept <= stage.query)
                              & ((accept == 0) | (stage.reads[: accept.numel()] > 0)))
        torch._assert_async(valid, "Double-buffer acceptance is out of range")
        self.pending = accept
        self.submitted = False

    def finish(self, commit):
        if not commit:
            if self.in_place and (self.submitted or self.poisoned):
                self.poisoned = True
                raise RuntimeError(
                    "In-place SSM commit has started; abort requires fail-stop"
                )
            # A failed peer or local launch may have written scratch. Since
            # C++ has not published the bank swap, dropping it is sufficient.
            self.pending = None
            self.submitted = False
            self.active = None
            return
        if self.poisoned or self.pending is None or self.submitted:
            raise RuntimeError("Double-buffer commit must be prepared exactly once")
        # Mark before the launch: an exception may be raised after some writes
        # have been enqueued. No retry or rollback can recover the original A.
        self.submitted = True
        try:
            self._commit(self.active, self.pending)
        except Exception:
            if self.in_place:
                self.poisoned = True
            raise
        # Triton launches on the caller's current stream. The native executor
        # waits only for that stream before commit consensus; do not add a
        # second synchronization or change streams here.

    def finalize(self):
        if self.poisoned or self.pending is None or not self.submitted:
            raise RuntimeError("Double-buffer finalize requires a successful finish")
        self.pending = None
        self.submitted = False
        self.active = None
