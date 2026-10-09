"""Compact CP transport preserves compressor arithmetic and replicated state tails.

Per-call handles own transient tensors; scratch is scoped to the forward."""

import logging
import os
from dataclasses import dataclass, replace
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8._cp_packed_rows import (
    RowLayout,
    pack_rows,
    scatter_packed_rows,
)

FLAG = "DSV4_CP_COMPACT_COMPRESSOR"

# Process-level caches for the chunk/request-invariant plan constants.
# Both default ON; the values they cache are pure functions of the geometry
# (no request state), so caching is byte-identical.  Kill switches exist so a
# bad deployment can revert to the per-forward rebuild without a rebuild.
_PLAN_CACHE_FLAG = "DSV4_CP_COMPACT_PLAN_CACHE"
_DEFERRED_VERIFY_FLAG = "DSV4_CP_COMPACT_DEFERRED_VERIFY"


def enabled():
    return os.environ.get(FLAG, "0") == "1"


def _flag_enabled(name: str, default: str) -> bool:
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _plan_cache_enabled() -> bool:
    # Enabled by default; set the flag to 0 to use the uncached path.
    return _flag_enabled(_PLAN_CACHE_FLAG, "1")


def _deferred_verify_enabled() -> bool:
    # Enabled by default; set the flag to 0 to use the uncached path.
    return _flag_enabled(_DEFERRED_VERIFY_FLAG, "1")


# (order, argsort(order)) for the one validated padded geometry (CP4 zigzag,
# 4096 padded tokens).  Pure host constant — built once per process instead of
# per chunk, avoiding repeated host arange/cat/sort operations.
_ZIGZAG_CONSTANTS: Optional[tuple] = None


def _zigzag_constants():
    global _ZIGZAG_CONSTANTS
    if _ZIGZAG_CONSTANTS is None:
        order = torch.cat(
            [
                torch.cat(
                    (
                        torch.arange(r * 512, (r + 1) * 512),
                        torch.arange((7 - r) * 512, (8 - r) * 512),
                    )
                )
                for r in range(4)
            ]
        )
        _ZIGZAG_CONSTANTS = (order, torch.argsort(order))
    return _ZIGZAG_CONSTANTS


def _verify_content(mask, restore):
    """The content verdict: identical ops on identical values as the original
    blocking path (the order constant is the same pure geometry value, now
    process-cached)."""
    _, expected = _zigzag_constants()
    return bool(
        mask.shape == (4096,)
        and torch.all(mask == 1)
        and restore.shape == (4096,)
        and torch.equal(restore, expected)
    )


class _PendingVerify:
    """Deferred content verdict for device-resident mask/restore.

    The two blocking pageable DtoH reads (16 KB each, with full stream
    drains) are replaced by one non_blocking pinned DtoH pair enqueued on the
    current stream at context-build time plus an event.  ``resolve`` runs at
    the first ``select_geometry`` call — the compact path's first consumer —
    long after the copy completed, so the host never drains the queue.  The
    comparison itself is the unchanged ``_verify_content`` over the pinned
    (byte-exact) copies.
    """

    def __init__(self, padding_mask, restore_indices):
        # Hold the sources so a resolve-time failure can fall back to the
        # legacy blocking read (they are framework-owned for the forward).
        self._mask_src = padding_mask
        self._restore_src = restore_indices
        self._pinned_mask = torch.empty(
            tuple(padding_mask.shape), dtype=padding_mask.dtype, pin_memory=True
        )
        self._pinned_restore = torch.empty(
            tuple(restore_indices.shape), dtype=restore_indices.dtype, pin_memory=True
        )
        self._pinned_mask.copy_(padding_mask, non_blocking=True)
        self._pinned_restore.copy_(restore_indices, non_blocking=True)
        self._event = torch.cuda.Event()
        self._event.record()

    def resolve(self) -> bool:
        try:
            self._event.synchronize()
            mask = self._pinned_mask.reshape(-1)
            restore = self._pinned_restore.reshape(-1).to(torch.long)
        except Exception:
            # Same legacy blocking reads as the pre-deferral path.
            mask = self._mask_src.detach().to(device="cpu").reshape(-1)
            restore = (
                self._restore_src.detach()
                .to(device="cpu", dtype=torch.long)
                .reshape(-1)
            )
        return _verify_content(mask, restore)


def resolve_compact_geometry_verified(cp) -> bool:
    """Single reader-side entry point for the (possibly deferred) verdict.

    Only ``select_geometry`` consumes the verdict; it runs at the first
    CSA/HCA layer's compressor, well after the context build, so the deferred
    readback has completed and ``resolve`` never stalls the hot path.  A
    forward without a pending check keeps the eager verdict; a forward whose
    verdict was never resolved fails safe (``False`` = generic path).
    """
    if getattr(cp, "compact_geometry_verified", False):
        return True
    pending = getattr(cp, "_compact_geometry_pending", None)
    if pending is None:
        return False
    cp._compact_geometry_pending = None
    verdict = bool(pending.resolve())
    cp.compact_geometry_verified = verdict
    if not getattr(resolve_compact_geometry_verified, "_logged", False):
        logging.getLogger(__name__).info(
            "[dsv4-compact-cp] deferred geometry verify resolved (verdict=%s)",
            verdict,
        )
        resolve_compact_geometry_verified._logged = True
    return verdict


def verified_geometry(cp, padding_mask, restore_indices):
    """Once per forward, never per layer. No inference from flag presence."""
    if not enabled() or cp.cp_size != 4 or cp.chunk_length != 1024:
        return False
    if (
        cp.padded_seq_len != 4096
        or cp.seq_len_full != 4096
        or cp.chunk_lengths_per_req != (1024,)
        or cp.kv_cache_sharded
        or cp.prefix_length < 0
        or cp.prefix_length % 4096
        or cp.prefix_length > 7 * 4096
    ):
        return False
    # CPContext generates the two local intervals itself. Validate the two
    # externally supplied maps once, including actual gathered-rank ordering.
    # Device-resident sources take the deferred readback: the verdict is
    # provisional until ``resolve_compact_geometry_verified`` (called by
    # ``select_geometry``, the only consumer) forces it.  Provisional False
    # fails safe — a non-resolving reader keeps the generic path.
    if (
        _deferred_verify_enabled()
        and padding_mask.is_cuda
        and restore_indices.is_cuda
        and torch.cuda.is_available()
        and not torch.cuda.is_current_stream_capturing()
    ):
        try:
            cp._compact_geometry_pending = _PendingVerify(padding_mask, restore_indices)
            return False
        except Exception:
            # Pinned alloc / copy launch failure: keep the legacy verdict.
            cp._compact_geometry_pending = None
    mask = padding_mask.detach().to(device="cpu").reshape(-1)
    restore = restore_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    return _verify_content(mask, restore)


@dataclass(frozen=True)
class Geometry:
    rank: int
    start: int
    state_entries: int
    state_block: int
    ratio: int
    width: int

    def __post_init__(self):
        if not (
            type(self.rank) is int
            and 0 <= self.rank < 4
            and type(self.start) is int
            and 0 <= self.start < 32768
            and self.start % 4096 == 0
            and self.ratio in (4, 128)
            and self.width in (512, 1024, 2048)
            and type(self.state_block) is int
            and 0 < self.state_block <= 256
            and 512 % self.state_block == 0
            and type(self.state_entries) is int
            and 4 <= self.state_entries <= self.state_block
        ):
            raise ValueError("unsupported compact CP geometry")

    def intervals(self, rank=None):
        r = self.rank if rank is None else rank
        return ((r * 512, (r + 1) * 512), ((7 - r) * 512, (8 - r) * 512))

    def tails(self, rank=None):
        return tuple(
            p
            for lo, hi in self.intervals(rank)
            for end in range(lo + self.state_block, hi + 1, self.state_block)
            for p in range(end - self.state_entries, end)
        )

    def wire_boundaries(self):
        return tuple(
            p
            for r in range(4)
            for lo, hi in self.intervals(r)
            for p in range(lo + self.ratio - 1, hi, self.ratio)
        )

    def bytes_per_rank(self):
        data = 132 if self.width == 512 else 584
        return {
            "raw_tail_send": len(self.tails()) * self.width * 4,
            "raw_tail_receive": 4 * len(self.tails()) * self.width * 4,
            "full_raw_send_reference": 1024 * self.width * 4,
            "compressed_send": 1024 // self.ratio * (data + 8),
        }


# Process-level cache for the geometry-pure plan constants.  Every entry is a
# pure function of (rank, state_entries, state_block, ratio, width, device) —
# the values are chunk- and request-invariant, so one build per process
# replaces the per-forward host list build + pageable ``torch.tensor(...,
# device=cuda)`` upload (the deep queue drain in the chunk head).  Consumers
# treat every cached tensor as read-only (verified by tests).
_GEOMETRY_PLAN_CACHE: dict = {}


def _build_geometry_plan(g: "Geometry", device) -> tuple:
    """Construct the geometry-pure plan tensors (one process build)."""
    tail_index_vals = []
    tail_dest_vals = []
    local_map = tuple(p for lo, hi in g.intervals() for p in range(lo, hi))
    lut = {p: i for i, p in enumerate(local_map)}
    tail_index_vals = [lut[p] for p in g.tails()]
    tail_dest_vals = [p for r in range(4) for p in g.tails(r)]
    boundary_vals = list(g.wire_boundaries())
    flat = torch.tensor(
        tail_index_vals + tail_dest_vals + boundary_vals,
        dtype=torch.long,
        device=device,
    )
    n_tail = len(tail_index_vals)
    n_dest = len(tail_dest_vals)
    tail_indices = flat[:n_tail]
    tail_dest = flat[n_tail : n_tail + n_dest]
    boundaries = flat[n_tail + n_dest :]
    local_boundaries = boundaries[
        g.rank * (1024 // g.ratio) : (g.rank + 1) * (1024 // g.ratio)
    ]
    covered_state = torch.zeros(4096, dtype=torch.bool, device=device)
    covered_state[tail_dest] = True
    arange_base = torch.arange(4096, device=device, dtype=torch.long)
    return (
        tail_indices,
        tail_dest,
        boundaries,
        local_boundaries,
        covered_state,
        arange_base,
    )


def _geometry_plan(g: "Geometry", device) -> tuple:
    """Process-cached geometry-pure plan constants (byte-identical values).

    Kill switch ``DSV4_CP_COMPACT_PLAN_CACHE=0`` rebuilds per call, matching
    the pre-cache per-forward behavior exactly.
    """
    if not _plan_cache_enabled():
        return _build_geometry_plan(g, device)
    key = (
        g.rank,
        g.state_entries,
        g.state_block,
        g.ratio,
        g.width,
        str(device),
    )
    cached = _GEOMETRY_PLAN_CACHE.get(key)
    if cached is None:
        cached = _build_geometry_plan(g, device)
        _GEOMETRY_PLAN_CACHE[key] = cached
        logging.getLogger(__name__).info(
            "[dsv4-compact-cp] geometry plan cached (rank=%d ratio=%d device=%s); "
            "per-chunk plan uploads eliminated",
            g.rank,
            g.ratio,
            device,
        )
    return cached


def _build_chunk_plan(g: "Geometry", device, meta, pool_cap) -> tuple:
    """The per-forward chunk plan: geometry constants from the process cache,
    start-dependent vectors and meta-dependent gathers computed per forward.

    Byte-faithful: the geometry constants are the same values the pre-cache
    code built per forward; the per-forward parts are the unchanged device
    ops (index_select / sort / adds) over them and the per-chunk meta.
    """
    (
        tail_indices,
        tail_dest,
        boundaries,
        local_boundaries,
        covered_state,
        arange_base,
    ) = _geometry_plan(g, device)
    # The original fused writer checks a negative KV slot only after
    # reading raw operands. Use ONLY owned boundary programs, not a
    # full grid with remote slots masked; otherwise it reads unstaged
    # bytes. The separate state writer retains its full original grid.
    masked = replace(
        meta,
        positions=meta.positions.index_select(0, local_boundaries),
        b_idx=meta.b_idx.index_select(0, local_boundaries),
        state_slots=meta.state_slots.index_select(0, local_boundaries),
        kv_slots=meta.kv_slots.index_select(0, local_boundaries),
        token_to_req=meta.token_to_req.index_select(0, local_boundaries),
    )
    expected_positions = arange_base + g.start
    receiver = meta.kv_slots.index_select(0, boundaries)
    ordered_slots = receiver.sort().values
    # Forward-constant pack slots for this role's owned boundaries;
    # the metadata snapshot is immutable for the forward.
    boundary_kv_slots = masked.kv_slots
    expected_wire_ids = boundaries + g.start
    local_boundary_ids = local_boundaries + g.start
    _assert_fused(
        _plan_metadata_checks(
            meta,
            expected_positions,
            covered_state,
            ordered_slots,
            receiver,
            pool_cap,
            g,
        )
    )
    return (
        tail_indices,
        tail_dest,
        boundaries,
        local_boundaries,
        masked,
        receiver,
        boundary_kv_slots,
        expected_wire_ids,
        local_boundary_ids,
    )


def select_geometry(module, cp, meta, fused):
    if not enabled() or cp is None or not resolve_compact_geometry_verified(cp):
        return None
    if (
        not fused.is_cuda
        or torch.cuda.is_current_stream_capturing()
        or tuple(fused.shape) != (1024, 2 * (1 + int(module.overlap)) * module.head_dim)
        or fused.dtype != torch.float32
        or not fused.is_contiguous()
        or cp.kv_cache_sharded
        or meta is None
        or meta.positions.shape != (4096,)
        or module._state_tokens_per_block <= 0
        or module._kv_pool_view is None
    ):
        return None
    if module.head_dim not in (128, 512) or module.compress_ratio not in (4, 128):
        return None
    if module.head_dim == 128 and module.compress_ratio != 4:
        return None
    try:
        return Geometry(
            cp.cp_rank,
            cp.prefix_length,
            module._state_eb,
            module._state_tokens_per_block,
            module.compress_ratio,
            fused.shape[1],
        )
    except ValueError:
        return None


def _assert_device(condition, message):
    # Asynchronous device assertion, before every dependent read/write. No
    # per-layer DtoH conversion or treating invalid slots as a permissible skip.
    torch._assert_async(condition, message)


def _plan_metadata_checks(
    meta,
    expected_positions,
    covered_state,
    ordered_slots,
    receiver_slots,
    pool_cap,
    geometry,
):
    """(condition, message) battery for the per-forward compact CP plan cache.

    This list is the single source of truth for the guard predicates: the
    production path asserts their fused conjunction in ONE async device
    assert, and tests drive the same list unfused.  Every predicate and
    message is the original one — no check is skipped, weakened, or made
    host-synchronous.

    ``receiver_slots`` (the forward-constant wire-destination plan) and
    ``pool_cap`` (guarded per layer by ``assert_binding``) are checked here
    once per forward instead of once per layer; ``assert_binding`` revalidates
    the pool identity before every replicate, so the capacity operand cannot
    change without the battery re-running (pool_cap is part of the cache key).
    """
    checks = [
        (
            (meta.positions == expected_positions).all(),
            "compact CP nonsequential full metadata",
        ),
        ((meta.b_idx == 0).all(), "compact CP multiple requests"),
        (
            (meta.token_to_req == 0).all(),
            "compact CP foreign token request mapping",
        ),
        (
            ((meta.state_slots < 0) | covered_state).all(),
            "compact CP missing state raw rows",
        ),
        (
            (ordered_slots[1:] != ordered_slots[:-1]).all(),
            "compact CP duplicate destinations",
        ),
    ]
    if pool_cap is not None:
        checks.append(
            (
                ((receiver_slots >= 0) & (receiver_slots < pool_cap)).all(),
                "compact CP invalid destination",
            )
        )
    if meta.is_batched:
        if meta.seq_start_per_req is None or meta.cu_seq_per_req is None:
            raise ValueError("compact CP missing B1 raw windows")
        checks.append(
            (
                (meta.seq_start_per_req == geometry.start).all(),
                "compact CP foreign prefix",
            )
        )
        cu = meta.cu_seq_per_req.reshape(-1)
        if cu.numel() != 2:
            # The pre-fusion code compared against ``torch.tensor([0, 4096],
            # device=...)``: numel==1 broadcast to a never-true comparison and
            # numel>2 raised a broadcast RuntimeError.  Both are malformed-meta
            # rejections before any write; fail closed with the guard message.
            raise ValueError("compact CP bad raw windows")
        checks.append(
            (
                (cu[0] == 0) & (cu[1] == 4096),
                "compact CP bad raw windows",
            )
        )
    return checks


def _assert_fused(checks):
    """One async device assert for a whole battery of independent conditions.

    ``torch._assert_async`` requires a scalar (multi-element input raises), so
    the per-check 0-dim conditions are stacked and reduced first.  The fused
    message concatenates every original message so failure logs keep the full
    predicate content.

    The CUDA device-assert path rejects a message of 255+ chars
    ("Message length must be smaller than 255").  Batteries whose joined
    message would exceed that are split into the fewest chunks that fit, each
    chunk carrying whole messages — every predicate's text is still asserted
    verbatim, and small batteries stay at exactly one assert.
    """
    if not checks:
        return
    if len(checks) == 1:
        _assert_device(checks[0][0], checks[0][1])
        return
    # Greedy whole-message packing under the 255-char device-assert limit;
    # each chunk asserts its own sub-conjunction so the message names the
    # predicates that actually fail.
    chunks: list = []
    cur_conds: list = []
    cur_msg = ""
    for condition, message in checks:
        candidate = message if not cur_msg else cur_msg + "; " + message
        if len(candidate) > 250 and cur_conds:
            chunks.append((cur_conds, cur_msg))
            cur_conds, cur_msg = [condition], message
        else:
            cur_conds.append(condition)
            cur_msg = candidate
    if cur_conds:
        chunks.append((cur_conds, cur_msg))
    for conds, msg in chunks:
        if len(conds) == 1:
            _assert_device(conds[0], msg)
        else:
            _assert_device(torch.stack(conds).all(), msg)


def split_pool(pool, head_dim):
    db, sb = (128, 4) if head_dim == 128 else (576, 8)
    if pool.dtype != torch.uint8 or pool.ndim != 3 or pool.stride(2) != 1:
        raise ValueError("compact CP requires production uint8 split-page pool")
    blocks, entries = pool.shape[:2]
    stride, offset = pool.stride(0), pool.storage_offset()
    required = offset + (blocks - 1) * stride + entries * (db + sb)
    if (
        blocks <= 0
        or entries <= 0
        or stride < entries * (db + sb)
        or required > pool.untyped_storage().nbytes()
    ):
        raise ValueError("compact CP split-page capacity/stride")
    data = pool.as_strided((blocks, entries, db), (stride, db, 1), offset)
    scales = pool.as_strided(
        (blocks, entries, sb), (stride, sb, 1), offset + entries * db
    )
    return data, scales, RowLayout("compact", db, sb, 4)


def _tensor_identity(t):
    if t is None:
        return None
    # PyWrappedModel holds c10::InferenceMode(true); those inputs have no
    # version counter. Snapshot their metadata under a normal tensor guard.
    version = None if t.is_inference() else t._version
    return (
        id(t),
        t.data_ptr(),
        version,
        tuple(t.shape),
        tuple(t.stride()),
        t.device,
        t.dtype,
    )


def _snapshot_meta(meta, workspace):
    cache = getattr(workspace, "_compact_cp_meta_snapshots", None)
    if cache is None:
        cache = workspace._compact_cp_meta_snapshots = {}
    key = _meta_identity(meta)
    if key not in cache:
        # The hoisted caller metadata is immutable for this forward; a private
        # snapshot owns its tensor contents across side-stream work and later
        # layer reuse, including inference-mode inputs without version counters.
        with torch.inference_mode(False):
            copied = {
                name: (
                    None if getattr(meta, name) is None else getattr(meta, name).clone()
                )
                for name in (
                    "positions",
                    "b_idx",
                    "state_slots",
                    "kv_slots",
                    "token_to_req",
                    "seq_start_per_req",
                    "cu_seq_per_req",
                )
            }
        cache[key] = replace(meta, **copied)
    return cache[key]


def _meta_identity(meta):
    return tuple(
        _tensor_identity(getattr(meta, key))
        for key in (
            "positions",
            "b_idx",
            "state_slots",
            "kv_slots",
            "token_to_req",
            "seq_start_per_req",
            "cu_seq_per_req",
        )
    ) + (meta.is_batched,)


def _pool_identity(module):
    # Pool contents legitimately change in the original writer. Bind their
    # storage/layout but not data-version; tables must remain immutable.
    def storage(t):
        return (
            None
            if t is None
            else (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.device)
        )

    return (
        storage(module._kv_pool_view),
        storage(module._state_pool_3d),
        _tensor_identity(module._kv_block_table),
        _tensor_identity(module._state_block_table),
    )


class CompactPending:
    def __init__(
        self, *, geometry, local, meta, workspace, role, stream, group, pool_cap
    ):
        self.geometry = g = geometry
        meta = _snapshot_meta(meta, workspace)
        self.local, self.meta, self.workspace = local, meta, workspace
        self.role, self.stream, self.group = role, stream, group
        self.state = "new"
        self.owner = (id(workspace), local.device, id(meta), id(group))
        self.meta_identity = _meta_identity(meta)
        self.binding = None
        device = local.device
        # This cache is owned by ONE PrefillWorkspace/forward, not by a layer,
        # module, request-global singleton or communicator. Metadata mutations
        # change the key; cached plans never survive a forward or PP transfer.
        cache = getattr(workspace, "_compact_cp_indices", None)
        if cache is None:
            cache = workspace._compact_cp_indices = {}

        def identity(t):
            return _tensor_identity(t)

        key = (
            g,
            device,
            pool_cap,
            identity(meta.positions),
            identity(meta.b_idx),
            identity(meta.state_slots),
            identity(meta.kv_slots),
            identity(meta.seq_start_per_req),
            identity(meta.cu_seq_per_req),
        )
        indices = cache.get(key)
        if indices is None:
            indices = _build_chunk_plan(g, device, meta, pool_cap)
            cache[key] = indices
        (
            self.tail_indices,
            self.tail_dest,
            self.boundaries,
            self.local_boundaries,
            self.masked_meta,
            self.receiver_slots,
            self.boundary_kv_slots,
            self.expected_wire_ids,
            self.local_boundary_ids,
        ) = indices
        self.send = local.index_select(0, self.tail_indices)
        getter = workspace.cp_gather_main if role == "main" else workspace.cp_gather_idx
        restore = (
            workspace.cp_restore_main if role == "main" else workspace.cp_restore_idx
        )
        self.gathered = getter(self.send.shape[0] * 4, g.width, torch.float32)
        self.scratch = restore(4096, g.width, torch.float32)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        self.send.record_stream(stream)
        # Protect the workspace storage even when an exception prevents the
        # usual forward drain (allocator lifetime, not a replacement for events).
        self.gathered.record_stream(stream)
        with torch.cuda.stream(stream):
            self.work = torch.distributed.all_gather_into_tensor(
                self.gathered, self.send, group=group, async_op=True
            )
            self.event = torch.cuda.Event()
            self.event.record(stream)
        self.state = "gathering"

    def assert_binding(self, module=None):
        if _meta_identity(self.meta) != self.meta_identity:
            raise RuntimeError("compact CP metadata mutated while pending")
        if (
            module is not None
            and self.binding is not None
            and _pool_identity(module) != self.binding
        ):
            raise RuntimeError("compact CP pool/table rebound while pending")

    def stage(self):
        if self.state == "staged":
            self.assert_binding()
            return self.scratch
        if self.state != "gathering":
            raise RuntimeError("compact CP handle already consumed")
        current = torch.cuda.current_stream(self.local.device)
        current.wait_event(self.event)
        self.work.wait()
        # Fence outstanding work even if caller metadata was illegally changed;
        # only then reject, before touching scratch or any output pool.
        self.assert_binding()
        for i, (lo, hi) in enumerate(self.geometry.intervals()):
            self.scratch[lo:hi].copy_(self.local[i * 512 : (i + 1) * 512])
        self.scratch.index_copy_(0, self.tail_dest, self.gathered)
        self.state = "staged"
        return self.scratch

    def owned_meta(self):
        self.assert_binding()
        if self.state != "staged":
            raise RuntimeError("compact CP writer before staging")
        return self.masked_meta

    def fail(self):
        # A partially executed numerical writer may never be replayed through
        # this handle, even when an outer caller catches the original error.
        if self.state != "finished":
            self.state = "failed"

    def replicate(self, module):
        self.assert_binding(module)
        if self.state != "staged":
            raise RuntimeError("compact CP publication before writer/after consumption")
        if (
            id(self.workspace),
            self.local.device,
            id(self.meta),
            id(self.group),
        ) != self.owner:
            raise RuntimeError("compact CP handle ownership changed")
        data, scales, layout = split_pool(module._kv_pool_view, module.head_dim)
        # Forward-constant pack slots / wire ids, computed once at plan
        # creation (the metadata snapshot and the geometry are immutable for
        # the forward).  The receiver range guard runs in that same per-forward
        # battery: ``assert_binding`` above revalidates the pool identity per
        # layer, so the capacity operand of that check cannot drift.
        slots = self.boundary_kv_slots
        receiver = self.receiver_slots
        fused = os.environ.get("DSV4_CP_COMPACT_TRANSPORT_FUSION", "0") == "1"
        if fused:
            from ._compact_cp_transport_triton import (
                pack_compact_bytes,
                scatter_compact_bytes,
            )

            payload = pack_compact_bytes(
                data, scales, slots, self.local_boundaries, self.geometry.start
            )
        else:
            payload = pack_rows(
                data,
                scales,
                slots,
                layout,
                num_blocks=data.shape[0],
                entries_per_block=data.shape[1],
                check=False,
            )
            # Wire identity is absolute logical boundary, NEVER sender physical slot.
            payload[:, :8].copy_(
                self.local_boundary_ids.contiguous().view(torch.uint8).reshape(-1, 8)
            )
        received = torch.empty(
            (payload.shape[0] * 4, payload.shape[1]),
            dtype=torch.uint8,
            device=payload.device,
        )
        torch.distributed.all_gather_into_tensor(received, payload, group=self.group)
        actual_ids = received[:, :8].contiguous().view(torch.int64).reshape(-1)
        _assert_device(
            (actual_ids == self.expected_wire_ids).all(),
            "compact CP foreign logical rows",
        )
        if fused:
            # Logical-ID and destination checks above stay intact. Directly use
            # receiver-local slots; do not rewrite a temporary packet header.
            scatter_compact_bytes(data, scales, received, receiver)
        else:
            received[:, :8].copy_(
                receiver.contiguous().view(torch.uint8).reshape(-1, 8)
            )
            scatter_packed_rows(
                data,
                scales,
                received,
                layout,
                num_blocks=data.shape[0],
                entries_per_block=data.shape[1],
                check=False,
            )
        self.state = "finished"


def start(module, local, cp, meta, workspace, role, stream):
    g = select_geometry(module, cp, meta, local)
    if g is None:
        return None
    from rtp_llm.models_py.distributed import collective_torch

    group = collective_torch._get_group(collective_torch.Group.TP)
    if (
        torch.distributed.get_world_size(group) != 4
        or torch.distributed.get_rank(group) != g.rank
    ):
        raise RuntimeError("compact CP stage group/rank mismatch")
    stream = stream if stream is not None else torch.cuda.current_stream(local.device)
    # Host metadata only: the pool's row capacity for the per-forward receiver
    # range guard.  ``split_pool`` still validates the pool layout at replicate
    # time, and ``assert_binding`` pins the pool identity to this module, so a
    # different pool can never silently reuse the cached battery.
    pool = module._kv_pool_view
    pool_cap = (
        int(pool.shape[0]) * int(pool.shape[1])
        if pool is not None and pool.dim() == 3
        else None
    )
    handle = CompactPending(
        geometry=g,
        local=local,
        meta=meta,
        workspace=workspace,
        role=role,
        stream=stream,
        group=group,
        pool_cap=pool_cap,
    )
    handle.binding = _pool_identity(module)
    return handle
