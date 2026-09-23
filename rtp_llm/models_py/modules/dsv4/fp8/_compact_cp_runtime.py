"""Compact CP transport preserves compressor arithmetic and replicated state tails.

Per-call handles own transient tensors; scratch is scoped to the forward."""

import os
from dataclasses import dataclass, replace

import torch

from rtp_llm.models_py.modules.dsv4.fp8._cp_packed_rows import (
    RowLayout,
    pack_rows,
    scatter_packed_rows,
)

FLAG = "DSV4_CP_COMPACT_COMPRESSOR"


def enabled():
    return os.environ.get(FLAG, "0") == "1"


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
    mask = padding_mask.detach().to(device="cpu").reshape(-1)
    restore = restore_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
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
    return bool(
        mask.shape == (4096,)
        and torch.all(mask == 1)
        and restore.shape == (4096,)
        and torch.equal(restore, torch.argsort(order))
    )


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


def select_geometry(module, cp, meta, fused):
    if (
        not enabled()
        or cp is None
        or not getattr(cp, "compact_geometry_verified", False)
    ):
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
    def __init__(self, *, geometry, local, meta, workspace, role, stream, group):
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
            identity(meta.positions),
            identity(meta.b_idx),
            identity(meta.state_slots),
            identity(meta.kv_slots),
            identity(meta.seq_start_per_req),
            identity(meta.cu_seq_per_req),
        )
        indices = cache.get(key)
        if indices is None:
            local_map = tuple(p for lo, hi in g.intervals() for p in range(lo, hi))
            lut = {p: i for i, p in enumerate(local_map)}
            tail_indices = torch.tensor(
                [lut[p] for p in g.tails()], dtype=torch.long, device=device
            )
            tail_dest = torch.tensor(
                [p for r in range(4) for p in g.tails(r)],
                dtype=torch.long,
                device=device,
            )
            boundaries = torch.tensor(
                g.wire_boundaries(), dtype=torch.long, device=device
            )
            local_boundaries = boundaries[
                g.rank * (1024 // g.ratio) : (g.rank + 1) * (1024 // g.ratio)
            ]
            # The original fused writer checks a negative KV slot only after
            # reading raw operands. Use ONLY owned boundary programs, not a
            # full grid with remote slots masked; otherwise it reads unstaged
            # bytes. The separate state writer retains its full original grid.
            own = torch.zeros_like(meta.kv_slots, dtype=torch.bool)
            masked = replace(
                meta,
                positions=meta.positions.index_select(0, local_boundaries),
                b_idx=meta.b_idx.index_select(0, local_boundaries),
                state_slots=meta.state_slots.index_select(0, local_boundaries),
                kv_slots=meta.kv_slots.index_select(0, local_boundaries),
                token_to_req=meta.token_to_req.index_select(0, local_boundaries),
            )
            expected_positions = (
                torch.arange(4096, device=device, dtype=torch.long) + g.start
            )
            covered_state = torch.zeros_like(own)
            covered_state[tail_dest] = True
            _assert_device(
                (meta.positions == expected_positions).all(),
                "compact CP nonsequential full metadata",
            )
            _assert_device((meta.b_idx == 0).all(), "compact CP multiple requests")
            _assert_device(
                (meta.token_to_req == 0).all(),
                "compact CP foreign token request mapping",
            )
            _assert_device(
                ((meta.state_slots < 0) | covered_state).all(),
                "compact CP missing state raw rows",
            )
            receiver = meta.kv_slots.index_select(0, boundaries)
            ordered_slots = receiver.sort().values
            _assert_device(
                (ordered_slots[1:] != ordered_slots[:-1]).all(),
                "compact CP duplicate destinations",
            )
            if meta.is_batched:
                if meta.seq_start_per_req is None or meta.cu_seq_per_req is None:
                    raise ValueError("compact CP missing B1 raw windows")
                _assert_device(
                    (meta.seq_start_per_req == g.start).all(),
                    "compact CP foreign prefix",
                )
                _assert_device(
                    (
                        meta.cu_seq_per_req == torch.tensor([0, 4096], device=device)
                    ).all(),
                    "compact CP bad raw windows",
                )
            indices = (
                tail_indices,
                tail_dest,
                boundaries,
                local_boundaries,
                masked,
                receiver,
            )
            cache[key] = indices
        (
            self.tail_indices,
            self.tail_dest,
            self.boundaries,
            self.local_boundaries,
            self.masked_meta,
            self.receiver_slots,
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
        slots = self.meta.kv_slots.index_select(0, self.local_boundaries)
        receiver = self.receiver_slots
        cap = data.shape[0] * data.shape[1]
        _assert_device(
            ((receiver >= 0) & (receiver < cap)).all(), "compact CP invalid destination"
        )
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
            identity = self.local_boundaries + self.geometry.start
            payload[:, :8].copy_(identity.contiguous().view(torch.uint8).reshape(-1, 8))
        received = torch.empty(
            (payload.shape[0] * 4, payload.shape[1]),
            dtype=torch.uint8,
            device=payload.device,
        )
        torch.distributed.all_gather_into_tensor(received, payload, group=self.group)
        actual_ids = received[:, :8].contiguous().view(torch.int64).reshape(-1)
        _assert_device(
            (actual_ids == self.boundaries + self.geometry.start).all(),
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
    handle = CompactPending(
        geometry=g,
        local=local,
        meta=meta,
        workspace=workspace,
        role=role,
        stream=stream,
        group=group,
    )
    handle.binding = _pool_identity(module)
    return handle
