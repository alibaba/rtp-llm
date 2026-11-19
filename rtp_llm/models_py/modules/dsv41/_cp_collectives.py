"""Single-node communication for the eager DeepSeek V4.1 CP4 prefill loop."""

import logging
import socket
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import prod
from threading import Lock
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

# The packed main/index KV gather can consume 139.06 MiB at B4. Its producer
# and consumer can share this window in place. Larger batches use bounded tiles.
_WORKSPACE_BYTES = 144 * 1024 * 1024
_PROJECTED_BYTES = 80 * 1024 * 1024
_active = ContextVar("dsv41_cp_collectives", default=None)


# The main model owns the workspace. The matching DSpARK commit can borrow
# it after target forward returns, without allocating a second symmetric window.
_warm_workspaces = WeakValueDictionary()
_commit_session = ContextVar("dsv41_commit_session", default=None)


def _all(group, value):
    values = [None] * group.size()
    dist.all_gather_object(values, value, group=group)
    return values


class _Window:
    def __init__(self, device, buffer, handle):
        self.device = device
        self.buffer = buffer
        self.handle = handle
        self.last_stream = None
        self.done = torch.cuda.Event()
        self.lock = Lock()

    @contextmanager
    def use(self):
        # CP projections and history reads can run on different CUDA streams.
        # Serialize scratch reuse across streams while leaving the caller's
        # output on its own stream. No host wait for device completion.
        with self.lock:
            stream = torch.cuda.current_stream(self.device)
            if self.last_stream is not None and stream != self.last_stream:
                stream.wait_event(self.done)
            try:
                yield
            finally:
                self.done.record(stream)
                self.last_stream = stream


class _Workspace(_Window):
    def __init__(self, group, device, buffer, handle, projected, runtime, copy_attr):
        super().__init__(device, buffer, handle)
        self.group = group
        self.rank = group.rank()
        self.peers = [
            handle.get_buffer(i, buffer.shape, torch.uint8) for i in range(group.size())
        ]
        self.peer_addresses = tuple(peer.data_ptr() for peer in self.peers)
        self.runtime = runtime
        self.copy_attributes = [copy_attr]
        self.projected = projected


def _allocate_window(sm, group, device, size):
    try:
        buffer = sm.empty((size,), dtype=torch.uint8, device=device)
    except RuntimeError as exc:
        logging.warning("V4.1 CP symmetric workspace allocation failed: %s", exc)
        buffer = None
    if not all(_all(group, buffer is not None)):
        return None, None
    handle = sm.rendezvous(buffer, group=group.group_name)
    if not all(_all(group, handle is not None and bool(handle.multicast_ptr))):
        return None, None
    return buffer, handle


def _initialize(group, device, *, cp_rank):
    group_rank = group.rank()
    if group_rank != cp_rank:
        logging.warning(
            "V4.1 CP rank %s differs from TP group rank %s", cp_rank, group_rank
        )
    try:
        import torch.distributed._symmetric_memory as sm
        from cuda.bindings import runtime

        driver_status, driver_version = runtime.cudaDriverGetVersion()
        runtime_status, runtime_version = runtime.cudaRuntimeGetVersion()
        copy_attr = runtime.cudaMemcpyAttributes()
        copy_attr.srcAccessOrder = (
            runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
        )
        copy_attr.flags = int(
            runtime.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute
        )
        props = torch.cuda.get_device_properties(device)
        # Restrict enrollment to the tested single-node multicast topology.
        local = str(props.uuid)
        peers = {
            str(torch.cuda.get_device_properties(i).uuid)
            for i in range(torch.cuda.device_count())
            if i == device.index or torch.cuda.can_device_access_peer(device.index, i)
        }
        ready = (
            props.major == 10
            and group_rank == cp_rank
            and int(driver_status) == int(runtime_status) == 0
            and min(driver_version, runtime_version) >= 13000
            and hasattr(runtime, "cudaMemcpyBatchAsync")
            and hasattr(torch.ops.symm_mem, "multimem_all_gather_out")
            and hasattr(torch.ops.symm_mem, "one_shot_all_reduce_copy_out")
        )
    except (ImportError, AttributeError, RuntimeError):
        ready, local, peers = False, "", set()
    info = _all(group, (socket.gethostname(), local, peers, ready))
    ids = {uuid for _, uuid, _, _ in info}
    if (
        len({host for host, _, _, _ in info}) != 1
        or len(ids) != 4
        or not all(ok and ids <= reachable for _, _, reachable, ok in info)
    ):
        return None
    buffer, handle = _allocate_window(sm, group, device, _WORKSPACE_BYTES)
    if buffer is None:
        return None
    # This gather overlaps the SWA collectives in the general window. A
    # distinct rendezvous supplies independent cross-rank signal pads.
    projected, projected_handle = _allocate_window(sm, group, device, _PROJECTED_BYTES)
    window = (
        _Window(device, projected, projected_handle) if projected is not None else None
    )
    return _Workspace(group, device, buffer, handle, window, runtime, copy_attr)


def initialize_prefill_cp_collectives(owner, device, *, cp_rank):
    """Register once on the model that survives the temporary warmup executor."""
    if hasattr(owner, "_dsv41_cp_collectives"):
        return
    workspace = None
    if device.type == "cuda" and torch.version.hip is None and dist.is_initialized():
        with torch.cuda.device(device):
            group = _get_group(Group.TP)
            if group.size() == 4:
                device = torch.device("cuda", torch.cuda.current_device())
                workspace = _initialize(group, device, cp_rank=cp_rank)
                if workspace is not None:
                    _warm_workspaces[group] = workspace
                    logging.info(
                        "V4.1 CP symmetric workspace registered on %s (CP rank %d, main=%d bytes, projected=%d bytes)",
                        device,
                        cp_rank,
                        workspace.buffer.nbytes,
                        workspace.projected.buffer.nbytes if workspace.projected else 0,
                    )
    owner._dsv41_cp_collectives = workspace


@contextmanager
def prefill_cp_collectives(model, device, eager):
    workspace = getattr(model, "_dsv41_cp_collectives", None) if eager else None
    if workspace is not None and (device != workspace.device or model._cp_size != 4):
        workspace = None
    token = _active.set(workspace)
    try:
        yield
    finally:
        _active.reset(token)


def _workspace(group):
    ws = _active.get()
    if ws is not None and (group is ws.group or group is Group.TP):
        return ws
    return None


def _can_gather(ws, local):
    # The participating call sites use rank-uniform padded contiguous shards.
    return (
        ws is not None
        and local.device == ws.device
        and local.is_contiguous()
        and local.nbytes > 0
    )


def _direct_output(ws, local, offset):
    """Gather into a view that is valid until the caller releases the window."""
    size = local.nbytes
    if size % 16 or 4 * size > ws.buffer.numel() - offset:
        return None
    shape = (4 * local.shape[0], *local.shape[1:])
    out = ws.buffer.narrow(0, offset, 4 * size).view(local.dtype).view(shape)
    # Common model producers supply aligned contiguous shards. A local view at
    # an odd byte offset still enters the same collective on all four ranks.
    if local.data_ptr() % 16:
        local = local.clone(memory_format=torch.contiguous_format)
    torch.ops.symm_mem.multimem_all_gather_out(local, ws.group.group_name, out=out)
    return out


@contextmanager
def borrowed_all_gather(local, group):
    """Enqueue the consumer on this stream before leaving the scope."""
    ws = _workspace(group)
    if (
        _can_gather(ws, local)
        and local.nbytes % 16 == 0
        and 4 * local.nbytes <= _WORKSPACE_BYTES
    ):
        with ws.use():
            yield _direct_output(ws, local, 0)
    else:
        yield None


@contextmanager
def borrowed_gather_buffer(reference, group, *, numel, dtype):
    """Lend a symmetric destination from the first producer to the final consumer."""
    ws = _workspace(group)
    size = numel * dtype.itemsize
    if (
        ws is not None
        and reference.device == ws.device
        and 0 < size <= _WORKSPACE_BYTES
    ):
        with ws.use():
            yield ws.buffer[:size].view(dtype)
    else:
        yield None


@contextmanager
def projected_gather_buffer(reference, group, *, numel, dtype):
    ws = _workspace(group)
    size = numel * dtype.itemsize
    if (
        ws is not None
        and ws.projected is not None
        and reference.device == ws.device
        and 0 < size <= _PROJECTED_BYTES
        and size % 64 == 0
    ):
        with ws.projected.use():
            yield ws.projected.buffer[:size].view(dtype)
    else:
        yield None


def gather_inplace(region, send, group):
    """Gather from this rank's aligned slice of a borrowed symmetric region."""
    torch.ops.symm_mem.multimem_all_gather_out(
        send, _workspace(group).group.group_name, out=region
    )


@dataclass
class _CommitSession:
    workspace: _Workspace
    retained_bytes: int = 0


@contextmanager
def reuse_prefill_cp_collectives(device):
    """Borrow the target's workspace for one eager, single-stream draft commit."""
    ws = _warm_workspaces.get(_get_group(Group.TP)) if _warm_workspaces else None
    if (
        ws is None
        or device.index not in (None, ws.device.index)
        or torch.cuda.is_current_stream_capturing()
    ):
        yield
        return
    # Request IDs and positions remain live across all draft layers. Each
    # layer enqueues its cache write before the next layer reuses the KV area.
    with ws.use():
        token = _commit_session.set(_CommitSession(ws))
        try:
            yield
        finally:
            _commit_session.reset(token)


def try_commit_all_gather(local, *, retain=False):
    """Gather a contiguous DSpark commit shard from its rank-local producer."""
    session = _commit_session.get()
    if session is None:
        return None
    ws, offset = session.workspace, session.retained_bytes
    if local.device != ws.device or local.nbytes == 0:
        return None
    size = 4 * local.nbytes
    # Keep at least half the window available to gather arbitrarily large KV
    # in tiles. A metadata array exceeding this budget gets independent output.
    if not retain or offset + size <= _WORKSPACE_BYTES // 2:
        gathered = _direct_output(ws, local, offset)
        if gathered is not None:
            if retain:
                session.retained_bytes += size
            return gathered
    gathered = local.new_empty((4 * local.shape[0], *local.shape[1:]))
    _gather_into(ws, gathered, local, ws.buffer[offset:])
    return gathered


def _gather_into(ws, out, local, window):
    size = local.nbytes
    if size % 16 == 0 and 4 * size <= window.nbytes:
        if local.data_ptr() % 16:
            local = local.clone(memory_format=torch.contiguous_format)
        scratch = window[: 4 * size].view(out.dtype).view_as(out)
        torch.ops.symm_mem.multimem_all_gather_out(
            local, ws.group.group_name, out=scratch
        )
        out.copy_(scratch)
        return

    # Large or unaligned private outputs cannot borrow a multicast region.
    # Stage one local shard per rank. The copy engine reads each peer directly
    # into the final rank-major layout, including uneven byte tails.
    runtime = ws.runtime
    stream = torch.cuda.current_stream(ws.device).cuda_stream
    # Batch copies require a non-legacy stream. The per-thread default stream
    # is ordered with the two barriers on the legacy stream in this context.
    batch_stream = (
        runtime.cudaStreamPerThread
        if stream in (0, runtime.cudaStreamLegacy)
        else stream
    )
    output = out.data_ptr()
    source = local.data_ptr()
    peers = [i for i in range(4) if i != ws.rank or source != output + i * size]
    # Draft commit may have reserved the beginning of each window for metadata.
    sources = [ws.peer_addresses[i] + window.storage_offset() for i in peers]
    stage = window.data_ptr()
    for start in range(0, size, window.nbytes):
        count = min(window.nbytes, size - start)
        (status,) = runtime.cudaMemcpyAsync(
            stage,
            source + start,
            count,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            stream,
        )
        if int(status):
            raise RuntimeError(f"V4.1 CP peer staging failed: {status}")
        ws.handle.barrier()
        destinations = [output + i * size + start for i in peers]
        (status,) = runtime.cudaMemcpyBatchAsync(
            destinations,
            sources,
            [count] * len(peers),
            len(peers),
            ws.copy_attributes,
            [0],
            1,
            batch_stream,
        )
        if int(status):
            raise RuntimeError(f"V4.1 CP peer gather failed: {status}")
        # All peers must finish reading before the next use can reuse staging.
        ws.handle.barrier()


def _gather(ws, out, local):
    with ws.use():
        _gather_into(ws, out, local, ws.buffer)


def try_all_gather_into(out, local, group):
    ws = _workspace(group)
    if not _can_gather(ws, local):
        return False
    _gather(ws, out, local)
    return True


def try_all_gather(local, group):
    ws = _workspace(group)
    if not _can_gather(ws, local):
        return None
    out = local.new_empty((4 * local.shape[0], *local.shape[1:]))
    _gather(ws, out, local)
    return out


def try_all_reduce_sum_(value, group):
    ws = _workspace(group)
    if (
        ws is None
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or not 0 < value.nbytes <= _WORKSPACE_BYTES
    ):
        return False
    with ws.use():
        torch.ops.symm_mem.one_shot_all_reduce_copy_out(
            ws.buffer[: value.nbytes].view(torch.float32).view_as(value),
            value,
            "sum",
            ws.group.group_name,
            out=value,
        )
    return True


def try_all_to_all_into(
    received, value, group, splits_matrix, *, rows, transpose=False
):
    ws = _workspace(group)
    row_bytes = value.element_size() * prod(value.shape[1:])
    if ws is None or row_bytes == 0 or row_bytes > _WORKSPACE_BYTES:
        return False
    # All ranks see the full layout, so even an empty source enters both
    # rendezvous in every tile. The existing row selection writes directly
    # into this rank's peer-visible window.
    splits = tuple(zip(*splits_matrix)) if transpose else splits_matrix
    rank = ws.rank
    dest = received.view(torch.uint8).reshape(-1)
    max_rows = max(map(sum, splits))
    rows_per_tile = _WORKSPACE_BYTES // row_bytes
    with ws.use():
        for start in range(0, max_rows, rows_per_tile):
            end = start + rows_per_tile
            count = min(rows_per_tile, max(0, rows.numel() - start))
            if count:
                send = (
                    ws.buffer[: count * row_bytes]
                    .view(value.dtype)
                    .view(count, *value.shape[1:])
                )
                torch.index_select(value, 0, rows.narrow(0, start, count), out=send)
            ws.handle.barrier()
            offset = 0
            for peer, sizes in enumerate(splits):
                begin = sum(sizes[:rank])
                size = sizes[rank]
                lo, hi = max(begin, start), min(begin + size, end)
                if hi > lo:
                    dst = (offset + lo - begin) * row_bytes
                    src = (lo - start) * row_bytes
                    length = (hi - lo) * row_bytes
                    dest.narrow(0, dst, length).copy_(
                        ws.peers[peer].narrow(0, src, length)
                    )
                offset += size
            # No peer may overwrite its segment until all readers finish.
            ws.handle.barrier()
    return True
