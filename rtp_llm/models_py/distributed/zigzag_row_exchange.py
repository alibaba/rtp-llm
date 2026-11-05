"""Private, bounded BF16 row pulls between natural and Zigzag local shards.

Only the local source is published. No full-H all-gather or baseline fallback
exists here. Initialization and invocation order must match on every group
rank. Eager calls are serialized; CUDA graph capture/replay is unsupported.
"""

import ctypes
from datetime import timedelta
import json
from operator import index
import socket
import threading

import torch
import torch.distributed as dist


_INSTANCE_COUNTS = {}
_INSTANCE_LOCK = threading.Lock()
_TIMEOUT_SECONDS = 120
_PREPARED_PLAN_LIMIT = 16


def _capacities(max_rows, hidden_size, chunk_capacity):
    max_rows, hidden_size, chunk_capacity = map(index, (max_rows, hidden_size, chunk_capacity))
    if min(max_rows, hidden_size, chunk_capacity) <= 0:
        raise ValueError("row, hidden and chunk capacities must be positive")
    return max_rows, hidden_size, min(max_rows, chunk_capacity)


def payload_bytes(max_rows, hidden_size, chunk_capacity=1024):
    """Logical persistent BF16 bytes: one local source and two receive slots."""
    rows, hidden, chunk = _capacities(max_rows, hidden_size, chunk_capacity)
    return 2 * rows * hidden + 4 * chunk * hidden


def chunk_intersections(segments, start, end):
    """Yield (peer, source offset, chunk-local destination offset, row count)."""
    if start < 0 or end < start:
        raise ValueError("invalid destination chunk")
    for segment in segments:
        left = max(start, segment.destination_offset)
        right = min(end, segment.destination_offset + segment.rows)
        if left < right:
            yield (segment.peer, segment.source_offset + left - segment.destination_offset,
                   left - start, right - left)


def validate_plan(plan, direction, size, rank, max_rows):
    """Validate the complete incoming interval cover before any GPU submission."""
    if direction not in ("to_zigzag", "to_natural"):
        raise ValueError("direction must be to_zigzag or to_natural")
    if plan.size != size or plan.rank != rank:
        raise ValueError("plan belongs to another group/rank")
    if not 0 <= plan.local_rows <= max_rows:
        raise ValueError("plan exceeds local physical row capacity")
    if plan.physical_rows != plan.local_rows * size:
        raise ValueError("invalid physical row geometry")
    cursor = 0
    for segment in getattr(plan, direction):
        if (segment.rows <= 0 or segment.destination_offset != cursor
                or not 0 <= segment.peer < size
                or not 0 <= segment.source_offset <= plan.local_rows - segment.rows):
            raise ValueError("incoming segments must cover each destination exactly once")
        cursor += segment.rows
    if cursor != plan.local_rows:
        raise ValueError("incoming segments omit destination rows")


def _fabric(uuid):
    """Same common-NVLink-domain evidence used by collective_torch workspaces."""
    import pynvml

    pynvml.nvmlInit()
    try:
        gpu = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + uuid.removeprefix("GPU-"))
        info = pynvml.c_nvmlGpuFabricInfoV_t()
        pynvml.nvmlDeviceGetGpuFabricInfoV(gpu, ctypes.byref(info))
        field = type(info).clusterUuid
        cluster = ctypes.string_at(ctypes.addressof(info) + field.offset, field.size)
        if (info.state == pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
                and info.status == pynvml.NVML_SUCCESS and any(cluster)):
            return [cluster.hex(), int(info.cliqueId)]
    except (AttributeError, pynvml.NVMLError):
        pass
    finally:
        pynvml.nvmlShutdown()
    return None


def _check_topology(records):
    uuids = [record["uuid"] for record in records]
    if len(set(uuids)) != len(records):
        raise RuntimeError("directed exchange requires distinct GPUs")
    cross_host = len({record["host"] for record in records}) > 1
    if cross_host:
        fabrics = [record["fabric"] for record in records]
        if fabrics[0] is None or any(fabric != fabrics[0] for fabric in fabrics):
            raise RuntimeError("cross-host exchange requires a common NVLink fabric domain")
        return True
    import pynvml

    pynvml.nvmlInit()
    try:
        handles = [pynvml.nvmlDeviceGetHandleByUUID("GPU-" + uuid.removeprefix("GPU-")) for uuid in uuids]
        if not all(
            pynvml.nvmlDeviceGetP2PStatus(a, b, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
            == pynvml.NVML_P2P_STATUS_OK
            for i, a in enumerate(handles) for j, b in enumerate(handles) if i != j
        ):
            raise RuntimeError("directed exchange requires NVLink P2P access")
    finally:
        pynvml.nvmlShutdown()
    return False


class DirectedRowExchange:
    """One model-lifetime source allocation plus two bounded receive slots.

    ``resident_bytes`` excludes raw allocation overhead: Torch adds a signal
    pad after 16-byte payload alignment, rounds to CUDA allocation granularity,
    and owns two world-size device pointer arrays. These bytes are not reliably
    visible in caching-allocator statistics. Caller-owned outputs also count
    separately. Peer views map remote storage; they are not extra local H copies.

    A consumer must queue every chunk read on the current caller stream before
    returning None, retain no slot/view, and launch no unjoined stream work.
    Arbitrary Python callbacks cannot be fully policed: the caller owns this
    contract. A changed current stream or non-None return fails the transport.
    Once asynchronous work has been queued, any failure is fail-stop for the
    request/job; never resume a baseline path on just one rank.
    """

    def __init__(self, group, device, max_rows, hidden_size, *, chunk_capacity=1024):
        self.enabled = False
        self.group = group
        self.rank, self.size = dist.get_rank(group), dist.get_world_size(group)
        if self.rank < 0:
            raise ValueError("directed exchange requires group membership")
        self._lock = threading.Lock()
        self._prepared = {}
        self._prepare_epoch = 0
        # Group-local instance sequence, independent of other groups' setup.
        with _INSTANCE_LOCK:
            instance = _INSTANCE_COUNTS.get(group, 0)
            _INSTANCE_COUNTS[group] = instance + 1
        self._store = dist.PrefixStore(
            f"rtp_llm/directed_rows/{instance}/",
            dist.distributed_c10d._get_process_group_store(group),
        )
        error = ""
        metadata = None
        try:
            import torch.distributed._symmetric_memory as symm

            self.device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
            if self.device.type != "cuda":
                raise ValueError("directed exchange requires CUDA")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            self.max_rows, self.hidden_size, self.chunk_capacity = _capacities(max_rows, hidden_size, chunk_capacity)
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("initialize directed exchange outside CUDA graph capture")
                if symm.get_backend(self.device) != "CUDA":
                    raise RuntimeError("requires the existing CUDA symmetric-memory backend")
                props = torch.cuda.get_device_properties(self.device)
            metadata = dict(
                contract=[self.max_rows, self.hidden_size, self.chunk_capacity, "bfloat16", "CUDA",
                          dist.get_process_group_ranks(group)],
                host=socket.gethostname(), uuid=str(props.uuid), fabric=_fabric(str(props.uuid)),
            )
        except Exception as exc:
            error = str(exc)
        records = self._agree("capability", metadata, error)
        error = ""
        try:
            if any(item["contract"] != metadata["contract"] for item in records):
                raise ValueError("all ranks must agree on group and capacity")
            cross_host = _check_topology(records)
        except Exception as exc:
            error = str(exc)
        self._agree("topology", None, error)

        error = ""
        try:
            with torch.cuda.device(self.device):
                # Bypass public symm.empty's implicit pool. Do not mutate global
                # backend, signal-pad configuration, or pool flags.
                self._source = symm._SymmetricMemory.empty_strided_p2p(
                    (self.max_rows, self.hidden_size), (self.hidden_size, 1),
                    torch.bfloat16, self.device,
                )
                if cross_host:
                    from rtp_llm.ops.compute_ops import rtp_llm_ops

                    if not rtp_llm_ops.is_cuda_fabric_allocation(self._source):
                        raise RuntimeError("cross-host exchange requires a FABRIC allocation")
                self._slots = torch.empty((2, self.chunk_capacity, self.hidden_size),
                                          dtype=torch.bfloat16, device=self.device)
                self._stream = torch.cuda.Stream(device=self.device)
                # Same-stream allocator reuse may carry old writes queued on
                # the constructor's stream. Transfer initial slot ownership;
                # record_stream during calls only protects later reclamation.
                self._stream.wait_stream(torch.cuda.current_stream(self.device))
                self._ready = [torch.cuda.Event() for _ in range(2)]
                self._free = [torch.cuda.Event() for _ in range(2)]
                self._readers_done = torch.cuda.Event()
                self._done = torch.cuda.Event()
                self._entry = torch.cuda.Event()
                # Force event creation now, before the allocation vote.
                for event in (*self._ready, *self._free, self._readers_done, self._done, self._entry):
                    event.record(self._stream)
                self._stream.synchronize()
        except Exception as exc:
            error = str(exc)
        self._agree("allocation", None, error)

        error = ""
        try:
            with torch.cuda.device(self.device):
                self._handle = symm.rendezvous(self._source, group=group)
                h = self._handle
                if (h.offset != 0 or h.rank != self.rank or h.world_size != self.size
                        or torch.device(h.device) != self.device
                        or h.buffer_size < self.max_rows * self.hidden_size * 2
                        or h.signal_pad_size // (4 * self.size) < 2
                        or len(h.buffer_ptrs) != self.size
                        or h.buffer_ptrs[self.rank] != self._source.data_ptr()):
                    raise RuntimeError("raw symmetric allocation base/device/group/size validation failed")
                self._peers = [h.get_buffer(peer, (self.max_rows, self.hidden_size), torch.bfloat16)
                               for peer in range(self.size)]
                for peer, tensor in enumerate(self._peers):
                    if (tensor.data_ptr() == 0 or tensor.data_ptr() != h.buffer_ptrs[peer]
                            or tensor.device != self.device or tensor.dtype != torch.bfloat16
                            or tensor.shape != self._source.shape or not tensor.is_contiguous()):
                        raise RuntimeError("symmetric peer mapping validation failed")
        except Exception as exc:
            error = str(exc)
        # The bounded votes do not bound a rank wedged inside Torch rendezvous;
        # the owning launcher must also provide a bounded process-group timeout.
        self._agree("mapping", None, error)
        self.enabled = True

    def _agree(self, phase, payload, error=""):
        key = f"{phase}/{self.rank}"
        self._store.set(key, json.dumps(dict(payload=payload, error=error)).encode())
        keys = [f"{phase}/{rank}" for rank in range(self.size)]
        try:
            self._store.wait(keys, timedelta(seconds=_TIMEOUT_SECONDS))
            records = [json.loads(self._store.get(key)) for key in keys]
            # The last reader can release the round's keys: every participant
            # has already copied every record before incrementing this counter.
            # No waiter reads the counter or keys after this acknowledgement.
            completed = phase + "/readers"
            if self._store.add(completed, 1) == self.size:
                for key in (*keys, completed):
                    self._store.delete_key(key)
            errors = [(rank, record["error"]) for rank, record in enumerate(records) if record["error"]]
            if errors:
                raise RuntimeError(f"directed exchange {phase} rejected collectively: {errors}")
            return [record["payload"] for record in records]
        except Exception:
            self.enabled = False
            raise

    @property
    def source(self):
        """Published storage for accounting only; never write/resize externally."""
        return self._source

    @property
    def resident_bytes(self):
        return payload_bytes(self.max_rows, self.hidden_size, self.chunk_capacity)

    @property
    def signal_pad_bytes(self):
        return self._handle.signal_pad_size

    @staticmethod
    def _plan_key(plan):
        return (plan.size, plan.rank, plan.local_rows, plan.physical_rows, tuple(plan.lengths),
                tuple((s.peer, s.source_offset, s.destination_offset, s.rows) for s in plan.to_zigzag),
                tuple((s.peer, s.source_offset, s.destination_offset, s.rows) for s in plan.to_natural))

    def prepare_plan(self, plan, *, chunk_rows=None):
        """Collectively prepare both directions once per geometry/chunk contract.

        Call from common batch setup, before forward. Every rank must prepare
        the same contracts in the same order; cached calls require identical
        cache history across ranks. The LRU holds at most 16 geometries; an
        evicted geometry requires another collective preparation. This
        authorizes the requested consumer
        chunk size plus exchange()'s capacity-sized chunks. Normal calls never
        perform host-store synchronization. The shared dispatcher must preserve
        invocation order, including empty/padding-only local shards.
        """
        with self._lock:
            if not self.enabled:
                raise RuntimeError("cannot prepare a disabled directed exchange")
            error, contract, key = "", None, None
            try:
                chunk_rows = self.chunk_capacity if chunk_rows is None else index(chunk_rows)
                if not 1 <= chunk_rows <= self.chunk_capacity:
                    raise ValueError("chunk_rows exceeds fixed receive capacity")
                for direction in ("to_zigzag", "to_natural"):
                    validate_plan(plan, direction, self.size, self.rank, self.max_rows)
                key = self._plan_key(plan)
                if chunk_rows in self._prepared.get(key, ()):
                    self._prepared[key] = self._prepared.pop(key)
                    return
                contract = [plan.size, plan.local_rows, plan.physical_rows, list(plan.lengths),
                            self.max_rows, self.chunk_capacity, chunk_rows]
            except Exception as exc:
                error = str(exc)
            phase = f"prepare/{self._prepare_epoch}"
            self._prepare_epoch += 1
            contracts = self._agree(phase, contract, error)
            if any(item != contract for item in contracts):
                self.enabled = False
                raise RuntimeError("directed exchange prepared contracts differ across ranks")
            chunks = self._prepared.pop(key, set())
            chunks.update((chunk_rows, self.chunk_capacity))
            self._prepared[key] = chunks
            while len(self._prepared) > _PREPARED_PLAN_LIMIT:
                self._prepared.pop(next(iter(self._prepared)))

    def _require_prepared(self, plan, chunk_rows):
        key = self._plan_key(plan)
        if chunk_rows not in self._prepared.get(key, ()):
            raise ValueError("missing or changed directed exchange contract; call prepare_plan collectively before forward")
        self._prepared[key] = self._prepared.pop(key)

    def _validate_tensor(self, tensor, rows, label):
        if (not isinstance(tensor, torch.Tensor) or tensor.device != self.device
                or tensor.dtype != torch.bfloat16 or tensor.shape != (rows, self.hidden_size)
                or not tensor.is_contiguous() or tensor.requires_grad):
            raise ValueError(f"{label} must be contiguous inference BF16 [local_rows, hidden_size] on {self.device}")
        for private in (self._source, self._slots):
            if tensor.untyped_storage().data_ptr() == private.untyped_storage().data_ptr():
                raise ValueError(f"{label} must not alias private transport storage")

    def exchange(self, values, plan, direction, out=None):
        """Return an independently owned complete local output, or validated out.

        The caller's stream joins the complete producer before publication.
        Reverse O routing therefore starts after the whole O producer, not an
        unfinished GEMM. Original-input aliasing is safe; private storage is not.
        """
        return self._run(values, plan, direction, self.chunk_capacity, out=out)

    def consume_chunks(self, values, plan, direction, chunk_rows, consumer):
        """Call consumer(start, slot) in destination order on the caller stream."""
        self._run(values, plan, direction, chunk_rows, consumer=consumer, consuming=True)

    def _run(self, values, plan, direction, chunk_rows, *, out=None, consumer=None, consuming=False):
        with self._lock:
            if not self.enabled:
                raise RuntimeError("directed exchange is disabled after a setup or asynchronous failure")
            try:
                with torch.cuda.device(self.device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("directed exchange does not support CUDA graph capture")
                    caller = torch.cuda.current_stream(self.device)
                    chunk_rows = index(chunk_rows)
                    if not 1 <= chunk_rows <= self.chunk_capacity:
                        raise ValueError("chunk_rows exceeds fixed receive capacity")
                    validate_plan(plan, direction, self.size, self.rank, self.max_rows)
                    self._require_prepared(plan, chunk_rows)
                    self._validate_tensor(values, plan.local_rows, "values")
                    if consuming:
                        if not callable(consumer):
                            raise TypeError("consumer must be callable")
                    else:
                        if out is None:
                            out = torch.empty_like(values)
                        self._validate_tensor(out, plan.local_rows, "out")
                    self._submit(values, plan, direction, chunk_rows, caller, out, consumer, consuming)
            except Exception:
                self.enabled = False
                raise
            return out

    def _submit(self, values, plan, direction, chunk_rows, caller, out, consumer, consuming):
        comm = self._stream
        self._entry.record(caller)
        with torch.cuda.stream(comm):
            comm.wait_event(self._done)
            comm.wait_event(self._entry)
            values.record_stream(comm)
            self._source[:plan.local_rows].copy_(values)
            self._handle.barrier(channel=0, timeout_ms=_TIMEOUT_SECONDS * 1000)
            if not consuming:
                out.record_stream(comm)
            self._slots.record_stream(comm)
        slots_used = [False, False]
        for number, start in enumerate(range(0, plan.local_rows, chunk_rows)):
            end = min(start + chunk_rows, plan.local_rows)
            slot = number % 2
            with torch.cuda.stream(comm):
                if consuming and slots_used[slot]:
                    comm.wait_event(self._free[slot])
                target = self._slots[slot, :end - start] if consuming else out[start:end]
                for peer, source, destination, rows in chunk_intersections(getattr(plan, direction), start, end):
                    target[destination:destination + rows].copy_(self._peers[peer][source:source + rows])
                if consuming:
                    self._ready[slot].record(comm)
            if consuming:
                with torch.cuda.stream(caller):
                    caller.wait_event(self._ready[slot])
                    self._slots.record_stream(caller)
                    result = consumer(start, target)
                    if result is not None or torch.cuda.current_stream(self.device) != caller:
                        raise RuntimeError("consumer must return None and keep the caller stream; unjoined work is unsupported")
                    self._free[slot].record(caller)
                slots_used[slot] = True
        with torch.cuda.stream(comm):
            # Protect remote sources until every peer has finished its pulls.
            self._handle.barrier(channel=1, timeout_ms=_TIMEOUT_SECONDS * 1000)
            self._readers_done.record(comm)
        with torch.cuda.stream(caller):
            caller.wait_event(self._readers_done)
            self._done.record(caller)
