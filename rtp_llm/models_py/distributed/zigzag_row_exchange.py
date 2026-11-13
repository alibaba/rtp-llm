"""Private row pulls between natural and Zigzag local shards.

Only the local source is published. No full-H all-gather or baseline fallback
exists here. Initialization and invocation order must match on every group
rank. Eager calls are serialized; CUDA graph capture/replay is unsupported.
"""

import ctypes
import json
import socket
import threading
from datetime import timedelta
from operator import index

import torch
import torch.distributed as dist

from rtp_llm.models_py.utils.module_profiler import profile_scope

_INSTANCE_COUNTS = {}
_INSTANCE_LOCK = threading.Lock()
_TIMEOUT_SECONDS = 120
_PREPARED_PLAN_LIMIT = 16
_HIDDEN_ROUTE_SCOPE = "RTP::attention.dsa.hidden_natural_to_zigzag"
_OUTPUT_ROUTE_SCOPE = "RTP::row_exchange.zigzag_to_natural"
_FP8_PRE_O_PIPELINE_MIN_ROWS = 512
# Per-rank sizes tuned on TP8 with the GPU queue prefilled.
_FP8_PRE_O_SHORT_CHUNKS = {2560: 1280, 3072: 768, 3584: 1280}


def _capacities(max_rows, hidden_size, chunk_capacity):
    max_rows, hidden_size, chunk_capacity = map(
        index, (max_rows, hidden_size, chunk_capacity)
    )
    if min(max_rows, hidden_size, chunk_capacity) <= 0:
        raise ValueError("row, hidden and chunk capacities must be positive")
    return max_rows, hidden_size, min(max_rows, chunk_capacity)


def payload_bytes(max_rows, hidden_size, chunk_capacity=1024):
    """Logical persistent BF16 bytes: one local source and two receive slots."""
    rows, hidden, chunk = _capacities(max_rows, hidden_size, chunk_capacity)
    return 2 * rows * hidden + 4 * chunk * hidden


def _fp8_output_bytes(rows, width):
    return rows * width + 4 * (width // 512) * ((rows + 3) // 4 * 4)


def _fp8_output_views(storage, rows, width):
    raw = storage.view(torch.uint8).flatten()
    value_end = rows * width
    scale_end = _fp8_output_bytes(rows, width)
    values = raw[:value_end].view(torch.float8_e4m3fn).view(rows, width)
    scales = raw[value_end:scale_end].view(torch.int32).view(width // 512, -1)
    return values, scales


def chunk_intersections(segments, start, end):
    """Yield (peer, source offset, chunk-local destination offset, row count)."""
    if start < 0 or end < start:
        raise ValueError("invalid destination chunk")
    for segment in segments:
        left = max(start, segment.destination_offset)
        right = min(end, segment.destination_offset + segment.rows)
        if left < right:
            yield (
                segment.peer,
                segment.source_offset + left - segment.destination_offset,
                left - start,
                right - left,
            )


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
        if (
            segment.rows <= 0
            or segment.destination_offset != cursor
            or not 0 <= segment.peer < size
            or not 0 <= segment.source_offset <= plan.local_rows - segment.rows
        ):
            raise ValueError(
                "incoming segments must cover each destination exactly once"
            )
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
        if (
            info.state == pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
            and info.status == pynvml.NVML_SUCCESS
            and any(cluster)
        ):
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
            raise RuntimeError(
                "cross-host exchange requires a common NVLink fabric domain"
            )
        return True
    import pynvml

    pynvml.nvmlInit()
    try:
        handles = [
            pynvml.nvmlDeviceGetHandleByUUID("GPU-" + uuid.removeprefix("GPU-"))
            for uuid in uuids
        ]
        if not all(
            pynvml.nvmlDeviceGetP2PStatus(a, b, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
            == pynvml.NVML_P2P_STATUS_OK
            for i, a in enumerate(handles)
            for j, b in enumerate(handles)
            if i != j
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

    def __init__(
        self,
        group,
        device,
        max_rows,
        hidden_size,
        *,
        chunk_capacity=1024,
        enable_producer_pipeline=True,
        fp8_output_width=0,
    ):
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

            self.device = (
                torch.device("cuda", device)
                if isinstance(device, int)
                else torch.device(device)
            )
            if self.device.type != "cuda":
                raise ValueError("directed exchange requires CUDA")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            self.max_rows, self.hidden_size, self.chunk_capacity = _capacities(
                max_rows, hidden_size, chunk_capacity
            )
            self.fp8_output_width = index(fp8_output_width)
            if self.fp8_output_width and not (
                self.can_route_fp8_output(self.max_rows, self.fp8_output_width)
                and _fp8_output_bytes(1, self.fp8_output_width)
                <= 2 * self.chunk_capacity * self.hidden_size
            ):
                raise ValueError(
                    "FP8 output does not fit the shared source or receive slot"
                )
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "initialize directed exchange outside CUDA graph capture"
                    )
                if symm.get_backend(self.device) != "CUDA":
                    raise RuntimeError(
                        "requires the existing CUDA symmetric-memory backend"
                    )
                props = torch.cuda.get_device_properties(self.device)
            metadata = dict(
                contract=[
                    self.max_rows,
                    self.hidden_size,
                    self.chunk_capacity,
                    "bfloat16",
                    "CUDA",
                    self.fp8_output_width,
                    dist.get_process_group_ranks(group),
                ],
                host=socket.gethostname(),
                uuid=str(props.uuid),
                fabric=_fabric(str(props.uuid)),
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
                    (self.max_rows, self.hidden_size),
                    (self.hidden_size, 1),
                    torch.bfloat16,
                    self.device,
                )
                if cross_host:
                    from rtp_llm.ops.compute_ops import rtp_llm_ops

                    if not rtp_llm_ops.is_cuda_fabric_allocation(self._source):
                        raise RuntimeError(
                            "cross-host exchange requires a FABRIC allocation"
                        )
                self._slots = torch.empty(
                    (2, self.chunk_capacity, self.hidden_size),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                self._stream = torch.cuda.Stream(device=self.device)
                # Same-stream allocator reuse may carry old writes queued on
                # the constructor's stream. Transfer initial slot ownership;
                # record_stream during calls only protects later reclamation.
                self._stream.wait_stream(torch.cuda.current_stream(self.device))
                self._ready = [torch.cuda.Event() for _ in range(2)]
                self._free = [torch.cuda.Event() for _ in range(2)]
                self._readers_done = torch.cuda.Event()
                self._pulls_done = torch.cuda.Event()
                self._done = torch.cuda.Event()
                self._entry = torch.cuda.Event()
                # Force event creation now, before the allocation vote.
                for event in (
                    *self._ready,
                    *self._free,
                    self._readers_done,
                    self._pulls_done,
                    self._done,
                    self._entry,
                ):
                    event.record(self._stream)
                self._stream.synchronize()
        except Exception as exc:
            error = str(exc)
        self._agree("allocation", None, error)

        error, can_pipeline = "", False
        try:
            with torch.cuda.device(self.device):
                self._handle = symm.rendezvous(self._source, group=group)
                h = self._handle
                if (
                    h.offset != 0
                    or h.rank != self.rank
                    or h.world_size != self.size
                    or torch.device(h.device) != self.device
                    or h.buffer_size < self.max_rows * self.hidden_size * 2
                    or h.signal_pad_size // (4 * self.size) < 2
                    or len(h.buffer_ptrs) != self.size
                    or h.buffer_ptrs[self.rank] != self._source.data_ptr()
                ):
                    raise RuntimeError(
                        "raw symmetric allocation base/device/group/size validation failed"
                    )
                self._peers = [
                    h.get_buffer(
                        peer, (self.max_rows, self.hidden_size), torch.bfloat16
                    )
                    for peer in range(self.size)
                ]
                for peer, tensor in enumerate(self._peers):
                    if (
                        tensor.data_ptr() == 0
                        or tensor.data_ptr() != h.buffer_ptrs[peer]
                        or tensor.device != self.device
                        or tensor.dtype != torch.bfloat16
                        or tensor.shape != self._source.shape
                        or not tensor.is_contiguous()
                    ):
                        raise RuntimeError("symmetric peer mapping validation failed")
                can_pipeline = (
                    enable_producer_pipeline
                    and h.signal_pad_size // (4 * self.size) >= 4
                    and callable(getattr(h, "put_signal", None))
                    and callable(getattr(h, "wait_signal", None))
                )
        except Exception as exc:
            error = str(exc)
        # The bounded votes do not bound a rank wedged inside Torch rendezvous;
        # the owning launcher must also provide a bounded process-group timeout.
        self.can_pipeline_producer = all(self._agree("mapping", can_pipeline, error))
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
            errors = [
                (rank, record["error"])
                for rank, record in enumerate(records)
                if record["error"]
            ]
            if errors:
                raise RuntimeError(
                    f"directed exchange {phase} rejected collectively: {errors}"
                )
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
        return (
            plan.size,
            plan.rank,
            plan.local_rows,
            plan.physical_rows,
            tuple(plan.lengths),
            tuple(
                (s.peer, s.source_offset, s.destination_offset, s.rows)
                for s in plan.to_zigzag
            ),
            tuple(
                (s.peer, s.source_offset, s.destination_offset, s.rows)
                for s in plan.to_natural
            ),
        )

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
            chunks, direct = set(), None
            try:
                chunk_rows = (
                    self.chunk_capacity if chunk_rows is None else index(chunk_rows)
                )
                if not 1 <= chunk_rows <= self.chunk_capacity:
                    raise ValueError("chunk_rows exceeds fixed receive capacity")
                for direction in ("to_zigzag", "to_natural"):
                    validate_plan(plan, direction, self.size, self.rank, self.max_rows)
                key = self._plan_key(plan)
                chunks, direct = self._prepared.get(key, (set(), None))
                if chunk_rows in chunks:
                    self._prepared[key] = self._prepared.pop(key)
                    return
                if (
                    not chunks
                    and self.fp8_output_width
                    and plan.local_rows
                    and self.fp8_pre_o_chunk_rows(plan) is None
                ):
                    from rtp_llm.models_py.triton_kernels.mla.fp8_row_gather import (
                        Fp8RowGather,
                    )

                    with torch.cuda.stream(self._stream):
                        peers = [
                            _fp8_output_views(
                                peer, plan.local_rows, self.fp8_output_width
                            )
                            for peer in self._peers
                        ]
                        direct = Fp8RowGather(
                            plan, peers, self.fp8_output_width, self.device
                        )
                contract = [
                    plan.size,
                    plan.local_rows,
                    plan.physical_rows,
                    list(plan.lengths),
                    self.max_rows,
                    self.chunk_capacity,
                    chunk_rows,
                ]
            except Exception as exc:
                error = str(exc)
            phase = f"prepare/{self._prepare_epoch}"
            self._prepare_epoch += 1
            contracts = self._agree(phase, contract, error)
            if any(item != contract for item in contracts):
                self.enabled = False
                raise RuntimeError(
                    "directed exchange prepared contracts differ across ranks"
                )
            self._prepared.pop(key, None)
            chunks.update((chunk_rows, self.chunk_capacity))
            self._prepared[key] = (chunks, direct)
            while len(self._prepared) > _PREPARED_PLAN_LIMIT:
                self._prepared.pop(next(iter(self._prepared)))

    def _require_prepared(self, plan, chunk_rows):
        key = self._plan_key(plan)
        chunks, direct = self._prepared.get(key, ((), None))
        if chunk_rows not in chunks:
            raise ValueError(
                "missing or changed directed exchange contract; call prepare_plan collectively before forward"
            )
        self._prepared[key] = self._prepared.pop(key)
        return direct

    def _validate_tensor(self, tensor, rows, label):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != self.device
            or tensor.dtype != torch.bfloat16
            or tensor.shape != (rows, self.hidden_size)
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ValueError(
                f"{label} must be contiguous inference BF16 [local_rows, hidden_size] on {self.device}"
            )
        for private in (self._source, self._slots):
            if (
                tensor.untyped_storage().data_ptr()
                == private.untyped_storage().data_ptr()
            ):
                raise ValueError(f"{label} must not alias private transport storage")

    def exchange(self, values, plan, direction, out=None, *, before_consume=None):
        """Return an independently owned complete local output, or validated out.

        The caller's stream joins the complete producer before publication.
        Reverse O routing therefore starts after the whole O producer, not an
        unfinished GEMM. Original-input aliasing is safe; private storage is not.
        """
        return self._run(
            values,
            plan,
            direction,
            self.chunk_capacity,
            out=out,
            before_consume=before_consume,
        )

    def consume_chunks(
        self, values, plan, direction, chunk_rows, consumer, *, before_consume=None
    ):
        """Call consumer(start, slot) in destination order on the caller stream."""
        self._run(
            values,
            plan,
            direction,
            chunk_rows,
            consumer=consumer,
            consuming=True,
            before_consume=before_consume,
        )

    def consume_fp8(
        self, activation, plan, consumer, *, before_consume, project_heads=None
    ):
        """Call consumer once with the complete Zigzag FP8 rows and scales."""
        return self._run(
            activation,
            plan,
            "to_zigzag",
            self.chunk_capacity,
            consumer=consumer,
            consuming=True,
            before_consume=before_consume,
            fp8=True,
            project_heads=project_heads,
        )

    def consume_nvfp4(
        self, activation, plan, consumer, *, before_consume, project_heads=None
    ):
        """Deliver Zigzag intervals with their original source's scalar scale."""
        return self._run(
            activation,
            plan,
            "to_zigzag",
            self.chunk_capacity,
            consumer=consumer,
            consuming=True,
            before_consume=before_consume,
            nvfp4=True,
            project_heads=project_heads,
        )

    def _fp8_views(self, storage, rows):
        # These sections fit inside the BF16 allocation used by the inverse route.
        raw = storage.view(torch.uint8).flatten()
        values_bytes = rows * self.hidden_size
        scale_rows = (rows + 3) // 4 * 4
        groups = self.hidden_size // 512
        scale_end = values_bytes + 4 * groups * scale_rows
        values = (
            raw[:values_bytes].view(torch.float8_e4m3fn).view(rows, self.hidden_size)
        )
        scales = raw[values_bytes:scale_end].view(torch.int32).view(groups, scale_rows)
        heads = (
            raw[scale_end : scale_end + rows * 128].view(torch.float32).view(rows, 32)
        )
        return values, scales, heads

    def produce_and_exchange(self, plan, direction, producer, out=None):
        """Queue a complete producer into owned publication storage, without staging copy."""
        return self._run(
            None, plan, direction, self.chunk_capacity, out=out, producer=producer
        )

    def can_route_fp8_output(self, rows, width):
        return (
            0 < rows <= self.max_rows
            and width > 0
            and width % 512 == 0
            and _fp8_output_bytes(rows, width) <= 2 * self.max_rows * self.hidden_size
        )

    def fp8_pre_o_chunk_rows(self, plan):
        rows = plan.local_rows
        width = self.fp8_output_width
        slot_bytes = 2 * self.chunk_capacity * self.hidden_size
        prefer_direct = rows < _FP8_PRE_O_PIPELINE_MIN_ROWS or len(plan.lengths) != 1
        if prefer_direct and _fp8_output_bytes(rows, width) <= slot_bytes:
            return None
        if rows < 4096:
            # Two chunks near the crossover. Keep scale strides four-row aligned.
            chunk = _FP8_PRE_O_SHORT_CHUNKS.get(rows, (rows + 7) // 8 * 4)
        else:
            chunk = ((rows + 4095) // 4096) * 1024
        chunk = min(self.chunk_capacity, chunk)
        if _fp8_output_bytes(chunk, width) > slot_bytes:
            # Only scale padding can exceed the slot once chunk <= chunk_capacity.
            chunk = max(1, chunk // 4 * 4)
        return chunk

    def route_fp8_output_and_project(self, plan, quantize, project, *, out=None):
        """Publish quantized Zigzag rows, then project into Natural output order.

        The quantizer writes directly into the shared source. The consumer
        projects each complete destination chunk while comm fills the next slot.
        Short or fragmented plans gather the full shard with one kernel.
        """
        rows = plan.local_rows
        width = self.fp8_output_width
        chunk_rows = self.fp8_pre_o_chunk_rows(plan)
        receive_rows = rows if chunk_rows is None else index(chunk_rows)
        with self._lock:
            if not self.enabled:
                raise RuntimeError(
                    "directed exchange is disabled after a setup or asynchronous failure"
                )
            try:
                with torch.cuda.device(self.device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "directed exchange does not support CUDA graph capture"
                        )
                    validate_plan(
                        plan, "to_natural", self.size, self.rank, self.max_rows
                    )
                    direct = self._require_prepared(plan, self.chunk_capacity)
                    if (
                        not self.can_route_fp8_output(rows, width)
                        or not 0 < receive_rows <= rows
                    ):
                        raise ValueError("invalid FP8 reverse route geometry")
                    if chunk_rows is None and direct is None:
                        raise ValueError("missing prepared FP8 Direct gather")
                    if not callable(quantize) or not callable(project):
                        raise TypeError(
                            "FP8 reverse route requires quantize and project callbacks"
                        )
                    if (
                        _fp8_output_bytes(receive_rows, width)
                        > 2 * self.chunk_capacity * self.hidden_size
                    ):
                        raise ValueError(
                            "FP8 reverse chunk exceeds fixed receive capacity"
                        )
                    if out is None:
                        out = torch.empty(
                            (rows, self.hidden_size),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                    self._validate_tensor(out, rows, "out")
                    caller = torch.cuda.current_stream(self.device)
                    comm = self._stream
                    timeout_ms = _TIMEOUT_SECONDS * 1000
                    source = _fp8_output_views(self._source, rows, width)
                    peers = (
                        [_fp8_output_views(peer, rows, width) for peer in self._peers]
                        if direct is None
                        else None
                    )
                    caller.wait_event(self._readers_done)
                    if (
                        quantize(source) is not None
                        or torch.cuda.current_stream(self.device) != caller
                    ):
                        raise RuntimeError(
                            "quantize must keep the caller stream and return None"
                        )
                    self._entry.record(caller)
                    with torch.cuda.stream(comm), profile_scope(
                        f"{_OUTPUT_ROUTE_SCOPE}.fp8_publish"
                    ):
                        comm.wait_event(self._done)
                        comm.wait_event(self._entry)
                        self._handle.barrier(channel=0, timeout_ms=timeout_ms)
                    for step, start in enumerate(range(0, rows, receive_rows)):
                        end = min(start + receive_rows, rows)
                        count_rows = end - start
                        slot = step % 2
                        storage = self._slots[slot]
                        values, scales = _fp8_output_views(storage, count_rows, width)
                        with torch.cuda.stream(comm), profile_scope(
                            f"{_OUTPUT_ROUTE_SCOPE}.fp8_pull"
                        ):
                            if step >= 2:
                                comm.wait_event(self._free[slot])
                            storage.record_stream(comm)
                            if direct is not None:
                                direct(values, scales)
                            else:
                                for peer, src, dst, count in chunk_intersections(
                                    plan.to_natural, start, end
                                ):
                                    peer_values, peer_scales = peers[peer]
                                    values[dst : dst + count].copy_(
                                        peer_values[src : src + count]
                                    )
                                    scales[:, dst : dst + count].copy_(
                                        peer_scales[:, src : src + count]
                                    )
                                if count_rows % 4:
                                    scales[:, count_rows:].fill_(0x7F7F7F7F)
                            self._ready[slot].record(comm)
                        caller.wait_event(self._ready[slot])
                        storage.record_stream(caller)
                        result = project(
                            values, scales[:, :count_rows].T, out[start:end]
                        )
                        if (
                            result is not None
                            or torch.cuda.current_stream(self.device) != caller
                        ):
                            raise RuntimeError(
                                "project must keep the caller stream and return None"
                            )
                        self._free[slot].record(caller)
                    with torch.cuda.stream(comm), profile_scope(
                        f"{_OUTPUT_ROUTE_SCOPE}.peer_reads_done"
                    ):
                        self._handle.barrier(channel=1, timeout_ms=timeout_ms)
                        self._readers_done.record(comm)
                    self._done.record(caller)
            except Exception:
                self.enabled = False
                raise
            return out

    def produce_chunks_and_exchange(self, plan, chunk_rows, producer, out=None):
        """Call producer(start, target) on alternating streams, then pull those rows.

        Chunk size and call order must agree across the group. The caller's
        stream joins both producers and all local pulls before consuming out.
        """
        return self._run(
            None,
            plan,
            "to_natural",
            self.chunk_capacity,
            out=out,
            producer=producer,
            producer_rows=index(chunk_rows),
        )

    def _run(
        self,
        values,
        plan,
        direction,
        chunk_rows,
        *,
        out=None,
        consumer=None,
        consuming=False,
        before_consume=None,
        producer=None,
        fp8=False,
        nvfp4=False,
        project_heads=None,
        producer_rows=None,
    ):
        with self._lock:
            if not self.enabled:
                raise RuntimeError(
                    "directed exchange is disabled after a setup or asynchronous failure"
                )
            try:
                with torch.cuda.device(self.device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "directed exchange does not support CUDA graph capture"
                        )
                    caller = torch.cuda.current_stream(self.device)
                    chunk_rows = index(chunk_rows)
                    if not 1 <= chunk_rows <= self.chunk_capacity:
                        raise ValueError("chunk_rows exceeds fixed receive capacity")
                    validate_plan(plan, direction, self.size, self.rank, self.max_rows)
                    self._require_prepared(plan, chunk_rows)
                    if fp8 or nvfp4:
                        if (
                            self.hidden_size != 7168
                            or values.device != self.device
                            or values.shape != (plan.local_rows, self.hidden_size)
                        ):
                            raise ValueError(
                                "Quantized directed exchange requires local K3.1 hidden rows"
                            )
                    elif producer is None:
                        self._validate_tensor(values, plan.local_rows, "values")
                    else:
                        if not callable(producer) or consuming:
                            raise TypeError(
                                "producer must be callable and cannot consume chunks"
                            )
                        # This source may still be read by a peer after the
                        # previous return. A direct producer must join that
                        # global lifetime fence before overwriting it.
                        caller.wait_event(self._readers_done)
                        values = self._source[: plan.local_rows]
                        if producer_rows is None:
                            result = producer(values)
                            if (
                                result is not None
                                or torch.cuda.current_stream(self.device) != caller
                            ):
                                raise RuntimeError(
                                    "producer must return None and keep the caller stream"
                                )
                        elif not self.can_pipeline_producer or producer_rows <= 0:
                            raise ValueError(
                                "unsupported pipelined producer chunk geometry"
                            )
                    if before_consume is not None and not callable(before_consume):
                        raise TypeError("before_consume must be callable")
                    if consuming:
                        if not callable(consumer):
                            raise TypeError("consumer must be callable")
                    else:
                        if out is None:
                            out = torch.empty_like(values)
                        self._validate_tensor(out, plan.local_rows, "out")
                    if fp8 or nvfp4:
                        return self._submit_quantized(
                            values,
                            plan,
                            caller,
                            consumer,
                            before_consume,
                            project_heads,
                            nvfp4=nvfp4,
                        )
                    if producer_rows is not None:
                        self._submit_produced_chunks(
                            values, plan, producer_rows, producer, caller, out
                        )
                    else:
                        self._submit(
                            values,
                            plan,
                            direction,
                            chunk_rows,
                            caller,
                            out,
                            consumer,
                            consuming,
                            before_consume,
                            producer is None,
                        )
            except Exception:
                self.enabled = False
                raise
            return out

    def _submit_quantized(
        self,
        activation,
        plan,
        caller,
        consumer,
        before_consume,
        project_heads,
        *,
        nvfp4,
    ):
        rows = plan.local_rows
        comm = self._stream
        source, source_scales, source_heads = self._fp8_views(
            self._source, self.max_rows
        )
        peers = [self._fp8_views(peer, self.max_rows) for peer in self._peers]
        # Allocate on the caller before recording entry so comm also waits
        # for any earlier uses of memory recycled by the caller's allocator.
        if nvfp4:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
                PackedNvfp4Activation,
                nvfp4_packet_capacity,
            )
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
                Fp4Activation,
            )
            from rtp_llm.models_py.triton_kernels.common.nvfp4 import pull_nvfp4_rows

            packet_bytes = nvfp4_packet_capacity(rows, self.hidden_size)
            if packet_bytes > source_heads.data_ptr() - self._source.data_ptr():
                raise ValueError("NVFP4 packet exceeds its publication region")
            wire = activation.wire
            if (
                wire.shape != (packet_bytes,)
                or wire.dtype != torch.uint8
                or not wire.is_contiguous()
            ):
                raise ValueError("Invalid NVFP4 source packet")
            local_wire = self._source.view(torch.uint8).flatten()[:packet_bytes]
            remote_packets = [
                PackedNvfp4Activation(
                    peer.view(torch.uint8).flatten()[:packet_bytes], activation.shape
                )
                for peer in self._peers
            ]
            target_packet = PackedNvfp4Activation.empty(
                rows, self.hidden_size, self.device
            )
            values, block_scales, _ = target_packet.unpack(target_packet.wire)
            target = Fp4Activation(
                values,
                block_scales,
                values.new_empty((rows,), dtype=torch.float32),
            )
        else:
            target = torch.empty_like(activation.values)
            scales = activation.scale_wire.new_empty(
                (source_scales.shape[0], (rows + 3) // 4 * 4)
            )
        self._entry.record(caller)
        if project_heads is not None:
            if not hasattr(self, "_heads_ready"):
                self._heads_ready = torch.cuda.Event()
            # Reuse the caller stream for projection. Comm can publish hidden
            # from the entry event while this stream prepares the head weights.
            caller.wait_event(self._readers_done)
            project_heads(source_heads[:rows])
            self._heads_ready.record(caller)
            heads = torch.empty((rows, 32), device=self.device, dtype=torch.float32)
        else:
            heads = None

        with torch.cuda.stream(comm), profile_scope(f"{_HIDDEN_ROUTE_SCOPE}.publish"):
            comm.wait_event(self._done)
            comm.wait_event(self._entry)
            if nvfp4:
                wire.record_stream(comm)
                local_wire.copy_(wire)
            else:
                activation.values.record_stream(comm)
                activation.scale_wire.record_stream(comm)
                source[:rows].copy_(activation.values)
                source_scales[:, :rows].copy_(activation.scale_wire[:, :rows])
            self._handle.barrier(channel=0, timeout_ms=_TIMEOUT_SECONDS * 1000)
        with torch.cuda.stream(comm), profile_scope(f"{_HIDDEN_ROUTE_SCOPE}.pull"):
            if nvfp4:
                target_packet.wire.record_stream(comm)
                target.scale.record_stream(comm)
                for segment in plan.to_zigzag:
                    pull_nvfp4_rows(remote_packets[segment.peer], target, segment)
            else:
                target.record_stream(comm)
                scales.record_stream(comm)
                for segment in plan.to_zigzag:
                    src, dst, count = (
                        segment.source_offset,
                        segment.destination_offset,
                        segment.rows,
                    )
                    remote, remote_scales, _ = peers[segment.peer]
                    target[dst : dst + count].copy_(remote[src : src + count])
                    scales[:, dst : dst + count].copy_(
                        remote_scales[:, src : src + count]
                    )
                scales[:, rows:].fill_(0x7F7F7F7F)
            self._ready[0].record(comm)

        with torch.cuda.stream(comm):
            if heads is not None:
                with profile_scope(
                    "RTP::attention.dsa.indexer.head_weights_natural_to_zigzag"
                ):
                    comm.wait_event(self._heads_ready)
                    self._handle.barrier(channel=0, timeout_ms=_TIMEOUT_SECONDS * 1000)
                    heads.record_stream(comm)
                    for peer, src, dst, count in chunk_intersections(
                        plan.to_zigzag, 0, rows
                    ):
                        heads[dst : dst + count].copy_(
                            peers[peer][2][src : src + count]
                        )
            self._pulls_done.record(comm)
            with profile_scope(f"{_HIDDEN_ROUTE_SCOPE}.peer_reads_done"):
                self._handle.barrier(channel=1, timeout_ms=_TIMEOUT_SECONDS * 1000)
                self._readers_done.record(comm)
        if before_consume is not None:
            result = before_consume()
            if result is not None or torch.cuda.current_stream(self.device) != caller:
                raise RuntimeError(
                    "before_consume must return None and keep the caller stream"
                )
        # Only hidden is needed for the query GEMMs. Head weights join after
        # the consumer has submitted them.
        caller.wait_event(self._ready[0])
        result = consumer(target) if nvfp4 else consumer(target, scales[:, :rows].T)
        if result is not None or torch.cuda.current_stream(self.device) != caller:
            raise RuntimeError(
                "Quantized route consumer must return None and keep the caller stream"
            )
        caller.wait_event(self._pulls_done)
        self._done.record(caller)
        return heads

    def _submit_produced_chunks(self, source, plan, chunk_rows, producer, caller, out):
        comm = self._stream
        self._entry.record(caller)
        comm.wait_event(self._entry)
        out.record_stream(comm)
        streams = (caller, comm)
        timeout_ms = _TIMEOUT_SECONDS * 1000

        for step, start in enumerate(range(0, plan.local_rows, chunk_rows)):
            end = min(start + chunk_rows, plan.local_rows)
            lane = step % 2
            stream = streams[lane]
            # Channels 0 and 1 belong to the whole-shard paths and source reuse.
            channel = 2 + lane
            # The forward route names the natural owners of our Zigzag rows.
            destinations = dict.fromkeys(
                peer
                for peer, _, _, _ in chunk_intersections(plan.to_zigzag, start, end)
                if peer != self.rank
            )
            pulls = {}
            for segment in plan.to_natural:
                left = max(start, segment.source_offset)
                right = min(end, segment.source_offset + segment.rows)
                if left < right:
                    dst = segment.destination_offset + left - segment.source_offset
                    pulls.setdefault(segment.peer, []).append((left, dst, right - left))

            with torch.cuda.stream(stream):
                if step:
                    stream.wait_event(self._ready[1 - lane])
                result = producer(start, source[start:end])
                if (
                    result is not None
                    or torch.cuda.current_stream(self.device) != stream
                ):
                    raise RuntimeError(
                        "chunk producer must return None and keep its assigned stream"
                    )
                # The next producer waits for the GEMM, not for this lane's pulls.
                self._ready[lane].record(stream)
                with profile_scope(f"{_OUTPUT_ROUTE_SCOPE}.publish_chunk"):
                    for peer in destinations:
                        self._handle.put_signal(
                            peer, channel=channel, timeout_ms=timeout_ms
                        )
                with profile_scope(f"{_OUTPUT_ROUTE_SCOPE}.pull_chunk"):
                    for peer, segments in pulls.items():
                        if peer != self.rank:
                            self._handle.wait_signal(
                                peer, channel=channel, timeout_ms=timeout_ms
                            )
                        for src, dst, rows in segments:
                            out[dst : dst + rows].copy_(
                                self._peers[peer][src : src + rows]
                            )

        caller.wait_stream(comm)
        self._done.record(caller)
        with torch.cuda.stream(comm), profile_scope(
            f"{_OUTPUT_ROUTE_SCOPE}.peer_reads_done"
        ):
            comm.wait_event(self._done)
            self._handle.barrier(channel=1, timeout_ms=timeout_ms)
            self._readers_done.record(comm)

    def _submit(
        self,
        values,
        plan,
        direction,
        chunk_rows,
        caller,
        out,
        consumer,
        consuming,
        before_consume=None,
        publish_source=True,
    ):
        comm = self._stream
        route_scope = (
            "RTP::row_exchange.natural_to_zigzag"
            if direction == "to_zigzag"
            else _OUTPUT_ROUTE_SCOPE
        )
        self._entry.record(caller)
        with torch.cuda.stream(comm), profile_scope(f"{route_scope}.publish"):
            comm.wait_event(self._done)
            comm.wait_event(self._entry)
            if publish_source:
                values.record_stream(comm)
                self._source[: plan.local_rows].copy_(values)
            self._handle.barrier(channel=0, timeout_ms=_TIMEOUT_SECONDS * 1000)
            if not consuming:
                out.record_stream(comm)
            self._slots.record_stream(comm)
        if before_consume is not None:
            with torch.cuda.stream(caller):
                result = before_consume()
                if (
                    result is not None
                    or torch.cuda.current_stream(self.device) != caller
                ):
                    raise RuntimeError(
                        "before_consume must return None and keep the caller stream"
                    )
        slots_used = [False, False]
        pull_rows = chunk_rows if consuming else max(1, plan.local_rows)
        for number, start in enumerate(range(0, plan.local_rows, pull_rows)):
            end = min(start + pull_rows, plan.local_rows)
            slot = number % 2
            with torch.cuda.stream(comm), profile_scope(f"{route_scope}.pull"):
                if consuming and slots_used[slot]:
                    comm.wait_event(self._free[slot])
                target = (
                    self._slots[slot, : end - start] if consuming else out[start:end]
                )
                for peer, source, destination, rows in chunk_intersections(
                    getattr(plan, direction), start, end
                ):
                    target[destination : destination + rows].copy_(
                        self._peers[peer][source : source + rows]
                    )
                if consuming:
                    self._ready[slot].record(comm)
            if consuming:
                with torch.cuda.stream(caller):
                    caller.wait_event(self._ready[slot])
                    self._slots.record_stream(caller)
                    result = consumer(start, target)
                    if (
                        result is not None
                        or torch.cuda.current_stream(self.device) != caller
                    ):
                        raise RuntimeError(
                            "consumer must return None and keep the caller stream; unjoined work is unsupported"
                        )
                    self._free[slot].record(caller)
                slots_used[slot] = True
        with torch.cuda.stream(comm):
            self._pulls_done.record(comm)
            # Protect remote sources until every peer has finished its pulls.
            with profile_scope(f"{route_scope}.peer_reads_done"):
                self._handle.barrier(channel=1, timeout_ms=_TIMEOUT_SECONDS * 1000)
                self._readers_done.record(comm)
        with torch.cuda.stream(caller):
            # Only local pulls are needed to consume the owned output. Keep
            # the global readers fence on comm, where it protects source reuse
            # without delaying unrelated downstream caller work.
            caller.wait_event(self._pulls_done)
            self._done.record(caller)
