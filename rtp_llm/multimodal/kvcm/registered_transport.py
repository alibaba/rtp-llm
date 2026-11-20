"""Registered CUDA slots with shared MR ownership and isolated failures."""

import ctypes
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from bisect import bisect_left
from contextlib import ExitStack, contextmanager, nullcontext

import torch

from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


def _enable_sync_memops(arena):
    # GPUDirect mappings must remain consistent with CUDA copies of this
    # allocation. DMA-BUF export alone does not enable SYNC_MEMOPS.
    driver = ctypes.CDLL("libcuda.so.1")
    attribute = 6  # CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
    pointer = arena.data_ptr()
    setter = driver.cuPointerSetAttribute
    setter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    setter.restype = ctypes.c_int
    enabled = ctypes.c_uint(1)
    result = setter(ctypes.byref(enabled), attribute, pointer)
    if result != 0:
        raise RuntimeError(f"CUDA SYNC_MEMOPS setup failed: {result}")
    getter = driver.cuPointerGetAttribute
    getter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    getter.restype = ctypes.c_int
    actual = ctypes.c_uint()
    result = getter(ctypes.byref(actual), attribute, pointer)
    if result != 0 or actual.value != 1:
        raise RuntimeError(f"CUDA SYNC_MEMOPS verification failed: {result}")


def _safe_error_message(error):
    """Bounded diagnostic text without URI credentials or common secret fields."""
    try:
        message = str(error)[:4096]
        message = re.sub(
            r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+", "<redacted-uri>", message
        )
        message = re.sub(
            r"(?i)\b(?:Bearer|Basic)\s+[^\s,;]+", "<redacted-auth>", message
        )
        message = re.sub(
            r"(?i)\b(password|passwd|pwd|token|secret|credentials?|authorization|api[_-]?key|access[_-]?key|cookie|signature)"
            r"[\"']?\s*[:=]\s*(?:\"(?:\\.|[^\"\\])*(?:\"|$)|'(?:\\.|[^'\\])*(?:'|$)|[^\s,;]+)",
            r"\1=<redacted>",
            message,
        )
        message = re.sub(r"[^\s:/@]+:[^\s@]+@", "<redacted-userinfo>@", message)
        return " ".join(message.split())[:384] or type(error).__name__
    except Exception:
        return type(error).__name__


def _safe_error_context(error):
    fields = {
        "error_type": type(error).__name__,
        "error_message": _safe_error_message(error),
    }
    try:
        code = getattr(error, "code", None)
        if callable(code):
            code = code()
        name = getattr(code, "name", None)
        if isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", name):
            fields["error_code"] = name
        elif type(code) is int:
            fields["error_code"] = code
        numeric = getattr(code, "value", None)
        if isinstance(numeric, tuple) and numeric:
            numeric = numeric[0]  # grpc.StatusCode.value is (number, description).
        if type(numeric) is int:
            fields["error_numeric_code"] = numeric
        details = getattr(error, "details", None)
        if callable(details):
            fields["error_details"] = _safe_error_message(details())
        reason = getattr(error, "reason", None)
        if isinstance(reason, str) and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,79}", reason
        ):
            fields["error_reason"] = reason
        inflight = getattr(error, "data_transfer_may_be_inflight", None)
        if type(inflight) is bool:
            fields["data_transfer_may_be_inflight"] = inflight
        if isinstance(error, RegisteredTransportUnavailableError):
            fields["restart_required"] = error.restart_required
            if error.quarantine_id is not None:
                fields["quarantine_id"] = error.quarantine_id
            for key, value in error.initial_failure.items():
                if key == "data_transfer_may_be_inflight" and type(value) is bool:
                    fields["initial_" + key] = value
                elif key == "error_code" and type(value) is int:
                    fields["initial_" + key] = value
                elif (
                    key in ("error_type", "error_code", "error_reason")
                    and isinstance(value, str)
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", value)
                ):
                    fields["initial_" + key] = value
                elif key == "error_message" and isinstance(value, str):
                    fields["initial_error_message"] = _safe_error_message(value)
    except Exception:
        pass  # Logging cannot change cache fallback or memory ownership.
    # Wrapper exceptions may intentionally suppress traceback printing; retain
    # bounded sanitized cause text without logging an unfiltered traceback.
    causes, seen, cause = [], {id(error)}, error
    try:
        for _ in range(2):
            cause = getattr(cause, "__cause__", None) or getattr(
                cause, "__context__", None
            )
            if cause is None or id(cause) in seen:
                break
            seen.add(id(cause))
            causes.append(
                {
                    "error_type": type(cause).__name__,
                    "error_message": _safe_error_message(cause),
                }
            )
    except Exception:
        pass
    if causes:
        fields["error_causes"] = causes
    return fields


class RegisteredTransportUnavailableError(RuntimeError):
    def __init__(
        self, message, *, quarantined, initial_failure=None, quarantine_id=None
    ):
        super().__init__(message)
        self.reason = (
            "gdr_transport_quarantined" if quarantined else "gdr_transport_closed"
        )
        self.restart_required = quarantined
        self.initial_failure = dict(initial_failure or {})
        self.quarantine_id = quarantine_id


class RegisteredObjectTransport:
    transport_mode = "gdr"
    _quarantine_lock = threading.RLock()
    _quarantined_transports = []

    def __init__(
        self, owner, buffer, device, observe_stage, *, lazy_stream=False, extent=None
    ):
        self._owner = owner
        self._buffer = buffer
        self._extent = extent
        self._capacity = buffer.numel() if extent is None else extent[1]
        self._device = device
        self._observe_stage = observe_stage
        self._lock = threading.Lock()
        self._poisoned = False
        self.device_failed = False
        self._host_staging = (
            None
            if owner.direct_payload
            else torch.empty(
                self._capacity, dtype=torch.uint8, device="cpu", pin_memory=True
            )
        )
        self._stream = None if lazy_stream else torch.cuda.Stream(device=device)

    @property
    def _client(self):
        return self._owner._client

    @property
    def quarantined(self):
        return self._poisoned

    def _check_available(self):
        if self._owner._closed or self._poisoned or self._buffer is None:
            raise RegisteredTransportUnavailableError(
                "registered transport unavailable; restart worker after GDR failure",
                quarantined=self._poisoned,
                initial_failure=getattr(self, "_quarantine_cause", {}),
                quarantine_id=getattr(self, "_quarantine_id", None),
            )

    def _record_quarantine(self, error, **context):
        self._quarantine_started = time.monotonic()
        self._quarantine_operation = context.get("operation", "unknown")
        self._quarantine_stage = context.get("stage", "unknown")
        self._quarantine_cause = _safe_error_context(error)
        self._quarantine_cause.setdefault("data_transfer_may_be_inflight", True)
        self._quarantine_id = uuid.uuid4().hex
        self._poisoned = True
        try:
            logging.error(
                "MM_EMB_GDR_QUARANTINE %s",
                json.dumps(
                    dict(
                        event="gdr_transport_quarantined",
                        reason="gdr_transport_quarantined",
                        action="hold_registered_slot",
                        message="GDR completion is unknown; isolate the registered slot and use encoder fallback",
                        quarantine_id=self._quarantine_id,
                        **self._quarantine_cause,
                        **context,
                    ),
                    sort_keys=True,
                ),
            )
        except Exception:
            pass  # Diagnostics must never mask the original failure.

        self._owner._report_hold(self._quarantine_operation, "entered")

    def _quarantine(self, error, **context):
        if not self._poisoned:
            with self._quarantine_lock:
                self._quarantined_transports.append(self)
                self._record_quarantine(error, **context)

    def reclaim_quarantine(self, ttl_seconds):
        # The pool excludes leased slots. Never wait behind a live IO worker.
        if not self._lock.acquire(blocking=False):
            return False
        try:
            # A checksum-free object cannot tolerate unknown DMA overwriting
            # a reused address. Native BufferSafe is checked at failure time.
            if (
                not self._poisoned
                or self._owner.direct_payload
                or self.device_failed
                or self._quarantine_stage != "sdk_transfer"
                or time.monotonic() - self._quarantine_started < ttl_seconds
            ):
                return False
            # For a serving pool, the operator-selected TTL returns this Tensor
            # to its fixed registration. It is not a BufferSafe completion.
            # No allocator reuse or MR replacement occurs. The codec verifies
            # the complete checksum and exact key before exposing a late read.
            with self._quarantine_lock:
                self._quarantined_transports.remove(self)
            self._poisoned = False
            return True
        finally:
            self._lock.release()

    def is_transfer_quiescent(self, error):
        if self._owner._closed or self._poisoned:
            return False
        check = getattr(self._client, "buffer_safe_after_failure", None)
        try:
            return callable(check) and check(error) is True
        except Exception:
            return False

    def _handle_failure(self, error, **context):
        # Legacy SDKs and unclassified CUDA/native exceptions remain unsafe.
        if context.get("stage") != "sdk_transfer":
            self.device_failed = True
        elif self.is_transfer_quiescent(error):
            return
        self._quarantine(error, **context)

    def _view(self, tensor):
        self._check_available()
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != torch.uint8
            or tensor.ndim != 1
            or not tensor.is_contiguous()
            or not 0 < tensor.numel() <= self._capacity
        ):
            raise ValueError("invalid registered object buffer")
        return self._buffer[: tensor.numel()]

    def _stage(self, operation, name):
        observe = getattr(self, "_observe_stage", None)
        return observe(operation, name) if observe is not None else nullcontext()

    @contextmanager
    def _serialized(self, operation):
        # Time the existing waits without adding a CUDA synchronization.
        with ExitStack() as stack:
            with self._stage(operation, "transport_lock_wait"):
                stack.enter_context(self._lock)
            self._check_available()
            with self._stage(operation, "cuda_gate_wait"):
                stack.enter_context(cuda_graph_gate.operation())
            stack.enter_context(torch.cuda.device(self._device))
            if self._stream is None:
                self._stream = torch.cuda.Stream(device=self._device)
            stack.enter_context(torch.cuda.stream(self._stream))
            yield

    def object_size(self, key, **kwargs):
        # Metadata lookup uses its own client lock and never touches the arena.
        self._check_available()
        return self._client.object_size(key, **kwargs)

    def pack(self, plan):
        from .tensor_object import pack_object

        try:
            if self._owner.direct_payload:
                with self._serialized("write"):
                    packed = pack_object(
                        plan, out=self._buffer, observe_stage=self._observe_stage
                    )
                    with self._stage("write", "pack_ready_wait"):
                        self._wait_for_stream_completion()
                    return packed
            return pack_object(
                plan, out=self._host_staging, observe_stage=self._observe_stage
            )
        except Exception as error:
            self._handle_failure(error, operation="write", stage="pack")
            raise

    def save_one(self, key, buffer, **kwargs):
        with self._serialized("write"):
            direct = self._owner.direct_payload
            if direct:
                if (
                    buffer.device != self._buffer.device
                    or buffer.dtype != torch.uint8
                    or buffer.ndim != 1
                    or not buffer.is_contiguous()
                    or not 0 < buffer.numel() <= self._capacity
                    or buffer.data_ptr() != self._buffer.data_ptr()
                ):
                    raise ValueError("invalid registered object buffer")
                transfer = buffer
            else:
                transfer = self._view(buffer)
            failure_stage = "host_copy"
            try:
                if not direct:
                    host = self._host_staging[: buffer.numel()]
                    with self._stage("write", "host_copy"):
                        if host.data_ptr() != buffer.data_ptr():
                            # Avoid dispatching a CPU thread team for a byte copy.
                            ctypes.memmove(
                                host.data_ptr(), buffer.data_ptr(), buffer.numel()
                            )
                    failure_stage = "gpu_copy"
                    with self._stage("write", "h2d_staging"):
                        transfer.copy_(host, non_blocking=True)
                        self._stream.synchronize()
                # Exact-object native IO returns after completion on success.
                failure_stage = "sdk_transfer"
                with self._stage("write", "native_save"):
                    self._client.save_one(key, transfer, **kwargs)
                if not direct:
                    failure_stage = "gpu_ready_wait"
                    with self._stage("write", "gpu_ready_wait"):
                        self._stream.synchronize()
            except Exception as error:
                self._handle_failure(
                    error,
                    operation="write",
                    stage=failure_stage,
                    key=key,
                    trace_id=kwargs.get("trace_id"),
                    nbytes=buffer.numel(),
                )
                raise

    def load_one(self, key, buffer, **kwargs):
        return self._load("load_one", key, buffer, key=key, **kwargs)

    def load_from_meta(self, meta, buffer, **kwargs):
        return self._load("load_from_meta", meta, buffer, **kwargs)

    def load_object_from_meta(
        self,
        meta,
        *,
        consumer_stream,
        expected_key=None,
        trace_id=None,
        deadline=None,
    ):
        from .tensor_object import unpack_object

        size = meta.size_bytes
        if not 0 < size <= self._capacity:
            raise ValueError("invalid registered object size")
        with self._serialized("read"):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("registered EMB read expired before transfer")
            transfer = self._buffer[:size]
            failure_stage = "sdk_transfer"
            try:
                with self._stage("read", "native_load"):
                    self._client.load_from_meta(meta, transfer, trace_id=trace_id)
                # Native I/O has completed. No CUDA copy has been submitted
                # and this timeout can safely return the lease to the arena.
                failure_stage = "post_transfer"
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("registered EMB read expired before restore")
                failure_stage = "restore"
                value = unpack_object(
                    transfer,
                    cuda_device=self._device,
                    max_bytes=self._capacity,
                    observe_stage=self._observe_stage,
                    consumer_stream=consumer_stream,
                    expected_key=expected_key,
                )
                with self._stage("read", "restore_ready_wait"):
                    self._wait_for_stream_completion()
                return value
            except Exception as error:
                # Layout validation completes before any payload copy. Header
                # copies to CPU are synchronous and cannot still use the slot.
                if failure_stage == "sdk_transfer" or (
                    failure_stage == "restore" and not isinstance(error, ValueError)
                ):
                    self._handle_failure(
                        error,
                        operation="read",
                        stage=failure_stage,
                        trace_id=trace_id,
                        nbytes=size,
                    )
                raise

    def _wait_for_stream_completion(self):
        event = (
            torch.cuda.Event(blocking=True)
            if self._owner.blocking_cuda_events
            else None
        )
        self._stream.record_event(event).synchronize()

    def _load(self, method, value, buffer, *, key=None, **kwargs):
        with self._serialized("read"):
            transfer = self._view(buffer)
            failure_stage = "sdk_transfer"
            try:
                with self._stage("read", "native_load"):
                    getattr(self._client, method)(value, transfer, **kwargs)
                failure_stage = "gpu_ready_wait"
                stream = self._stream
                with self._stage("read", "gpu_ready_wait"):
                    stream.synchronize()
                failure_stage = "gpu_copy"
                host = self._host_staging[: buffer.numel()]
                with self._stage("read", "d2h_staging"):
                    host.copy_(transfer, non_blocking=True)
                    stream.synchronize()
                failure_stage = "host_copy"
                with self._stage("read", "host_copy"):
                    ctypes.memmove(buffer.data_ptr(), host.data_ptr(), buffer.numel())
                return buffer
            except Exception as error:
                self._handle_failure(
                    error,
                    operation="read",
                    stage=failure_stage,
                    key=key,
                    trace_id=kwargs.get("trace_id"),
                    nbytes=buffer.numel(),
                )
                raise


class RegisteredTransportPool:
    """One registered GPU arena and KVCM client shared by disjoint slot leases."""

    _GUARD_BYTES = 65536  # PACE requires a nonzero MR-relative transfer offset.

    def __init__(
        self,
        factory,
        *,
        max_object_bytes,
        device,
        slots=1,
        observe_stage=None,
        quarantine_ttl_seconds=300,
        observe_hold=None,
        observe_lease=None,
        observe_arena=None,
        observe_acquire_failure=None,
        direct_payload=False,
        arena_bytes=None,
        min_block_bytes=None,
        observe_lock_hold=None,
    ):
        if type(slots) is not int or slots <= 0:
            raise ValueError("slots must be a positive integer")
        if type(quarantine_ttl_seconds) is not int or quarantine_ttl_seconds <= 0:
            raise ValueError("quarantine_ttl_seconds must be a positive integer")
        self.direct_payload = direct_payload
        self.blocking_cuda_events = os.environ.get("MM_EMB_BLOCKING_CUDA_EVENTS") == "1"
        self._quarantine_ttl_seconds = quarantine_ttl_seconds
        self._observe_hold = observe_hold
        self._observe_lease = observe_lease
        self._observe_arena = observe_arena
        self._observe_acquire_failure = observe_acquire_failure
        self._observe_lock_hold = observe_lock_hold
        self._lock_hold_rng = random.Random()
        self._gc_stop = threading.Event()
        self._condition = threading.Condition()
        self._accepting = True
        self._closed = False
        self._in_use = {}  # slot -> whether the lease is for a write
        self._pending_gc_success = []
        self._arena_bytes = arena_bytes
        self._min_block_bytes = min_block_bytes
        self._free_ranges = None
        self._idle_streams = []
        self._create_slots(factory, slots, max_object_bytes, device, observe_stage)
        self.instance_id = self._client.instance_id
        self.instance_group = self._client.instance_group
        self._gc_thread = threading.Thread(
            target=self._gc_loop, name="mm-emb-quarantine-gc", daemon=True
        )
        self._gc_thread.start()

    def _create_slots(self, factory, slots, capacity, device, observe_stage):
        if not 0 < capacity <= 1024**3:
            raise ValueError("invalid registered object capacity")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GDR transport requires a CUDA device")

        # Dtype views in the direct codec need aligned slot addresses even
        # when the configured maximum object size is not a multiple of 64.
        stride = (capacity + 63) // 64 * 64 if self.direct_payload else capacity
        capacity_bytes = slots * stride
        if self._arena_bytes is not None or self._min_block_bytes is not None:
            block = (
                self._min_block_bytes
                if self._min_block_bytes is not None
                else 1024**2
            )
            if (
                not self.direct_payload
                or type(block) is not int
                or block < 64
                or block % 64
            ):
                raise ValueError(
                    "sized GDR requires direct payload and a 64-byte aligned block"
                )
            self._min_block_bytes = block
            self._device = device
            self._observe_stage = observe_stage
            self._max_object_bytes = capacity
            max_extent = self._round_up(capacity)
            # When only a granularity is set, keep the old registration budget.
            capacity_bytes = (
                self._arena_bytes
                if self._arena_bytes is not None
                else capacity_bytes // block * block
            )
            if (
                type(capacity_bytes) is not int
                or capacity_bytes < max_extent
                or capacity_bytes % block
            ):
                raise ValueError(
                    "GDR arena must be block-aligned and fit the largest object"
                )
            self._free_ranges = [(self._GUARD_BYTES, capacity_bytes)]
            # Leave a contiguous range for a maximum-sized read when the arena
            # has capacity for at least one maximum-sized read and write.
            reserve = max_extent if capacity_bytes >= 2 * max_extent else 0
            self._write_end = self._GUARD_BYTES + capacity_bytes - reserve
        with cuda_graph_gate.operation(), RegisteredObjectTransport._quarantine_lock:
            held = RegisteredObjectTransport._quarantined_transports
            if held:
                raise RegisteredTransportUnavailableError(
                    "GDR transport is quarantined; restart the worker",
                    quarantined=True,
                    initial_failure=held[0]._quarantine_cause,
                    quarantine_id=held[0]._quarantine_id,
                )
            with torch.cuda.device(device):
                self._allocator = torch.cuda.MemPool()
                with torch.cuda.use_mem_pool(self._allocator):
                    self._arena = torch.empty(
                        self._GUARD_BYTES + capacity_bytes,
                        dtype=torch.uint8,
                        device=device,
                    )
                _enable_sync_memops(self._arena)
                self._slots = [
                    RegisteredObjectTransport(
                        self,
                        self._arena.narrow(
                            0, self._GUARD_BYTES + i * stride, capacity
                        ),
                        device,
                        observe_stage,
                    )
                    for i in range(slots if self._free_ranges is None else 0)
                ]
                try:
                    # Register the allocation root exactly once. IO passes only
                    # the leased slot's view, trimmed to the object size.
                    self._client = factory(gpu_tensors=(self._arena,))
                except Exception as error:
                    # Native initialization may have partially registered the MR.
                    self._quarantine_registration(error, "registration")
                    raise

    def _quarantine_registration(self, error, stage):
        if self._free_ranges is not None:
            # Registration failure can affect the entire MR, including free space.
            self._slots.append(
                RegisteredObjectTransport(
                    self, self._arena, self._device, self._observe_stage, lazy_stream=True
                )
            )
        for slot in self._slots:
            slot._quarantine(error, stage=stage)

    def _report_hold(self, operation, event):
        if self._observe_hold is None:
            return
        held = [slot for slot in self._slots if slot.quarantined]
        arena = self._arena
        registered_bytes = arena.numel() if arena is not None else 0
        held_bytes = sum(slot._capacity for slot in held)
        age = max(
            (time.monotonic() - slot._quarantine_started for slot in held), default=0
        )
        try:
            self._observe_hold(
                operation, event, held_bytes, age, registered_bytes, len(held)
            )
        except Exception as error:
            logging.error("MM_EMB_GDR_GC_METRIC_ERROR %s", _safe_error_context(error))

    def _gc_loop(self):
        gc_interval = min(10, self._quarantine_ttl_seconds)
        elapsed = 0
        while not self._gc_stop.wait(1):
            if self._observe_arena is not None:
                self._report_arena()
            elapsed += 1
            if elapsed == gc_interval:
                self._gc_tick()
                elapsed = 0

    def _report_arena(self):
        with self._condition:
            if self._arena is None:
                return
            total = self._arena.numel() - self._GUARD_BYTES
            if self._free_ranges is not None:
                sizes = [size for _, size in self._free_ranges]
            else:
                sizes = [
                    slot._capacity
                    for slot in self._slots
                    if slot not in self._in_use and not slot.quarantined
                ]
        self._observe_arena(total, sum(sizes), max(sizes, default=0))

    def _gc_tick(self):
        with self._condition:
            recovered = [
                slot._quarantine_operation
                for slot in self._slots
                if slot not in self._in_use
                and slot.quarantined
                and slot.reclaim_quarantine(self._quarantine_ttl_seconds)
            ]
            if recovered:
                self._condition.notify_all()
            if self._closed:
                # Closed slots share one MR: count every expired slot only
                # after native teardown succeeds, even across several ticks.
                self._pending_gc_success.extend(recovered)
                recovered = []
                if self._close_registration_if_safe():
                    recovered, self._pending_gc_success = self._pending_gc_success, []
        for operation in recovered:
            self._report_hold(operation, "gc_success")
        self._report_hold("unknown", "held")

    @property
    def quarantined(self):
        return any(slot.device_failed is True for slot in self._slots) or (
            not self._free_ranges and all(slot.quarantined for slot in self._slots)
        )

    def object_size(self, key, **kwargs):
        if self._free_ranges is not None:
            client = self._client
            return None if client is None else client.object_size(key, **kwargs)
        for slot in self._slots:
            if not slot.quarantined:
                return slot.object_size(key, **kwargs)
        return None

    def get_object_meta(self, key, **kwargs):
        client = self._client
        if client is None:
            return None
        return client.get_object_meta(key, **kwargs)

    def _round_up(self, nbytes):
        block = self._min_block_bytes
        return (nbytes + block - 1) // block * block

    def _take_range(self, nbytes, for_write):
        capacity = self._round_up(nbytes)
        for index, (offset, free_bytes) in enumerate(self._free_ranges):
            end = offset + capacity
            if capacity > free_bytes or (for_write and end > self._write_end):
                continue
            slot = RegisteredObjectTransport(
                self,
                None,
                self._device,
                self._observe_stage,
                lazy_stream=True,
                extent=(offset, capacity),
            )
            if self._idle_streams:
                slot._stream = self._idle_streams.pop()
            self._slots.append(slot)
            if capacity == free_bytes:
                self._free_ranges.pop(index)
            else:
                self._free_ranges[index] = (end, free_bytes - capacity)
            return slot
        return None

    def _return_range(self, slot):
        offset, capacity = slot._extent
        ranges = self._free_ranges
        index = bisect_left(ranges, (offset,))
        if index > 0:
            start, size = ranges[index - 1]
            if start + size == offset:
                index -= 1
                ranges.pop(index)
                offset, capacity = start, size + capacity
        if index < len(ranges):
            start, size = ranges[index]
            if offset + capacity == start:
                ranges.pop(index)
                capacity += size
        ranges.insert(index, (offset, capacity))
        self._slots.remove(slot)
        if slot._stream is not None:
            self._idle_streams.append(slot._stream)
        slot._buffer = slot._stream = None

    def acquire(self, *, nbytes=None, for_write=False, timeout=0):
        """Lease a transfer buffer, optionally waiting up to timeout seconds."""
        sized = self._free_ranges is not None
        if sized and (
            type(nbytes) is not int or not 0 < nbytes <= self._max_object_bytes
        ):
            raise ValueError("invalid GDR object size")
        wait_seconds = max(timeout, 0)
        deadline = time.monotonic() + wait_seconds
        # Report about one in 64 acquired locks, outside the lock.
        sample = (
            self._observe_lock_hold is not None and self._lock_hold_rng.getrandbits(6) == 0
        )
        if not self._condition.acquire(timeout=wait_seconds):
            self._report_acquire_failure(for_write, "lock_timeout")
            return None
        held_ns = 0
        held_since = time.perf_counter_ns() if sample else 0
        slot = None
        failure = "extent_unavailable"
        try:
            while True:
                if not self._accepting:
                    failure = "closed"
                    break
                if self.quarantined:
                    failure = "quarantined"
                    break
                # Condition.wait() reacquires the lock before returning.
                if wait_seconds and time.monotonic() >= deadline:
                    failure = "deadline_with_lock"
                    break
                if sized:
                    slot = self._take_range(nbytes, for_write)
                else:
                    healthy = [slot for slot in self._slots if not slot.quarantined]
                    # A multi-slot pool reserves one healthy slot for read leases.
                    writable = len(healthy)
                    if for_write and len(self._slots) > 1:
                        if len(healthy) <= 1:
                            failure = "read_reserve"
                            break
                        writable -= 1 + sum(
                            self._in_use.get(slot, False) for slot in healthy
                        )
                    if writable > 0:
                        slot = next(
                            (item for item in healthy if item not in self._in_use),
                            None,
                        )
                if slot is not None:
                    self._in_use[slot] = for_write
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if sample:
                    held_ns += time.perf_counter_ns() - held_since
                notified = self._condition.wait(remaining)
                if sample:
                    held_since = time.perf_counter_ns()
                if not notified:
                    break
        finally:
            if sample:
                held_ns += time.perf_counter_ns() - held_since
            self._condition.release()
        if sample:
            self._observe_lock_hold(for_write, "gdr_acquire_lock_hold", held_ns)
        if slot is None:
            self._report_acquire_failure(for_write, failure)
            return None
        if self._observe_lease is not None:
            self._observe_lease(for_write, 1, slot._capacity)
        if sized:
            # The in-use entry keeps the arena and this extent alive until
            # release. PyTorch dispatcher work need not hold the pool lock.
            offset, capacity = slot._extent
            try:
                slot._buffer = self._arena.narrow(0, offset, capacity)
            except BaseException:
                self.release(slot)
                raise
        return slot

    def _report_acquire_failure(self, for_write, reason):
        if self._observe_acquire_failure is not None:
            self._observe_acquire_failure(for_write, reason)

    def stop_acquiring(self):
        with self._condition:
            self._accepting = False
            self._condition.notify_all()

    def release(self, slot):
        sample = (
            self._observe_lock_hold is not None and self._lock_hold_rng.getrandbits(6) == 0
        )
        with self._condition:
            held_since = time.perf_counter_ns() if sample else 0
            for_write = self._in_use.pop(slot)
            if self._free_ranges is not None and not slot.quarantined:
                self._return_range(slot)
            self._condition.notify_all()
            held_ns = time.perf_counter_ns() - held_since if sample else 0
        if sample:
            self._observe_lock_hold(for_write, "gdr_release_lock_hold", held_ns)
        if self._observe_lease is not None:
            self._observe_lease(for_write, -1, -slot._capacity)

    def close(self):
        self.stop_acquiring()
        with self._condition:
            self._condition.wait_for(lambda: not self._in_use)
            if not self._closed:
                self._closed = True
                self._close_registration_if_safe()

    def _close_registration_if_safe(self):
        # Called under _condition after all leases drain, including late IO.
        if self._client is None:
            return True
        if any(slot.quarantined for slot in self._slots):
            return False
        try:
            self._client.close()
        except Exception as error:
            # A subsequent SDK close can be a no-op even if deregistration
            # failed. Retain the entire allocation and its allocator until exit.
            logging.error(
                "MM_EMB_GDR_GC_ERROR closing registration: %s",
                _safe_error_context(error),
            )
            self._quarantine_registration(error, "close")
            self._report_hold("unknown", "gc_error")
        else:
            self._client = None
            for slot in self._slots:
                slot._buffer = None
                slot._host_staging = None
                slot._stream = None
            self._idle_streams.clear()
            self._arena = None
            self._allocator = None
        self._gc_stop.set()
        return self._client is None
