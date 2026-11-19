"""Bounded registered CUDA transport for the existing CPU object codec.

The wire payload uses PACE GDR. Serialization and SHA256 validation still run
on CPU; this is not a claim that the tensor-tree codec is GPU-only.
"""

import ctypes
import json
import logging
import re
import threading
import time
import uuid
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

    def __init__(self, owner, buffer, device, observe_stage):
        self._owner = owner
        self._buffer = buffer
        self._capacity = buffer.numel()
        self._device = device
        self._observe_stage = observe_stage
        self._lock = threading.Lock()
        self._poisoned = False
        self.device_failed = False
        self._host_staging = torch.empty(
            self._capacity, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self._stream = torch.cuda.Stream(device=device)

    @property
    def _client(self):
        return self._owner._client

    @property
    def quarantined(self):
        return self._poisoned

    def _check_available(self):
        if self._owner._closed or self._poisoned:
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
            if (
                not self._poisoned
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
            stack.enter_context(torch.cuda.stream(self._stream))
            yield

    def object_size(self, key, **kwargs):
        # Metadata lookup uses its own client lock and never touches the arena.
        self._check_available()
        return self._client.object_size(key, **kwargs)

    def pack(self, plan):
        from .tensor_object import pack_object

        try:
            return pack_object(
                plan, out=self._host_staging, observe_stage=self._observe_stage
            )
        except Exception as error:
            self._handle_failure(error, operation="write", stage="pack")
            raise

    def save_one(self, key, buffer, **kwargs):
        with self._serialized("write"):
            transfer = self._view(buffer)
            failure_stage = "host_copy"
            try:
                host = self._host_staging[: buffer.numel()]
                with self._stage("write", "host_copy"):
                    if host.data_ptr() != buffer.data_ptr():
                        # These are contiguous CPU byte buffers. Use a single
                        # native copy so parallel slots do not each fan out to
                        # the PyTorch CPU intra-op workers.
                        ctypes.memmove(
                            host.data_ptr(), buffer.data_ptr(), buffer.numel()
                        )
                failure_stage = "gpu_copy"
                stream = self._stream
                with self._stage("write", "h2d_staging"):
                    transfer.copy_(host, non_blocking=True)
                    stream.synchronize()
                # Includes the native client's metadata transaction, not just IO.
                failure_stage = "sdk_transfer"
                with self._stage("write", "native_save"):
                    self._client.save_one(key, transfer, **kwargs)
                failure_stage = "gpu_ready_wait"
                with self._stage("write", "gpu_ready_wait"):
                    stream.synchronize()
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
    ):
        if type(slots) is not int or slots <= 0:
            raise ValueError("slots must be a positive integer")
        if type(quarantine_ttl_seconds) is not int or quarantine_ttl_seconds <= 0:
            raise ValueError("quarantine_ttl_seconds must be a positive integer")
        self._quarantine_ttl_seconds = quarantine_ttl_seconds
        self._observe_hold = observe_hold
        self._gc_stop = threading.Event()
        self._condition = threading.Condition()
        self._accepting = True
        self._closed = False
        self._in_use = {}  # slot -> whether the lease is for a write
        self._pending_gc_success = []
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
                        self._GUARD_BYTES + slots * capacity,
                        dtype=torch.uint8,
                        device=device,
                    )
                _enable_sync_memops(self._arena)
                self._slots = [
                    RegisteredObjectTransport(
                        self,
                        self._arena.narrow(
                            0, self._GUARD_BYTES + i * capacity, capacity
                        ),
                        device,
                        observe_stage,
                    )
                    for i in range(slots)
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
        while not self._gc_stop.wait(min(10, self._quarantine_ttl_seconds)):
            self._gc_tick()

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
        return any(slot.device_failed is True for slot in self._slots) or all(
            slot.quarantined for slot in self._slots
        )

    def object_size(self, key, **kwargs):
        for slot in self._slots:
            if not slot.quarantined:
                return slot.object_size(key, **kwargs)
        return None

    def get_object_meta(self, key, **kwargs):
        client = self._client
        if client is None:
            return None
        return client.get_object_meta(key, **kwargs)

    def acquire(self, *, for_write=False, timeout=0):
        """Lease an available slot, optionally waiting up to timeout seconds."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._accepting and not self.quarantined:
                healthy = [slot for slot in self._slots if not slot.quarantined]
                # A multi-slot pool reserves one healthy slot for read leases.
                writable = len(healthy)
                if for_write and len(self._slots) > 1:
                    if len(healthy) <= 1:
                        return None
                    writable -= 1 + sum(
                        self._in_use.get(slot, False) for slot in healthy
                    )
                if writable > 0:
                    for slot in healthy:
                        if slot not in self._in_use:
                            self._in_use[slot] = for_write
                            return slot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if not self._condition.wait(remaining):
                    break
        return None

    def stop_acquiring(self):
        with self._condition:
            self._accepting = False
            self._condition.notify_all()

    def release(self, slot):
        with self._condition:
            self._in_use.pop(slot)
            self._condition.notify_all()

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
            self._arena = None
            self._allocator = None
        self._gc_stop.set()
        return self._client is None
