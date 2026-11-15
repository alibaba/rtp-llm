"""Bounded registered CUDA transport for the existing CPU object codec.

The wire payload uses PACE GDR. Serialization and SHA256 validation still run
on CPU; this is not a claim that the tensor-tree codec is GPU-only.
"""

import ctypes
import json
import logging
import re
import threading
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
    _GUARD_BYTES = 65536
    _quarantine_lock = threading.Lock()
    _quarantined_transports = []

    def __init__(self, factory, *, max_object_bytes, device, observe_stage=None):
        if max_object_bytes <= 0 or max_object_bytes > 1024**3:
            raise ValueError("invalid registered object capacity")
        self._observe_stage = observe_stage
        self._lock = threading.Lock()
        self._closed = False
        self._poisoned = False
        self._capacity = max_object_bytes
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError("GDR transport requires a CUDA device")
        # A dedicated allocation pool isolates registered storage from the
        # model allocator. Reserve a nonzero MR-relative offset: PACE's normal
        # single-IOV interface treats a zero IOVA as an invalid address.
        # A failed transfer/registration may leave DMA outstanding. Refuse new
        # arenas after failure so retries cannot accumulate quarantined memory.
        with self._quarantine_lock:
            if self._quarantined_transports:
                raise RegisteredTransportUnavailableError(
                    "GDR transport is quarantined; restart the worker",
                    quarantined=True,
                    initial_failure=getattr(
                        self._quarantined_transports[0], "_quarantine_cause", {}
                    ),
                    quarantine_id=getattr(
                        self._quarantined_transports[0], "_quarantine_id", None
                    ),
                )
            with cuda_graph_gate.operation(), torch.cuda.device(self._device):
                # Shared by reads and writes under _lock. Keep pageable codec
                # buffers out of CUDA copies to the registered GPU allocation.
                self._host_staging = torch.empty(
                    max_object_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
                )
                try:
                    self._pool = torch.cuda.MemPool()
                    with torch.cuda.use_mem_pool(self._pool):
                        self._arena = torch.empty(
                            self._GUARD_BYTES + max_object_bytes,
                            dtype=torch.uint8,
                            device=self._device,
                        )
                    _enable_sync_memops(self._arena)
                    self._client = factory(gpu_tensors=(self._arena,))
                except Exception as error:
                    self._poisoned = True
                    self._quarantined_transports.append(self)
                    self._record_quarantine(error, stage="registration")
                    raise
        self.instance_id = self._client.instance_id
        self.instance_group = self._client.instance_group

    def _check_available(self):
        if self._closed or self._poisoned:
            raise RegisteredTransportUnavailableError(
                "registered transport unavailable; restart worker after GDR failure",
                quarantined=self._poisoned,
                initial_failure=getattr(self, "_quarantine_cause", {}),
                quarantine_id=getattr(self, "_quarantine_id", None),
            )

    def _record_quarantine(self, error, **context):
        self._quarantine_cause = _safe_error_context(error)
        self._quarantine_cause.setdefault("data_transfer_may_be_inflight", True)
        self._quarantine_id = uuid.uuid4().hex
        try:
            logging.error(
                "MM_EMB_GDR_QUARANTINE %s",
                json.dumps(
                    dict(
                        event="gdr_transport_quarantined",
                        reason="gdr_transport_quarantined",
                        action="restart_worker",
                        message="GDR completion is not proven safe; arena retained; restart worker",
                        quarantine_id=self._quarantine_id,
                        **self._quarantine_cause,
                        **context,
                    ),
                    sort_keys=True,
                ),
            )
        except Exception:
            pass  # Diagnostics must never mask the original failure.

    def _quarantine(self, error, **context):
        if not self._poisoned:
            self._poisoned = True
            with self._quarantine_lock:
                self._quarantined_transports.append(self)
            self._record_quarantine(error, **context)

    def is_transfer_quiescent(self, error):
        if self._closed or self._poisoned:
            return False
        check = getattr(self._client, "is_transfer_quiescent", None)
        try:
            return callable(check) and check(error) is True
        except Exception:
            return False

    def _handle_failure(self, error, **context):
        # Legacy SDKs and unclassified CUDA/native exceptions remain unsafe.
        if not self.is_transfer_quiescent(error):
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
        return self._arena[self._GUARD_BYTES : self._GUARD_BYTES + tensor.numel()]

    def _stage(self, operation, name):
        observe = getattr(self, "_observe_stage", None)
        return observe(operation, name) if observe is not None else nullcontext()

    @contextmanager
    def _serialized(self, operation):
        # Time the existing waits without adding a CUDA synchronization.
        with ExitStack() as stack:
            with self._stage(operation, "transport_lock_wait"):
                stack.enter_context(self._lock)
            with self._stage(operation, "cuda_gate_wait"):
                stack.enter_context(cuda_graph_gate.operation())
            stack.enter_context(torch.cuda.device(self._device))
            yield

    def object_size(self, key, **kwargs):
        with ExitStack() as stack:
            with self._stage("read", "metadata_lock_wait"):
                stack.enter_context(self._lock)
            self._check_available()
            return self._client.object_size(key, **kwargs)

    def save_one(self, key, buffer, **kwargs):
        with self._serialized("write"):
            transfer = self._view(buffer)
            failure_stage = "host_copy"
            try:
                host = self._host_staging[: buffer.numel()]
                with self._stage("write", "host_copy"):
                    host.copy_(buffer)
                failure_stage = "gpu_copy"
                stream = torch.cuda.current_stream(self._device)
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
        with self._serialized("read"):
            transfer = self._view(buffer)
            failure_stage = "sdk_transfer"
            try:
                # Includes native metadata lookup; PACE's own RT is separate.
                with self._stage("read", "native_load"):
                    self._client.load_one(key, transfer, **kwargs)
                failure_stage = "gpu_ready_wait"
                stream = torch.cuda.current_stream(self._device)
                with self._stage("read", "gpu_ready_wait"):
                    stream.synchronize()
                failure_stage = "gpu_copy"
                host = self._host_staging[: buffer.numel()]
                with self._stage("read", "d2h_staging"):
                    host.copy_(transfer, non_blocking=True)
                    stream.synchronize()
                failure_stage = "host_copy"
                with self._stage("read", "host_copy"):
                    buffer.copy_(host)
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

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._poisoned:
                # A timeout is not proof that peer DMA stopped. Keep the native
                # client, registration, arena AND allocator pool alive together.
                # This transport cannot be reused or recreated in this worker.
                self._closed = True
                return
            try:
                self._client.close()
            except Exception as error:
                self._quarantine(error, stage="close")
                raise
            self._closed = True
            self._host_staging = None
            self._arena = None
            self._pool = None
