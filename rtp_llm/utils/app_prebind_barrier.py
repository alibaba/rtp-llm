"""Application pre-bind barrier used by the optional sCR lifecycle.

The barrier deliberately has no dependency on ``epsilon`` or
``scr_controller``.  A process reports that its static initialization is
complete before creating a business listener, then waits for an external
coordinator to release the generation.  With the feature disabled (the
default) the client is a no-op, preserving the normal RTP-LLM startup path.

The production transport is a small JSON-lines Unix socket protocol.  Tests
can inject :class:`BarrierTransport` implementations without opening sockets
or talking to a scheduler.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol


SCR_APP_BARRIER_ENABLE_ENV = "RTPLLM_ENABLE_SCR"
SCR_APP_BARRIER_ENABLE_ALIASES = (SCR_APP_BARRIER_ENABLE_ENV, "RTP_LLM_ENABLE_SCR")
SCR_APP_BARRIER_ENABLE_VALUES = {"1", "true", "yes", "on"}
APP_BARRIER_UDS_ENV = "RTPLLM_APP_PREBIND_BARRIER_UDS"
APP_BARRIER_GENERATION_ENV = "RTPLLM_APP_PREBIND_GENERATION"
APP_BARRIER_TIMEOUT_ENV = "RTPLLM_APP_PREBIND_TIMEOUT_SECONDS"
DEFAULT_APP_BARRIER_TIMEOUT_SECONDS = 600.0


class AppPreBindPhase(str, Enum):
    """Lifecycle phases understood by the coordinator."""

    PREBIND_READY = "PREBIND_READY"
    RESTORE_FIXUP_READY = "RESTORE_FIXUP_READY"
    FINAL_RELEASE = "FINAL_RELEASE"


class BarrierTransport(Protocol):
    """Minimal request transport, intentionally independent of scheduler."""

    def request(self, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class BarrierResult:
    """Normalized coordinator response."""

    status: str
    generation: str = ""
    phase: str = ""
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def released(self) -> bool:
        return self.status.lower() in {"release", "released", "ok", "ready"}

    @property
    def aborted(self) -> bool:
        return self.status.lower() in {"abort", "aborted", "failed", "error"}


class UnixSocketBarrierTransport:
    """JSON-lines Unix socket transport for an external barrier coordinator."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path

    def request(self, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
        wire = (json.dumps(dict(payload), separators=(",", ":")) + "\n").encode()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(max(0.001, float(timeout)))
            sock.connect(self.socket_path)
            sock.sendall(wire)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
        if not chunks:
            raise ConnectionError("app pre-bind barrier closed the connection")
        line = b"".join(chunks).splitlines()[0]
        response = json.loads(line.decode())
        if not isinstance(response, Mapping):
            raise ValueError("barrier response must be a JSON object")
        return response


def _timeout_from_env() -> float:
    raw = os.environ.get(APP_BARRIER_TIMEOUT_ENV, "")
    if not raw:
        return DEFAULT_APP_BARRIER_TIMEOUT_SECONDS
    try:
        return max(0.001, float(raw))
    except ValueError:
        logging.warning("Invalid %s=%r; using %.1fs", APP_BARRIER_TIMEOUT_ENV, raw,
                        DEFAULT_APP_BARRIER_TIMEOUT_SECONDS)
        return DEFAULT_APP_BARRIER_TIMEOUT_SECONDS


def _response(value: Mapping[str, Any], *, generation: str, phase: str) -> BarrierResult:
    metadata = value.get("metadata", {})
    status = value.get("status", "")
    if not status:
        if value.get("ok") is True:
            status = "released"
        elif value.get("error"):
            status = "error"
    return BarrierResult(
        status=str(status),
        generation=str(value.get("generation", generation)),
        phase=str(value.get("phase", phase)),
        reason=str(value.get("reason", value.get("message", ""))),
        metadata=metadata if isinstance(metadata, Mapping) else {},
    )


class AppPreBindBarrierClient:
    """Participant-side client for one barrier generation.

    ``transport`` is injectable for unit tests.  Production callers normally
    use :meth:`from_env`; it returns ``None`` unless both the SCR gate and an
    explicit barrier UDS are configured.  Transport errors are fail-open to
    the ordinary cold-start path, but an ``ABORT`` response is surfaced as a
    failed result so the coordinator can reject the generation.
    """

    def __init__(
        self,
        role: str,
        instance: str | int,
        *,
        generation: str = "default",
        transport: Optional[BarrierTransport] = None,
        timeout: float = DEFAULT_APP_BARRIER_TIMEOUT_SECONDS,
        enabled: bool = True,
    ) -> None:
        self.role = str(role)
        self.instance = str(instance)
        self.generation = str(generation)
        self.transport = transport
        self.timeout = max(0.001, float(timeout))
        self.enabled = bool(enabled and transport is not None)
        self._lock = threading.Lock()
        self._arrived: set[str] = set()
        self._released: set[str] = set()
        self._aborted = False

    @classmethod
    def from_env(cls, role: str, instance: str | int) -> Optional["AppPreBindBarrierClient"]:
        if not any(
            os.environ.get(key, "").strip().lower() in SCR_APP_BARRIER_ENABLE_VALUES
            for key in SCR_APP_BARRIER_ENABLE_ALIASES
        ):
            return None
        path = os.environ.get(APP_BARRIER_UDS_ENV, "").strip()
        if not path:
            logging.warning("%s=1 but %s is unset; using normal startup",
                            SCR_APP_BARRIER_ENABLE_ENV, APP_BARRIER_UDS_ENV)
            return None
        generation = os.environ.get(APP_BARRIER_GENERATION_ENV, "default")
        return cls(role, instance, generation=generation,
                   transport=UnixSocketBarrierTransport(path), timeout=_timeout_from_env())

    def _call(self, operation: str, phase: AppPreBindPhase, metadata: Optional[Mapping[str, Any]],
              timeout: Optional[float]) -> BarrierResult:
        if not self.enabled or self.transport is None:
            return BarrierResult("released", self.generation, phase.value)
        payload = {
            "op": operation,
            "role": self.role,
            "instance": self.instance,
            "generation": self.generation,
            "phase": phase.value,
            "metadata": dict(metadata or {}),
        }
        request_timeout = self.timeout if timeout is None else max(0.001, timeout)
        value = self.transport.request(payload, request_timeout)
        return _response(value, generation=self.generation, phase=phase.value)

    def arrive(self, phase: AppPreBindPhase = AppPreBindPhase.PREBIND_READY,
               metadata: Optional[Mapping[str, Any]] = None,
               timeout: Optional[float] = None) -> BarrierResult:
        """Report readiness.  Repeated calls for a phase are idempotent."""
        with self._lock:
            if self._aborted:
                return BarrierResult("aborted", self.generation, phase.value, "client aborted")
            if phase.value in self._arrived:
                return BarrierResult("accepted", self.generation, phase.value)
        try:
            result = self._call("arrive", phase, metadata, timeout)
        except Exception as exc:
            logging.warning("App pre-bind barrier arrive failed role=%s instance=%s: %s",
                            self.role, self.instance, exc)
            return BarrierResult("error", self.generation, phase.value, str(exc))
        if not result.aborted:
            with self._lock:
                self._arrived.add(phase.value)
        return result

    def wait_release(self, phase: AppPreBindPhase = AppPreBindPhase.PREBIND_READY,
                     timeout: Optional[float] = None) -> BarrierResult:
        """Wait for coordinator release; disconnects are treated as failure."""
        with self._lock:
            if phase.value in self._released:
                return BarrierResult("released", self.generation, phase.value)
            if self._aborted:
                return BarrierResult("aborted", self.generation, phase.value, "client aborted")
        try:
            result = self._call("wait_release", phase, None, timeout)
        except Exception as exc:
            logging.warning("App pre-bind barrier wait failed role=%s instance=%s: %s",
                            self.role, self.instance, exc)
            return BarrierResult("error", self.generation, phase.value, str(exc))
        if result.released:
            with self._lock:
                self._released.add(phase.value)
        return result

    def prebind_ready(self, metadata: Optional[Mapping[str, Any]] = None,
                      timeout: Optional[float] = None,
                      *, wait_for_release: bool = True) -> bool:
        """Arrive at ``PREBIND_READY`` and optionally wait for release.

        Returns ``True`` when disabled or explicitly released.  ``False`` lets
        callers choose the normal cold-start fallback on coordinator failure.

        GPU ranks use ``wait_for_release=False``: they must publish the same
        application-level arrival as CPU children, but their blocking
        steady-point operation is Epsilon's ``snapstart_checkpoint`` waiter.
        Waiting here as well would couple the two barriers and can deadlock the
        controller-triggered dump path.
        """
        if not self.enabled:
            return True
        arrived = self.arrive(AppPreBindPhase.PREBIND_READY, metadata, timeout)
        if arrived.aborted or arrived.status.lower() in {"error", "failed"}:
            self.abort("arrive failed: " + arrived.reason)
            return False
        if not wait_for_release:
            return True
        if arrived.released:
            with self._lock:
                self._released.add(AppPreBindPhase.PREBIND_READY.value)
            return True
        released = self.wait_release(AppPreBindPhase.PREBIND_READY, timeout)
        if not released.released:
            self.abort("release failed: " + released.reason)
            return False
        return True

    def restore_fixup_ready(self, metadata: Optional[Mapping[str, Any]] = None,
                            timeout: Optional[float] = None) -> bool:
        result = self.arrive(AppPreBindPhase.RESTORE_FIXUP_READY, metadata, timeout)
        return not result.aborted and result.status.lower() not in {"error", "failed"}

    def final_release(self, timeout: Optional[float] = None) -> bool:
        result = self.wait_release(AppPreBindPhase.FINAL_RELEASE, timeout)
        return result.released

    def abort(self, reason: str = "") -> None:
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
        if not self.enabled or self.transport is None:
            return
        try:
            self._call("abort", AppPreBindPhase.PREBIND_READY, {"reason": reason}, self.timeout)
        except Exception:
            logging.debug("App pre-bind barrier abort failed", exc_info=True)

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if close is not None:
            close()


def get_app_prebind_barrier_client(
    role: str, instance: str | int
) -> Optional[AppPreBindBarrierClient]:
    """Return an environment-configured client, or ``None`` when SCR is off."""
    return AppPreBindBarrierClient.from_env(role, instance)


__all__ = [
    "AppPreBindBarrierClient",
    "AppPreBindPhase",
    "BarrierResult",
    "BarrierTransport",
    "UnixSocketBarrierTransport",
    "get_app_prebind_barrier_client",
]
