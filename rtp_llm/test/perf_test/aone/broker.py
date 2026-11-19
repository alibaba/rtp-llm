"""Gang rendezvous broker for multi-machine AOne execution.

The broker runs on the submit host and lets N symmetric gang members
discover each other through a small authenticated HTTP API. Workers do not
self-discover their routable address: the broker records each
registration's HTTP source address and keeps the claimed hostname alongside
a best-effort reverse lookup for diagnostics. The broker serves only bounded run
metadata — no commands, no source files, no filesystem paths beyond the
state directory — and never persists the bearer token.
"""

from __future__ import annotations

import json
import logging
import secrets
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlsplit

from rtp_llm.test.perf_test.storage import atomic_write_json

_LOG = logging.getLogger(__name__)
_LOG.addHandler(logging.NullHandler())

REGISTER_DEADLINE_S = 3600.0
RUN_DEADLINE_S = 43200.0
HEARTBEAT_EXPIRY_S = 120.0

STATE_REGISTERING = "registering"
STATE_READY = "ready"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"

_STATE_FILE_NAME = "broker_state.json"
_EVENTS_FILE_NAME = "broker_events.jsonl"
_MAX_BODY_BYTES = 65536
_MAX_DETAIL_CHARS = 2000
_BARRIER_POLL_S = 0.5

Resolver = Callable[[str], Optional[str]]


@dataclass(frozen=True)
class BrokerMember:
    """One registered gang member as the broker sees it."""

    shard_index: int
    address: str
    hostname: str
    port: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "address": self.address,
            "hostname": self.hostname,
            "port": self.port,
        }


class _BrokerError(Exception):
    """A request the broker rejects with a stable status and error code."""

    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status = status
        self.code = code


def _default_resolver(address: str) -> Optional[str]:
    """Reverse-resolve one address to its canonical hostname, if possible."""
    try:
        return socket.gethostbyaddr(address)[0]
    except OSError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hostnames_equal(left: str, right: str) -> bool:
    return left.rstrip(".").lower() == right.rstrip(".").lower()


class GangBroker:
    """Rendezvous state machine plus loopback-safe HTTP transport."""

    def __init__(
        self,
        machine_count: int,
        *,
        state_dir: Path,
        host: str = "0.0.0.0",
        register_deadline_s: float = REGISTER_DEADLINE_S,
        run_deadline_s: float = RUN_DEADLINE_S,
        heartbeat_expiry_s: float = HEARTBEAT_EXPIRY_S,
        expected_config_fingerprint: Optional[str] = None,
        resolver: Optional[Resolver] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if type(machine_count) is not int or machine_count < 1:
            raise ValueError("machine_count must be a positive integer")
        for name, value in (
            ("register_deadline_s", register_deadline_s),
            ("run_deadline_s", run_deadline_s),
            ("heartbeat_expiry_s", heartbeat_expiry_s),
        ):
            if not value > 0:
                raise ValueError(f"{name} must be a positive number of seconds")
        if expected_config_fingerprint is not None and (
            not isinstance(expected_config_fingerprint, str)
            or not expected_config_fingerprint
        ):
            raise ValueError(
                "expected_config_fingerprint must be None or a non-empty string"
            )
        self._machine_count = machine_count
        self._state_dir = Path(state_dir)
        self._host = host
        self._register_deadline_s = register_deadline_s
        self._run_deadline_s = run_deadline_s
        self._heartbeat_expiry_s = heartbeat_expiry_s
        self._expected_config_fingerprint = expected_config_fingerprint
        self._resolver = _default_resolver if resolver is None else resolver
        self._clock = clock
        self._run_id = uuid.uuid4().hex
        self._token = secrets.token_urlsafe(32)
        self._condition = threading.Condition()
        self._members: Dict[int, BrokerMember] = {}
        self._last_heartbeat: Dict[int, float] = {}
        self._results: Dict[int, Dict[str, Any]] = {}
        self._lost_shards: set = set()
        self._gang_state = STATE_REGISTERING
        self._failure_reason: Optional[str] = None
        self._cleaned = False
        self._started = False
        self._stopped = False
        self._start_monotonic = clock()
        self._started_at = _utc_now()
        self._state_path = self._state_dir / _STATE_FILE_NAME
        self._events_path = self._state_dir / _EVENTS_FILE_NAME
        self._server: Optional[_BrokerServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # Lifecycle -----------------------------------------------------------

    def start(self) -> str:
        """Bind the HTTP server, start helper threads, and return the URL."""
        if self._stopped:
            raise RuntimeError("broker has been stopped")
        if self._started:
            return self.url
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._start_monotonic = self._clock()
        self._started_at = _utc_now()
        self._server = _BrokerServer((self._host, 0), self)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"gang-broker-{self._run_id}",
            daemon=True,
        )
        self._server_thread.start()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"gang-broker-monitor-{self._run_id}",
            daemon=True,
        )
        self._monitor_thread.start()
        self._started = True
        with self._condition:
            self._append_event_locked(
                "broker_started",
                machine_count=self._machine_count,
                host=self._host,
                port=self._server.server_address[1],
            )
            self._persist_state_locked()
        return self.url

    def stop(self) -> None:
        """Stop serving and wake every blocked rendezvous call."""
        if not self._started or self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        server = self._server
        if server is not None:
            server.shutdown()
            server.server_close()
        for thread in (self._server_thread, self._monitor_thread):
            if thread is not None:
                thread.join(timeout=5.0)
        with self._condition:
            self._append_event_locked("broker_stopped")
            self._persist_state_locked()
            self._condition.notify_all()

    def __enter__(self) -> "GangBroker":
        self.start()
        return self

    def __exit__(self, *_exc_info: Any) -> bool:
        self.stop()
        return False

    # Public state --------------------------------------------------------

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("broker is not started")
        host, port = self._server.server_address[:2]
        advertised = "127.0.0.1" if host in ("", "0.0.0.0") else host
        return f"http://{advertised}:{port}"

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def token(self) -> str:
        return self._token

    @property
    def members(self) -> Tuple[BrokerMember, ...]:
        with self._condition:
            return tuple(sorted(self._members.values(), key=lambda m: m.shard_index))

    @property
    def gang_state(self) -> str:
        with self._condition:
            return self._gang_state

    @property
    def failure_reason(self) -> Optional[str]:
        with self._condition:
            return self._failure_reason

    # Rendezvous operations -------------------------------------------------

    def register_member(
        self,
        *,
        shard_index: Any,
        hostname: Any,
        port: Any,
        address: str,
        config_fingerprint: Any = None,
    ) -> Dict[str, Any]:
        """Validate and record one member registration."""
        if type(shard_index) is not int or not 0 <= shard_index < self._machine_count:
            raise _BrokerError(400, "invalid_registration")
        if not isinstance(hostname, str) or not hostname.strip():
            raise _BrokerError(400, "invalid_registration")
        if type(port) is not int or not 0 < port < 65536:
            raise _BrokerError(400, "invalid_registration")
        if self._expected_config_fingerprint is not None:
            if config_fingerprint != self._expected_config_fingerprint:
                raise _BrokerError(400, "config_fingerprint_mismatch")
        claimed = hostname.strip()
        # Best-effort reverse lookup: in containerized worker pools the
        # claimed hostname is the container's own name while reverse DNS
        # returns the machine-level FQDN, so the two legitimately differ.
        # Record both for diagnostics; uniqueness checks below are the
        # same-host gate, not hostname equality.
        resolved = self._resolve_hostname(address)
        with self._condition:
            if self._cleaned:
                raise _BrokerError(409, "run_closed")
            if self._gang_state == STATE_FAILED:
                raise _BrokerError(409, "gang_failed")
            if self._gang_state in (STATE_READY, STATE_COMPLETED):
                raise _BrokerError(409, "gang_complete")
            if shard_index in self._members:
                raise _BrokerError(409, "duplicate_shard")
            for existing in self._members.values():
                if existing.address == address or _hostnames_equal(
                    existing.hostname, claimed
                ):
                    self._fail_locked("duplicate_host")
                    raise _BrokerError(409, "duplicate_host")
            member = BrokerMember(
                shard_index=shard_index, address=address, hostname=claimed, port=port
            )
            self._members[shard_index] = member
            self._last_heartbeat[shard_index] = self._clock()
            self._append_event_locked(
                "member_registered",
                shard_index=shard_index,
                address=address,
                hostname=claimed,
                resolved_hostname=resolved,
                port=port,
            )
            if len(self._members) >= self._machine_count:
                self._gang_state = STATE_READY
                self._append_event_locked("gang_ready")
            self._persist_state_locked()
            self._condition.notify_all()
            return {
                "status": "registered",
                "shard_index": shard_index,
                "registered": len(self._members),
                "machine_count": self._machine_count,
                "gang_state": self._gang_state,
            }

    def await_members(self) -> Tuple[int, Dict[str, Any]]:
        """Block until the gang is complete, the register deadline passes,
        or the run ends, and return the HTTP status and body."""
        with self._condition:
            while True:
                if self._cleaned:
                    return 409, {"error": "run_closed"}
                if self._gang_state in (STATE_READY, STATE_COMPLETED):
                    return 200, {
                        "run_id": self._run_id,
                        "gang_state": self._gang_state,
                        "machine_count": self._machine_count,
                        "members": [member.to_dict() for member in self.members],
                    }
                if self._gang_state == STATE_FAILED:
                    return 504, {"error": self._failure_reason}
                if self._stopped:
                    return 503, {"error": "broker_stopped"}
                remaining = (
                    self._start_monotonic + self._register_deadline_s
                ) - self._clock()
                if remaining <= 0:
                    self._fail_locked("register_deadline_exceeded")
                    return 504, {"error": "register_deadline_exceeded"}
                self._condition.wait(min(remaining, _BARRIER_POLL_S))

    def record_heartbeat(self, *, shard_index: Any) -> Dict[str, Any]:
        """Refresh one member's heartbeat."""
        if type(shard_index) is not int:
            raise _BrokerError(400, "invalid_heartbeat")
        with self._condition:
            if self._cleaned:
                raise _BrokerError(409, "run_closed")
            if self._gang_state == STATE_FAILED:
                raise _BrokerError(409, "gang_failed")
            if shard_index not in self._members:
                raise _BrokerError(404, "unknown_shard")
            self._last_heartbeat[shard_index] = self._clock()
            return {"status": "ok"}

    def record_result(
        self, *, shard_index: Any, state: Any, detail: Any = None
    ) -> Dict[str, Any]:
        """Record one member's terminal result and derive the gang outcome."""
        if type(shard_index) is not int:
            raise _BrokerError(400, "invalid_result")
        if state not in ("completed", "failed"):
            raise _BrokerError(400, "invalid_result")
        if detail is not None and not isinstance(detail, str):
            raise _BrokerError(400, "invalid_result")
        bounded_detail = None if detail is None else detail[:_MAX_DETAIL_CHARS]
        with self._condition:
            if self._cleaned:
                raise _BrokerError(409, "run_closed")
            if self._gang_state == STATE_FAILED:
                raise _BrokerError(409, "gang_failed")
            if shard_index not in self._members:
                raise _BrokerError(404, "unknown_shard")
            if shard_index in self._results:
                raise _BrokerError(409, "duplicate_result")
            self._results[shard_index] = {"state": state, "detail": bounded_detail}
            self._append_event_locked(
                "member_result",
                shard_index=shard_index,
                state=state,
                detail=bounded_detail,
            )
            if state == "failed":
                self._fail_locked("member_failed")
            elif len(self._results) >= self._machine_count and all(
                result["state"] == "completed" for result in self._results.values()
            ):
                self._gang_state = STATE_COMPLETED
                self._append_event_locked("gang_completed")
                self._persist_state_locked()
                self._condition.notify_all()
            else:
                self._persist_state_locked()
            return {"status": "recorded"}

    def cleanup(self) -> Dict[str, Any]:
        """Close the rendezvous; idempotent and valid for any gang state."""
        with self._condition:
            if not self._cleaned:
                self._cleaned = True
                self._append_event_locked("run_cleaned")
                self._persist_state_locked()
                self._condition.notify_all()
            return {"status": "cleaned"}

    # Internals --------------------------------------------------------------

    def _resolve_hostname(self, address: str) -> Optional[str]:
        try:
            return self._resolver(address)
        except OSError:
            return None

    def _monitor_loop(self) -> None:
        tick = max(0.005, min(0.25, self._heartbeat_expiry_s / 4.0))
        while not self._stop_event.wait(tick):
            try:
                with self._condition:
                    self._evaluate_deadlines_locked()
            except Exception:
                # The watchdog must survive transient persistence errors.
                continue

    def _evaluate_deadlines_locked(self) -> None:
        now = self._clock()
        if (
            self._gang_state == STATE_REGISTERING
            and now >= self._start_monotonic + self._register_deadline_s
        ):
            self._fail_locked("register_deadline_exceeded")
        if (
            self._gang_state not in (STATE_COMPLETED, STATE_FAILED)
            and now >= self._start_monotonic + self._run_deadline_s
        ):
            self._fail_locked("run_deadline_exceeded")
        if self._gang_state not in (STATE_COMPLETED, STATE_FAILED):
            for shard_index, member in sorted(self._members.items()):
                if shard_index in self._lost_shards:
                    continue
                if now >= self._last_heartbeat[shard_index] + self._heartbeat_expiry_s:
                    self._lost_shards.add(shard_index)
                    self._append_event_locked(
                        "member_lost",
                        shard_index=shard_index,
                        address=member.address,
                        hostname=member.hostname,
                    )
                    self._fail_locked("heartbeat_expired")
                    return

    def _fail_locked(self, reason: str) -> None:
        if self._gang_state in (STATE_COMPLETED, STATE_FAILED):
            return
        self._gang_state = STATE_FAILED
        self._failure_reason = reason
        self._append_event_locked("gang_failed", reason=reason)
        self._persist_state_locked()
        self._condition.notify_all()

    def _persist_state_locked(self) -> None:
        state = {
            "schema_version": 1,
            "run_id": self._run_id,
            "gang_state": self._gang_state,
            "machine_count": self._machine_count,
            "cleaned": self._cleaned,
            "failure_reason": self._failure_reason,
            "members": [member.to_dict() for member in self.members],
            "lost_shard_indices": sorted(self._lost_shards),
            "results": {
                str(shard_index): dict(result)
                for shard_index, result in sorted(self._results.items())
            },
            "started_at": self._started_at,
            "updated_at": _utc_now(),
            "listen_host": self._host,
            "listen_port": (
                None if self._server is None else self._server.server_address[1]
            ),
            "register_deadline_s": self._register_deadline_s,
            "run_deadline_s": self._run_deadline_s,
            "heartbeat_expiry_s": self._heartbeat_expiry_s,
        }
        atomic_write_json(self._state_path, state)

    def _append_event_locked(self, event: str, **fields: Any) -> None:
        record = {"event": event, "time": _utc_now()}
        record.update(fields)
        line = json.dumps(record, sort_keys=True) + "\n"
        with open(self._events_path, "a", encoding="utf-8") as stream:
            stream.write(line)


class _BrokerServer(ThreadingHTTPServer):
    """HTTP transport for one GangBroker on an ephemeral port."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: Tuple[str, int], broker: GangBroker):
        super().__init__(address, _BrokerHandler)
        self.broker = broker

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Keep routine client disconnects out of broker stderr."""
        exception = sys.exc_info()[1]
        if isinstance(exception, (BrokenPipeError, ConnectionResetError)):
            _LOG.debug("broker client %s disconnected: %r", client_address, exception)
            return
        super().handle_error(request, client_address)


class _BrokerHandler(BaseHTTPRequestHandler):
    """Loopback-safe request handler; errors are status codes plus JSON."""

    server_version = "perf-test-gang-broker"
    protocol_version = "HTTP/1.1"

    _OPERATIONS = frozenset({"register", "members", "heartbeat", "result", "cleanup"})

    def log_message(self, *_args: Any) -> None:
        """Keep test and service output pristine."""

    def version_string(self) -> str:
        return self.server_version

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_GET(self) -> None:
        self._dispatch("GET")

    def _dispatch(self, method: str) -> None:
        try:
            status, body = self._handle(method)
        except _BrokerError as error:
            status, body = error.status, {"error": error.code}
        except Exception:
            status, body = 500, {"error": "internal_error"}
        self._respond(status, body)

    def _handle(self, method: str) -> Tuple[int, Dict[str, Any]]:
        path = urlsplit(self.path).path
        parts = path.strip("/").split("/")
        if len(parts) != 4 or parts[0] != "v1" or parts[1] != "runs":
            raise _BrokerError(404, "unknown_operation")
        run_id, op = parts[2], parts[3]
        self._require_auth()
        broker = self.server.broker
        if run_id != broker.run_id:
            raise _BrokerError(404, "unknown_run")
        if op == "register" and method == "POST":
            payload = self._read_json()
            return 200, broker.register_member(
                shard_index=payload.get("shard_index"),
                hostname=payload.get("hostname"),
                port=payload.get("port"),
                address=self.client_address[0],
                config_fingerprint=payload.get("config_fingerprint"),
            )
        if op == "members" and method == "GET":
            return broker.await_members()
        if op == "heartbeat" and method == "POST":
            payload = self._read_json()
            return 200, broker.record_heartbeat(shard_index=payload.get("shard_index"))
        if op == "result" and method == "POST":
            payload = self._read_json()
            return 200, broker.record_result(
                shard_index=payload.get("shard_index"),
                state=payload.get("state"),
                detail=payload.get("detail"),
            )
        if op == "cleanup" and method == "POST":
            return 200, broker.cleanup()
        if op in self._OPERATIONS:
            raise _BrokerError(405, "method_not_allowed")
        raise _BrokerError(404, "unknown_operation")

    def _require_auth(self) -> None:
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.broker.token}"
        if not secrets.compare_digest(header.encode("utf-8"), expected.encode("utf-8")):
            raise _BrokerError(401, "unauthorized")

    def _read_json(self) -> Dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            raise _BrokerError(400, "invalid_json") from None
        if length < 0 or length > _MAX_BODY_BYTES:
            raise _BrokerError(400, "payload_too_large")
        raw = self.rfile.read(length) if length else b""
        try:
            value = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _BrokerError(400, "invalid_json") from None
        if not isinstance(value, dict):
            raise _BrokerError(400, "invalid_json")
        return value

    def _respond(self, status: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_gang_broker(
    machine_count: int,
    *,
    state_dir: Path,
    host: str = "0.0.0.0",
    register_deadline_s: float = REGISTER_DEADLINE_S,
    run_deadline_s: float = RUN_DEADLINE_S,
    heartbeat_expiry_s: float = HEARTBEAT_EXPIRY_S,
    expected_config_fingerprint: Optional[str] = None,
    resolver: Optional[Resolver] = None,
    clock: Callable[[], float] = time.monotonic,
) -> GangBroker:
    """Create (but do not start) a gang rendezvous broker."""
    return GangBroker(
        machine_count,
        state_dir=state_dir,
        host=host,
        register_deadline_s=register_deadline_s,
        run_deadline_s=run_deadline_s,
        heartbeat_expiry_s=heartbeat_expiry_s,
        expected_config_fingerprint=expected_config_fingerprint,
        resolver=resolver,
        clock=clock,
    )
