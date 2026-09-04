"""Out-of-process client and coordinator state machine for ``scr_controller``.

This module intentionally contains no worker-side hooks and performs no work at
import time.  An application-level coordinator can use :class:`ScrControllerClient`
to invoke the controller CLI, while GPU workers continue to use the Epsilon
Python API independently.

The controller is a versioned CLI rather than a stable Python ABI.  Commands
are therefore recorded as argv tuples and every subprocess return code is
preserved in :class:`ControllerResult`.  ``check`` is considered ready only
when its JSON has ``errno == 0`` *and* ``checkpoint_ready is True``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger(__name__)

DEFAULT_CONTROLLER = "/home/yuziqu.yzq/scr_controller"
DEFAULT_UDS = "/run/scr/socket"
COMMAND_TIMEOUT_GRACE_SECONDS = 30.0


class ScrControllerError(RuntimeError):
    """Base error for command, protocol, and state failures."""


class ScrControllerCommandError(ScrControllerError):
    """A controller process failed or returned an invalid response."""

    def __init__(self, message: str, result: "ControllerResult | None" = None):
        super().__init__(message)
        self.result = result


class ScrControllerNotReady(ScrControllerError):
    """The scheduler's JSON response says that checkpoint is not ready."""


class ScrControllerStateError(ScrControllerError):
    """An operation is not valid for the current coordinator phase."""


class ScrControllerBusy(ScrControllerError):
    """Another coordinator already owns this ``(uds, generation)`` lease."""


@dataclass(frozen=True)
class ControllerResult:
    """A complete invocation record, suitable for coordinator manifests."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    payload: Mapping[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def errno(self) -> int | None:
        value = self.payload.get("errno") if self.payload is not None else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @property
    def checkpoint_ready(self) -> bool | None:
        value = self.payload.get("checkpoint_ready") if self.payload is not None else None
        return value if isinstance(value, bool) else None


def _json_payload(output: str, argv: Sequence[str]) -> Mapping[str, Any]:
    """Parse the controller's JSON object, tolerating build banners.

    Some packaged controller builds print a release banner before the JSON
    response.  We only accept a complete JSON object beginning with ``{``;
    arbitrary text or a JSON array is rejected.
    """

    decoder = json.JSONDecoder()
    # The response is normally one compact line, but accepting a pretty
    # printed object keeps the parser compatible with controller builds that
    # use ``serde_json::to_string_pretty``.  Parse from the last opening brace
    # first so a release banner containing an unrelated object is ignored.
    parsed: list[Mapping[str, Any]] = []
    for offset in reversed([i for i, char in enumerate(output) if char == "{"]):
        try:
            value, _ = decoder.raw_decode(output[offset:].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
            # Status responses are identified by one of these top-level
            # fields.  Prefer that object over a nested metadata object.
            if "errno" in value or "checkpoint_ready" in value:
                return value
    if parsed:
        return parsed[0]
    raise ScrControllerCommandError(
        f"controller command did not return a JSON object: {' '.join(argv)}"
    )


def _require_success(result: ControllerResult) -> ControllerResult:
    if not result.ok:
        raise ScrControllerCommandError(
            f"controller exited with rc={result.returncode}: {' '.join(result.argv)}; "
            f"stderr={result.stderr.strip()!r}",
            result,
        )
    return result


class ScrControllerClient:
    """Thin, mockable wrapper around the ``scr_controller`` executable.

    ``uds`` is a directory (for example ``/run/scr/socket``); the executable
    itself appends ``ttrpc``.  No sudo is added implicitly: deployment may use
    a privileged wrapper, a dedicated service account, or a caller-provided
    executable path.
    """

    def __init__(
        self,
        controller: str = DEFAULT_CONTROLLER,
        uds: str = DEFAULT_UDS,
        sid: str | None = None,
        *,
        default_timeout: float = 30.0,
    ) -> None:
        if not uds:
            raise ValueError("uds must be a non-empty scheduler socket directory")
        if default_timeout <= 0:
            raise ValueError("default_timeout must be positive")
        self.controller = str(controller)
        self.uds = str(uds)
        self.sid = sid
        self.default_timeout = float(default_timeout)
        self.history: list[ControllerResult] = []

    def _argv(self, command: str, args: Sequence[str] = ()) -> tuple[str, ...]:
        argv = [self.controller, "--uds", self.uds]
        if self.sid is not None:
            argv.extend(("--sid", str(self.sid)))
        argv.append(command)
        argv.extend(str(arg) for arg in args)
        return tuple(argv)

    def run(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        timeout: float | None = None,
        json_response: bool = False,
    ) -> ControllerResult:
        """Run one command and return its complete invocation record.

        A timeout is converted to :class:`ScrControllerCommandError`; the
        attempted argv is still appended to ``history`` with returncode ``-1``
        so the coordinator can persist diagnostics.
        """

        argv = self._argv(command, args)
        effective_timeout = self.default_timeout if timeout is None else float(timeout)
        if effective_timeout <= 0:
            raise ValueError("timeout must be positive")
        try:
            completed = subprocess.run(
                list(argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            result = ControllerResult(
                argv=argv,
                returncode=-1,
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr),
            )
            self.history.append(result)
            raise ScrControllerCommandError(
                f"controller command timed out after {effective_timeout}s: {' '.join(argv)}",
                result,
            ) from exc
        except OSError as exc:
            result = ControllerResult(argv=argv, returncode=-1, stderr=str(exc))
            self.history.append(result)
            raise ScrControllerCommandError(
                f"failed to execute controller: {' '.join(argv)}: {exc}", result
            ) from exc

        payload: Mapping[str, Any] | None = None
        if json_response:
            try:
                payload = _json_payload(completed.stdout, argv)
            except ScrControllerCommandError as exc:
                result = ControllerResult(
                    argv=argv,
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
                self.history.append(result)
                exc.result = result
                raise
        result = ControllerResult(
            argv=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
        )
        self.history.append(result)
        return result

    def health(self, *, timeout: float | None = None) -> ControllerResult:
        return _require_success(self.run("health", timeout=timeout, json_response=True))

    def check(self, *, timeout: float | None = None) -> ControllerResult:
        result = self.run("check", timeout=timeout, json_response=True)
        _require_success(result)
        if result.errno != 0 or result.checkpoint_ready is not True:
            raise ScrControllerNotReady(
                f"checkpoint is not ready (errno={result.errno!r}, "
                f"checkpoint_ready={result.checkpoint_ready!r})"
            )
        return result

    def check_steady_state(self, *, timeout: float | None = None) -> ControllerResult:
        result = self.run("check-steady-state", timeout=timeout, json_response=True)
        _require_success(result)
        if (result.errno is not None and result.errno != 0) or (
            result.checkpoint_ready is not True
        ):
            raise ScrControllerNotReady(
                f"steady-state is not ready (errno={result.errno!r}, "
                f"checkpoint_ready={result.checkpoint_ready!r})"
            )
        return result

    def _simple(self, command: str, *, timeout: float | None = None) -> ControllerResult:
        return _require_success(self.run(command, timeout=timeout))

    def block(self, *, timeout: float | None = None) -> ControllerResult:
        return self._simple("block", timeout=timeout)

    def unblock(self, *, timeout: float | None = None) -> ControllerResult:
        return self._simple("unblock", timeout=timeout)

    def fallback(self, *, timeout: float | None = None) -> ControllerResult:
        return self._simple("fallback", timeout=timeout)

    def dump(
        self,
        path: str | Path,
        *,
        bypass_cr_path: str | Path | None = None,
        block_timeout_ms: int | None = None,
        bypass_direct_io: bool = False,
        cache_fs_speedup: bool = False,
        timeout: float | None = None,
    ) -> ControllerResult:
        args = ["--path", str(path)]
        if bypass_cr_path is not None:
            args += ["--bypass-cr-path", str(bypass_cr_path)]
        if block_timeout_ms is not None:
            if block_timeout_ms <= 0:
                raise ValueError("block_timeout_ms must be positive")
            args += ["--block-timeout-ms", str(block_timeout_ms)]
        if bypass_direct_io:
            args.append("--bypass-direct-io")
        if cache_fs_speedup:
            args.append("--cache-fs-speedup")
        return _require_success(self.run("dump", args, timeout=timeout))

    def wait_cr_done(
        self, *, timeout_seconds: int | None = None, timeout: float | None = None
    ) -> ControllerResult:
        args: list[str] = []
        if timeout_seconds is not None:
            if timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive")
            args += ["--timeout", str(timeout_seconds)]
        if timeout is None and timeout_seconds is not None:
            timeout = max(
                self.default_timeout,
                float(timeout_seconds) + COMMAND_TIMEOUT_GRACE_SECONDS,
            )
        return _require_success(self.run("wait-cr-done", args, timeout=timeout))

    def prepare_restore(
        self,
        bypass_cr_path: str | Path,
        *,
        cache_fs_speedup: bool = False,
        timeout_seconds: int | None = None,
        timeout: float | None = None,
    ) -> ControllerResult:
        args = ["--bypass-cr-path", str(bypass_cr_path)]
        if cache_fs_speedup:
            args.append("--cache-fs-speedup")
        if timeout_seconds is not None:
            if timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive")
            args += ["--timeout", str(timeout_seconds)]
            if timeout is None:
                timeout = max(
                    self.default_timeout,
                    float(timeout_seconds) + COMMAND_TIMEOUT_GRACE_SECONDS,
                )
        return _require_success(self.run("prepare-restore", args, timeout=timeout))

    def restore(
        self,
        path: str | Path,
        *,
        bypass_cr_path: str | Path | None = None,
        cache_fs_speedup: bool = False,
        timeout: float | None = None,
    ) -> ControllerResult:
        args = ["--path", str(path)]
        if bypass_cr_path is not None:
            args += ["--bypass-cr-path", str(bypass_cr_path)]
        if cache_fs_speedup:
            args.append("--cache-fs-speedup")
        return _require_success(self.run("restore", args, timeout=timeout))


class ControllerPhase(str, Enum):
    INIT = "INIT"
    CHECK_READY = "CHECK_READY"
    BLOCKED = "BLOCKED"
    DUMP_REQUESTED = "DUMP_REQUESTED"
    RESTORE_PREPARED = "RESTORE_PREPARED"
    RESTORE_REQUESTED = "RESTORE_REQUESTED"
    WAIT_CR_DONE = "WAIT_CR_DONE"
    SERVING = "SERVING"
    ABORTED = "ABORTED"
    FALLBACK = "FALLBACK"


class ScrControllerCoordinator:
    """Explicit, generation-scoped state machine for one external coordinator.

    The class never starts a thread and never calls the controller implicitly.
    Every transition is an explicit method call, making it safe to place in a
    sidecar or platform coordinator.  Idempotent retries return the cached
    command result and do not invoke the executable a second time.
    """

    _lease_lock = threading.Lock()
    _leases: dict[tuple[str, str, str], str] = {}

    def __init__(
        self,
        client: ScrControllerClient,
        generation: str,
        *,
        coordinator_id: str | None = None,
    ) -> None:
        if not generation:
            raise ValueError("generation must be non-empty")
        self.client = client
        self.generation = str(generation)
        self.coordinator_id = coordinator_id or uuid.uuid4().hex
        self._lease_key = (client.controller, client.uds, self.generation)
        with self._lease_lock:
            owner = self._leases.get(self._lease_key)
            if owner is not None and owner != self.coordinator_id:
                raise ScrControllerBusy(
                    f"generation {self.generation!r} is already owned by a coordinator"
                )
            self._leases[self._lease_key] = self.coordinator_id
        self.phase = ControllerPhase.INIT
        self.failure_reason: str | None = None
        self._results: dict[str, ControllerResult] = {}
        self._fallback_result: ControllerResult | None = None

    def close(self) -> None:
        with self._lease_lock:
            if self._leases.get(self._lease_key) == self.coordinator_id:
                self._leases.pop(self._lease_key, None)

    def __enter__(self) -> "ScrControllerCoordinator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _cached(self, key: str) -> ControllerResult | None:
        return self._results.get(key)

    def _remember(self, key: str, result: ControllerResult) -> ControllerResult:
        self._results[key] = result
        return result

    def check_ready(self, *, timeout: float | None = None) -> ControllerResult:
        cached = self._cached("check")
        if cached is not None:
            return cached
        try:
            result = self.client.check(timeout=timeout)
        except ScrControllerNotReady:
            raise
        except ScrControllerError as exc:
            self._mark_aborted(exc)
            raise
        self.phase = ControllerPhase.CHECK_READY
        return self._remember("check", result)

    def block(self, *, timeout: float | None = None) -> ControllerResult:
        cached = self._cached("block")
        if cached is not None:
            return cached
        if self.phase not in (ControllerPhase.CHECK_READY, ControllerPhase.BLOCKED):
            raise ScrControllerStateError(f"block is invalid in phase {self.phase.value}")
        try:
            result = self.client.block(timeout=timeout)
        except ScrControllerError as exc:
            self._mark_aborted(exc)
            raise
        self.phase = ControllerPhase.BLOCKED
        return self._remember("block", result)

    def dump(self, path: str | Path, **kwargs: Any) -> ControllerResult:
        cached = self._cached("dump")
        if cached is not None:
            return cached
        if self.phase not in (ControllerPhase.CHECK_READY, ControllerPhase.BLOCKED):
            raise ScrControllerStateError(f"dump is invalid in phase {self.phase.value}")
        try:
            result = self.client.dump(path, **kwargs)
        except ScrControllerError as exc:
            self._mark_aborted(exc)
            raise
        self.phase = ControllerPhase.DUMP_REQUESTED
        return self._remember("dump", result)

    def prepare_restore(self, bypass_cr_path: str | Path, **kwargs: Any) -> ControllerResult:
        cached = self._cached("prepare_restore")
        if cached is not None:
            return cached
        if self.phase not in (ControllerPhase.INIT, ControllerPhase.RESTORE_PREPARED):
            raise ScrControllerStateError(
                f"prepare_restore is invalid in phase {self.phase.value}"
            )
        try:
            result = self.client.prepare_restore(bypass_cr_path, **kwargs)
        except ScrControllerError as exc:
            self._mark_aborted(exc)
            raise
        self.phase = ControllerPhase.RESTORE_PREPARED
        return self._remember("prepare_restore", result)

    def restore(self, path: str | Path, **kwargs: Any) -> ControllerResult:
        cached = self._cached("restore")
        if cached is not None:
            return cached
        if self.phase != ControllerPhase.RESTORE_PREPARED:
            raise ScrControllerStateError(f"restore is invalid in phase {self.phase.value}")
        try:
            result = self.client.restore(path, **kwargs)
        except ScrControllerError as exc:
            self._mark_aborted(exc)
            raise
        self.phase = ControllerPhase.RESTORE_REQUESTED
        return self._remember("restore", result)

    def wait_cr_done(
        self,
        *,
        timeout_seconds: int | None = None,
        timeout: float | None = None,
    ) -> ControllerResult:
        key = "wait_restore" if self.phase == ControllerPhase.RESTORE_REQUESTED else "wait_dump"
        cached = self._cached(key)
        if cached is not None:
            return cached
        if self.phase not in (ControllerPhase.DUMP_REQUESTED, ControllerPhase.RESTORE_REQUESTED):
            raise ScrControllerStateError(f"wait_cr_done is invalid in phase {self.phase.value}")
        try:
            result = self.client.wait_cr_done(timeout_seconds=timeout_seconds, timeout=timeout)
        except ScrControllerError as exc:
            self._mark_aborted(exc)
            raise
        self.phase = ControllerPhase.WAIT_CR_DONE
        return self._remember(key, result)

    def unblock(self, *, timeout: float | None = None) -> ControllerResult:
        cached = self._cached("unblock")
        if cached is not None:
            return cached
        if self.phase not in (ControllerPhase.WAIT_CR_DONE, ControllerPhase.BLOCKED):
            raise ScrControllerStateError(f"unblock is invalid in phase {self.phase.value}")
        try:
            result = self.client.unblock(timeout=timeout)
        except ScrControllerError as exc:
            self._mark_aborted(exc)
            raise
        self.phase = ControllerPhase.SERVING
        return self._remember("unblock", result)

    def abort(
        self,
        reason: str,
        *,
        fallback: bool = True,
        timeout: float | None = None,
    ) -> ControllerResult | None:
        """Enter ABORTED and optionally issue the controller FALLBACK once."""

        if self.phase == ControllerPhase.FALLBACK:
            return self._fallback_result
        self.phase = ControllerPhase.ABORTED
        self.failure_reason = str(reason)
        if not fallback:
            return None
        if self._fallback_result is not None:
            return self._fallback_result
        try:
            self._fallback_result = self.client.fallback(timeout=timeout)
        except ScrControllerError as exc:
            # Preserve the ABORTED phase and original reason; callers still get
            # the command error and can perform platform-specific fallback.
            LOGGER.error("scr_controller fallback failed: %s", exc)
            raise
        self.phase = ControllerPhase.FALLBACK
        return self._fallback_result

    def _mark_aborted(self, exc: BaseException) -> None:
        self.phase = ControllerPhase.ABORTED
        self.failure_reason = str(exc)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


__all__ = [
    "ControllerPhase",
    "ControllerResult",
    "COMMAND_TIMEOUT_GRACE_SECONDS",
    "DEFAULT_CONTROLLER",
    "DEFAULT_UDS",
    "ScrControllerBusy",
    "ScrControllerClient",
    "ScrControllerCommandError",
    "ScrControllerCoordinator",
    "ScrControllerError",
    "ScrControllerNotReady",
    "ScrControllerStateError",
]
