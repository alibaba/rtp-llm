"""One shared serving-engine lifecycle for a local plan or remote shard."""

from __future__ import annotations

import signal
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from rtp_llm.test.perf_test.errors import EngineLifecycleError


@dataclass(frozen=True)
class EngineDiagnostics:
    log_path: Optional[str] = None
    timeline_files: Tuple[str, ...] = ()
    coredumps: Dict[str, Any] = field(default_factory=dict)
    exit_code: Optional[int] = None
    collection_errors: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_path": self.log_path,
            "timeline_files": list(self.timeline_files),
            "coredumps": dict(self.coredumps),
            "exit_code": self.exit_code,
            "collection_errors": list(self.collection_errors),
        }


class EngineSessionError(EngineLifecycleError):
    def __init__(self, message: str, diagnostics: EngineDiagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class EngineInterrupted(KeyboardInterrupt):
    def __init__(self, signal_number: int):
        super().__init__(f"engine session interrupted by signal {signal_number}")
        self.signal_number = signal_number


class EngineSession:
    """Own exactly one engine process and all lifecycle diagnostics."""

    def __init__(
        self,
        *,
        launcher: Callable[[], Any],
        readiness_timeout_seconds: int,
        timeline_collector: Callable[[], Any],
        coredump_collector: Callable[[], Any],
        install_signal_handlers: bool = True,
    ):
        self._launcher = launcher
        self._readiness_timeout_seconds = readiness_timeout_seconds
        self._timeline_collector = timeline_collector
        self._coredump_collector = coredump_collector
        self._install_signal_handlers = install_signal_handlers
        self._handle: Optional[Any] = None
        self._start_attempted = False
        self._started = False
        self._closed = False
        self._previous_signal_handlers: Dict[int, Any] = {}
        self._diagnostics = EngineDiagnostics()

    @property
    def diagnostics(self) -> EngineDiagnostics:
        return self._diagnostics

    @property
    def log_path(self) -> Optional[str]:
        return getattr(self._handle, "log_path", None)

    @property
    def client(self) -> Any:
        self.assert_healthy()
        return self._handle.client

    def _current_diagnostics(self) -> EngineDiagnostics:
        return EngineDiagnostics(
            log_path=getattr(self._handle, "log_path", None),
            timeline_files=self._diagnostics.timeline_files,
            coredumps=self._diagnostics.coredumps,
            exit_code=getattr(self._handle, "exit_code", None),
            collection_errors=self._diagnostics.collection_errors,
        )

    def _install_handlers(self) -> None:
        if not self._install_signal_handlers:
            return
        try:
            for signal_number in (signal.SIGINT, signal.SIGTERM):
                self._previous_signal_handlers[signal_number] = signal.getsignal(
                    signal_number
                )
                signal.signal(signal_number, self.handle_signal)
        except (OSError, ValueError) as error:
            self._restore_handlers()
            raise EngineSessionError(
                f"failed to install engine signal handlers: {error}",
                self._current_diagnostics(),
            ) from error

    def _restore_handlers(self) -> None:
        handlers = self._previous_signal_handlers
        self._previous_signal_handlers = {}
        for signal_number, handler in handlers.items():
            try:
                signal.signal(signal_number, handler)
            except (OSError, ValueError):
                pass

    def start(self) -> "EngineSession":
        if self._closed:
            raise EngineSessionError(
                "engine session is already closed", self._current_diagnostics()
            )
        if self._started:
            self.assert_healthy()
            return self
        self._start_attempted = True
        try:
            self._handle = self._launcher()
        except BaseException as error:
            self.close()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise EngineSessionError(
                f"engine launch failed: {error}", self._diagnostics
            ) from error
        self._install_handlers()
        try:
            ready = self._handle.wait_ready(self._readiness_timeout_seconds)
        except BaseException as error:
            self.close()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise EngineSessionError(
                f"engine readiness check failed: {error}", self._diagnostics
            ) from error
        if not ready:
            self.close()
            raise EngineSessionError("engine failed readiness check", self._diagnostics)
        self._started = True
        return self

    def assert_healthy(self) -> None:
        if not self._started or self._handle is None:
            raise EngineSessionError(
                "engine session is not started", self._current_diagnostics()
            )
        if not self._handle.is_alive():
            raise EngineSessionError(
                "engine exited during run", self._current_diagnostics()
            )

    def _collect_diagnostics(self) -> None:
        errors = []
        timelines: Tuple[str, ...] = ()
        coredumps: Dict[str, Any] = {}
        try:
            timelines = tuple(self._timeline_collector() or ())
        except Exception as error:
            errors.append(f"timeline collection failed: {error}")
        try:
            coredumps = dict(self._coredump_collector() or {})
        except Exception as error:
            errors.append(f"coredump collection failed: {error}")
        self._diagnostics = EngineDiagnostics(
            log_path=getattr(self._handle, "log_path", None),
            timeline_files=timelines,
            coredumps=coredumps,
            exit_code=getattr(self._handle, "exit_code", None),
            collection_errors=tuple(errors),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._restore_handlers()
        try:
            if self._handle is not None:
                self._handle.stop()
        finally:
            if self._start_attempted:
                self._collect_diagnostics()
            self._started = False

    def handle_signal(self, signal_number: int, _frame: Any = None) -> None:
        self.close()
        raise EngineInterrupted(signal_number)

    def __enter__(self) -> "EngineSession":
        return self.start()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> Optional[bool]:
        self.close()
        return None
