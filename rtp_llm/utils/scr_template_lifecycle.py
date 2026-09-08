"""Process-local lifecycle hooks for template checkpoint/restore.

The SCR barrier is a control-plane rendezvous.  Components that carry host
bound state must participate in the same rendezvous instead of inspecting SCR
environment variables independently.  Hooks are deliberately small so that
network, grammar and metrics implementations can be tested in isolation.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Protocol


LOGGER = logging.getLogger(__name__)


def template_phase_active() -> bool:
    """Return whether the external controller selected a template phase."""
    enabled = os.environ.get("RTPLLM_ENABLE_SCR", "").strip().lower()
    phase = os.environ.get("SCR_PHASE", "").strip().lower()
    return enabled in {"1", "true", "yes", "on"} and phase in {
        "checkpoint",
        "restore",
    }


class TemplateLifecycleHook(Protocol):
    """Optional operations run around one template barrier."""

    def prepare_for_template(self, generation: str) -> None: ...

    def restore_fixup(self, generation: str) -> None: ...

    def release_template(self, generation: str) -> None: ...

    def abort_template(self, generation: str) -> None: ...


@dataclass(frozen=True)
class TemplateLifecycleState:
    generation: str
    phase: str


class TemplateLifecycle:
    """Idempotent coordinator owned by one RTP-LLM process.

    Registration is process-local.  The control plane still owns dump/restore;
    this object only guarantees that every host-bound participant reaches the
    same local prepare/fixup/release sequence before the process announces
    readiness.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._hooks: dict[str, TemplateLifecycleHook] = {}
        self._state: TemplateLifecycleState | None = None

    def register(self, name: str, hook: TemplateLifecycleHook) -> None:
        if not name:
            raise ValueError("template lifecycle hook name must be non-empty")
        with self._lock:
            existing = self._hooks.get(name)
            if existing is not None and existing is not hook:
                raise ValueError(f"template lifecycle hook already registered: {name}")
            self._hooks[name] = hook

    def unregister(self, name: str) -> None:
        with self._lock:
            self._hooks.pop(name, None)

    def _snapshot(self) -> list[tuple[str, TemplateLifecycleHook]]:
        with self._lock:
            return list(self._hooks.items())

    def prepare_for_template(self, generation: str, phase: str) -> None:
        with self._lock:
            if self._state is not None:
                if self._state == TemplateLifecycleState(generation, phase):
                    return
                raise RuntimeError(
                    "template lifecycle already active "
                    f"generation={self._state.generation} phase={self._state.phase}"
                )
            self._state = TemplateLifecycleState(generation, phase)
        completed: list[tuple[str, TemplateLifecycleHook]] = []
        try:
            for name, hook in self._snapshot():
                LOGGER.info("template hook prepare name=%s generation=%s", name, generation)
                hook.prepare_for_template(generation)
                completed.append((name, hook))
        except BaseException:
            for name, hook in reversed(completed):
                try:
                    hook.abort_template(generation)
                except BaseException:
                    LOGGER.exception("template hook abort failed name=%s", name)
            with self._lock:
                self._state = None
            raise

    def restore_fixup(self, generation: str) -> None:
        with self._lock:
            state = self._state
        if state is None or state.generation != generation:
            raise RuntimeError(f"template lifecycle fixup without prepare: {generation}")
        for name, hook in self._snapshot():
            LOGGER.info("template hook fixup name=%s generation=%s", name, generation)
            hook.restore_fixup(generation)

    def release_template(self, generation: str) -> None:
        with self._lock:
            state = self._state
        if state is None or state.generation != generation:
            raise RuntimeError(f"template lifecycle release without prepare: {generation}")
        try:
            for name, hook in self._snapshot():
                LOGGER.info("template hook release name=%s generation=%s", name, generation)
                hook.release_template(generation)
        finally:
            with self._lock:
                self._state = None

    def abort_template(self, generation: str) -> None:
        with self._lock:
            state = self._state
        if state is None or state.generation != generation:
            return
        for name, hook in reversed(self._snapshot()):
            try:
                hook.abort_template(generation)
            except BaseException:
                LOGGER.exception("template hook abort failed name=%s", name)
        with self._lock:
            self._state = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._state is not None


_LIFECYCLE = TemplateLifecycle()


def get_template_lifecycle() -> TemplateLifecycle:
    return _LIFECYCLE


class CallbackHook:
    """Adapt callbacks while keeping lifecycle error handling uniform."""

    def __init__(
        self,
        prepare: Callable[[str], None] | None = None,
        fixup: Callable[[str], None] | None = None,
        release: Callable[[str], None] | None = None,
        abort: Callable[[str], None] | None = None,
    ) -> None:
        self._prepare = prepare or (lambda _generation: None)
        self._fixup = fixup or (lambda _generation: None)
        self._release = release or (lambda _generation: None)
        self._abort = abort or (lambda _generation: None)

    def prepare_for_template(self, generation: str) -> None:
        self._prepare(generation)

    def restore_fixup(self, generation: str) -> None:
        self._fixup(generation)

    def release_template(self, generation: str) -> None:
        self._release(generation)

    def abort_template(self, generation: str) -> None:
        self._abort(generation)
