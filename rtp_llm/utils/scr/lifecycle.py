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
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from rtp_llm.utils.scr.restore_context import RestoreContext

LOGGER = logging.getLogger(__name__)
RTPLLM_ENABLE_SCR_ENV = "RTPLLM_ENABLE_SCR"
SCR_PHASE_ENV = "SCR_PHASE"
SCR_PHASE_CHECKPOINT = "checkpoint"
SCR_PHASE_RESTORE = "restore"
SCR_PHASE_NORMAL = "normal"


def is_scr_enabled() -> bool:
    """Read the RTP-LLM participation switch without importing Epsilon."""
    return os.environ.get(RTPLLM_ENABLE_SCR_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def current_phase() -> str:
    """Return the normalized SCR phase supplied by the control plane."""
    return os.environ.get(SCR_PHASE_ENV, "").strip().lower()


def template_phase_active() -> bool:
    """Return whether the external controller selected a template phase."""
    return is_scr_enabled() and current_phase() in {
        SCR_PHASE_CHECKPOINT,
        SCR_PHASE_RESTORE,
    }


class RestoreFixup(Protocol):
    """Repair one component using this attempt's shared, fresh inputs."""

    def restore_fixup(self, context: RestoreContext) -> None: ...


class TemplateLifecycleHook(RestoreFixup, Protocol):
    """Optional operations run around one template barrier."""

    def prepare_for_template(self) -> None: ...

    def release_template(self) -> None: ...

    def abort_template(self) -> None: ...


def defer_until_release(name: str, start: Callable[[], None]) -> None:
    """Run ``start`` at template release, or immediately when SCR is inactive.

    This is the single entry point for components that must not open sockets,
    spawn workers or touch the network before the controller has captured the
    process tree.
    """
    if template_phase_active():
        # A process may rebuild a component (tests, re-init); the latest
        # instance owns the release action.
        get_template_lifecycle().register(
            name,
            CallbackHook(release=start),
            replace=True,
        )
        return
    start()


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
        self._phase: str | None = None
        self._fixup_complete = False

    def register(
        self, name: str, hook: TemplateLifecycleHook, *, replace: bool = False
    ) -> None:
        if not name:
            raise ValueError("template lifecycle hook name must be non-empty")
        with self._lock:
            existing = self._hooks.get(name)
            if existing is hook:
                return
            if self._phase is not None:
                raise RuntimeError("cannot register hooks during a template lifecycle")
            if existing is not None and existing is not hook and not replace:
                raise ValueError(f"template lifecycle hook already registered: {name}")
            self._hooks[name] = hook

    def register_fixup(
        self, name: str, fixup: RestoreFixup, *, replace: bool = False
    ) -> None:
        """Register a fixup-only component without three empty lifecycle methods.

        Fixups run in registration order. Register local identity/address repairs
        before service discovery or registration, and register before prepare.
        ``replace`` lets a component that is constructed more than once (for
        example a hook bound to one config object) overwrite its own entry
        instead of failing on the duplicate name.
        """
        with self._lock:
            existing = self._hooks.get(name)
            if isinstance(existing, _FixupHook) and existing.target is fixup:
                return
            self.register(name, _FixupHook(fixup), replace=replace)

    def _snapshot(self) -> list[tuple[str, TemplateLifecycleHook]]:
        with self._lock:
            return list(self._hooks.items())

    def prepare_for_template(self, phase: str) -> None:
        with self._lock:
            if self._phase is not None:
                if self._phase == phase:
                    return
                raise RuntimeError(
                    "template lifecycle already active " f"phase={self._phase}"
                )
            self._phase = phase
            self._fixup_complete = False
        completed: list[tuple[str, TemplateLifecycleHook]] = []
        try:
            for name, hook in self._snapshot():
                LOGGER.info("template hook prepare name=%s", name)
                hook.prepare_for_template()
                completed.append((name, hook))
        except BaseException:
            for name, hook in reversed(completed):
                try:
                    hook.abort_template()
                except BaseException:
                    LOGGER.exception("template hook abort failed name=%s", name)
            with self._lock:
                self._phase = None
            raise

    def restore_fixup(self, context: RestoreContext) -> None:
        with self._lock:
            phase = self._phase
            self._fixup_complete = False
        if phase is None:
            raise RuntimeError("template lifecycle fixup without prepare")
        for name, hook in self._snapshot():
            LOGGER.info("template hook fixup name=%s", name)
            hook.restore_fixup(context)
        with self._lock:
            self._fixup_complete = True

    def release_template(self) -> None:
        with self._lock:
            phase = self._phase
            fixed = self._fixup_complete
        if phase is None:
            raise RuntimeError("template lifecycle release without prepare")
        if not fixed:
            raise RuntimeError("template lifecycle release before successful fixup")
        for name, hook in self._snapshot():
            LOGGER.info("template hook release name=%s", name)
            hook.release_template()
        # Keep the state on release failure so the barrier can still abort.
        with self._lock:
            self._phase = None
            self._fixup_complete = False

    def abort_template(self) -> None:
        with self._lock:
            phase = self._phase
        if phase is None:
            return
        for name, hook in reversed(self._snapshot()):
            try:
                hook.abort_template()
            except BaseException:
                LOGGER.exception("template hook abort failed name=%s", name)
        with self._lock:
            self._phase = None
            self._fixup_complete = False

    @property
    def active(self) -> bool:
        with self._lock:
            return self._phase is not None


_LIFECYCLE = TemplateLifecycle()


def get_template_lifecycle() -> TemplateLifecycle:
    return _LIFECYCLE


class CallbackHook:
    """Adapt callbacks while keeping lifecycle error handling uniform."""

    def __init__(
        self,
        prepare: Callable[[], None] | None = None,
        fixup: Callable[[RestoreContext], None] | None = None,
        release: Callable[[], None] | None = None,
        abort: Callable[[], None] | None = None,
    ) -> None:
        self._prepare = prepare or (lambda: None)
        self._fixup = fixup or (lambda _context: None)
        self._release = release or (lambda: None)
        self._abort = abort or (lambda: None)

    def prepare_for_template(self) -> None:
        self._prepare()

    def restore_fixup(self, context: RestoreContext) -> None:
        self._fixup(context)

    def release_template(self) -> None:
        self._release()

    def abort_template(self) -> None:
        self._abort()


class _FixupHook(CallbackHook):
    def __init__(self, target: RestoreFixup) -> None:
        super().__init__(fixup=target.restore_fixup)
        self.target = target
