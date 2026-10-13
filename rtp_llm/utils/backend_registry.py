"""Deferred registration of out-of-tree model backends.

Every slot consumer loads the optional backend entrypoint through
``run_backend_registrations()`` before it initialises the slot. A backend
cannot call ``LinearFactory.register()`` or ``StrategyRegistry.register()``
at entrypoint import time because the factory does not exist yet. Importing the
factories early from the backend is not an option either: eager imports of the
MoE/attention factories pull in communication libraries and have broken server
startup for configurations that never use them.

Backends therefore record their intent here, and each factory runs the hooks
for its own slot once it has finished building its registry:

    # backend, at import time
    register_backend_hook(
        "linear", lambda factory: factory.register(MyLinear)
    )

    # factory, at the end of its __init__
    run_backend_registrations("linear", factory=LinearFactory)

This lives under ``rtp_llm.utils`` rather than next to the factories on
purpose: the server argument parser also consumes a slot, and importing
anything under ``models_py.modules.factory`` would execute that package's
``__init__``, which builds every factory. That is the eager import this
mechanism exists to avoid.

Hook exceptions are intentionally not swallowed. A backend that registered a
hook needs it: silently dropping the registration leaves the factory selecting
a different implementation, which shows up as wrong numerics rather than a
startup failure.
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

BackendHook = Callable[..., None]

_hooks: Dict[str, List[BackendHook]] = {}
_started: Set[str] = set()
_repeatable: Set[str] = set()
_lock = threading.RLock()
_condition = threading.Condition(_lock)
_inflight: Dict[str, Tuple[int, threading.Event]] = {}
_failures: Dict[str, BaseException] = {}


def ensure_backend_entrypoint_loaded() -> bool:
    """Load the optional model-backend entrypoint before consuming a slot."""
    from rtp_llm.utils.import_util import import_optional_internal_source_entrypoint

    return import_optional_internal_source_entrypoint("models_py")


def register_backend_hook(slot: str, hook: BackendHook) -> None:
    """Record ``hook`` to run once the ``slot`` owner is initialised.

    Raises if the slot already started, since late hooks would not be applied
    consistently to every owner of that slot.
    """
    with _condition:
        if slot in _started:
            raise RuntimeError(
                f"backend slot {slot!r} was already initialised; register the "
                "hook before its owner is initialized"
            )
        _hooks.setdefault(slot, []).append(hook)


def run_backend_registrations(
    slot: str, *, repeatable: bool = False, **context: Any
) -> None:
    """Run the hooks recorded for ``slot``, passing ``context`` to each.

    By default a slot runs at most once, so a factory re-imported under a
    different alias does not double-register. A repeatable slot freezes its
    hook set on the first call and replays those hooks for every new owner.
    """
    ensure_backend_entrypoint_loaded()

    # Loading the entrypoint is deliberately outside the condition: its import
    # body records hooks and therefore needs to acquire this same lock.
    while True:
        with _condition:
            inflight = _inflight.get(slot)
            if inflight is not None:
                owner_thread, event = inflight
                if owner_thread == threading.get_ident():
                    raise RuntimeError(
                        f"backend slot {slot!r} registration is re-entrant"
                    )
            elif slot in _started:
                was_repeatable = slot in _repeatable
                if repeatable != was_repeatable:
                    raise RuntimeError(
                        f"backend slot {slot!r} lifecycle changed after it started"
                    )
                if not repeatable:
                    return
                hooks = tuple(_hooks.get(slot, ()))
                event = threading.Event()
                _inflight[slot] = (threading.get_ident(), event)
                break
            else:
                _failures.pop(slot, None)
                _started.add(slot)
                if repeatable:
                    _repeatable.add(slot)
                event = threading.Event()
                _inflight[slot] = (threading.get_ident(), event)
                hooks = tuple(_hooks.get(slot, ()))
                break

        # Another owner is currently executing this slot. Wait for its result,
        # then re-evaluate the lifecycle so repeatable slots can replay safely.
        event.wait()
        with _condition:
            failure = _failures.get(slot)
        if failure is not None:
            raise RuntimeError(
                f"backend slot {slot!r} registration failed; retry is allowed"
            ) from failure

    try:
        # Hooks may import backend modules and must never execute while holding
        # the global registry lock. The snapshot keeps late registration
        # rejected while allowing independent slots to proceed concurrently.
        for hook in hooks:
            logger.debug("running backend registration hook for slot %r", slot)
            hook(**context)
    except BaseException as error:
        with _condition:
            _started.discard(slot)
            _repeatable.discard(slot)
            _failures[slot] = error
            _, event = _inflight.pop(slot)
            event.set()
        raise
    else:
        with _condition:
            _, event = _inflight.pop(slot)
            event.set()


def reset_backend_registrations() -> None:
    """Drop recorded hooks and lifecycle state. For tests only."""
    with _condition:
        if _inflight:
            raise RuntimeError("cannot reset backend registrations while hooks run")
        _hooks.clear()
        _started.clear()
        _repeatable.clear()
        _failures.clear()
