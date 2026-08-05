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
from typing import Any, Callable, Dict, List, Set

logger = logging.getLogger(__name__)

BackendHook = Callable[..., None]

_hooks: Dict[str, List[BackendHook]] = {}
_started: Set[str] = set()
_repeatable: Set[str] = set()
_lock = threading.RLock()


def ensure_backend_entrypoint_loaded() -> bool:
    """Load the optional model-backend entrypoint before consuming a slot."""
    from rtp_llm.utils.import_util import import_optional_internal_source_entrypoint

    return import_optional_internal_source_entrypoint("models_py")


def register_backend_hook(slot: str, hook: BackendHook) -> None:
    """Record ``hook`` to run once the ``slot`` owner is initialised.

    Raises if the slot already started, since late hooks would not be applied
    consistently to every owner of that slot.
    """
    with _lock:
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

    with _lock:
        if slot in _started:
            was_repeatable = slot in _repeatable
            if repeatable != was_repeatable:
                raise RuntimeError(
                    f"backend slot {slot!r} lifecycle changed after it started"
                )
            if not repeatable:
                return
        else:
            _started.add(slot)
            if repeatable:
                _repeatable.add(slot)
        for hook in tuple(_hooks.get(slot, ())):
            logger.debug("running backend registration hook for slot %r", slot)
            hook(**context)


def reset_backend_registrations() -> None:
    """Drop all recorded hooks and slot lifecycle state. For tests only."""
    with _lock:
        _hooks.clear()
        _started.clear()
        _repeatable.clear()
