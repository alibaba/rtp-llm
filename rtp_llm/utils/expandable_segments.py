"""Scope PyTorch ``expandable_segments`` to runtime allocations.

PyTorch's expandable allocator is useful for serving-time activation buffers,
but it is undesirable for allocations that happen while the engine starts: a
resident weight or cache allocation can pin a large expandable segment for the
life of the process.  This module removes the option before engine startup and
turns it on again once the engine has been created.

This follows vLLM's CuMemAllocator workaround for the VMM/MemPool conflict:
allocator settings are changed through PyTorch's live setter and restored after
the VMM-owned allocation window.  The startup window is deliberately broader
here because RDMA-registered cache memory must be allocated before expandable
segments can reserve the conflicting virtual-address range.

The helper is intentionally independent of the sleep implementation.  It is a
process-wide allocator setting, so callers should invoke
:func:`prepare_expandable_segments` once before startup allocations and
:func:`enable_runtime_expandable` once after the engine is ready.
"""

import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator


__all__ = [
    "enable_runtime_expandable",
    "expandable_segments_disabled",
    "is_runtime_expandable_active",
    "prepare_expandable_segments",
]


_EXPANDABLE_KEY = "expandable_segments"
# ``start_backend_server`` may prepare the parent before spawning rank
# processes.  The marker lets a spawned child recover the original request
# after the parent has removed the option from PYTORCH_CUDA_ALLOC_CONF.
_REQUESTED_ENV = "RTP_LLM_EXPANDABLE_SEGMENTS_REQUESTED"
_lock = threading.RLock()
_prepared = False
_requested = False
_active = False
_live = False
_base_conf = ""


def _split_conf(conf: str) -> list[str]:
    """Return non-empty allocator config entries with surrounding space removed."""
    return [part.strip() for part in conf.split(",") if part.strip()]


def _is_expandable_true(part: str) -> bool:
    key, separator, value = part.partition(":")
    return (
        separator
        and key.strip().lower() == _EXPANDABLE_KEY
        and value.strip().lower() == "true"
    )


def _alloc_conf_without_expandable(conf: str) -> str:
    """Remove ``expandable_segments`` while preserving other allocator options."""
    return ",".join(
        part
        for part in _split_conf(conf)
        if part.partition(":")[0].strip().lower() != _EXPANDABLE_KEY
    )


def _live_conf(enabled: bool) -> str:
    """Compose the complete allocator config for a live-setting update."""
    setting = f"{_EXPANDABLE_KEY}:{'True' if enabled else 'False'}"
    return f"{_base_conf},{setting}" if _base_conf else setting


def _set_live(enabled: bool) -> None:
    """Apply the live allocator setting and mirror it in this module."""
    import torch

    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is None:
        memory = getattr(torch.cuda, "memory", None)
        setter = getattr(memory, "_set_allocator_settings", None)
    if setter is None:
        raise RuntimeError("PyTorch has no live allocator-settings setter")

    setter(_live_conf(enabled))
    global _live
    _live = enabled


def prepare_expandable_segments() -> bool:
    """Disable expandable segments for engine-startup allocations.

    If ``PYTORCH_CUDA_ALLOC_CONF`` does not request
    ``expandable_segments:True``, this is an idempotent no-op.  When it does,
    the option is removed from the environment and the live allocator is
    forced off.  Removing the environment entry is needed before PyTorch's
    allocator parses it; forcing the live value off also covers processes that
    touched CUDA before this function was called.

    Returns ``True`` when the option was requested, otherwise ``False``.
    """
    global _prepared, _requested, _base_conf
    with _lock:
        if _prepared:
            return _requested
        _prepared = True

        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        entries = _split_conf(conf)
        requested = os.environ.get(_REQUESTED_ENV) == "1" or any(
            _is_expandable_true(part) for part in entries
        )
        if not requested:
            return False

        _base_conf = _alloc_conf_without_expandable(conf)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _base_conf
        os.environ[_REQUESTED_ENV] = "1"
        _requested = True
        try:
            _set_live(False)
        except Exception:
            # The stripped environment still makes a not-yet-initialized
            # allocator choose the safe default.  Keep startup fail-open when a
            # particular PyTorch build has no live setter.
            logging.warning(
                "expandable_segments requested but could not be disabled live; "
                "relying on the stripped PYTORCH_CUDA_ALLOC_CONF",
                exc_info=True,
            )
        return True


def enable_runtime_expandable() -> bool:
    """Enable expandable segments for allocations after engine startup."""
    global _active
    with _lock:
        if not _requested or _active:
            return False
        try:
            _set_live(True)
        except Exception:
            # Staying disabled is safe; it only gives up the runtime
            # fragmentation benefit on older/incompatible PyTorch builds.
            logging.warning(
                "could not enable expandable_segments for runtime allocations; "
                "continuing with it disabled",
                exc_info=True,
            )
            return False
        _active = True
        # Rank children no longer need to recover the request once their own
        # runtime state is established.  Avoid leaking the private marker to
        # unrelated subprocesses started by the serving process.
        os.environ.pop(_REQUESTED_ENV, None)
        logging.info("expandable_segments enabled for runtime allocations")
        return True


@contextmanager
def expandable_segments_disabled() -> Iterator[None]:
    """Temporarily disable expandable segments after runtime has started.

    This is useful for a rare persistent allocation made after startup.  The
    process-wide setting is held under the module lock across the context so
    another thread cannot allocate while the temporary value is in effect.
    """
    with _lock:
        if not _active or not _live:
            yield
            return
        _set_live(False)
        try:
            yield
        finally:
            _set_live(True)


def is_runtime_expandable_active() -> bool:
    """Return whether runtime expandable segments are currently enabled."""
    with _lock:
        return _active and _live


def _reset_for_testing() -> None:
    """Reset module state; intended for unit tests only."""
    global _prepared, _requested, _active, _live, _base_conf
    with _lock:
        _prepared = False
        _requested = False
        _active = False
        _live = False
        _base_conf = ""
