"""Comparer registry — predicate-driven resolver for smoke test result validation.

Replaces the if/elif chain in case_runner._get_comparer_cls(), which hardcoded
internal-only mainse comparer imports inside an OSS file. With a registry:

- OSS modules register their endpoint/q_r-driven comparers at import time
  (see case_runner.py module-level register_* calls).
- internal_source/rtp_llm/test/smoke/conftest.py registers mainse comparers — only
  loaded when internal_source is on the path (e.g., monorepo runs of
  test_smoke_internal.py).
- OSS-only checkouts that try to run a mainse case get a clear error
  ("comparer not registered: q_r mainse_module=True") instead of a
  ModuleNotFoundError on smoke.mainse.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

Predicate = Callable[[Dict[str, Any], str], bool]

_REGISTRY: List[Tuple[Predicate, Type]] = []
_FALLBACK: Optional[Type] = None


def register_comparer(predicate: Predicate, comparer_cls: Type) -> None:
    """Register a comparer with a predicate over (q_r, request_endpoint).

    Order matters — first-match-wins. Caller controls priority by registration
    order. Conventional ordering: most-specific first (mainse-flag, exact
    endpoint), generic last (OpenaiComparer for "messages" in q_r).
    """
    _REGISTRY.append((predicate, comparer_cls))


def set_default_comparer(comparer_cls: Type) -> None:
    """Set the fallback comparer when no predicate matches."""
    global _FALLBACK
    _FALLBACK = comparer_cls


def resolve_comparer(q_r: Dict[str, Any], request_endpoint: str) -> Type:
    """Resolve the comparer class for a (q_r, endpoint) pair.

    Raises RuntimeError if no predicate matches and no fallback is registered.
    For cases flagged with ``mainse_module=True``, the fallback is never used —
    a missing mainse comparer is a configuration error that must fail loudly
    rather than silently falling back to the OSS comparer.
    """
    for predicate, comparer_cls in _REGISTRY:
        if predicate(q_r, request_endpoint):
            return comparer_cls

    # Cases that require the internal mainse comparer must not silently fall
    # back to the OSS default — that would mask a missing mainse installation.
    is_mainse = q_r.get("mainse_module", False) or q_r.get("mainse", False)
    if is_mainse:
        raise RuntimeError(
            f"mainse comparer not registered for q_r keys={sorted(q_r.keys())} "
            f"endpoint={request_endpoint!r}; the internal mainse smoke package "
            "must be imported before this test ran. Refusing to fall back to "
            "OSS comparer for a mainse-flagged case."
        )

    if _FALLBACK is None:
        raise RuntimeError(
            f"comparer not registered: q_r keys={sorted(q_r.keys())} "
            f"endpoint={request_endpoint!r}; ensure the relevant smoke "
            "package (OSS / internal mainse) was imported before this test ran."
        )
    return _FALLBACK


# Helpers for tests/debugging — not part of the public API.
def _registry_size() -> int:
    return len(_REGISTRY)


def _reset_for_tests() -> None:
    global _FALLBACK
    _REGISTRY.clear()
    _FALLBACK = None


def _try_register_mainse_comparers() -> None:
    """Auto-register internal mainse comparers when the internal package exists.

    Registered before any OSS fallback so mainse-flagged cases pick the
    internal comparer first.
    """
    try:
        from rtp_llm.test.smoke.mainse.mainse_comparer import (
            MainseDecodeArpcComparer,
            MainseEmbeddingArpcComparer,
        )

        register_comparer(
            lambda q_r, ep: q_r.get("mainse_module", False)
            or q_r.get("mainse", False),
            MainseDecodeArpcComparer,
        )
        register_comparer(
            lambda q_r, ep: q_r.get("mainse_arpc", False),
            MainseEmbeddingArpcComparer,
        )
    except (ImportError, ModuleNotFoundError):
        # OSS-only checkouts legitimately lack internal mainse comparers.
        pass


_try_register_mainse_comparers()
