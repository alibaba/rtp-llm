"""Comparer registry — predicate-driven resolver for smoke test result validation.

Replaces the if/elif chain in case_runner._get_comparer_cls(), which hardcoded
internal-only mainse comparer imports inside an OSS file. With a registry:

- OSS modules register their endpoint/q_r-driven comparers at import time
  (see case_runner.py module-level register_* calls).
- Internal MainSE comparers are auto-registered when their overlay package is
  available.
- OSS-only checkouts that try to run a mainse case get a clear error
  ("comparer not registered: q_r mainse_module=True") instead of a
  ModuleNotFoundError on smoke.mainse.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
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


def _is_mainse(q_r: Dict[str, Any]) -> bool:
    return bool(q_r.get("mainse_module", False) or q_r.get("mainse", False))


def _register_mainse_comparer_classes(
    decode_arpc_cls: Type, embedding_arpc_cls: Type, mainse_cls: Type
) -> None:
    """Register MainSE modes from most specific to the generic HTTP mode."""
    register_comparer(
        lambda q_r, ep: _is_mainse(q_r) and q_r.get("use_decode_arpc", False),
        decode_arpc_cls,
    )
    register_comparer(
        lambda q_r, ep: _is_mainse(q_r) and q_r.get("use_emb_arpc", False),
        embedding_arpc_cls,
    )
    register_comparer(lambda q_r, ep: _is_mainse(q_r), mainse_cls)


def _install_legacy_smoke_import_aliases() -> None:
    """Map internal comparers' legacy ``smoke.*`` imports to this package.

    The internal MainSE modules predate the ``rtp_llm.test.smoke`` package
    migration. Aliasing their two direct dependencies avoids loading duplicate
    module objects with separate registry and data-root state.
    """
    smoke_package = importlib.import_module("rtp_llm.test.smoke")
    sys.modules.setdefault("smoke", smoke_package)
    for module_name in ("base_comparer", "common_def"):
        canonical_name = f"rtp_llm.test.smoke.{module_name}"
        module = importlib.import_module(canonical_name)
        sys.modules.setdefault(f"smoke.{module_name}", module)


def _try_register_mainse_comparers() -> None:
    """Auto-register internal mainse comparers when the internal package exists.

    Registered before any OSS fallback so mainse-flagged cases pick the
    internal comparer first.
    """
    module_name = "rtp_llm.test.smoke.mainse.mainse_comparer"
    try:
        module_spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        module_spec = None
    if module_spec is None:
        # OSS-only checkouts legitimately lack internal MainSE comparers.
        return

    _install_legacy_smoke_import_aliases()

    # Once the internal package is present, imports must fail loudly. Silently
    # swallowing an ImportError here only defers it into a misleading
    # "comparer not registered" error after the model server has started.
    from rtp_llm.test.smoke.mainse.mainse_comparer import MainseComparer
    from rtp_llm.test.smoke.mainse.mainse_decode_arpc_comparer import (
        MainseDecodeArpcComparer,
    )
    from rtp_llm.test.smoke.mainse.mainse_embedding_arpc_comparer import (
        MainseEmbeddingArpcComparer,
    )

    _register_mainse_comparer_classes(
        MainseDecodeArpcComparer,
        MainseEmbeddingArpcComparer,
        MainseComparer,
    )


_try_register_mainse_comparers()
