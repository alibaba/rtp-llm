"""RPC outcome classification for the dash_sc gRPC access path.

Pure functions that map a gRPC outcome — the handler exception, the context
status code, or a backend ``error_message`` frame — to the bounded
``(status, status_detail)`` tokens recorded on each access-log line and emitted
as the kmonitor ``error_code`` tag. Inputs are the raw gRPC signals, outputs are
classified strings: no record / logger / kmonitor dependency, so the policy is
shared by both servicers and unit-testable in isolation.

``GrpcAccessRecord.resolve_status`` is the sole caller — it collects the context /
exception inputs and writes the returned classification onto its own ``status``
fields. This module owns *how* an outcome is classified; the record owns *that*
it has a status.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import grpc


def rpc_code(exc: BaseException) -> Optional[Any]:
    try:
        return exc.code()
    except Exception:
        return None


def rpc_details(exc: BaseException) -> Optional[str]:
    try:
        return exc.details()
    except Exception:
        return None


# grpcio surfaces request-iterator exceptions under this exact ``details``
# string — matched verbatim rather than by regex so an unrelated error whose
# message happens to contain "iterating requests" doesn't silently get
# reclassified as a client fault.
_CLIENT_REQ_ITER_DETAILS = "Exception iterating requests!"


def classify_rpc_exception(
    exc: BaseException, *, req_count: int
) -> tuple[str, Optional[str]]:
    """Map a server-side RPC exception to ``(status, status_detail)``.

    ``req_count`` is the number of request frames seen so far — the only record
    state this classifier needs (to tell "peer closed before sending a frame"
    apart from a genuine error).

    Precedence (narrowest first):
    1. ``GeneratorExit`` — gRPC framework closed our generator because the
       client handler terminated. Route to CANCELLED so benign disconnects
       don't pollute the error bucket.
    2. ``details == "Exception iterating requests!"`` — grpcio's fixed
       marker for a failed client request iterator; code is UNKNOWN but
       semantics are "client gone", so override to CANCELLED.
    3. Bare ``RpcError()`` (no ``code()``, no ``details()``) on a
       frame-less RPC — backend-frontend view of "peer closed before
       sending a frame". CANCELLED.
    4. ``RpcError`` with a real ``code()`` — use ``code.name`` directly
       (CANCELLED / UNAVAILABLE / DEADLINE_EXCEEDED / …). This is the
       forwarder's common path; before this existed, every
       ``_MultiThreadedRendezvous`` dumped into UNKNOWN.
    5. Everything else — UNKNOWN.
    """
    if isinstance(exc, GeneratorExit):
        return "CANCELLED", "client closed generator"
    if isinstance(exc, asyncio.CancelledError):
        return "CANCELLED", "peer cancelled (async)"
    if not isinstance(exc, grpc.RpcError):
        return f"UNKNOWN_{type(exc).__name__}", repr(exc)

    code = rpc_code(exc)
    details = rpc_details(exc)

    if details == _CLIENT_REQ_ITER_DETAILS:
        return "CANCELLED", "client request iterator failed"
    if code is None and not details and req_count == 0:
        return "CANCELLED", "peer closed before request arrived"
    if code is not None and code != grpc.StatusCode.OK:
        return code.name, details or code.name
    # Preserve the Python class name so ``error_code`` on Grafana is actionable
    # (``UNKNOWN_RuntimeError`` / ``UNKNOWN_ValueError`` / …) instead of a
    # single opaque ``UNKNOWN`` bucket — class names are bounded (few dozen at
    # most) so tag cardinality stays tight.
    return f"UNKNOWN_{type(exc).__name__}", repr(exc)


# Short-form classifiers for free-form backend ``error_message`` strings.
# Matched in order; first hit wins. Kept intentionally small — every entry
# becomes a permanent Grafana ``error_code`` tag bucket, so new entries need
# to correspond to a distinct operational signal.
_ERROR_MESSAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("BACKEND_EMPTY_OUTPUTS", "empty outputs_list"),
)


def classify_error_message(msg: Optional[str]) -> str:
    """Map a backend ``error_message`` to a bounded ``error_code`` token.

    Primary path: the backend (``inference/servicer.py``
    ``iter_real_model_stream_infer``) formats exceptions as
    ``f"{type(e).__name__}: {e}"`` so the leading class name before ``":"`` is a
    stable, bounded categorizer — dozens of Python exception classes at most, no
    free-form cardinality blow-up. We extract that prefix and return
    ``BACKEND_<ClassName>``.

    Fallback path: bare phrases emitted without a type prefix (e.g.
    ``"empty outputs_list from backend"`` for the zero-chunk case) map through
    ``_ERROR_MESSAGE_PATTERNS``.

    Everything else collapses to ``BACKEND_INTERNAL`` — still more specific
    than the old blanket ``INTERNAL`` because the status itself now signals
    "backend error_message channel" vs "gRPC transport error".
    """
    if not msg:
        return "BACKEND_INTERNAL"
    head = msg.strip().split("\n", 1)[0]
    colon = head.find(":")
    if 0 < colon <= 48:
        prefix = head[:colon].strip()
        if prefix and all(c.isalnum() or c == "_" for c in prefix):
            return f"BACKEND_{prefix}"
    for code, needle in _ERROR_MESSAGE_PATTERNS:
        if needle in msg:
            return code
    return "BACKEND_INTERNAL"
