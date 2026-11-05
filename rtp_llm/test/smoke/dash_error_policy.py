"""Shared DashScope error-spec extraction and transient-retry policy.

Both DashSc channels (the plain Dash gRPC comparer and the DashSc-channel
comparer) talk to the same engine and observe the same engine-side
rejections, so the error-spec reader and the retry policy live here to keep
the two channels from drifting apart.
"""

import json
import time
from typing import Any, Dict, Optional

# Engine-side transient rejections seen in CI (overload, admission control,
# capacity preemption). Retried a bounded number of times inside a wall-clock
# budget; deterministic rejections (invalid parameter, schema violations,
# request-level errors) are never retried.
DASH_TRANSIENT_STATUS_CODES = frozenset({429, 503, 504})
DASH_MAX_ATTEMPTS = 3
DASH_RETRY_INTERVAL_S = 5.0
# Each attempt may itself block for the gRPC timeout, so the loop is bounded
# by wall time as well as by attempt count.
DASH_RETRY_BUDGET_S = 120.0


def dash_error_spec(resp: Any) -> Optional[Dict[str, Any]]:
    """Return the DashSc error spec carried by a stream response, if any.

    Engine-side rejections (grammar validation overload/timeout/unavailable,
    invalid parameter, capacity, abort, ...) arrive as
    ``finish_reason=USE_PARAMETER_STATUS`` with an explicit status described by
    ``infer_response.parameters["error_msg"]`` plus standalone status
    parameters.  Reading them here keeps the real error visible instead of
    letting a downstream grammar/response check hide it.
    """
    try:
        params = resp.infer_response.parameters
        raw = params.get("error_msg")
        if raw is None or not raw.string_param:
            return None
        spec = json.loads(raw.string_param)
        if not isinstance(spec, dict):
            return None
        for name, key in (
            ("status_code", "int64_param"),
            ("status_name", "string_param"),
            ("status_message", "string_param"),
        ):
            parameter = params.get(name)
            if parameter is not None and parameter.HasField(key):
                spec.setdefault(name, getattr(parameter, key))
        return spec
    except Exception:  # noqa: BLE001 - diagnostics must never break the stream
        return None


def should_retry_transient(
    *,
    retryable: bool,
    status_code: Optional[int],
    attempts: int,
    deadline: float,
) -> bool:
    """Whether a transient-retry loop should sleep and run another attempt.

    ``retryable`` carries the case-level veto (a golden that asserts an error,
    or a cold-cache golden whose first attempt already published blocks) so
    both DashSc channels share one decision function.
    """
    return (
        retryable
        and status_code in DASH_TRANSIENT_STATUS_CODES
        and attempts < DASH_MAX_ATTEMPTS
        and time.monotonic() < deadline
    )
