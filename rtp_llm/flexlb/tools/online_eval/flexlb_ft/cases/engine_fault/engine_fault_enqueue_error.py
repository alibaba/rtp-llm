from __future__ import annotations

from ...context import CaseContext
from ...registry import case
from ...support.engine_fault import ANOMALY_STREAM_TIMEOUT_S, _anomaly_error_case


@case(
    "engine_fault_enqueue_error", category="engine_fault", source="anomaly_smoke.py E3"
)
def engine_fault_enqueue_error(ctx: CaseContext):
    return _anomaly_error_case(
        ctx, {"enqueue_error": True}, ANOMALY_STREAM_TIMEOUT_S, True
    )
