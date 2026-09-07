from __future__ import annotations

from ...context import CaseContext
from ...registry import case
from ...support.engine_fault import TIMEOUT_WAIT_S, _anomaly_error_case


@case("engine_fault_no_respond", category="engine_fault", source="anomaly_smoke.py E2")
def engine_fault_no_respond(ctx: CaseContext):
    return _anomaly_error_case(ctx, {"no_respond": True}, TIMEOUT_WAIT_S, False)
