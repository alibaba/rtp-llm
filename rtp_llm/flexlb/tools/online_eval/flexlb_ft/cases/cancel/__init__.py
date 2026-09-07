"""cancel cases in stable execution order."""

from ...registry import collect_cases
from .cancel_after_terminal import cancel_after_terminal
from .cancel_anomaly_path import cancel_anomaly_path
from .cancel_basic import cancel_basic
from .cancel_deadline_exempt_inflight import cancel_deadline_exempt_inflight
from .cancel_decode_retire_closes_fence import cancel_decode_retire_closes_fence
from .cancel_engine_notfound_settle import cancel_engine_notfound_settle
from .cancel_engine_restarted_tombstoned_settle import (
    cancel_engine_restarted_tombstoned_settle,
)
from .cancel_fencing_lost_on_engine_restart import cancel_fencing_lost_on_engine_restart
from .cancel_idempotent import cancel_idempotent
from .cancel_phase_timing import cancel_phase_timing
from .cancel_preemption_victim import cancel_preemption_victim
from .cancel_prefill_dead_await_terminal import cancel_prefill_dead_await_terminal
from .cancel_schedule_drop_delivered import cancel_schedule_drop_delivered
from .cancel_sibling_isolation import cancel_sibling_isolation
from .cancel_stream_break_decode_autonomous import cancel_stream_break_decode_autonomous
from .cancel_stream_break_prefill_autonomous import (
    cancel_stream_break_prefill_autonomous,
)
from .cancel_transport_failure_one_shot import cancel_transport_failure_one_shot
from .cancel_unexpected_status_await_terminal import (
    cancel_unexpected_status_await_terminal,
)
from .cancel_unknown_rid import cancel_unknown_rid

CANCEL_CASES = collect_cases(
    "cancel",
    [
        cancel_basic,
        cancel_idempotent,
        cancel_sibling_isolation,
        cancel_after_terminal,
        cancel_unknown_rid,
        cancel_phase_timing,
        cancel_anomaly_path,
        cancel_deadline_exempt_inflight,
        cancel_schedule_drop_delivered,
        cancel_engine_notfound_settle,
        cancel_preemption_victim,
        cancel_stream_break_prefill_autonomous,
        cancel_stream_break_decode_autonomous,
        cancel_engine_restarted_tombstoned_settle,
        cancel_prefill_dead_await_terminal,
        cancel_decode_retire_closes_fence,
        cancel_fencing_lost_on_engine_restart,
        cancel_transport_failure_one_shot,
        cancel_unexpected_status_await_terminal,
    ],
)
