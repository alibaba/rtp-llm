"""engine_fault cases in stable execution order."""

from ...registry import collect_cases
from .engine_fault_crash_after import inject_crash_after
from .engine_fault_down_phases import engine_down_http_stop_prefill
from .engine_fault_enqueue_delay import inject_enqueue_delay
from .engine_fault_enqueue_error import engine_fault_enqueue_error
from .engine_fault_flap import engine_flap
from .engine_fault_generate_delay import inject_generate_delay
from .engine_fault_no_respond import engine_fault_no_respond
from .engine_fault_recovery_generation_bump import recovery_generation_bump
from .engine_fault_recovery_kv_resync import recovery_kv_resync
from .engine_fault_recovery_kv_usage_reset import recovery_kv_usage_reset
from .engine_fault_recovery_no_resurrect import recovery_no_resurrect
from .engine_fault_status_gap_long_retire import status_gap_long_retire
from .engine_fault_status_gap_no_bump import status_gap_no_bump

ENGINE_FAULT_CASES = collect_cases(
    "engine_fault",
    [
        engine_down_http_stop_prefill,
        engine_flap,
        inject_crash_after,
        engine_fault_no_respond,
        engine_fault_enqueue_error,
        inject_enqueue_delay,
        inject_generate_delay,
        recovery_generation_bump,
        recovery_kv_resync,
        recovery_no_resurrect,
        status_gap_no_bump,
        status_gap_long_retire,
        recovery_kv_usage_reset,
    ],
)
