"""priority cases in stable execution order."""

from ...registry import collect_cases
from .atpm_comparator_frozen_weak import atpm_comparator_frozen_weak
from .atpm_config_strict_reject import atpm_config_strict_reject
from .atpm_decode_reservation_priority import atpm_decode_reservation_priority
from .atpm_error_code_family import atpm_error_code_family
from .atpm_observability_integrity import atpm_observability_integrity
from .atpm_preempt_cancel_not_found import atpm_preempt_cancel_not_found
from .atpm_preempt_cancel_tombstoned import atpm_preempt_cancel_tombstoned
from .atpm_preempt_decode_engine_owned import atpm_preempt_decode_engine_owned
from .atpm_preempt_decode_reserved_live import atpm_preempt_decode_reserved_live
from .atpm_preempt_prefill_queued import atpm_preempt_prefill_queued
from .atpm_preempt_prefill_queued_live import atpm_preempt_prefill_queued_live
from .atpm_preemption_disabled_zero_eviction import (
    atpm_preemption_disabled_zero_eviction,
)
from .atpm_same_priority_zero_eviction import atpm_same_priority_zero_eviction
from .atpm_timeout_attribution import atpm_timeout_attribution
from .prio_low_no_starvation import prio_low_no_starvation
from .prio_normalize import prio_normalize
from .prio_order_basic import prio_order_basic
from .prio_queue_timeout_terminal import prio_queue_timeout_terminal
from .prio_same_level_fifo import prio_same_level_fifo

PRIORITY_CASES = collect_cases(
    "priority",
    [
        prio_order_basic,
        prio_same_level_fifo,
        prio_normalize,
        prio_low_no_starvation,
        prio_queue_timeout_terminal,
        atpm_preempt_prefill_queued,
        atpm_preempt_decode_engine_owned,
        atpm_same_priority_zero_eviction,
        atpm_preemption_disabled_zero_eviction,
        atpm_timeout_attribution,
        atpm_comparator_frozen_weak,
        atpm_error_code_family,
        atpm_config_strict_reject,
        atpm_decode_reservation_priority,
        atpm_observability_integrity,
        atpm_preempt_prefill_queued_live,
        atpm_preempt_decode_reserved_live,
        atpm_preempt_cancel_not_found,
        atpm_preempt_cancel_tombstoned,
    ],
)
