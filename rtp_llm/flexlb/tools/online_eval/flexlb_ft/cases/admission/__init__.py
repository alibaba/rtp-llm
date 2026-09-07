"""admission cases in stable execution order."""

from ...registry import collect_cases
from .admission_batcher_queue_capacity_park import admission_batcher_queue_capacity_park
from .admission_batcher_queue_deadline import admission_batcher_queue_deadline
from .admission_engine_kv_lack_mem_fast_reject import (
    admission_engine_kv_lack_mem_fast_reject,
)
from .admission_engine_waiting_batch_cap_reject import (
    admission_engine_waiting_batch_cap_reject,
)
from .admission_master_capacity_reject import admission_master_capacity
from .admission_placement_pool_wait import admission_placement_pool_wait
from .admission_priority_incomer_reject import admission_priority_incomer_reject
from .admission_queue_depth_reject import admission_queue_depth
from .admission_slo_queue_deadline import admission_slo_deadline
from .engine_decode_hard_gate_unbounded_park import (
    engine_decode_hard_gate_unbounded_park,
)
from .engine_prefill_concurrency_gate_park import engine_prefill_concurrency_gate_park
from .engine_prefill_regroup_disabled_verbatim import (
    engine_prefill_regroup_disabled_verbatim,
)
from .engine_prefill_token_budget_boundary import engine_prefill_token_budget_boundary
from .engine_prefill_token_budget_split import engine_prefill_token_budget_split
from .engine_prefill_token_budget_split_fifo import (
    engine_prefill_token_budget_split_fifo,
)

ADMISSION_CASES = collect_cases(
    "admission",
    [
        admission_queue_depth,
        admission_slo_deadline,
        admission_master_capacity,
        engine_prefill_concurrency_gate_park,
        engine_decode_hard_gate_unbounded_park,
        admission_priority_incomer_reject,
        admission_batcher_queue_capacity_park,
        admission_batcher_queue_deadline,
        admission_placement_pool_wait,
        admission_engine_waiting_batch_cap_reject,
        admission_engine_kv_lack_mem_fast_reject,
        engine_prefill_token_budget_split,
        engine_prefill_token_budget_split_fifo,
        engine_prefill_token_budget_boundary,
        engine_prefill_regroup_disabled_verbatim,
    ],
)
