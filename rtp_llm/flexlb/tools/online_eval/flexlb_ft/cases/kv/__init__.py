"""kv cases in stable execution order."""

from ...registry import collect_cases
from .kv_capacity_conflict_overflow import kv_capacity_conflict_overflow
from .kv_decode_capacity_park import kv_decode_capacity_park
from .kv_decode_pool_exhaustion_terminal import kv_decode_pool_exhaustion_terminal
from .kv_g_engine_down_cleanup import kv_g_engine_down_cleanup
from .kv_g_full_release_no_ghost import kv_g_full_release_no_ghost
from .kv_g_partial_release_redirect import kv_g_partial_release_redirect
from .kv_g_shared_block_both_match import kv_g_shared_block_both_match
from .kv_g_sync_convergence import kv_g_sync_convergence
from .kv_hot_prefix_tension import kv_hot_prefix_tension
from .kv_leader_saturation_spill import kv_leader_saturation_spill
from .kv_lru_eviction_affinity import kv_lru
from .kv_match_mixed import kv_match_mixed
from .kv_pe_admit_isolation import kv_pe_admit_isolation
from .kv_pe_evict_zero_match import kv_pe_evict_zero_match
from .kv_pe_prefix_continuity import kv_pe_prefix_continuity
from .kv_pool_saturation_evict_reject_recover import (
    kv_pool_saturation_evict_reject_recover,
)
from .kv_prefix_stickiness import kv_prefix_stickiness
from .kv_storm_hot_churn import kv_storm_hot_churn

KV_CASES = collect_cases(
    "kv",
    [
        kv_pe_admit_isolation,
        kv_pe_evict_zero_match,
        kv_pe_prefix_continuity,
        kv_g_shared_block_both_match,
        kv_g_partial_release_redirect,
        kv_g_full_release_no_ghost,
        kv_g_sync_convergence,
        kv_g_engine_down_cleanup,
        kv_storm_hot_churn,
        kv_capacity_conflict_overflow,
        kv_prefix_stickiness,
        kv_hot_prefix_tension,
        kv_match_mixed,
        kv_lru,
        kv_decode_capacity_park,
        kv_pool_saturation_evict_reject_recover,
        kv_decode_pool_exhaustion_terminal,
        kv_leader_saturation_spill,
    ],
)
