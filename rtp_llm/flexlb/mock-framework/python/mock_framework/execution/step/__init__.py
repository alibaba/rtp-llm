"""Case actions. Load them explicitly when assembling the default registry."""


def standard_actions():
    from .engine import engine_scale, engine_stop
    from .frontend import (
        frontend_playback_requests,
        frontend_start_playback,
        frontend_stop_playback,
    )
    from .master import (
        master_engine_count,
        master_list_addresses,
        master_list_engines,
        master_pv_requests,
        master_restart,
        master_start,
        master_stop,
    )
    from .metrics.engine import (
        query_cache_hit_rate,
        query_decode_tps,
        query_prefill_tps,
        query_prefill_tps_with_cache,
    )
    from .metrics.frontend import query_qps, query_success_rate, query_tpot, query_ttft
    from .metrics.prometheus import query_promql, query_promql_scalar

    return (
        master_list_addresses,
        master_list_engines,
        master_engine_count,
        master_pv_requests,
        master_start,
        master_stop,
        master_restart,
        engine_scale,
        engine_stop,
        query_promql,
        query_promql_scalar,
        query_cache_hit_rate,
        query_prefill_tps,
        query_prefill_tps_with_cache,
        query_decode_tps,
        query_qps,
        query_ttft,
        query_tpot,
        query_success_rate,
        frontend_start_playback,
        frontend_stop_playback,
        frontend_playback_requests,
    )
