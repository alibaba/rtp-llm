"""Case actions. Load them explicitly when assembling the default registry."""


def standard_actions():
    from .engine import engine_scale
    from .frontend import frontend_start_playback, frontend_stop_playback
    from .master import (
        master_list_addresses,
        master_list_engines,
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
    from .metrics.prometheus import query_promql

    return (
        master_list_addresses,
        master_list_engines,
        master_start,
        master_stop,
        master_restart,
        engine_scale,
        query_promql,
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
    )
