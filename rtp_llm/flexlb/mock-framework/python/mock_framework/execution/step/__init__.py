"""Case actions. Load them explicitly when assembling the default registry."""


def standard_actions():
    from .client_start_playback import client_start_playback
    from .client_stop_playback import client_stop_playback
    from .engine_scale import engine_scale
    from .master_list_addresses import master_list_addresses
    from .master_list_engines import master_list_engines
    from .master_restart import master_restart
    from .master_start import master_start
    from .master_stop import master_stop
    from .metrics.client import query_qps, query_success_rate, query_tpot, query_ttft
    from .metrics.engine import (
        query_cache_hit_rate,
        query_decode_tps,
        query_prefill_tps,
        query_prefill_tps_with_cache,
    )
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
        client_start_playback,
        client_stop_playback,
    )
