"""Case actions. Load them explicitly when assembling the default registry."""


def standard_actions():
    from .client_query_qps import client_query_qps
    from .client_query_requests import client_query_requests
    from .client_query_success_rate import client_query_success_rate
    from .client_query_tpot import client_query_tpot
    from .client_query_ttft import client_query_ttft
    from .client_start_playback import client_start_playback
    from .client_stop_playback import client_stop_playback
    from .engine_query_cache_hit_rate import engine_query_cache_hit_rate
    from .engine_query_decode_tps import engine_query_decode_tps
    from .engine_query_prefill_tps import engine_query_prefill_tps
    from .engine_query_prefill_tps_with_cache import engine_query_prefill_tps_with_cache
    from .engine_scale import engine_scale
    from .master_list_addresses import master_list_addresses
    from .master_list_engines import master_list_engines
    from .master_restart import master_restart
    from .master_start import master_start
    from .master_stop import master_stop
    from .prometheus_query import prometheus_query

    return (
        master_list_addresses,
        master_list_engines,
        master_start,
        master_stop,
        master_restart,
        engine_scale,
        prometheus_query,
        engine_query_cache_hit_rate,
        engine_query_prefill_tps,
        engine_query_prefill_tps_with_cache,
        engine_query_decode_tps,
        client_query_qps,
        client_query_ttft,
        client_query_tpot,
        client_query_success_rate,
        client_start_playback,
        client_stop_playback,
        client_query_requests,
    )
