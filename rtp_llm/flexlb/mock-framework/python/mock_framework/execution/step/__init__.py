"""Case actions. Load them explicitly when assembling the default registry."""


def standard_actions():
    from .client_qps import client_qps
    from .client_requests import client_requests
    from .client_start import client_start
    from .client_stop import client_stop
    from .client_success_rate import client_success_rate
    from .client_tpot import client_tpot
    from .client_ttft import client_ttft
    from .engine_cache_hit_rate import engine_cache_hit_rate
    from .engine_decode_tps import engine_decode_tps
    from .engine_prefill_tps import engine_prefill_tps
    from .engine_prefill_tps_with_cache import engine_prefill_tps_with_cache
    from .engine_scale import engine_scale
    from .master_engine_get import master_engine_get
    from .master_get import master_get
    from .master_restart import master_restart
    from .master_start import master_start
    from .master_stop import master_stop
    from .prometheus_query import prometheus_query

    return (
        master_get,
        master_engine_get,
        master_start,
        master_stop,
        master_restart,
        engine_scale,
        prometheus_query,
        engine_cache_hit_rate,
        engine_prefill_tps,
        engine_prefill_tps_with_cache,
        engine_decode_tps,
        client_qps,
        client_ttft,
        client_tpot,
        client_success_rate,
        client_start,
        client_stop,
        client_requests,
    )
