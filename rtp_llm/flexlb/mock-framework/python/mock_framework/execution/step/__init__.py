"""Case actions. Load them explicitly when assembling the default registry."""


def standard_actions():
    from .master_get import MasterGet
    from .master_engine_get import MasterEngineGet
    from .master_start import MasterStart
    from .master_stop import MasterStop
    from .master_restart import MasterRestart
    from .engine_scale import EngineScale
    from .prometheus_query import PrometheusQuery
    from .engine_cache_hit_rate import EngineCacheHitRate
    from .engine_prefill_tps import EnginePrefillTps
    from .engine_prefill_tps_with_cache import EnginePrefillTpsWithCache
    from .engine_decode_tps import EngineDecodeTps
    from .client_qps import ClientQps
    from .client_ttft import ClientTtft
    from .client_tpot import ClientTpot
    from .client_success_rate import ClientSuccessRate
    from .client_start import ClientStart
    from .client_stop import ClientStop
    from .client_requests import ClientRequests

    return (
        MasterGet,
        MasterEngineGet,
        MasterStart,
        MasterStop,
        MasterRestart,
        EngineScale,
        PrometheusQuery,
        EngineCacheHitRate,
        EnginePrefillTps,
        EnginePrefillTpsWithCache,
        EngineDecodeTps,
        ClientQps,
        ClientTtft,
        ClientTpot,
        ClientSuccessRate,
        ClientStart,
        ClientStop,
        ClientRequests,
    )
