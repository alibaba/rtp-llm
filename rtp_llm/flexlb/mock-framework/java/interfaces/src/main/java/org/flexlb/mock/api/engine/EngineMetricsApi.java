package org.flexlb.mock.api.engine;

import org.flexlb.mock.api.Endpoint;
import org.flexlb.mock.api.model.Types.*;
import static org.flexlb.mock.api.Endpoint.Kind.read;

/** Read interfaces can be bound without advertising topology actions. */
public interface EngineMetricsApi {
    @Endpoint(value="engine/cache/hit_rate", kind=read) MetricValue hitRate(CacheRange args);
    @Endpoint(value="engine/prefill/tps", kind=read) MetricValue prefillTps(EngineRange args);
    @Endpoint(value="engine/prefill/tps_with_cache", kind=read) MetricValue cachedTps(EngineRange args);
    @Endpoint(value="engine/decode/tps", kind=read) MetricValue decodeTps(EngineRange args);
}
