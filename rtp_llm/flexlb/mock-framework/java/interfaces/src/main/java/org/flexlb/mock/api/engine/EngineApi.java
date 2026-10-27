package org.flexlb.mock.api.engine;

import org.flexlb.mock.api.Endpoint;
import org.flexlb.mock.api.model.Types.*;
import static org.flexlb.mock.api.Endpoint.Kind.*;

public interface EngineApi {
    @Endpoint(value="engine/topology/scale_in", kind=action) ScaleReceipt scaleIn(Scale args);
    @Endpoint(value="engine/cache/hit_rate", kind=read) MetricValue hitRate(CacheRange args);
    @Endpoint(value="engine/prefill/tps", kind=read) MetricValue prefillTps(EngineRange args);
    @Endpoint(value="engine/prefill/tps_with_cache", kind=read) MetricValue cachedTps(EngineRange args);
    @Endpoint(value="engine/decode/tps", kind=read) MetricValue decodeTps(EngineRange args);
}
