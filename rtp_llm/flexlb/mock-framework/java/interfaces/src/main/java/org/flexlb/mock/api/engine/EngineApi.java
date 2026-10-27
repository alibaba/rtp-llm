package org.flexlb.mock.api.engine;

import org.flexlb.mock.api.Endpoint;
import org.flexlb.mock.api.model.Types.*;
import static org.flexlb.mock.api.Endpoint.Kind.*;

public interface EngineApi extends EngineMetricsApi {
    @Endpoint(value="engine/topology/scale_in", kind=action) ScaleReceipt scaleIn(Scale args);
}
