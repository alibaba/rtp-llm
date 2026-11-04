package org.flexlb.mock.api.master;

import org.flexlb.mock.api.Endpoint;
import org.flexlb.mock.api.model.Types.*;
import static org.flexlb.mock.api.Endpoint.Kind.*;

/** Read-only bindings may be available before process-control actions are connected. */
public interface MasterStateApi {
    @Endpoint(value="master/leadership/get", kind=read) MasterSet leadership(None args);
    @Endpoint(value="master/engine/get", kind=read) PerMasterEngines engines(EngineSelection args);
}
