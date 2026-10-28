package org.flexlb.mock.api.master;

import org.flexlb.mock.api.Endpoint;
import org.flexlb.mock.api.model.Types.*;
import static org.flexlb.mock.api.Endpoint.Kind.*;

public interface MasterApi extends MasterStateApi {
    @Endpoint(value="master/process/stop", kind=action) MasterReceipt stop(None args);
    @Endpoint(value="master/process/start", kind=action) MasterReceipt start(Start args);
}
