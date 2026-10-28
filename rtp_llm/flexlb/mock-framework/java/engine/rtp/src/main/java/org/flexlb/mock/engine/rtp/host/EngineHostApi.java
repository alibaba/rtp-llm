package org.flexlb.mock.engine.rtp.host;

import java.util.List;
import org.flexlb.mock.api.Endpoint;
import org.flexlb.mock.api.model.Types.None;
import static org.flexlb.mock.api.Endpoint.Kind.*;

/** Host-local operations. Selection and desired replica counts belong to Python. */
public interface EngineHostApi {
    record Create(String id, int rpcPort, int controlPort) {
        public Create {
            valid(id);
            if (rpcPort < 1 || rpcPort > 65535 || controlPort < 1 || controlPort > 65535 || rpcPort == controlPort)
                throw new IllegalArgumentException("Distinct valid RPC and control ports required");
        }
    }
    record Stop(String id) { public Stop { valid(id); } }
    record Instance(String id, String role, String rpc, String control, String metrics) {}
    @Endpoint(value="host/create", kind=action) Instance create(Create args) throws Exception;
    @Endpoint(value="host/stop", kind=action) Instance stop(Stop args) throws Exception;
    @Endpoint(value="host/status", kind=read) List<Instance> status(None args);
    private static void valid(String id) {
        if (id == null || !id.matches("(prefill|decode)-[0-9]{1,9}"))
            throw new IllegalArgumentException("Invalid Engine instance id");
    }
}
