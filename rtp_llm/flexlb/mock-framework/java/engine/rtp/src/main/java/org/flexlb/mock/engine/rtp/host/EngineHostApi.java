package org.flexlb.mock.engine.rtp.host;

import java.util.List;
import org.flexlb.mock.api.Endpoint;
import org.flexlb.mock.api.model.Types.None;
import static org.flexlb.mock.api.Endpoint.Kind.*;

/** Host-local operations. Selection and desired replica counts belong to Python. */
public interface EngineHostApi {
    record Create(String id, List<String> unavailable) {
        public Create { valid(id); unavailable = List.copyOf(unavailable); }
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
