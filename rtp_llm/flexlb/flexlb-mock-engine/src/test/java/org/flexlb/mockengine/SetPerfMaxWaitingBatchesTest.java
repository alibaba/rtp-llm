package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mockengine.MockEngineTestSupport.*;

class SetPerfMaxWaitingBatchesTest {
    @TempDir Path directory;
    @Test void calibrationEndpointCannotChangeBehavior() throws Exception {
        var scheduler = Executors.newScheduledThreadPool(1);
        var services = new ConcurrentHashMap<Integer, JavaMockEngineCluster.FastRpcService>();
        var service = new JavaMockEngineCluster.FastRpcService("prefill", org.flexlb.engine.grpc.EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                63250, services, scheduler, performanceModel(directory, "1"), 16, new JavaMockEngineCluster.ClusterStats());
        services.put(63250, service);
        var control = new MockControlServer(services, new ConcurrentHashMap<>(), null, null, "127.0.0.1", 63249);
        control.start();
        try {
            for (String field : new String[]{"max_waiting_batches", "max_prefill_concurrency", "cache_retention_blocks", "decode_retry_times"})
                assertEquals(400, httpPostResponse(63249, "/set_perf", "{\"port\":63250,\"" + field + "\":1}").statusCode());
            assertEquals(200, httpPostResponse(63249, "/set_perf", "{\"port\":63250,\"prefill_fixed_ms\":2}").statusCode());
        } finally { control.stop(); service.shutdown(); scheduler.shutdownNow(); }
    }
}
