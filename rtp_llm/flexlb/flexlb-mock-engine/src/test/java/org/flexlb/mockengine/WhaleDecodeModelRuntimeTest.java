package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;

class WhaleDecodeModelRuntimeTest {
    @TempDir Path directory;

    @Test void runtimeCoefficientsChangeStepAndAcceptanceWithoutChangingPrefill() throws Exception {
        var scheduler = Executors.newScheduledThreadPool(2);
        Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
        MockControlServer server = null;
        try {
            var model = MockEngineTestSupport.performanceModel(directory, "100", 1);
            int port = 63220;
            for (String role : new String[]{"prefill", "decode"}) {
                services.put(port, new JavaMockEngineCluster.FastRpcService(role + "-0", "127.0.0.1", role,
                        role.equals("prefill") ? EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                                : EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                        port, services, scheduler, model, 100, new JavaMockEngineCluster.ClusterStats(),
                        JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS,
                        JavaMockEngineCluster.DEFAULT_DECODE_MAX_CONCURRENCY));
                port++;
            }
            server = new MockControlServer(services, new ConcurrentHashMap<>(), null, null, "127.0.0.1", 0);
            server.start();
            int httpPort = server.getPort();
            var json = new ObjectMapper();
            String before = MockEngineTestSupport.httpGet(httpPort, "/decode_model");
            assertEquals(1, json.readTree(before).path("engines").size());
            String body = "{\"step_base_ms\":23,\"step_per_running_ms\":0.02,\"tokens_per_step\":2.35}";
            var response = MockEngineTestSupport.httpPostResponse(httpPort, "/decode_model", body);
            assertEquals(200, response.statusCode());
            var state = json.readTree(response.body()).path("engines").path("decode-0");
            assertTrue(state.path("runtime_override").asBoolean());
            assertEquals(2.35, state.path("tokens_per_step").asDouble());
            assertEquals(24, services.get(63221).getPerformance().decodeStepDelayMs(50));
            assertEquals(5, services.get(63221).getPerformance().decodeSteps(10));
            assertFalse(services.get(63220).getPerformance().decodeModelState()
                    .get("runtime_override").equals(true));
            String after = MockEngineTestSupport.httpGet(httpPort, "/decode_model");
            for (String bad : new String[]{"{}", body.replace("2.35", "0"),
                    body.replace("0.02", "-1"), body.replace("\"step_base_ms\"", "\"unknown\""),
                    body.replace("}", ",\"engine\":\"missing\"}")}) {
                assertEquals(400, MockEngineTestSupport.httpPostResponse(httpPort, "/decode_model", bad).statusCode());
                assertEquals(json.readTree(after), json.readTree(MockEngineTestSupport.httpGet(httpPort, "/decode_model")));
            }
        } finally {
            if (server != null) server.stop();
            for (var service : services.values()) service.shutdown();
            scheduler.shutdownNow();
        }
    }
}
