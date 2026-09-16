package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import static org.junit.jupiter.api.Assertions.*;

class WhaleOutputLengthRuntimeTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String EOS = "{\"enabled\":true,\"distribution\":\"geometric\",\"mean_tokens\":1,\"seed\":42}";
    @TempDir Path directory;

    @Test void updateChangesNewShapesButPreservesExistingTargetsAndRequestContracts() throws Exception {
        var model = MockEngineTestSupport.performanceModel(directory, "100", 1);
        var peer = model.forEngine();
        var cache = new MockLruBlockCache(100);
        cache.admit(List.of(10L, 20L));
        var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(1)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(100)).build();
        var old = model.shape(input, cache);
        model.setEosModel(MockEosModel.fromControl(JSON.readTree(EOS)));
        assertEquals(100, old.outputLen());
        assertEquals(1, model.shape(input, cache).outputLen());
        assertEquals(100, peer.shape(input, cache).outputLen());
        assertEquals(1, model.forEngine().shape(input, cache).outputLen());
        assertEquals(2, cache.lruKeyBlocks());
        assertEquals(100, model.prefillMs(List.of(old)));
        var min = input.toBuilder().setGenerateConfig(input.getGenerateConfig().toBuilder().setMinNewTokens(9)).build();
        assertEquals(9, model.shape(min, cache).outputLen());
        var ignore = input.toBuilder().setGenerateConfig(input.getGenerateConfig().toBuilder().setIgnoreEos(true)).build();
        assertEquals(100, model.shape(ignore, cache).outputLen());
        var replay = input.toBuilder().setGenerateConfig(input.getGenerateConfig().toBuilder()
                .setUniqueKey("flexlb_eval:{\"output_len\":37}")).build();
        assertEquals(37, model.shape(replay, cache).outputLen());
        model.setEosModel(MockEosModel.fromControl(JSON.readTree("{\"enabled\":false}")));
        assertEquals(100, model.shape(input, cache).outputLen());
    }

    @Test void httpUpdatesBothRolesAndInvalidReplacementIsNonMutating() throws Exception {
        var scheduler = Executors.newScheduledThreadPool(2);
        Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
        MockControlServer server = null;
        try {
            var model = MockEngineTestSupport.performanceModel(directory, "100", 1);
            int port = 63210;
            for (String role : List.of("prefill", "decode")) {
                services.put(port, new JavaMockEngineCluster.FastRpcService(role + "-0", "127.0.0.1", role,
                        role.equals("prefill") ? EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL : EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                        port, services, scheduler, model, 100, new JavaMockEngineCluster.ClusterStats(),
                        JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS, JavaMockEngineCluster.DEFAULT_DECODE_MAX_CONCURRENCY));
                port++;
            }
            server = new MockControlServer(services, new ConcurrentHashMap<>(), null, null, "127.0.0.1", 0);
            server.start();
            int httpPort = server.getPort();
            var response = MockEngineTestSupport.httpPostResponse(httpPort, "/output_length", "{\"eos\":" + EOS + "}");
            assertEquals(200, response.statusCode());
            var states = JSON.readTree(response.body()).path("engines");
            assertEquals(2, states.size());
            for (var state : states) {
                assertTrue(state.path("runtime_override").asBoolean());
                assertEquals(1, state.path("eos").path("mean_tokens").asInt());
            }
            String before = MockEngineTestSupport.httpGet(httpPort, "/output_length");
            for (String body : List.of("{}", "{\"eos\":{\"enabled\":true}}",
                    "{\"eos\":" + EOS.replace("\"mean_tokens\":1", "\"mean_tokens\":0") + "}",
                    "{\"eos\":" + EOS.replace("\"seed\":42", "\"seed\":1.5") + "}",
                    "{\"eos\":" + EOS + ",\"mean_token\":7}",
                    "{\"eos\":" + EOS + ",\"engine\":\"missing\"}")) {
                assertEquals(400, MockEngineTestSupport.httpPostResponse(httpPort, "/output_length", body).statusCode());
                assertEquals(JSON.readTree(before), JSON.readTree(MockEngineTestSupport.httpGet(httpPort, "/output_length")));
            }
            assertEquals(200, MockEngineTestSupport.httpPostResponse(httpPort, "/output_length",
                    "{\"engine\":\"prefill-0\",\"eos\":{\"enabled\":false}}").statusCode());
            states = JSON.readTree(MockEngineTestSupport.httpGet(httpPort, "/output_length")).path("engines");
            assertFalse(states.path("prefill-0").path("eos").path("enabled").asBoolean());
            assertTrue(states.path("decode-0").path("eos").path("enabled").asBoolean());
        } finally {
            if (server != null) server.stop();
            for (var service : services.values()) service.shutdown();
            scheduler.shutdownNow();
        }
    }
}
