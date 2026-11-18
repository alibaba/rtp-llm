package org.flexlb.mock.engine.cache;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.Map;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.scheduler.LLMMockEngineTest;
import org.flexlb.mock.protocol.Role;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks that Remote Cache identity follows deployment and cache compatibility, not the Pod lifecycle.
 */
class KvcmRemoteCacheTest {
    @Test void replicasAndRestartsShareIdentityButDeploymentsAndLayoutsDoNot() throws Exception {
        var config = config();
        var env = Map.of("BIZ_NAME", "mock-engine", "DEPLOYMENT_NAME", "dsv41", "HIPPO_POD_IP", "10.0.0.1");
        var first = KvcmRemoteCache.registration(config, env);
        assertEquals(first, KvcmRemoteCache.registration(config(), Map.of(
            "BIZ_NAME", "mock-engine", "DEPLOYMENT_NAME", "dsv41", "HIPPO_POD_IP", "10.0.0.2")));
        assertNotEquals(first.path("instance_id"), KvcmRemoteCache.registration(config, Map.of(
            "BIZ_NAME", "mock-engine", "DEPLOYMENT_NAME", "other")).path("instance_id"));
        var changed = root();
        ((ObjectNode) changed.path("component").path("config")).put("prefill_cp_size", 4);
        assertNotEquals(first.path("instance_id"), KvcmRemoteCache.registration(RtpConfig.load(changed), env).path("instance_id"));
        assertEquals(config.tokensPerCacheKey(), first.path("block_size").asInt());
        assertEquals("default", first.path("instance_group").asText());
    }

    @Test void nativeGroupAndSaltAreAcceptedFromWhaleEnvironment() throws Exception {
        var config = RtpConfig.fromEnvironment("p-1", Role.prefill, Map.of(
            "TEST_BLOCK_NUM", "100", "MOCK_COMPUTE_FORWARD_EXPRESSION", "1", "RECO_INSTANCE_GROUP", "shared", "RECO_INSTANCE_ID_SALT", "scope"));
        var registration = KvcmRemoteCache.registration(config, Map.of());
        assertEquals("shared", registration.path("instance_group").asText());
        assertNotEquals(registration.path("instance_id"), KvcmRemoteCache.registration(config(), Map.of()).path("instance_id"));
    }

    @Test void transientReadFailureDoesNotDisableFollowingRemoteLookup() throws Exception {
        var json = new ObjectMapper();
        var calls = new java.util.concurrent.atomic.AtomicInteger();
        var server = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress("127.0.0.1", 0), 0);
        var registered = new java.util.concurrent.CountDownLatch(1);
        server.createContext("/api/", exchange -> {
            try (exchange) {
                var response = json.createObjectNode();
                response.putObject("header").putObject("status").put("code", "OK");
                if (exchange.getRequestURI().getPath().equals("/api/registerInstance")) registered.countDown();
                else if (calls.incrementAndGet() == 1) {
                    exchange.sendResponseHeaders(503, -1);
                    return;
                } else response.put("cache_location_len", "1");
                byte[] bytes = json.writeValueAsBytes(response);
                exchange.sendResponseHeaders(200, bytes.length);
                exchange.getResponseBody().write(bytes);
            }
        });
        server.start();
        var root = root();
        ((ObjectNode) root.path("component").path("config"))
            .put("enable_remote_cache", true).put("reuse_cache", true)
            .put("mock.cache.remote.endpoint", "http://127.0.0.1:" + server.getAddress().getPort())
            .put("mock.cache.remote.read.timeout.ms", 1000);
        var config = RtpConfig.load(root);
        try (var cache = new KvCacheManager(2, 20, 0);
             var remote = KvcmRemoteCache.start(config, cache, new org.flexlb.mock.metrics.EventMetrics("prefill"))) {
            assertTrue(registered.await(3, java.util.concurrent.TimeUnit.SECONDS));
            var keys = new org.flexlb.mock.protocol.PromptCacheKeys(2, java.util.List.of(1, 2, 3), java.util.List.of());
            var input = new org.flexlb.mock.engine.scheduler.EngineScheduler.Input(
                "test", 1, 0, 3, 1, 0, System.nanoTime(), Long.MAX_VALUE, keys, true, false, 0, false);
            // Await the registration response, without publishing or triggering the writer's recovery path.
            for (int attempt = 0; calls.get() == 0 && attempt < 100; attempt++) {
                remote.lookup(input);
                if (calls.get() == 0) Thread.sleep(10);
            }
            assertEquals(1, calls.get());
            assertEquals(1, remote.lookup(input).remotePrefix().endBlock());
            assertEquals(2, calls.get());
        } finally { server.stop(0); }
    }

    private RtpConfig config() throws Exception { return RtpConfig.load(root()); }

    private ObjectNode root() {
        return new ObjectMapper().valueToTree(LLMMockEngineTest.configuration("prefill"));
    }
}
