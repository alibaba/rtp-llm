package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.List;
import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class CacheFixtureControlTest {
    @TempDir Path directory;
    private static EngineRpcService.CacheStatusPB status(JavaMockEngineCluster.FastRpcService service) {
        return unary(o -> service.getCacheStatus(EngineRpcService.CacheVersionPB.newBuilder()
                .setNeedCacheKeys(true).build(), o));
    }
    @Test void externalKeysDefaultOffAndConfigurationRequiresExplicitOptIn() throws Exception {
        try (var cluster = MockEngineTestCluster.start(performanceModel(directory, "10"), 61420, 1, 0)) {
            int port = cluster.controlPort();
            String shape = "\"port\":61420,\"operation\":\"configure\",\"worker\":0,\"workers\":1,"
                    + "\"unique\":17,\"shared\":3,\"delta\":0,\"interval_ms\":1000";
            assertTrue(status(cluster.prefill(0)).getCacheKeysMap().isEmpty());
            for (String option : new String[]{"", ",\"enabled\":false", ",\"enabled\":\"true\""})
                assertEquals(400, httpPostResponse(port, "/cache_fixture", "{" + shape + option + "}").statusCode());
            assertTrue(status(cluster.prefill(0)).getCacheKeysMap().isEmpty());
            assertEquals(400, httpPostResponse(port, "/cache_fixture", "{" + shape.replace("\"delta\":0", "\"delta\":18")
                    + ",\"enabled\":true}").statusCode());
            assertEquals(200, httpPostResponse(port, "/cache_seed", "{\"port\":61420,\"keys\":[7]}").statusCode());
            assertEquals(200, httpPostResponse(port, "/cache_fixture", "{" + shape + ",\"enabled\":true}").statusCode());
            assertEquals(21, status(cluster.prefill(0)).getCacheKeysCount());
            assertEquals(400, httpPostResponse(port, "/cache_fixture", "{" + shape + ",\"enabled\":true}").statusCode());
            for (String old : new String[]{"seed", "grow", "clock"})
                assertEquals(400, httpPostResponse(port, "/cache_fixture", "{\"port\":61420,\"operation\":\"" + old + "\"}").statusCode());
            assertEquals(400, httpPostResponse(port, "/cache_fixture",
                    "{\"port\":61420,\"operation\":\"churn\",\"delta\":1,\"interval_ms\":1000,\"every\":1}").statusCode());
            for (int delta : new int[]{1, 0})
                assertEquals(200, httpPostResponse(port, "/cache_fixture",
                    "{\"port\":61420,\"operation\":\"churn\",\"delta\":" + delta + ",\"interval_ms\":1000}").statusCode());
        }
    }
    @Test void backgroundGrowthAndEvictionRemainVisibleAlongsideExternalKeys() throws Exception {
        try (var cluster = MockEngineTestCluster.start(performanceModel(directory, "10"), 61430, 1, 0)) {
            var p = cluster.prefill(0);
            assertEquals(0, enqueueAndFetch(p, batch(1, slot(0,
                    inputWithBlockKeys(81, 1024, List.of(7L))))).getErrorsCount());
            cluster.awaitCompleted(1, 3000); cluster.awaitNoInflight(p, 3000);
            var old = status(p);
            p.cacheFixture("configure", 0, 1, 17, 3, 0, 1000);
            var before = status(p);
            assertTrue(before.getVersion() > old.getVersion());
            assertEquals(21, before.getCacheKeysCount());
            p.cacheFixture("churn", 0, 0, 0, 0, 1, 1);
            assertEquals(0, enqueueAndFetch(p, batch(1, slot(0,
                    inputWithBlockKeys(82, 2048, List.of(7L, 8L))))).getErrorsCount());
            cluster.awaitCompleted(2, 3000); cluster.awaitNoInflight(p, 3000);
            p.cacheFixture("churn", 0, 0, 0, 0, 0, 1);
            var after = status(p);
            assertTrue(after.getVersion() > before.getVersion());
            assertEquals(22, after.getCacheKeysCount());
            assertTrue(after.getCacheKeysMap().keySet().containsAll(List.of(7L, 8L)));
            var inspected = p.cacheFixture("inspect", 0, 0, 0, 0, 0, 0);
            assertEquals(20, inspected.get("keys"));
            assertTrue(((Number) inspected.get("fixture_version")).longValue() > 1);
            assertEquals(22L, inspected.get("reported_keys"));
            assertEquals(after.getVersion(), inspected.get("version"));
            assertTrue(p.evictCacheKeys(List.of(7L)));
            assertEquals(21, status(p).getCacheKeysCount());
            assertTrue(status(p).getVersion() > after.getVersion());
        }
    }
}
