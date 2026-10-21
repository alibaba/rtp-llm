package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.List;
import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class CacheFixtureControlTest {
    @TempDir Path directory;
    @Test void configurationIsAtomicExplicitAndExclusiveWithNaturalSeed() throws Exception {
        try (var cluster = MockEngineTestCluster.start(performanceModel(directory, "10"), 61420, 2, 0)) {
            String shape = "\"operation\":\"configure\",\"worker\":0,\"workers\":2,\"unique\":17,\"shared\":3,\"interval_ms\":1000";
            int port = cluster.controlPort();
            assertEquals(400, httpPostResponse(port, "/cache_fixture", "{\"port\":61420," + shape + ",\"delta\":18}").statusCode());
            assertEquals(200, httpPostResponse(port, "/cache_seed", "{\"port\":61420,\"keys\":[7]}").statusCode());
            assertEquals(400, httpPostResponse(port, "/cache_fixture", "{\"port\":61420," + shape + ",\"delta\":0}").statusCode());
            String configured = httpPost(port, "/cache_fixture", "{\"port\":61421," + shape + ",\"delta\":0}");
            assertTrue(configured.contains("controlled"));
            assertEquals(20, cluster.prefill(1).cacheFixture("inspect", 0, 0, 0, 0, 0, 0).get("keys"));
            assertThrows(IllegalArgumentException.class, () -> cluster.prefill(1).seedNaturalCache(List.of(8L)));
            assertEquals(400, httpPostResponse(port, "/cache_fixture", "{\"port\":61421," + shape + ",\"delta\":0}").statusCode());
            for (String old : new String[]{"seed", "grow", "clock"})
                assertEquals(400, httpPostResponse(port, "/cache_fixture", "{\"port\":61421,\"operation\":\"" + old + "\"}").statusCode());
            assertEquals(400, httpPostResponse(port, "/cache_fixture",
                    "{\"port\":61421,\"operation\":\"churn\",\"delta\":1,\"interval_ms\":1000,\"every\":1}").statusCode());
            assertEquals(200, httpPostResponse(port, "/cache_fixture",
                    "{\"port\":61421,\"operation\":\"churn\",\"delta\":1,\"interval_ms\":1000}").statusCode());
            assertEquals(200, httpPostResponse(port, "/cache_fixture",
                    "{\"port\":61421,\"operation\":\"churn\",\"delta\":0,\"interval_ms\":1000}").statusCode());
            var controlled = cluster.prefill(1);
            org.flexlb.engine.grpc.EngineRpcService.CacheStatusPB before = unary(o -> controlled.getCacheStatus(
                    org.flexlb.engine.grpc.EngineRpcService.CacheVersionPB.newBuilder().setNeedCacheKeys(true).build(), o));
            assertEquals(0, enqueueAndFetch(controlled, batch(1, slot(0,
                    inputWithBlockKeys(81, 1024, List.of(12345L))))).getErrorsCount());
            cluster.awaitCompleted(1, 3000); cluster.awaitNoInflight(controlled, 3000);
            org.flexlb.engine.grpc.EngineRpcService.CacheStatusPB after = unary(o -> controlled.getCacheStatus(
                    org.flexlb.engine.grpc.EngineRpcService.CacheVersionPB.newBuilder().setNeedCacheKeys(true).build(), o));
            assertEquals(before.getCacheKeysMap(), after.getCacheKeysMap(),
                    "background inference must not alter controlled sync keys");
            assertEquals(1, ((Number) controlled.getSnapshot().get("cache_keys")).intValue());
        }
    }
}
