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
                .setNeedCacheKeys(true).setLatestCacheVersion(-1).build(), o));
    }
    @Test void sameVersionRefreshesCapacityAndReportsRpcBeforeResponse() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "10"), 61410, 1, 0)) {
            var service = cluster.prefill(0);
            service.seedNaturalCache(List.of(7L));
            var initial = status(service);
            assertEquals(0, initial.getVersion(), "first logical node advances -1 to 0");
            var reports = new java.util.concurrent.atomic.AtomicInteger();
            service.cacheMetricReporter = latency -> { assertTrue(latency >= 0); reports.incrementAndGet(); };
            var request = EngineRpcService.CacheVersionPB.newBuilder().setNeedCacheKeys(true)
                    .setLatestCacheVersion(initial.getVersion()).build();
            service.setFaultConfig(FaultInjectionConfig.builder().kvPressureTokens(64).build());
            EngineRpcService.CacheStatusPB same = unary(observer -> service.getCacheStatus(request, observer));
            assertEquals(1, reports.get());
            assertEquals(initial.getVersion(), same.getVersion());
            assertEquals(0, same.getCacheKeysCount());
            assertEquals(initial.getAvailableKvCache() - 64, same.getAvailableKvCache());
            EngineRpcService.CacheStatusPB scalar = unary(observer -> service.getCacheStatus(
                    request.toBuilder().setNeedCacheKeys(false).setLatestCacheVersion(919).build(), observer));
            assertEquals(919, scalar.getVersion());
            assertEquals(0, scalar.getCacheKeysCount());
            assertEquals(2, reports.get());
            assertTrue(service.evictCacheKeys(List.of(7L)));
            EngineRpcService.CacheStatusPB removed = unary(observer -> service.getCacheStatus(request, observer));
            assertEquals(same.getVersion() + 1, removed.getVersion());
            assertEquals(0, removed.getCacheKeysCount(), "changed empty snapshot clears the client index");
        }
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
    @Test void processCrashReseedsFinishedVersionAndStartsAnEmptyKeyIndex() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "10"), 61415, 1, 0)) {
            var service = cluster.prefill(0);
            service.seedNaturalCache(List.of(7L, 8L));
            service.cacheFixture("configure", 0, 1, 17, 3, 0, 1000);
            assertEquals(22, status(service).getCacheKeysCount());
            service.crashNow();
            var empty = status(service);
            assertEquals(-1, empty.getVersion());
            assertEquals(0, empty.getCacheKeysCount());
            var noRecords = workerStatus(service, Long.MAX_VALUE);
            assertEquals(Long.MAX_VALUE, noRecords.getLatestFinishedVersion());
            assertEquals(0, noRecords.getFinishedTaskListCount());
            assertThrows(IllegalArgumentException.class, () -> service.cacheFixture("inspect", 0, 0, 0, 0, 0, 0));
        }
    }

    @Test void controlInspectionAndRpcSnapshotsUseTheSameLockOrder() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "10"), 61417, 1, 0)) {
            var service = cluster.prefill(0);
            service.cacheFixture("configure", 0, 1, 17, 3, 0, 1000);
            var rpc = java.util.concurrent.CompletableFuture.runAsync(() -> {
                for (int i = 0; i < 200; i++) assertEquals(20, status(service).getCacheKeysCount());
            });
            var control = java.util.concurrent.CompletableFuture.runAsync(() -> {
                for (int i = 0; i < 200; i++) assertEquals(20L, service.cacheFixture("inspect", 0, 0, 0, 0, 0, 0).get("reported_keys"));
            });
            java.util.concurrent.CompletableFuture.allOf(rpc, control).get(5, java.util.concurrent.TimeUnit.SECONDS);
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
