package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class NaturalCacheSeedTest {
    @TempDir Path directory;

    private static EngineRpcService.CacheStatusPB status(JavaMockEngineCluster.FastRpcService p) {
        return unary(o -> p.getCacheStatus(EngineRpcService.CacheVersionPB.newBuilder()
                .setNeedCacheKeys(true).setLatestCacheVersion(-1).build(), o));
    }

    @Test void seededKeysParticipateInNormalGrowthReuseAndEviction() throws Exception {
        try (var cluster = MockEngineTestCluster.start(performanceModel(directory, "10"), 61360, 1, 0)) {
            var p = cluster.prefill(0);
            assertEquals(200, httpPostResponse(cluster.controlPort(), "/cache_seed",
                    "{\"port\":61360,\"keys\":[7]}").statusCode());
            var seeded = status(p);
            assertEquals(Set.of(7L), seeded.getCacheKeysMap().keySet());
            assertEquals(0, enqueueAndFetch(p, batch(1, slot(0,
                    inputWithBlockKeys(71, 2048, List.of(7L, 8L))))).getErrorsCount());
            cluster.awaitCompleted(1, 3000);
            cluster.awaitNoInflight(p, 3000);
            var grown = status(p);
            assertEquals(Set.of(7L, 8L), grown.getCacheKeysMap().keySet());
            assertTrue(grown.getVersion() > seeded.getVersion());
            assertTrue(((Number) p.getMetricsSnapshot().get("cache_key_hits")).longValue() > 0);
            assertTrue(p.evictCacheKeys(List.of(7L)));
            assertEquals(Set.of(8L), status(p).getCacheKeysMap().keySet());
            assertTrue(status(p).getVersion() > grown.getVersion());
            assertEquals(400, httpPostResponse(cluster.controlPort(), "/cache_seed",
                    "{\"port\":61360,\"keys\":[9]}").statusCode());
            p.cacheFixture("configure", 0, 1, 2, 1, 0, 1000);
            assertTrue(status(p).getCacheKeysMap().containsKey(8L));
        }
    }

    @Test void seedHonorsRetentionBudgetWithoutSelectingModeOnFailure() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "10"), 61390, 1, 0)) {
            var p = cluster.prefill(0);
            p.setCacheRetentionBlocks(1);
            assertThrows(IllegalArgumentException.class, () -> p.seedNaturalCache(List.of(7L, 8L)));
            assertTrue(status(p).getCacheKeysMap().isEmpty());
            p.seedNaturalCache(List.of(7L));
            assertEquals(Set.of(7L), status(p).getCacheKeysMap().keySet());
        }
    }

    @Test void externalKeysCoexistWithNaturalSeedAndInvalidSeedIsAtomic() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "10"), 61380, 2, 0)) {
            var controlled = cluster.prefill(0);
            controlled.cacheFixture("configure", 0, 2, 2, 1, 0, 1000);
            controlled.seedNaturalCache(List.of(7L));
            assertTrue(status(controlled).getCacheKeysMap().containsKey(7L));
            var natural = cluster.prefill(1);
            assertThrows(IllegalArgumentException.class, () -> natural.seedNaturalCache(List.of(7L, 7L)));
            assertThrows(IllegalArgumentException.class, () -> natural.seedNaturalCache(List.of()));
            assertTrue(status(natural).getCacheKeysMap().isEmpty());
            natural.seedNaturalCache(List.of(7L));
            natural.cacheFixture("configure", 1, 2, 2, 1, 0, 1000);
            assertTrue(status(natural).getCacheKeysMap().containsKey(7L));
        }
    }
}
