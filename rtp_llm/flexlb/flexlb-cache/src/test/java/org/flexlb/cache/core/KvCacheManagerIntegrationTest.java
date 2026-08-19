package org.flexlb.cache.core;

import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class KvCacheManagerIntegrationTest {

    private static final String ENGINE = "127.0.0.1:8080";

    @Test
    void retriesPartiallyFailedClearBeforePublishingNewGeneration() {
        FailOnceGlobalCacheIndex global = new FailOnceGlobalCacheIndex();
        EngineLocalView local = new EngineLocalView();
        KvCacheManager manager = new KvCacheManager();
        CacheMetricsReporter metrics = new NoOpCacheMetricsReporter();
        ReflectionTestUtils.setField(local, "cacheMetricsReporter", metrics);
        ReflectionTestUtils.setField(local, "dynamicIntervalManager",
                (DynamicCacheIntervalService) diffSize -> { });
        ReflectionTestUtils.setField(manager, "globalCacheIndex", global);
        ReflectionTestUtils.setField(manager, "engineLocalView", local);
        ReflectionTestUtils.setField(manager, "cacheMetricsReporter", metrics);

        Set<Long> oldGeneration = Set.of(1L, 2L);
        Set<Long> newGeneration = Set.of(3L, 4L);
        manager.updateEngineCache(ENGINE, "prefill", oldGeneration);
        assertEquals(2L, global.totalBlocks());
        assertEquals(2L, global.totalMappings());

        assertThrows(IllegalStateException.class,
                () -> manager.clearEngineCache(ENGINE));
        assertEquals(oldGeneration, local.getEngineCacheBlocks(ENGINE),
                "a partial global clear must not advance the local commit marker");

        manager.updateEngineCache(ENGINE, "prefill", newGeneration);

        assertEquals(newGeneration, local.getEngineCacheBlocks(ENGINE));
        assertEquals(0, prefixMatch(global, 1L));
        assertEquals(0, prefixMatch(global, 2L));
        assertEquals(2, prefixMatch(global, 3L, 4L));
        assertEquals(2L, global.totalBlocks());
        assertEquals(2L, global.totalMappings());
    }

    private static int prefixMatch(GlobalCacheIndex global, Long... blocks) {
        return global.batchCalculatePrefixMatchLength(
                List.of(ENGINE), List.of(blocks)).get(ENGINE);
    }

    private static final class FailOnceGlobalCacheIndex extends GlobalCacheIndex {
        private boolean failNextClear = true;

        @Override
        void removeEngineCacheBlocks(String engineIpPort, Set<Long> cacheBlocks) {
            if (failNextClear) {
                failNextClear = false;
                removeCacheBlock(engineIpPort, cacheBlocks.iterator().next());
                throw new IllegalStateException("injected failure after partial clear");
            }
            super.removeEngineCacheBlocks(engineIpPort, cacheBlocks);
        }
    }

    private static final class NoOpCacheMetricsReporter extends CacheMetricsReporter {
        @Override
        public void reportEngineLocalMetrics(String engineIp, String role, int cacheCount) {
        }

        @Override
        public void reportGlobalCacheMetrics(long totalBlocks, long totalMappings) {
        }

        @Override
        public void reportCacheDiffMetrics(String role, int addedBlocksSize,
                                           int removedBlocksSize) {
        }

        @Override
        public void reportEngineViewsMapSize(int mapSize) {
        }
    }
}
