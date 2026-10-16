package org.flexlb.cache.core;

import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

class KvCacheRetirementMetricTest {
    @Test
    void reportsOnceAfterLocalAndGlobalRetirementPreservingSurvivors() {
        GlobalCacheIndex global = new GlobalCacheIndex();
        EngineLocalView local = new EngineLocalView();
        CacheMetricsReporter reporter = mock(CacheMetricsReporter.class);
        KvCacheManager manager = manager(global, local, reporter);
        for (long key = 0; key < 513; key++) {
            global.addCacheBlock(key, "retired");
            global.addCacheBlock(key, "survivor");
            local.addOrUpdateCacheBlock("retired", key);
        }
        doAnswer(invocation -> {
            double elapsedMs = invocation.getArgument(0);
            assertTrue(Double.isFinite(elapsedMs) && elapsedMs >= 0);
            assertEquals(0, local.size("retired"));
            assertEquals(513, global.totalMappings());
            return null;
        }).when(reporter).reportEngineCacheRemovalTime(anyDouble());

        manager.removeEngineCache("retired");

        verify(reporter).reportEngineCacheRemovalTime(anyDouble());
        verify(reporter).reportGlobalCacheMetrics(513, 513);
    }

    @Test
    void nullEngineDoesNotPublishRetirement() {
        CacheMetricsReporter reporter = mock(CacheMetricsReporter.class);
        manager(new GlobalCacheIndex(), new EngineLocalView(), reporter)
                .removeEngineCache(null);
        verifyNoInteractions(reporter);
    }

    private KvCacheManager manager(GlobalCacheIndex global, EngineLocalView local,
                                   CacheMetricsReporter reporter) {
        KvCacheManager manager = new KvCacheManager();
        ReflectionTestUtils.setField(manager, "globalCacheIndex", global);
        ReflectionTestUtils.setField(manager, "engineLocalView", local);
        ReflectionTestUtils.setField(manager, "cacheMetricsReporter", reporter);
        return manager;
    }
}
