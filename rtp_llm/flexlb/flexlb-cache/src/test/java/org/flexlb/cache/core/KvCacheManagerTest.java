package org.flexlb.cache.core;

import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class KvCacheManagerTest {
    private final GlobalCacheIndex index = spy(new GlobalCacheIndex());
    private final EngineLocalView local = new EngineLocalView();
    private final KvCacheManager manager = new KvCacheManager();

    @BeforeEach
    void setUp() {
        CacheMetricsReporter metrics = mock(CacheMetricsReporter.class);
        ReflectionTestUtils.setField(local, "cacheMetricsReporter", metrics);
        ReflectionTestUtils.setField(local, "dynamicIntervalManager", mock(DynamicCacheIntervalService.class));
        ReflectionTestUtils.setField(manager, "globalCacheIndex", index);
        ReflectionTestUtils.setField(manager, "engineLocalView", local);
        ReflectionTestUtils.setField(manager, "cacheMetricsReporter", metrics);
    }

    @Test
    void retirementPreservesSharedKeysAndAllowsReplacement() {
        manager.updateEngineCache("retiring:1", "PREFILL", Set.of(1L, 2L));
        manager.updateEngineCache("healthy:2", "PREFILL", Set.of(1L, 3L));
        manager.removeEngineCache("retiring:1");
        manager.removeEngineCache("retiring:1");
        assertEquals(2, index.totalBlocks());
        assertEquals(2, index.totalMappings());
        assertEquals(1, local.getEngineViewsMapSize());
        assertEquals(Map.of("healthy:2", 2), manager.findMatchingEngines(
                List.of(1L, 3L), List.of("retiring:1", "healthy:2")));
        // Fleet-wide removal must never be used for generation retirement.
        verify(index, never()).removeAllCacheBlockOfEngine(anyString());
        manager.updateEngineCache("retiring:1", "PREFILL", Set.of(2L, 3L));
        assertEquals(4, index.totalMappings());
        manager.removeEngineCache("retiring:1");
        assertEquals(2, index.totalMappings());
        assertEquals(Set.of(1L, 3L), local.getEngineCacheBlocks("healthy:2"));
    }

    @Test
    void failedCleanupRetainsKeysForAnIdempotentRetry() {
        manager.updateEngineCache("retiring:1", "PREFILL", Set.of(1L, 2L));
        doThrow(new IllegalStateException("injected cleanup failure"))
                .doCallRealMethod().when(index).removeCacheBlock("retiring:1", 2L);
        assertThrows(IllegalStateException.class, () -> manager.removeEngineCache("retiring:1"));
        assertEquals(Set.of(1L, 2L), local.getEngineCacheBlocks("retiring:1"));
        manager.removeEngineCache("retiring:1");
        assertEquals(0, index.totalBlocks());
        assertEquals(0, index.totalMappings());
        assertEquals(0, local.getEngineViewsMapSize());
    }
}
