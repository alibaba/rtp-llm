package org.flexlb.cache.core;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.MetricsCardinalityConfig;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_BYTES;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_VIEWS_MAP_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_GLOBAL_TOTAL_COUNT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anySet;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.anySet;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doCallRealMethod;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;

class KvCacheManagerRetirementTest {
    private static final String ADDRESS = "127.0.0.1:8080";

    @Test
    void delayedViewsReportCannotOverwriteConcurrentRetirement() throws Exception {
        var registry = new SimpleMeterRegistry();
        var local = new EngineLocalView();
        var captured = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        try (var executor = Executors.newSingleThreadExecutor()) {
            var monitor = spy(new MicrometerFlexMonitor(registry));
            var reporter = new CacheMetricsReporter();
            inject(reporter, "monitor", monitor);
            reporter.init();
            inject(local, "cacheMetricsReporter", reporter);
            inject(
                    local,
                    "dynamicIntervalManager",
                    mock(org.flexlb.cache.service.DynamicCacheIntervalService.class));
            var manager = manager(local, new GlobalCacheIndex());
            inject(manager, "cacheMetricsReporter", reporter);
            manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(1L));
            doAnswer(
                            call -> {
                                captured.countDown();
                                assertTrue(release.await(5, TimeUnit.SECONDS));
                                return call.callRealMethod();
                            })
                    .when(monitor)
                    .report(CACHE_ENGINE_VIEWS_MAP_SIZE, 2.0);
            var update =
                    executor.submit(
                            () ->
                                    manager.updateEngineCache(
                                            "127.0.0.2:8080", "PREFILL", Set.of(2L)));
            try {
                assertTrue(captured.await(5, TimeUnit.SECONDS));
                manager.removeEngineCache(ADDRESS);
            } finally {
                release.countDown();
            }
            update.get(5, TimeUnit.SECONDS);
            assertEquals(1, local.getEngineViewsMapSize());
            assertEquals(1, registry.get("flexlb." + CACHE_ENGINE_VIEWS_MAP_SIZE).gauge().value());
            assertEquals(1, registry.get("flexlb." + CACHE_GLOBAL_TOTAL_COUNT).gauge().value());
        } finally {
            release.countDown();
            var field = EngineLocalView.class.getDeclaredField("customPool");
            field.setAccessible(true);
            ((ForkJoinPool) field.get(local)).shutdownNow();
            registry.close();
        }
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void retirementRemovesOnlyItsSourceAndRefreshesViews(boolean collapsed, boolean clear)
            throws Exception {
        var registry = new SimpleMeterRegistry();
        var local = new EngineLocalView();
        try {
            if (collapsed) {
                registry.config()
                        .meterFilter(new MetricsCardinalityConfig().removeEngineTagsFilter());
            }
            var reporter = new CacheMetricsReporter();
            inject(reporter, "monitor", new MicrometerFlexMonitor(registry));
            reporter.init();
            inject(local, "cacheMetricsReporter", reporter);
            inject(
                    local,
                    "dynamicIntervalManager",
                    mock(org.flexlb.cache.service.DynamicCacheIntervalService.class));
            var global = new GlobalCacheIndex();
            var manager = manager(local, global);
            inject(manager, "cacheMetricsReporter", reporter);
            String otherPort = "127.0.0.1:8081";
            manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(1L, 2L, 3L));
            manager.updateEngineCache(otherPort, "PREFILL", Set.of(4L, 5L));
            assertEquals(
                    5,
                    registry.find("flexlb." + CACHE_ENGINE_LOCAL_COUNT).gauges().stream()
                            .mapToDouble(io.micrometer.core.instrument.Gauge::value)
                            .sum());

            manager.removeEngineCache(ADDRESS);
            manager.removeEngineCache(ADDRESS);
            // An unchanged surviving worker must not be needed to repair retirement metrics.
            manager.updateEngineCache(otherPort, "PREFILL", Set.of(4L, 5L));
            assertEquals(1, registry.find("flexlb." + CACHE_ENGINE_LOCAL_COUNT).gauges().size());
            assertEquals(2, registry.get("flexlb." + CACHE_ENGINE_LOCAL_COUNT).gauge().value());
            assertEquals(272, registry.get("flexlb." + CACHE_ENGINE_LOCAL_BYTES).gauge().value());
            assertEquals(1, registry.get("flexlb." + CACHE_ENGINE_VIEWS_MAP_SIZE).gauge().value());
            assertEquals(2, global.totalBlocks());
            // A replacement at the same address publishes fresh values, not the retired ones.
            manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(6L));
            assertEquals(
                    3,
                    registry.find("flexlb." + CACHE_ENGINE_LOCAL_COUNT).gauges().stream()
                            .mapToDouble(io.micrometer.core.instrument.Gauge::value)
                            .sum());
            if (clear) {
                manager.clear();
            } else {
                manager.removeEngineCache(ADDRESS);
                manager.removeEngineCache(otherPort);
            }
            assertTrue(registry.find("flexlb." + CACHE_ENGINE_LOCAL_COUNT).gauges().isEmpty());
            assertTrue(registry.find("flexlb." + CACHE_ENGINE_LOCAL_BYTES).gauges().isEmpty());
            assertEquals(0, registry.get("flexlb." + CACHE_ENGINE_VIEWS_MAP_SIZE).gauge().value());
            assertEquals(0, registry.get("flexlb." + CACHE_GLOBAL_TOTAL_COUNT).gauge().value());
        } finally {
            var field = EngineLocalView.class.getDeclaredField("customPool");
            field.setAccessible(true);
            ((ForkJoinPool) field.get(local)).shutdownNow();
            registry.close();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void partialUpdateKeepsEveryPotentialGlobalMappingDiscoverableForCleanup(boolean addition)
            throws Exception {
        var local = spy(new EngineLocalView());
        var global = spy(new GlobalCacheIndex());
        var manager = manager(local, global);
        if (addition) {
            doReturn(DiffResult.builder().addedBlocks(Set.of(1L)).build())
                    .when(local)
                    .calculateDiff(eq(ADDRESS), anySet(), anyString());
            doAnswer(
                            call -> {
                                call.callRealMethod();
                                throw new IllegalStateException("failed after global insertion");
                            })
                    .when(global)
                    .addCacheBlock(1L, ADDRESS);
        } else {
            local.addOrUpdateCacheBlock(ADDRESS, 1L);
            global.addCacheBlock(1L, ADDRESS);
            doReturn(DiffResult.builder().removedBlocks(Set.of(1L)).build())
                    .when(local)
                    .calculateDiff(eq(ADDRESS), anySet(), anyString());
            doThrow(new IllegalStateException("failed before global removal"))
                    .when(global)
                    .removeCacheBlock(ADDRESS, 1L);
        }
        assertThrows(
                IllegalStateException.class,
                () ->
                        manager.updateEngineCache(
                                ADDRESS, "PREFILL", addition ? Set.of(1L) : Set.of()));
        assertEquals(Set.of(1L), local.getEngineCacheBlocks(ADDRESS));
        assertEquals(1L, global.totalMappings());
        manager.removeEngineCache(ADDRESS);
        assertEquals(0L, global.totalMappings());
        assertEquals(0, local.size(ADDRESS));
    }

    @Test
    void failedInsertionIsReplayedBeforeAcknowledgingTheNextSnapshot() throws Exception {
        var local = spy(new EngineLocalView());
        var global = spy(new GlobalCacheIndex());
        var manager = manager(local, global);
        doReturn(DiffResult.builder().addedBlocks(Set.of(1L)).build(), DiffResult.empty())
                .when(local)
                .calculateDiff(eq(ADDRESS), anySet(), anyString());
        doThrow(new IllegalStateException("before insertion"))
                .doCallRealMethod()
                .when(global)
                .addCacheBlock(1L, ADDRESS);
        assertThrows(
                IllegalStateException.class,
                () -> manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(1L)));
        assertEquals(0L, global.totalMappings());
        manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(1L));
        assertEquals(1L, global.totalMappings());
        assertEquals(Set.of(1L), local.getEngineCacheBlocks(ADDRESS));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void interruptedAdditionRecoversAgainstChangedSnapshot(boolean localFailure) throws Exception {
        var local = spy(new EngineLocalView());
        inject(local, "cacheMetricsReporter", mock(CacheMetricsReporter.class));
        inject(
                local,
                "dynamicIntervalManager",
                mock(org.flexlb.cache.service.DynamicCacheIntervalService.class));
        var global = spy(new GlobalCacheIndex());
        var manager = manager(local, global);
        // The first snapshot has several changes; failure occurs at an arbitrary key.
        if (localFailure) {
            doThrow(new IllegalStateException("before local insertion"))
                    .doCallRealMethod()
                    .when(local)
                    .addOrUpdateCacheBlock(ADDRESS, 2L);
        } else {
            doThrow(new IllegalStateException("before global insertion"))
                    .doCallRealMethod()
                    .when(global)
                    .addCacheBlock(2L, ADDRESS);
        }
        assertThrows(
                IllegalStateException.class,
                () -> manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(1L, 2L, 3L)));

        // The failed key is already gone from the next engine snapshot.
        manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(3L, 4L));
        assertEquals(Set.of(3L, 4L), local.getEngineCacheBlocks(ADDRESS));
        assertEquals(2L, global.totalMappings());
        for (long key : List.of(1L, 2L)) {
            assertTrue(
                    global.batchCalculatePrefixMatchLength(List.of(ADDRESS), List.of(key))
                            .isEmpty());
        }
        for (long key : List.of(3L, 4L)) {
            assertEquals(
                    java.util.Map.of(ADDRESS, 1),
                    global.batchCalculatePrefixMatchLength(List.of(ADDRESS), List.of(key)));
        }
        manager.removeEngineCache(ADDRESS);
        assertEquals(0L, global.totalMappings());
    }

    @Test
    void partialCleanupFailureRetainsKeysForIdempotentRetryAndPreservesOtherOwners()
            throws Exception {
        var local = new EngineLocalView();
        var global = spy(new GlobalCacheIndex());
        var manager = manager(local, global);
        for (long key : List.of(1L, 2L)) {
            local.addOrUpdateCacheBlock(ADDRESS, key);
            global.addCacheBlock(key, ADDRESS);
        }
        global.addCacheBlock(2L, "survivor");
        doAnswer(
                        call -> {
                            global.removeCacheBlock(ADDRESS, 1L);
                            throw new IllegalStateException("partial cleanup failure");
                        })
                .when(global)
                .removeAllCacheBlockOfEngine(eq(ADDRESS), anySet());
        assertThrows(IllegalStateException.class, () -> manager.removeEngineCache(ADDRESS));
        assertEquals(Set.of(1L, 2L), local.getEngineCacheBlocks(ADDRESS));
        doCallRealMethod().when(global).removeAllCacheBlockOfEngine(eq(ADDRESS), anySet());
        manager.removeEngineCache(ADDRESS);
        assertEquals(0, local.size(ADDRESS));
        assertEquals(1L, global.totalMappings());
        assertEquals(
                java.util.Map.of("survivor", 1),
                global.batchCalculatePrefixMatchLength(List.of(ADDRESS, "survivor"), List.of(2L)));
    }

    private static KvCacheManager manager(EngineLocalView local, GlobalCacheIndex global)
            throws Exception {
        var manager = new KvCacheManager();
        inject(manager, "engineLocalView", local);
        inject(manager, "globalCacheIndex", global);
        inject(manager, "cacheMetricsReporter", mock(CacheMetricsReporter.class));
        return manager;
    }

    private static void inject(Object target, String name, Object value) throws Exception {
        var field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }
}
