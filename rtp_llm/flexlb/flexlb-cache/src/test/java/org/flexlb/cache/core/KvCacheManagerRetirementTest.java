package org.flexlb.cache.core;

import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class KvCacheManagerRetirementTest {
    private static final String ADDRESS = "127.0.0.1:8080";

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void partialUpdateKeepsEveryPotentialGlobalMappingDiscoverableForCleanup(boolean addition) throws Exception {
        var local = spy(new EngineLocalView());
        var global = spy(new GlobalCacheIndex());
        var manager = manager(local, global);
        if (addition) {
            doReturn(DiffResult.builder().addedBlocks(Set.of(1L)).build())
                    .when(local).calculateDiff(eq(ADDRESS), anySet(), anyString());
            doAnswer(call -> {
                call.callRealMethod();
                throw new IllegalStateException("failed after global insertion");
            }).when(global).addCacheBlock(1L, ADDRESS, null);
        } else {
            local.addOrUpdateCacheBlock(ADDRESS, 1L);
            global.addCacheBlock(1L, ADDRESS, null);
            doReturn(DiffResult.builder().removedBlocks(Set.of(1L)).build())
                    .when(local).calculateDiff(eq(ADDRESS), anySet(), anyString());
            doThrow(new IllegalStateException("failed before global removal"))
                    .when(global).removeCacheBlock(ADDRESS, 1L, null);
        }
        assertThrows(IllegalStateException.class,
                () -> manager.updateEngineCache(ADDRESS, "PREFILL", addition ? Set.of(1L) : Set.of(), null));
        assertEquals(Set.of(1L), local.getEngineCacheBlocks(ADDRESS));
        assertEquals(1L, global.totalMappings());
        manager.removeEngineCache(ADDRESS, null);
        assertEquals(0L, global.totalMappings());
        assertEquals(0, local.size(ADDRESS));
    }

    @Test
    void failedInsertionIsReplayedBeforeAcknowledgingTheNextSnapshot() throws Exception {
        var local = spy(new EngineLocalView());
        var global = spy(new GlobalCacheIndex());
        var manager = manager(local, global);
        doReturn(DiffResult.builder().addedBlocks(Set.of(1L)).build(), DiffResult.empty(ADDRESS))
                .when(local).calculateDiff(eq(ADDRESS), anySet(), anyString());
        doThrow(new IllegalStateException("before insertion")).doCallRealMethod()
                .when(global).addCacheBlock(1L, ADDRESS, null);
        assertThrows(IllegalStateException.class,
                () -> manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(1L), null));
        assertEquals(0L, global.totalMappings());
        manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(1L), null);
        assertEquals(1L, global.totalMappings());
        assertEquals(Set.of(1L), local.getEngineCacheBlocks(ADDRESS));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void interruptedAdditionRecoversAgainstChangedSnapshot(boolean localFailure) throws Exception {
        var local = spy(new EngineLocalView());
        inject(local, "cacheMetricsReporter", mock(CacheMetricsReporter.class));
        inject(local, "dynamicIntervalManager", mock(org.flexlb.cache.match.localsync.DynamicCacheIntervalService.class));
        var global = spy(new GlobalCacheIndex());
        var manager = manager(local, global);
        // The first snapshot has several changes; failure occurs at an arbitrary key.
        if (localFailure) {
            doThrow(new IllegalStateException("before local insertion")).doCallRealMethod()
                    .when(local).addOrUpdateCacheBlock(ADDRESS, 2L);
        } else {
            doThrow(new IllegalStateException("before global insertion")).doCallRealMethod()
                    .when(global).addCacheBlock(2L, ADDRESS, null);
        }
        assertThrows(IllegalStateException.class,
                () -> manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(1L, 2L, 3L), null));

        // The failed key is already gone from the next engine snapshot.
        manager.updateEngineCache(ADDRESS, "PREFILL", Set.of(3L, 4L), null);
        assertEquals(Set.of(3L, 4L), local.getEngineCacheBlocks(ADDRESS));
        assertEquals(2L, global.totalMappings());
        for (long key : List.of(1L, 2L)) {
            assertTrue(global.batchCalculatePrefixMatchLength(List.of(ADDRESS), List.of(key)).isEmpty());
        }
        for (long key : List.of(3L, 4L)) {
            assertEquals(java.util.Map.of(ADDRESS, 1),
                    global.batchCalculatePrefixMatchLength(List.of(ADDRESS), List.of(key)));
        }
        manager.removeEngineCache(ADDRESS, null);
        assertEquals(0L, global.totalMappings());
    }

    @Test
    void partialCleanupFailureRetainsKeysForIdempotentRetryAndPreservesOtherOwners() throws Exception {
        var local = new EngineLocalView();
        var global = spy(new GlobalCacheIndex());
        var manager = manager(local, global);
        for (long key : List.of(1L, 2L)) {
            local.addOrUpdateCacheBlock(ADDRESS, key);
            global.addCacheBlock(key, ADDRESS, null);
        }
        global.addCacheBlock(2L, "survivor", null);
        doAnswer(call -> {
            global.removeCacheBlock(ADDRESS, 1L, null);
            throw new IllegalStateException("partial cleanup failure");
        }).when(global).removeAllCacheBlockOfEngine(eq(ADDRESS), anySet(), isNull());
        assertThrows(IllegalStateException.class, () -> manager.removeEngineCache(ADDRESS, null));
        assertEquals(Set.of(1L, 2L), local.getEngineCacheBlocks(ADDRESS));
        doCallRealMethod().when(global).removeAllCacheBlockOfEngine(eq(ADDRESS), anySet(), isNull());
        manager.removeEngineCache(ADDRESS, null);
        assertEquals(0, local.size(ADDRESS));
        assertEquals(1L, global.totalMappings());
        assertEquals(java.util.Map.of("survivor", 1),
                global.batchCalculatePrefixMatchLength(List.of(ADDRESS, "survivor"), List.of(2L)));
    }

    private static KvCacheManager manager(EngineLocalView local, GlobalCacheIndex global) throws Exception {
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
