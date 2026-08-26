package org.flexlb.cache.core;

import org.flexlb.cache.domain.CacheMatch;
import org.flexlb.cache.domain.EngineGeneration;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.cache.service.impl.DefaultCacheAwareService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class CacheGenerationIsolationTest {

    @Test
    void delayedOldCallbackAndRetirementCannotMutateReplacementGeneration() {
        String engine = "127.0.0.1:61000";
        long oldGeneration = 1L;
        long replacementGeneration = 2L;

        CacheMetricsReporter metrics = mock(CacheMetricsReporter.class);
        GlobalCacheIndex globalIndex = new GlobalCacheIndex();
        KvCacheManager manager = new KvCacheManager(
                globalIndex,
                metrics,
                mock(DynamicCacheIntervalService.class));
        DefaultCacheAwareService service =
                new DefaultCacheAwareService(manager, metrics);

        assertTrue(service.activateEngineGeneration(
                engine, RoleType.PREFILL, oldGeneration));
        assertTrue(service.updateEngineBlockCache(
                engine,
                RoleType.PREFILL,
                oldGeneration,
                cacheStatus(1L, Set.of(11L, 12L))).isSuccess());

        // Replacement publication clears the old full view before its first
        // callback arrives.
        assertTrue(service.activateEngineGeneration(
                engine, RoleType.PREFILL, replacementGeneration));
        assertEquals(0, prefixMatch(
                globalIndex, engine, replacementGeneration, 11L));
        assertTrue(service.updateEngineBlockCache(
                engine,
                RoleType.PREFILL,
                replacementGeneration,
                cacheStatus(1L, Set.of(21L, 22L))).isSuccess());

        // The old asynchronous callback completes last.
        WorkerCacheUpdateResult staleUpdate = service.updateEngineBlockCache(
                engine,
                RoleType.PREFILL,
                oldGeneration,
                cacheStatus(2L, Set.of(11L, 13L)));
        assertEquals(
                WorkerCacheUpdateResult.Outcome.STALE_GENERATION,
                staleUpdate.getOutcome());
        assertFalse(staleUpdate.isSuccess());
        assertEquals(0, prefixMatch(
                globalIndex, engine, replacementGeneration, 11L));
        assertEquals(2, prefixMatch(
                globalIndex, engine, replacementGeneration, 21L, 22L));

        // An old exact retirement is equally harmless.
        assertFalse(service.retireEngineGeneration(
                engine, RoleType.PREFILL, oldGeneration));
        assertEquals(2, prefixMatch(
                globalIndex, engine, replacementGeneration, 21L, 22L));
        assertFalse(globalIndex.batchCalculatePrefixMatches(
                List.of(new EngineGeneration(engine, oldGeneration)),
                List.of(21L, 22L)).containsKey(
                new EngineGeneration(engine, oldGeneration)));
    }

    private static CacheStatus cacheStatus(long version, Set<Long> keys) {
        return CacheStatus.builder()
                .version(version)
                .cachedKeys(keys)
                .cacheKeySize(keys.size())
                .build();
    }

    private static int prefixMatch(
            GlobalCacheIndex index,
            String engine,
            long generation,
            Long... keys) {
        EngineGeneration identity = new EngineGeneration(engine, generation);
        CacheMatch match = index.batchCalculatePrefixMatches(
                List.of(identity), List.of(keys)).get(identity);
        return match == null ? 0 : match.prefixMatchLength();
    }
}
