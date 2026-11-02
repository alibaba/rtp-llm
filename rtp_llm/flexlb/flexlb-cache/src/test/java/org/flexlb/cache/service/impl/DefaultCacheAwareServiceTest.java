package org.flexlb.cache.service.impl;

import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DefaultCacheAwareServiceTest {

    @Test
    void cachePollAndRetirementKeepTheirExactDiagnosticOwner() {
        var cache = mock(KvCacheManager.class);
        var service = localService(cache, mock(CacheMetricsReporter.class));
        var worker = WorkerStatus.createDiscovered(RoleType.PREFILL, "group", "127.0.0.1", 8080, 8081, "site");
        var status = new CacheStatus();
        Set<Long> keys = Set.of(1L, 2L);
        status.setCachedKeys(keys);
        worker.cacheMutationLock.lock();
        try {
            worker.publishCacheStatus(status);
        } finally {
            worker.cacheMutationLock.unlock();
        }
        var lease = worker.tryBeginCachePoll(1L);
        worker.cacheMutationLock.lock();
        try {
            org.junit.jupiter.api.Assertions.assertTrue(service.updateEngineBlockCache(worker, lease.pollTrace()).isSuccess());
            verify(cache).updateEngineCache(worker.getIpPort(), RoleType.PREFILL.getCode(), keys, lease.pollTrace());
            service.removeEngineBlockCache(worker, worker.trace());
            verify(cache).removeEngineCache(worker.getIpPort(), worker.trace());
        } finally {
            worker.cacheMutationLock.unlock();
            lease.close();
        }
    }

    @Test
    void mutationWithoutTheGenerationCacheLockFailsBeforeReadingOrChangingKeys() {
        var cache = mock(KvCacheManager.class);
        var metrics = mock(CacheMetricsReporter.class);
        var service = localService(cache, metrics);
        var worker = WorkerStatus.createDiscovered(RoleType.PREFILL, "group", "127.0.0.1", 8080, 8081, "site");

        assertThrows(IllegalStateException.class, () -> service.updateEngineBlockCache(worker, null));
        assertThrows(IllegalStateException.class, () -> service.removeEngineBlockCache(worker, worker.trace()));
        // Holding the health lock must not be mistaken for cache ownership.
        worker.lock.lock();
        try {
            assertThrows(IllegalStateException.class, () -> service.updateEngineBlockCache(worker, null));
            assertThrows(IllegalStateException.class, () -> service.removeEngineBlockCache(worker, worker.trace()));
        } finally {
            worker.lock.unlock();
        }
        org.mockito.Mockito.verifyNoInteractions(cache, metrics);
    }

    @Test
    void cacheLookupFailureIsNotConvertedIntoACacheMiss() {
        KvCacheManager cache = mock(KvCacheManager.class);
        CacheMetricsReporter metrics = mock(CacheMetricsReporter.class);
        DefaultCacheAwareService service =
                localService(cache, metrics);
        IllegalStateException failure =
                new IllegalStateException("index unavailable");
        List<Long> keys = List.of(1L);
        List<String> candidates = List.of("127.0.0.1:8080");
        when(cache.findMatchingEngines(keys, candidates)).thenThrow(failure);

        IllegalStateException actual = assertThrows(
                IllegalStateException.class,
                () -> service.findMatchingEngines(
                        keys, RoleType.PREFILL, candidates, 0L));

        assertSame(failure, actual);
        verify(metrics).reportFindMatchingEnginesRT(
                eq(RoleType.PREFILL), anyLong(), eq("1"));
    }

    @Test void enabledFailureUsesStandbyInsteadOfSynchronizedIndex() {
        var cache = mock(KvCacheManager.class);
        var kvcm = mock(org.flexlb.engine.grpc.client.KvcmGrpcClient.class);
        var config = mock(org.flexlb.config.CacheMatchConfiguration.class);
        var failover = mock(org.flexlb.cache.match.CacheMatchFailoverManager.class);
        var standby = mock(org.flexlb.cache.match.localstandby.LocalStandbyCacheManager.class);
        when(config.isKvcmEnabled()).thenReturn(true);
        when(failover.activeSource()).thenReturn(org.flexlb.cache.domain.CacheMatchSource.KVCM);
        var service = new DefaultCacheAwareService(cache, mock(CacheMetricsReporter.class), kvcm, config, failover, standby);
        var keys = List.of(11L, 22L); var candidates = List.of("127.0.0.1:18080");
        when(kvcm.findMatchingEngines("", keys, 16, RoleType.PREFILL, null))
                .thenReturn(java.util.Map.of(candidates.getFirst() + "@0", new org.flexlb.dao.cache.HostCacheMatch(2, 2)));
        org.junit.jupiter.api.Assertions.assertEquals(java.util.Map.of(candidates.getFirst(), 2),
                service.findMatchingEngines(keys, RoleType.PREFILL, candidates, 16));
        org.mockito.Mockito.verifyNoInteractions(cache);
        when(kvcm.findMatchingEngines("", keys, 16, RoleType.PREFILL, null))
                .thenThrow(new org.flexlb.exception.KvcmQueryException("unavailable"));
        when(standby.findMatchingEngines(keys, RoleType.PREFILL, candidates, 16)).thenReturn(java.util.Map.of(candidates.getFirst(), 1));
        org.junit.jupiter.api.Assertions.assertEquals(java.util.Map.of(candidates.getFirst(), 1),
                service.findMatchingEngines(keys, RoleType.PREFILL, candidates, 16));
        org.mockito.Mockito.verifyNoInteractions(cache);
    }
    private static DefaultCacheAwareService localService(KvCacheManager cache, CacheMetricsReporter metrics) {
        return new DefaultCacheAwareService(cache, metrics,
                mock(org.flexlb.engine.grpc.client.KvcmGrpcClient.class),
                mock(org.flexlb.config.CacheMatchConfiguration.class),
                mock(org.flexlb.cache.match.CacheMatchFailoverManager.class),
                mock(org.flexlb.cache.match.localstandby.LocalStandbyCacheManager.class));
    }
}
