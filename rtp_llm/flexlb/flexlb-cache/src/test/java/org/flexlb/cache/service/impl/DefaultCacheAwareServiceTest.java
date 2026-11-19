package org.flexlb.cache.service.impl;

import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.KvcmGrpcClient.MatchObservation;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies worker cache updates and the routing lookup's local and KVCM paths.
 */
class DefaultCacheAwareServiceTest {

    @Test
    void cachePollAndRetirementKeepTheirExactDiagnosticOwner() {
        var cache = mock(KvCacheManager.class);
        var service = new DefaultCacheAwareService(cache, mock(CacheMetricsReporter.class));
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
        var service = new DefaultCacheAwareService(cache, metrics);
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
                new DefaultCacheAwareService(cache, metrics);
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

    @Test void enabledUsesKvcmAndFailureFallsBackToStandby() {
        var cache = mock(KvCacheManager.class);
        var kvcm = mock(org.flexlb.engine.grpc.KvcmGrpcClient.class);
        var standby = mock(org.flexlb.cache.match.localstandby.LocalStandbyCacheManager.class);
        var service = new DefaultCacheAwareService(cache, mock(CacheMetricsReporter.class), kvcm, standby);
        var keys = List.of(11L, 22L); var candidates = List.of("127.0.0.1:18080");
        when(kvcm.enabled()).thenReturn(true);
        when(kvcm.lookup(keys, 16, candidates)).thenReturn(
                new MatchObservation(Map.of(candidates.getFirst(), 1), Map.of(candidates.getFirst(), 2L)));
        org.junit.jupiter.api.Assertions.assertEquals(Map.of(candidates.getFirst(), 1),
                service.findMatchingEngines(keys, RoleType.PREFILL, candidates, 16));
        org.junit.jupiter.api.Assertions.assertEquals(Map.of(candidates.getFirst(), 2L),
                service.lookup(keys, RoleType.PREFILL, candidates, 16).globalMatches());
        org.mockito.Mockito.verifyNoInteractions(cache);
        when(kvcm.lookup(keys, 16, candidates)).thenThrow(new IllegalStateException("unavailable"));
        when(standby.findMatchingEngines(keys, RoleType.PREFILL, candidates, 16)).thenReturn(java.util.Map.of(candidates.getFirst(), 1));
        org.junit.jupiter.api.Assertions.assertEquals(java.util.Map.of(candidates.getFirst(), 1),
                service.findMatchingEngines(keys, RoleType.PREFILL, candidates, 16));
    }

    @Test
    void reportsPairedCountersOnlyForSuccessfulRoutesWithGlobalEvidence() {
        var metrics = mock(CacheMetricsReporter.class);
        var service = new DefaultCacheAwareService(mock(KvCacheManager.class), metrics);
        Request request = new Request();
        request.setSeqLen(64);
        request.setCacheKeyBlockSize(16);
        ServerStatus selected = new ServerStatus();
        selected.setRole(RoleType.PREFILL);
        selected.setServerIp("127.0.0.1");
        selected.setHttpPort(8080);
        selected.setSuccess(true);
        selected.setKvcmMatchedBlocks(1);
        service.recordRoutedRequest(request, List.of(selected));
        org.mockito.Mockito.verifyNoInteractions(metrics);

        selected.setKvcmGlobalMatchTokens(48L);
        service.recordRoutedRequest(request, List.of(selected));
        verify(metrics).reportKvcmSelectedMatch(RoleType.PREFILL, "127.0.0.1:8080", 16L, 48L, 64L);
    }

    @Test void disabledBypassesKvcmEvenWithConfiguredTopology() {
        var cache = mock(KvCacheManager.class);
        var kvcm = mock(org.flexlb.engine.grpc.KvcmGrpcClient.class);
        var standby = mock(org.flexlb.cache.match.localstandby.LocalStandbyCacheManager.class);
        var service = new DefaultCacheAwareService(cache, mock(CacheMetricsReporter.class), kvcm, standby);
        service.findMatchingEngines(List.of(11L), RoleType.PREFILL, List.of("host:1"), 16);
        org.mockito.Mockito.verify(kvcm).enabled();
        org.mockito.Mockito.verifyNoMoreInteractions(kvcm);
        verify(cache).findMatchingEngines(List.of(11L), List.of("host:1"));
    }
}
