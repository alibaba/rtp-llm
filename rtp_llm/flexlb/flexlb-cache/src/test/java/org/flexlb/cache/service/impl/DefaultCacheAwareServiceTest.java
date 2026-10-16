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
        try {
            org.junit.jupiter.api.Assertions.assertTrue(service.updateEngineBlockCache(worker, lease.pollTrace()).isSuccess());
            verify(cache).updateEngineCache(worker.getIpPort(), RoleType.PREFILL.getCode(), keys, lease.pollTrace());
            service.removeEngineBlockCache(worker.getIpPort(), worker.trace());
            verify(cache).removeEngineCache(worker.getIpPort(), worker.trace());
        } finally {
            lease.close();
        }
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
                        keys, RoleType.PREFILL, candidates));

        assertSame(failure, actual);
        verify(metrics).reportFindMatchingEnginesRT(
                eq(RoleType.PREFILL), anyLong(), eq("1"));
    }
}
