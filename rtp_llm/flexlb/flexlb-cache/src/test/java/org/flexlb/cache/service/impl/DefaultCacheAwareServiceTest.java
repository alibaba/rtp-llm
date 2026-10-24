package org.flexlb.cache.service.impl;

import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DefaultCacheAwareServiceTest {

    @Test
    void reportingFailureCannotTurnAnAppliedCacheUpdateIntoFailure() {
        var cache = mock(KvCacheManager.class);
        var metrics = mock(CacheMetricsReporter.class);
        org.mockito.Mockito.doThrow(new IllegalStateException("monitor unavailable"))
                .when(metrics).reportUpdateEngineBlockCacheRT(org.mockito.ArgumentMatchers.anyString(), anyLong(), org.mockito.ArgumentMatchers.anyString());
        var worker = org.flexlb.dao.master.WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null);
        var status = new org.flexlb.dao.master.CacheStatus();
        status.setCachedKeys(java.util.Set.of(1L));
        worker.cacheMutationLock.lock();
        try { worker.publishCacheStatus(status); } finally { worker.cacheMutationLock.unlock(); }
        var result = new DefaultCacheAwareService(cache, metrics).updateEngineBlockCache(worker);
        org.junit.jupiter.api.Assertions.assertTrue(result.isSuccess());
        verify(cache).updateEngineCache("127.0.0.1:8080", RoleType.PREFILL.getCode(), java.util.Set.of(1L));
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
