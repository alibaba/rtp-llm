package org.flexlb.cache.localsync;

import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.service.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks that monitoring failures cannot invalidate an applied cache update.
 */
class LocalSyncCacheMatchProviderSafetyTest {

    @Test
    void reportingFailureCannotTurnAnAppliedCacheUpdateIntoFailure() {
        var cache = mock(KvCacheManager.class);
        var metrics = mock(CacheMetricsReporter.class);
        org.mockito.Mockito.doThrow(new IllegalStateException("monitor unavailable"))
                .when(metrics)
                .reportUpdateEngineBlockCacheRT(
                        org.mockito.ArgumentMatchers.anyString(),
                        org.mockito.ArgumentMatchers.anyString(),
                        anyLong(),
                        org.mockito.ArgumentMatchers.anyString());
        var worker =
                org.flexlb.worker.status.WorkerStatus.createDiscovered(
                        RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null);
        var status = new org.flexlb.worker.cache.CacheStatus();
        status.setCachedKeys(java.util.Set.of(1L));
        worker.cacheMutationLock.lock();
        try {
            worker.publishCacheStatus(status);
        } finally {
            worker.cacheMutationLock.unlock();
        }
        var result = new LocalSyncCacheMatchProvider(cache, metrics).updateFromWorkerStatus(worker);
        org.junit.jupiter.api.Assertions.assertTrue(result.isSuccess());
        verify(cache).updateEngineCache(worker, RoleType.PREFILL.name(), java.util.Set.of(1L));
    }

    @Test
    void cacheLookupFailureIsNotConvertedIntoACacheMiss() {
        KvCacheManager cache = mock(KvCacheManager.class);
        CacheMetricsReporter metrics = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider service = new LocalSyncCacheMatchProvider(cache, metrics);
        IllegalStateException failure = new IllegalStateException("index unavailable");
        List<Long> keys = List.of(1L);
        when(cache.findMatchingEngines(keys, RoleType.PREFILL, "group")).thenThrow(failure);

        IllegalStateException actual =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                service.findMatchingEngines(
                                        "request", keys, 16, RoleType.PREFILL, "group"));

        assertSame(failure, actual);
    }
}
