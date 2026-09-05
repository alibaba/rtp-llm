package org.flexlb.cache.match.localsync;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerIdentity;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

class LocalSyncCacheMatchProviderTest {

    @Test
    void localProviderUpdatesLocalCache() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider provider =
                new LocalSyncCacheMatchProvider(kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatus();

        WorkerCacheUpdateResult result = provider.updateFromWorkerStatus(workerStatus);

        assertTrue(result.isSuccess());
        assertEquals(2, result.getCacheBlockCount());
        assertEquals(100L, result.getAvailableKvCache());
        assertEquals(200L, result.getTotalKvCache());
        assertEquals(3L, result.getCacheVersion());
        verify(kvCacheManager).updateEngineCache(
                new WorkerIdentity("127.0.0.1", 8080, 0), "PREFILL", Set.of(11L, 22L));
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1@0"), eq("PREFILL"), anyLong(), eq("1"));
    }

    @Test
    void rejectsMissingCacheStatus() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider provider =
                new LocalSyncCacheMatchProvider(kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatus();
        workerStatus.setCacheStatus(null);

        WorkerCacheUpdateResult result = provider.updateFromWorkerStatus(workerStatus);

        assertFalse(result.isSuccess());
        assertEquals("127.0.0.1:8080@0", result.getLogicalIpPort());
        assertEquals("Worker Cache Status is null", result.getErrorMessage());
        verifyNoInteractions(kvCacheManager);
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1@0"), eq("PREFILL"), anyLong(), eq("0"));
    }

    @Test
    void rejectsMissingCachedKeys() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider provider =
                new LocalSyncCacheMatchProvider(kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatus();
        workerStatus.getCacheStatus().setCachedKeys(null);

        WorkerCacheUpdateResult result = provider.updateFromWorkerStatus(workerStatus);

        assertFalse(result.isSuccess());
        assertEquals("Worker Cached Keys is null", result.getErrorMessage());
        verifyNoInteractions(kvCacheManager);
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1@0"), eq("PREFILL"), anyLong(), eq("0"));
    }

    @Test
    void acceptsEmptyCachedKeys() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider provider =
                new LocalSyncCacheMatchProvider(kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatus();
        workerStatus.getCacheStatus().setCachedKeys(Set.of());

        WorkerCacheUpdateResult result = provider.updateFromWorkerStatus(workerStatus);

        assertTrue(result.isSuccess());
        assertEquals(0, result.getCacheBlockCount());
        verify(kvCacheManager).updateEngineCache(
                new WorkerIdentity("127.0.0.1", 8080, 0), "PREFILL", Set.of());
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1@0"), eq("PREFILL"), anyLong(), eq("1"));
    }

    @Test
    void reportsCacheManagerFailure() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider provider =
                new LocalSyncCacheMatchProvider(kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatus();
        doThrow(new IllegalStateException("cache update failed"))
                .when(kvCacheManager)
                .updateEngineCache(
                        new WorkerIdentity("127.0.0.1", 8080, 0), "PREFILL", Set.of(11L, 22L));

        WorkerCacheUpdateResult result = provider.updateFromWorkerStatus(workerStatus);

        assertFalse(result.isSuccess());
        assertEquals("cache update failed", result.getErrorMessage());
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1@0"), eq("PREFILL"), anyLong(), eq("0"));
    }

    private WorkerStatus workerStatus() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);
        workerStatus.setRole(RoleType.PREFILL);
        workerStatus.setCacheStatus(CacheStatus.builder()
                .cachedKeys(Set.of(11L, 22L))
                .availableKvCache(100L)
                .totalKvCache(200L)
                .version(3L)
                .build());
        return workerStatus;
    }
}
