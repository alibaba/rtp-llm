package org.flexlb.cache.service.impl;

import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

class CacheMatchProviderUpdateTest {

    @Test
    void localProviderUpdatesLocalCache() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalCacheMatchProvider provider = new LocalCacheMatchProvider(
                kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatus();

        WorkerCacheUpdateResult result = provider.updateEngineBlockCache(workerStatus);

        assertTrue(result.isSuccess());
        assertEquals(2, result.getCacheBlockCount());
        verify(kvCacheManager).updateEngineCache(
                "127.0.0.1:8080", "PREFILL", Set.of(11L, 22L));
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1:8080"), eq("PREFILL"), anyLong(), eq("1"));
    }

    @Test
    void kvcmProviderRejectsLocalCacheUpdatesWithoutCallingKvcm() {
        KvcmGrpcClient kvcmGrpcClient = mock(KvcmGrpcClient.class);
        KvcmCacheMatchProvider provider = new KvcmCacheMatchProvider(kvcmGrpcClient);

        WorkerCacheUpdateResult result = provider.updateEngineBlockCache(workerStatus());

        assertFalse(result.isSuccess());
        assertEquals("Local cache updates are disabled when KVCM is enabled",
                result.getErrorMessage());
        verifyNoInteractions(kvcmGrpcClient);
    }

    private WorkerStatus workerStatus() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);
        workerStatus.setRole("PREFILL");
        workerStatus.setCacheStatus(CacheStatus.builder()
                .cachedKeys(Set.of(11L, 22L))
                .availableKvCache(100L)
                .totalKvCache(200L)
                .version(3L)
                .build());
        return workerStatus;
    }
}
