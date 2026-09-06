package org.flexlb.cache.match.localsync;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

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
        verify(kvCacheManager).updateEngineCache("127.0.0.1:8080", "PREFILL", Set.of(11L, 22L));
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(eq("127.0.0.1:8080"), eq("PREFILL"), anyLong(), eq("1"));
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
