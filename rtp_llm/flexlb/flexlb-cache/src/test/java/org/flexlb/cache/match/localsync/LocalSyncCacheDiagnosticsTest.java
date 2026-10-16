package org.flexlb.cache.match.localsync;

import org.flexlb.cache.match.localsync.KvCacheManager;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
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

class LocalSyncCacheDiagnosticsTest {

    @Test
    void cachePollAndRetirementKeepTheirExactDiagnosticOwner() {
        var cache = mock(KvCacheManager.class);
        var service = new LocalSyncCacheMatchProvider(cache, mock(CacheMetricsReporter.class));
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
            org.junit.jupiter.api.Assertions.assertTrue(service.updateFromWorkerStatus(worker, lease.pollTrace()).isSuccess());
            verify(cache).updateEngineCache(worker, RoleType.PREFILL.getCode(), keys, lease.pollTrace());
            service.removeEngineBlockCache(worker.getLogicalIpPort(), worker.trace());
            verify(cache).removeEngineCache(worker.getLogicalIpPort(), worker.trace());
        } finally {
            lease.close();
        }
    }

    @Test
    void cacheLookupFailureIsNotConvertedIntoACacheMiss() {
        KvCacheManager cache = mock(KvCacheManager.class);
        CacheMetricsReporter metrics = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider service =
                new LocalSyncCacheMatchProvider(cache, metrics);
        IllegalStateException failure =
                new IllegalStateException("index unavailable");
        List<Long> keys = List.of(1L);
        List<String> candidates = List.of("127.0.0.1:8080");
        when(cache.findMatchingEngines(keys, RoleType.PREFILL, "group")).thenThrow(failure);

        IllegalStateException actual = assertThrows(
                IllegalStateException.class,
                () -> service.findMatchingEngines(
                        "request", keys, 1L, RoleType.PREFILL, "group"));

        assertSame(failure, actual);

    }
}
