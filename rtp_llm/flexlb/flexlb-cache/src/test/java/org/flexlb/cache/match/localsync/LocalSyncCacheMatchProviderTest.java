package org.flexlb.cache.match.localsync;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.flexlb.cache.CacheMatchTestConfigurations.kvcm;
import static org.flexlb.cache.CacheMatchTestConfigurations.localSync;
import static org.flexlb.cache.WorkerStatusTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

class LocalSyncCacheMatchProviderTest {

    @Test
    void subscribesToAddressCleanupOnlyInLocalSyncMode() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        LocalSyncCacheMatchProvider provider = new LocalSyncCacheMatchProvider(
                kvCacheManager,
                mock(CacheMetricsReporter.class),
                addressResolver,
                localSync(modelMetaConfig(false)));
        verify(addressResolver).subscribe(provider);

        provider.onAddressUpdate(List.of(new WorkerHost("10.0.0.1", 8080)));

        verify(kvCacheManager).removeStaleEngineCaches(List.of("10.0.0.1:8080"));
    }

    @Test
    void doesNotSubscribeToAddressCleanupInKvcmMode() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);

        new LocalSyncCacheMatchProvider(
                mock(KvCacheManager.class),
                mock(CacheMetricsReporter.class),
                addressResolver,
                kvcm(modelMetaConfig(true)));

        verify(addressResolver, never()).subscribe(any());
    }

    @Test
    void localProviderUpdatesLocalCache() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider provider =
                new LocalSyncCacheMatchProvider(kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatusFixture();

        WorkerCacheUpdateResult result = provider.updateFromWorkerStatus(workerStatus);

        assertTrue(result.isSuccess());
        assertEquals(2, result.getCacheBlockCount());
        assertEquals(100L, result.getAvailableKvCache());
        assertEquals(200L, result.getTotalKvCache());
        assertEquals(3L, result.getCacheVersion());
        verify(kvCacheManager).updateEngineCache(
                workerStatus, "PREFILL", Set.of(11L, 22L));
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1:8080"), eq("PREFILL"), anyLong(), eq("1"));
    }

    @Test
    void rejectsMissingCacheStatus() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider provider =
                new LocalSyncCacheMatchProvider(kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatus(
                "127.0.0.1", 8080, RoleType.PREFILL, false, null);

        WorkerCacheUpdateResult result = provider.updateFromWorkerStatus(workerStatus);

        assertFalse(result.isSuccess());
        assertEquals("127.0.0.1:8080@0", result.getLogicalIpPort());
        assertEquals("Worker Cache Status is null", result.getErrorMessage());
        verifyNoInteractions(kvCacheManager);
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1:8080"), eq("PREFILL"), anyLong(), eq("0"));
    }

    @Test
    void rejectsMissingCachedKeys() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider provider =
                new LocalSyncCacheMatchProvider(kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatusWithCachedKeys(null);

        WorkerCacheUpdateResult result = provider.updateFromWorkerStatus(workerStatus);

        assertFalse(result.isSuccess());
        assertEquals("Worker Cached Keys is null", result.getErrorMessage());
        verifyNoInteractions(kvCacheManager);
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1:8080"), eq("PREFILL"), anyLong(), eq("0"));
    }

    @Test
    void acceptsEmptyCachedKeys() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider provider =
                new LocalSyncCacheMatchProvider(kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatusWithCachedKeys(Set.of());

        WorkerCacheUpdateResult result = provider.updateFromWorkerStatus(workerStatus);

        assertTrue(result.isSuccess());
        assertEquals(0, result.getCacheBlockCount());
        verify(kvCacheManager).updateEngineCache(
                workerStatus, "PREFILL", Set.of());
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1:8080"), eq("PREFILL"), anyLong(), eq("1"));
    }

    @Test
    void reportsCacheManagerFailure() {
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalSyncCacheMatchProvider provider =
                new LocalSyncCacheMatchProvider(kvCacheManager, metricsReporter);
        WorkerStatus workerStatus = workerStatusFixture();
        doThrow(new IllegalStateException("cache update failed"))
                .when(kvCacheManager)
                .updateEngineCache(workerStatus, "PREFILL", Set.of(11L, 22L));

        WorkerCacheUpdateResult result = provider.updateFromWorkerStatus(workerStatus);

        assertFalse(result.isSuccess());
        assertEquals("cache update failed", result.getErrorMessage());
        verify(metricsReporter).reportUpdateEngineBlockCacheRT(
                eq("127.0.0.1:8080"), eq("PREFILL"), anyLong(), eq("0"));
    }

    private WorkerStatus workerStatusFixture() {
        return workerStatus("127.0.0.1", 8080, RoleType.PREFILL, false,
                CacheStatus.builder()
                .cachedKeys(Set.of(11L, 22L))
                .availableKvCache(100L)
                .totalKvCache(200L)
                .version(3L)
                .build());
    }

    private WorkerStatus workerStatusWithCachedKeys(Set<Long> cachedKeys) {
        return workerStatus("127.0.0.1", 8080, RoleType.PREFILL, false,
                CacheStatus.builder().cachedKeys(cachedKeys).build());
    }

    private ModelMetaConfig modelMetaConfig(boolean kvcmEnabled) {
        ServiceRoute serviceRoute = new ServiceRoute();
        serviceRoute.setServiceId("test-service");
        if (kvcmEnabled) {
            serviceRoute.setKvcm(new KvcmConfig());
        }

        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(serviceRoute.getServiceId(), serviceRoute);
        return modelMetaConfig;
    }
}
