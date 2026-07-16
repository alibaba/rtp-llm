package org.flexlb.cache.service.impl;

import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheMatchProvider;
import org.flexlb.cache.service.CacheMatchResult;
import org.flexlb.cache.service.CacheMatchSource;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DefaultCacheAwareServiceTest {

    private final CacheMatchProvider localProvider = Mockito.mock(CacheMatchProvider.class);
    private final CacheMatchProvider kvcmProvider = Mockito.mock(CacheMatchProvider.class);
    private final CacheMetricsReporter metricsReporter = Mockito.mock(CacheMetricsReporter.class);

    @Test
    void usesKvcmWithoutFallingBackToLocalWhenRemoteQueryFails() {
        when(localProvider.source()).thenReturn(CacheMatchSource.LOCAL);
        when(kvcmProvider.source()).thenReturn(CacheMatchSource.KVCM);
        when(kvcmProvider.findMatchingEngines(
                "request-1", List.of(1L), RoleType.PREFILL, "default"))
                .thenThrow(new IllegalStateException("KVCM unavailable"));

        DefaultCacheAwareService service = new DefaultCacheAwareService(
                List.of(localProvider, kvcmProvider), modelMetaConfig(true), metricsReporter);

        CacheMatchResult result = service.findMatchingEngines(
                "request-1", List.of(1L), RoleType.PREFILL, "default");

        assertTrue(result.matches().isEmpty());
        assertEquals(CacheMatchSource.KVCM, result.source());
        assertTrue(result.queryTimeUs() >= 0);
        verify(localProvider, never()).findMatchingEngines(
                "request-1", List.of(1L), RoleType.PREFILL, "default");
    }

    @Test
    void usesLocalProviderWhenKvcmIsDisabled() {
        when(localProvider.source()).thenReturn(CacheMatchSource.LOCAL);
        when(kvcmProvider.source()).thenReturn(CacheMatchSource.KVCM);
        when(localProvider.findMatchingEngines(
                "request-1", List.of(1L), RoleType.PREFILL, "default"))
                .thenReturn(Map.of("127.0.0.1:8080", 1));

        DefaultCacheAwareService service = new DefaultCacheAwareService(
                List.of(localProvider, kvcmProvider), modelMetaConfig(false), metricsReporter);

        CacheMatchResult result = service.findMatchingEngines(
                "request-1", List.of(1L), RoleType.PREFILL, "default");

        assertEquals(1, result.matches().get("127.0.0.1:8080"));
        assertEquals(CacheMatchSource.LOCAL, result.source());
        verify(kvcmProvider, never()).findMatchingEngines(
                "request-1", List.of(1L), RoleType.PREFILL, "default");
    }

    @Test
    void delegatesCacheUpdatesToSelectedLocalProvider() {
        when(localProvider.source()).thenReturn(CacheMatchSource.LOCAL);
        when(kvcmProvider.source()).thenReturn(CacheMatchSource.KVCM);
        WorkerStatus workerStatus = new WorkerStatus();
        WorkerCacheUpdateResult expected = WorkerCacheUpdateResult.builder()
                .success(true)
                .build();
        when(localProvider.updateEngineBlockCache(workerStatus)).thenReturn(expected);

        DefaultCacheAwareService service = new DefaultCacheAwareService(
                List.of(localProvider, kvcmProvider), modelMetaConfig(false), metricsReporter);

        WorkerCacheUpdateResult actual = service.updateEngineBlockCache(workerStatus);

        assertSame(expected, actual);
        verify(kvcmProvider, never()).updateEngineBlockCache(workerStatus);
    }

    private ModelMetaConfig modelMetaConfig(boolean kvcmEnabled) {
        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(kvcmEnabled);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);

        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return config;
    }
}
