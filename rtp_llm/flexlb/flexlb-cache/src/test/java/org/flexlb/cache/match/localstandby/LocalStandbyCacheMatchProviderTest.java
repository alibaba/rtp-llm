package org.flexlb.cache.match.localstandby;

import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.flexlb.cache.CacheMatchTestConfigurations.kvcm;
import static org.flexlb.cache.CacheMatchTestConfigurations.localSync;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class LocalStandbyCacheMatchProviderTest {

    private final CacheMetricsReporter reporter = mock(CacheMetricsReporter.class);

    @Test
    void disabledLocalStandbyDoesNotAllocateExecutors() {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyCacheMatchProvider provider = new LocalStandbyCacheMatchProvider(
                localSync(modelMetaConfig()), cacheManager, reporter);
        try {
            assertNull(ReflectionTestUtils.getField(provider, "asyncMatchExecutor"));
            assertNull(ReflectionTestUtils.getField(provider, "updateExecutor"));
            assertTrue(provider.asyncLocalStandbyMatch(new CacheMatchQuery(
                    "request-disabled", List.of(11L), 2192L, RoleType.PREFILL, "default")).join().hostMatches().isEmpty());
            verifyNoInteractions(cacheManager);
        } finally {
            provider.shutdown();
        }
    }

    @Test
    void rejectsInvalidQueueCapacityWhenEnabled() {
        for (int capacity : new int[]{0, -1}) {
            var configuration = kvcm(modelMetaConfig(),
                    runtime -> runtime.getLocalStandby().setAsyncQueueCapacity(capacity));
            IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                    () -> new LocalStandbyCacheMatchProvider(
                            configuration, mock(LocalStandbyCacheManager.class), reporter));
            assertTrue(error.getMessage().contains("asyncQueueCapacity"));
        }
    }

    @Test
    void matchesClientProvidedKeysWithClientBlockSize() throws Exception {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyCacheMatchProvider provider = new LocalStandbyCacheMatchProvider(
                kvcm(modelMetaConfig()), cacheManager, reporter);
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default");
        when(cacheManager.findMatchingEngines(List.of(11L), RoleType.PREFILL, "default"))
                .thenReturn(Map.of("10.0.0.1:8080@0", 1));

        try {
            CacheMatchResult result = provider.asyncLocalStandbyMatch(query).get(1, TimeUnit.SECONDS);
            assertEquals(1, result.exactHostMatch("10.0.0.1:8080@0").localMatchBlocks());
            assertEquals(2192, result.blockSize());
            verify(reporter).reportLocalStandbyBlockSize(RoleType.PREFILL, 2192L);
            verify(cacheManager).findMatchingEngines(List.of(11L), RoleType.PREFILL, "default");
        } finally {
            provider.shutdown();
        }
    }

    @Test
    void emptyClientKeysAreAValidZeroMatch() throws Exception {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyCacheMatchProvider provider = new LocalStandbyCacheMatchProvider(
                kvcm(modelMetaConfig()), cacheManager, reporter);
        CacheMatchQuery query = new CacheMatchQuery(
                "short-request", List.of(), 4096, RoleType.PREFILL, "default");
        try {
            CacheMatchResult result = provider.asyncLocalStandbyMatch(query).get(1, TimeUnit.SECONDS);
            assertEquals(4096, result.blockSize());
            assertEquals(Map.of(), result.hostMatches());
            verifyNoInteractions(cacheManager);
        } finally {
            provider.shutdown();
        }
    }

    @Test
    void unavailableMatchExecutorReturnsFailedPrediction() throws Exception {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyCacheMatchProvider provider = new LocalStandbyCacheMatchProvider(
                kvcm(modelMetaConfig()), cacheManager, reporter);
        provider.shutdown();
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1", List.of(11L), 4096, RoleType.PREFILL, "default");
        assertFalse(provider.asyncLocalStandbyMatch(query).get(1, TimeUnit.SECONDS).querySucceeded());
        verifyNoInteractions(cacheManager);
    }

    @ParameterizedTest
    @CsvSource({"0, 1, 10.0.0.1:8080@0", "1, 2, 10.0.0.1:8080@1"})
    void updatesIndexWithClientProvidedKeys(int engineIndex, int multiEngineNum, String logicalIpPort) {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyCacheMatchProvider provider = new LocalStandbyCacheMatchProvider(
                kvcm(modelMetaConfig()), cacheManager, reporter);
        Request request = request();
        ServerStatus selectedWorker = worker(RoleType.PREFILL);
        selectedWorker.setSelectedEngineIndex(engineIndex, multiEngineNum);
        try {
            provider.updateFromRoutedRequest(request, List.of(selectedWorker));
            verify(cacheManager, timeout(1_000)).addRoutedRequestBlocks(logicalIpPort, List.of(11L, 22L));
        } finally {
            provider.shutdown();
        }
    }

    @Test
    void ignoresCacheMetadataForNonPrefillWorkers() {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyCacheMatchProvider provider = new LocalStandbyCacheMatchProvider(
                kvcm(modelMetaConfig()), cacheManager, reporter);
        try {
            provider.updateFromRoutedRequest(request(), List.of(worker(RoleType.DECODE)));
        } finally {
            provider.shutdown();
        }
        verify(cacheManager, after(200).never()).addRoutedRequestBlocks(anyString(), anyList());
    }

    @Test
    void missingClientKeysOrSelectedWorkersDoNotPopulateIndex() {
        LocalStandbyCacheManager cacheManager = mock(LocalStandbyCacheManager.class);
        LocalStandbyCacheMatchProvider provider = new LocalStandbyCacheMatchProvider(
                kvcm(modelMetaConfig()), cacheManager, reporter);
        Request request = request();
        request.setBlockCacheKeys(null);
        try {
            provider.updateFromRoutedRequest(request(), null);
            provider.updateFromRoutedRequest(request(), List.of());
            provider.updateFromRoutedRequest(request, List.of(worker(RoleType.PREFILL)));
            verifyNoInteractions(cacheManager);
        } finally {
            provider.shutdown();
        }
    }

    private Request request() {
        Request request = new Request();
        request.setRequestId("1");
        request.setBlockSize(4096);
        request.setBlockCacheKeys(List.of(11L, 22L));
        return request;
    }

    private ServerStatus worker(RoleType role) {
        ServerStatus worker = new ServerStatus();
        worker.setSuccess(true);
        worker.setServerIp("10.0.0.1");
        worker.setHttpPort(8080);
        worker.setRole(role);
        worker.setGroup("default");
        return worker;
    }

    private ModelMetaConfig modelMetaConfig() {
        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(new KvcmConfig());
        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return config;
    }
}
