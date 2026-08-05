package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheHitFeedback;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheAwareServiceTest {

    private final CacheMetricsReporter metricsReporter = Mockito.mock(CacheMetricsReporter.class);
    private final CacheMatchQueryOrchestrator queryOrchestrator =
            Mockito.mock(CacheMatchQueryOrchestrator.class);
    private final LocalStandbyComparisonService comparisonService =
            Mockito.mock(LocalStandbyComparisonService.class);
    private final CacheMetadataUpdateOrchestrator updateOrchestrator =
            Mockito.mock(CacheMetadataUpdateOrchestrator.class);

    @Test
    void delegatesCacheQueriesToOrchestrator() {
        when(queryOrchestrator.effectiveSource()).thenReturn(CacheMatchSource.KVCM);
        when(queryOrchestrator.findMatchingEngines(any(CacheMatchQuery.class)))
                .thenReturn(new CacheMatchResult(
                        Map.of("127.0.0.1:8080", HostCacheMatch.local(1)), CacheMatchSource.KVCM, 10, 2192));

        CacheMatchResult result = service().findMatchingEngines(new CacheMatchQuery(
                "request-1",
                List.of(1L),
                2192L,
                List.of(1L),
                2192L,
                RoleType.PREFILL,
                "default"));

        assertEquals(1, result.hostMatch("127.0.0.1:8080").localMatchBlocks());
        assertEquals(CacheMatchSource.KVCM, result.source());
        verify(queryOrchestrator).findMatchingEngines(any(CacheMatchQuery.class));
    }

    @Test
    void delegatesEmptyCacheKeysToOrchestrator() {
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1",
                List.of(),
                2192,
                List.of(),
                4096,
                RoleType.PREFILL,
                "default");
        CacheMatchResult expected =
                CacheMatchResult.empty(CacheMatchSource.LOCAL_STANDBY);
        when(queryOrchestrator.findMatchingEngines(query)).thenReturn(expected);
        CacheMatchResult result = service().findMatchingEngines(query);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        assertEquals(0, result.blockSize());
        verify(queryOrchestrator).findMatchingEngines(query);
    }

    @Test
    void pendingLocalStandbyHashDelegatesToOrchestrator() {
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1",
                List.of(),
                2192,
                null,
                1024,
                RoleType.PREFILL,
                "default");
        CacheMatchResult expected = new CacheMatchResult(
                Map.of("127.0.0.1:8080", HostCacheMatch.local(1)), CacheMatchSource.LOCAL_STANDBY, 10, 1024);
        when(queryOrchestrator.effectiveSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        when(queryOrchestrator.findMatchingEngines(query)).thenReturn(expected);
        CacheMatchResult result = service().findMatchingEngines(query);

        assertSame(expected, result);
        verify(queryOrchestrator).findMatchingEngines(query);
    }

    @Test
    void failedCacheQueryHasNoBlockSize() {
        when(queryOrchestrator.effectiveSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        when(queryOrchestrator.findMatchingEngines(any(CacheMatchQuery.class)))
                .thenThrow(new IllegalStateException("query failed"));
        CacheMatchResult result = service().findMatchingEngines(new CacheMatchQuery(
                "request-1",
                List.of(1L),
                2192,
                List.of(1L),
                4096,
                RoleType.PREFILL,
                "default"));

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        assertEquals(0, result.blockSize());
    }

    @Test
    void delegatesWorkerStatusUpdatesToUpdateOrchestrator() {
        WorkerStatus workerStatus = new WorkerStatus();
        WorkerCacheUpdateResult expected = WorkerCacheUpdateResult.builder()
                .success(true)
                .build();
        when(updateOrchestrator.updateFromWorkerStatus(workerStatus))
                .thenReturn(expected);

        WorkerCacheUpdateResult actual = service().updateFromWorkerStatus(workerStatus);

        assertSame(expected, actual);
        verify(updateOrchestrator).updateFromWorkerStatus(workerStatus);
    }

    @Test
    void delegatesRoutedRequestUpdatesToUpdateOrchestrator() {
        Request request = new Request();
        request.setRequestId("request-1");
        List<ServerStatus> selectedWorkers = List.of(new ServerStatus());

        service().updateFromRoutedRequest(request, selectedWorkers);

        verify(updateOrchestrator).updateFromRoutedRequest(request, selectedWorkers);
    }

    @Test
    void delegatesCacheHitComparisonBuildingToComparisonService() {
        CacheHitFeedback feedback = new CacheHitFeedback(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                "127.0.0.1", 8080, "running", 200, 64, 100, 120, 20);
        CacheHitComparisonResult expected = new CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                "127.0.0.1", 8080, "running", 200, 64, 4096,
                100, 0, true, 120, 20, 120);
        when(comparisonService.buildCacheHitComparison(feedback))
                .thenReturn(CompletableFuture.completedFuture(expected));

        CompletableFuture<CacheHitComparisonResult> result =
                service().buildCacheHitComparison(feedback);

        assertSame(expected, result.join());
        verify(comparisonService).buildCacheHitComparison(feedback);
    }

    private CacheAwareService service() {
        return new CacheAwareService(
                metricsReporter,
                queryOrchestrator,
                comparisonService,
                updateOrchestrator);
    }
}
