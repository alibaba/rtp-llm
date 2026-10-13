package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.flexlb.cache.WorkerStatusTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class CacheAwareServiceTest {

    private final CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
    private final CacheMatchQueryOrchestrator queryOrchestrator =
            mock(CacheMatchQueryOrchestrator.class);
    private final CacheMetadataUpdateOrchestrator updateOrchestrator =
            mock(CacheMetadataUpdateOrchestrator.class);
    private final LocalStandbyComparisonService comparisonService = mock(LocalStandbyComparisonService.class);
    private final CacheAwareService service = new CacheAwareService(
            metricsReporter,
            queryOrchestrator,
            comparisonService,
            updateOrchestrator);

    @Test
    void delegatesCacheQueriesToOrchestrator() {
        CacheMatchQuery query = new CacheMatchQuery(
                "1", List.of(11L), 2192L,
                RoleType.PREFILL, "default");
        CacheMatchResult expected = new CacheMatchResult(
                Map.of("127.0.0.1:8080", HostCacheMatch.local(1)),
                CacheMatchSource.KVCM, 10, 2192);
        when(queryOrchestrator.findMatchingEngines(query)).thenReturn(expected);

        CacheMatchResult actual = service.findMatchingEngines(query);

        assertSame(expected, actual);
        verify(queryOrchestrator).findMatchingEngines(query);
        verify(metricsReporter).reportFindMatchingEnginesRT(
                org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.eq("0"));
    }

    @Test
    void reportsFailedCacheQueryResultAsFailure() {
        CacheMatchQuery query = new CacheMatchQuery(
                "failed-request", List.of(11L), 2192L,
                RoleType.PREFILL, "default");
        CacheMatchResult failed = CacheMatchResult.failed(CacheMatchSource.LOCAL_STANDBY, 10);
        when(queryOrchestrator.findMatchingEngines(query)).thenReturn(failed);

        assertSame(failed, service.findMatchingEngines(query));

        verify(metricsReporter).reportFindMatchingEnginesRT(
                org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.eq("1"));
    }

    @Test
    void delegatesWorkerStatusUpdates() {
        WorkerStatus workerStatus = workerStatus("127.0.0.1", 8080, RoleType.PREFILL);
        WorkerCacheUpdateResult expected = WorkerCacheUpdateResult.builder()
                .success(true)
                .build();
        when(updateOrchestrator.updateFromWorkerStatus(workerStatus)).thenReturn(expected);

        assertSame(expected, service.updateFromWorkerStatus(workerStatus));
        verify(updateOrchestrator).updateFromWorkerStatus(workerStatus);
    }

    @Test
    void convertsUnexpectedQueryFailureToFailedResult() {
        CacheMatchQuery query = new CacheMatchQuery(
                "2", List.of(11L), 2192L,
                RoleType.PREFILL, "default");
        when(queryOrchestrator.findMatchingEngines(query))
                .thenThrow(new IllegalStateException("failed"));
        when(queryOrchestrator.effectiveSource()).thenReturn(CacheMatchSource.KVCM);

        CacheMatchResult result = service.findMatchingEngines(query);

        assertEquals(CacheMatchSource.KVCM, result.source());
        assertEquals(Map.of(), result.hostMatches());
        assertFalse(result.querySucceeded());
    }

    @Test
    void reportsZeroKvcmPredictionBeforeWorkerFeedback() {
        WorkerStatus worker = workerStatus("127.0.0.1", 8080, RoleType.PREFILL);

        service.trackRoutingPrediction("short-request", RoleType.PREFILL, "default", worker,
                100, 0, CacheMatchResult.empty(CacheMatchSource.KVCM));

        verify(metricsReporter).reportKvcmPrediction(RoleType.PREFILL, worker.getMetricIpPort(), 0, 100);
    }

    @Test
    void failedKvcmQueryDoesNotBecomeZeroPrediction() {
        WorkerStatus worker = workerStatus("127.0.0.1", 8080, RoleType.PREFILL);

        service.trackRoutingPrediction("request-1", RoleType.PREFILL, "default", worker,
                100, 0, CacheMatchResult.failed(CacheMatchSource.KVCM, 10));

        verifyNoInteractions(metricsReporter);
    }

    @Test
    void localStandbySelectionIsRecordedWhenKvcmMetricReportingFails() {
        WorkerStatus worker = workerStatus("127.0.0.1", 8080, RoleType.PREFILL);
        doThrow(new IllegalStateException("monitor unavailable")).when(metricsReporter)
                .reportKvcmPrediction(RoleType.PREFILL, worker.getMetricIpPort(), 0, 100);

        service.trackRoutingPrediction("request-1", RoleType.PREFILL, "default", worker,
                100, 0, CacheMatchResult.empty(CacheMatchSource.KVCM));

        verify(comparisonService).recordSelectedWorker("request-1", RoleType.PREFILL, worker, 100);
    }
}
