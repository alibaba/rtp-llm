package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.hash.RequestBlockHashService;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheAwareServiceTest {

    private final CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
    private final CacheMatchQueryOrchestrator queryOrchestrator =
            mock(CacheMatchQueryOrchestrator.class);
    private final CacheMetadataUpdateOrchestrator updateOrchestrator =
            mock(CacheMetadataUpdateOrchestrator.class);
    private final RequestBlockHashService requestBlockHashService =
            mock(RequestBlockHashService.class);
    private final CacheAwareService service = new CacheAwareService(
            metricsReporter,
            queryOrchestrator,
            mock(LocalStandbyComparisonService.class),
            updateOrchestrator,
            requestBlockHashService);

    @Test
    void delegatesCacheQueriesToOrchestrator() {
        CacheMatchQuery query = new CacheMatchQuery(
                "1", List.of(11L), 2192L, List.of(), 0,
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
    void delegatesWorkerStatusUpdates() {
        WorkerStatus workerStatus = new WorkerStatus();
        WorkerCacheUpdateResult expected = WorkerCacheUpdateResult.builder()
                .success(true)
                .build();
        when(updateOrchestrator.updateFromWorkerStatus(workerStatus)).thenReturn(expected);

        assertSame(expected, service.updateFromWorkerStatus(workerStatus));
        verify(updateOrchestrator).updateFromWorkerStatus(workerStatus);
    }

    @Test
    void delegatesRequestBlockHashPreparation() {
        BalanceContext context = new BalanceContext();
        when(requestBlockHashService.prepareBlockCacheKeys(context))
                .thenReturn(Mono.empty());

        service.prepareBlockCacheKeys(context).join();

        verify(requestBlockHashService).prepareBlockCacheKeys(context);
    }

    @Test
    void convertsUnexpectedQueryFailureToFailedResult() {
        CacheMatchQuery query = new CacheMatchQuery(
                "2", List.of(11L), 2192L, List.of(), 0,
                RoleType.PREFILL, "default");
        when(queryOrchestrator.findMatchingEngines(query))
                .thenThrow(new IllegalStateException("failed"));
        when(queryOrchestrator.effectiveSource()).thenReturn(CacheMatchSource.KVCM);

        CacheMatchResult result = service.findMatchingEngines(query);

        assertEquals(CacheMatchSource.KVCM, result.source());
        assertEquals(Map.of(), result.hostMatches());
    }
}
