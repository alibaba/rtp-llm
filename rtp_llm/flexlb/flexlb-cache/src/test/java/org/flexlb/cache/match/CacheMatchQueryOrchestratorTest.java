package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.kvcm.KvcmCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheMatchQueryOrchestratorTest {

    private final LocalSyncCacheMatchProvider localSyncProvider =
            mock(LocalSyncCacheMatchProvider.class);
    private final KvcmCacheMatchProvider kvcmProvider = mock(KvcmCacheMatchProvider.class);
    private final CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
    private final LocalStandbyCacheMatchProvider localStandbyProvider =
            mock(LocalStandbyCacheMatchProvider.class);
    private final LocalStandbyCacheManager localStandbyCacheManager =
            mock(LocalStandbyCacheManager.class);
    private final CacheMatchFailoverManager failoverManager =
            mock(CacheMatchFailoverManager.class);
    private final LocalStandbyComparisonService comparisonService =
            mock(LocalStandbyComparisonService.class);
    private final CacheMetricsReporter cacheMetricsReporter =
            mock(CacheMetricsReporter.class);
    private final CacheMatchQuery query = new CacheMatchQuery(
            "request-1", List.of(11L, 22L), 2192L,
            List.of(), 0, RoleType.PREFILL, "default");

    @Test
    void usesLocalSyncWhenKvcmIsDisabled() {
        when(configuration.isKvcmEnabled()).thenReturn(false);
        when(localSyncProvider.findMatchingEngines(
                "request-1", List.of(11L, 22L), 2192L,
                RoleType.PREFILL, "default"))
                .thenReturn(Map.of("10.0.0.1:8080", HostCacheMatch.local(2)));

        CacheMatchResult result = orchestrator().findMatchingEngines(query);

        assertEquals(CacheMatchSource.LOCAL_SYNC, result.source());
        assertEquals(2, result.exactHostMatch("10.0.0.1:8080").localMatchBlocks());
        verify(kvcmProvider, never()).findMatchingEngines(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any());
    }

    @Test
    void usesKvcmWhenEnabled() {
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        when(kvcmProvider.findMatchingEngines(
                "request-1", List.of(11L, 22L), 2192L,
                RoleType.PREFILL, "default"))
                .thenReturn(Map.of("10.0.0.2:8080", HostCacheMatch.local(1)));

        CacheMatchResult result = orchestrator().findMatchingEngines(query);

        assertEquals(CacheMatchSource.KVCM, result.source());
        assertEquals(1, result.exactHostMatch("10.0.0.2:8080").localMatchBlocks());
        verify(comparisonService).trackLocalStandbyPrediction(query);
    }

    @Test
    void tracksResolvedPredictionWhenLocalStandbyIsActive() {
        CacheMatchQuery standbyQuery = standbyQuery();
        CacheMatchResult standbyResult = new CacheMatchResult(
                Map.of("10.0.0.3:8080", HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                4096);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        when(localStandbyProvider.asyncLocalStandbyMatch(standbyQuery))
                .thenReturn(CompletableFuture.completedFuture(standbyResult));

        CacheMatchResult result = orchestrator().findMatchingEngines(standbyQuery);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        verify(comparisonService)
                .trackResolvedLocalStandbyPrediction(standbyQuery, standbyResult);
        verify(cacheMetricsReporter).reportStandbyFallback("active_source");
    }

    @Test
    void fallsBackCurrentRequestAndTracksResolvedPredictionOnKvcmFailure() {
        CacheMatchQuery standbyQuery = standbyQuery();
        CacheMatchResult standbyResult = new CacheMatchResult(
                Map.of("10.0.0.3:8080", HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                4096);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        when(kvcmProvider.findMatchingEngines(
                standbyQuery.requestId(),
                standbyQuery.blockCacheKeys(),
                standbyQuery.blockSize(),
                standbyQuery.roleType(),
                standbyQuery.group()))
                .thenThrow(new IllegalStateException("KVCM unavailable"));
        when(localStandbyProvider.asyncLocalStandbyMatch(standbyQuery))
                .thenReturn(CompletableFuture.completedFuture(standbyResult));

        CacheMatchResult result = orchestrator().findMatchingEngines(standbyQuery);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        verify(comparisonService)
                .trackResolvedLocalStandbyPrediction(standbyQuery, standbyResult);
        verify(cacheMetricsReporter).reportStandbyFallback("kvcm_query_failure");
    }

    @Test
    void keepsKvcmResultWhenPredictionTrackingFails() {
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        when(kvcmProvider.findMatchingEngines(
                query.requestId(),
                query.blockCacheKeys(),
                query.blockSize(),
                query.roleType(),
                query.group()))
                .thenReturn(Map.of("10.0.0.2:8080", HostCacheMatch.local(1)));
        doThrow(new IllegalStateException("comparison unavailable"))
                .when(comparisonService).trackLocalStandbyPrediction(query);

        CacheMatchResult result = orchestrator().findMatchingEngines(query);

        assertEquals(CacheMatchSource.KVCM, result.source());
        assertEquals(1, result.exactHostMatch("10.0.0.2:8080").localMatchBlocks());
    }

    @Test
    void keepsLocalStandbyResultWhenResolvedPredictionTrackingFails() {
        CacheMatchQuery standbyQuery = standbyQuery();
        CacheMatchResult standbyResult = new CacheMatchResult(
                Map.of("10.0.0.3:8080", HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                4096);
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        when(localStandbyProvider.asyncLocalStandbyMatch(standbyQuery))
                .thenReturn(CompletableFuture.completedFuture(standbyResult));
        doThrow(new IllegalStateException("comparison unavailable"))
                .when(comparisonService)
                .trackResolvedLocalStandbyPrediction(standbyQuery, standbyResult);

        CacheMatchResult result = orchestrator().findMatchingEngines(standbyQuery);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        assertEquals(1, result.exactHostMatch("10.0.0.3:8080").localMatchBlocks());
    }

    @Test
    void skipsProviderForEmptyKeys() {
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        CacheMatchQuery empty = new CacheMatchQuery(
                "request-2", List.of(), 2192L,
                List.of(), 0, RoleType.PREFILL, "default");

        CacheMatchResult result = orchestrator().findMatchingEngines(empty);

        assertEquals(CacheMatchSource.KVCM, result.source());
        assertEquals(Map.of(), result.hostMatches());
        verify(kvcmProvider, never()).findMatchingEngines(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any());
    }

    private CacheMatchQueryOrchestrator orchestrator() {
        return new CacheMatchQueryOrchestrator(
                localSyncProvider,
                kvcmProvider,
                localStandbyProvider,
                localStandbyCacheManager,
                failoverManager,
                comparisonService,
                cacheMetricsReporter,
                configuration);
    }

    private CacheMatchQuery standbyQuery() {
        return new CacheMatchQuery(
                "request-standby",
                List.of(11L, 22L),
                2192L,
                List.of(101L),
                4096,
                RoleType.PREFILL,
                "default");
    }
}
