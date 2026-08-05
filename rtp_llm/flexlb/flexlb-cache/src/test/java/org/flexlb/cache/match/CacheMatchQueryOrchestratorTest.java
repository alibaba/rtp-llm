package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchFailoverAction;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.LocalStandbyHashResult;
import org.flexlb.cache.hash.LocalStandbyHashService;
import org.flexlb.cache.match.kvcm.KvcmCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheMatchQueryOrchestratorTest {

    private final LocalSyncCacheMatchProvider localSyncProvider =
            mock(LocalSyncCacheMatchProvider.class);
    private final KvcmCacheMatchProvider kvcmProvider =
            mock(KvcmCacheMatchProvider.class);
    private final LocalStandbyCacheMatchProvider localStandbyProvider =
            mock(LocalStandbyCacheMatchProvider.class);
    private final LocalStandbyCacheManager localStandbyCacheManager =
            mock(LocalStandbyCacheManager.class);
    private final CacheMatchFailoverManager failoverManager =
            mock(CacheMatchFailoverManager.class);
    private final LocalStandbyComparisonService comparisonService =
            mock(LocalStandbyComparisonService.class);
    private final LocalStandbyHashService localStandbyHashService =
            mock(LocalStandbyHashService.class);
    private final CacheMetricsReporter cacheMetricsReporter =
            mock(CacheMetricsReporter.class);
    private final CacheMatchQuery query = new CacheMatchQuery(
            "request-1",
            List.of(11L, 22L),
            2192,
            List.of(101L),
            4096,
            RoleType.PREFILL,
            "default");

    @Test
    void usesLocalSyncWhenKvcmIsDisabled() {
        when(localSyncProvider.findMatchingEngines("request-1", List.of(11L, 22L), 2192, RoleType.PREFILL, "default"))
                .thenReturn(Map.of("10.0.0.1:8080", HostCacheMatch.local(2)));

        CacheMatchResult result = orchestrator(false, false).findMatchingEngines(query);

        assertEquals(CacheMatchSource.LOCAL_SYNC, result.source());
        assertEquals(2192, result.blockSize());
        assertEquals(2, result.hostMatch("10.0.0.1:8080").localMatchBlocks());
        verify(kvcmProvider, never()).findMatchingEngines(any(), any(), any(Long.class), any(), any());
    }

    @Test
    void fallsBackCurrentRequestToLocalStandbyAfterFinalKvcmFailure() {
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        RuntimeException failure = new RuntimeException("KVCM unavailable");
        when(kvcmProvider.findMatchingEngines("request-1", List.of(11L, 22L), 2192, RoleType.PREFILL, "default"))
                .thenThrow(failure);
        when(localStandbyHashService.getHashResult("request-1", List.of(101L), 4096))
                .thenReturn(CompletableFuture.completedFuture(
                        new LocalStandbyHashResult(List.of(101L), 4096)));
        when(localStandbyProvider.findMatchingEngines("request-1", List.of(101L), 4096, RoleType.PREFILL, "default"))
                .thenReturn(Map.of("10.0.0.2:8080", HostCacheMatch.local(1)));

        CacheMatchResult result =
                orchestrator(true, false).findMatchingEngines(query);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        assertEquals(4096, result.blockSize());
        assertEquals(1, result.hostMatch("10.0.0.2:8080").localMatchBlocks());
        verify(comparisonService, never()).trackLocalStandbyPrediction(query);
        verify(cacheMetricsReporter).reportStandbyFallback("kvcm_query_failure");
    }

    @Test
    void startsAsyncComparisonAfterKvcmSuccess() {
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        when(kvcmProvider.findMatchingEngines("request-1", List.of(11L, 22L), 2192, RoleType.PREFILL, "default"))
                .thenReturn(Map.of("10.0.0.1:8080", new HostCacheMatch(2, 8, 10)));

        CacheMatchResult result = orchestrator().findMatchingEngines(query);

        assertEquals(CacheMatchSource.KVCM, result.source());
        assertEquals(2192, result.blockSize());
        assertEquals(2, result.hostMatch("10.0.0.1:8080").localMatchBlocks());
        assertEquals(10, result.hostMatch("10.0.0.1:8080").p2pTotalMatchBlocks());
        verify(comparisonService).trackLocalStandbyPrediction(query);
        verify(localStandbyHashService, never()).getHashResult(any(), any(), anyLong());
    }

    @Test
    void skipsKvcmQueryWhenPrimaryHashesAreEmpty() {
        CacheMatchQuery emptyQuery = new CacheMatchQuery(
                "request-1",
                List.of(),
                2192,
                List.of(101L),
                4096,
                RoleType.PREFILL,
                "default");
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);

        CacheMatchResult result = orchestrator().findMatchingEngines(emptyQuery);

        assertEquals(CacheMatchSource.KVCM, result.source());
        assertEquals(0, result.blockSize());
        assertEquals(Map.of(), result.hostMatches());
        verify(kvcmProvider, never()).findMatchingEngines(any(), any(), any(Long.class), any(), any());
        verify(comparisonService, never()).trackLocalStandbyPrediction(any(CacheMatchQuery.class));
    }

    @Test
    void skipsLocalSyncQueryWhenPrimaryHashesAreEmpty() {
        CacheMatchQuery emptyQuery = new CacheMatchQuery(
                "request-1",
                List.of(),
                2192,
                List.of(),
                4096,
                RoleType.PREFILL,
                "default");

        CacheMatchResult result =
                orchestrator(false, false).findMatchingEngines(emptyQuery);

        assertEquals(CacheMatchSource.LOCAL_SYNC, result.source());
        assertEquals(0, result.blockSize());
        assertEquals(Map.of(), result.hostMatches());
        verify(localSyncProvider, never()).findMatchingEngines(any(), any(), any(Long.class), any(), any());
    }

    @Test
    void skipsCompletedEmptyLocalStandbyHash() {
        CacheMatchQuery emptyQuery = new CacheMatchQuery(
                "request-1",
                List.of(11L),
                2192,
                List.of(),
                4096,
                RoleType.PREFILL,
                "default");
        when(failoverManager.activeSource())
                .thenReturn(CacheMatchSource.LOCAL_STANDBY);

        CacheMatchResult result = orchestrator().findMatchingEngines(emptyQuery);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        assertEquals(0, result.blockSize());
        assertEquals(Map.of(), result.hostMatches());
        verify(localStandbyHashService, never()).getHashResult(any(), any(), anyLong());
        verify(localStandbyProvider, never())
                .findMatchingEngines(any(), any(), any(Long.class), any(), any());
        verify(cacheMetricsReporter).reportStandbyFallback("active_source");
    }

    @Test
    void waitsForInFlightHashOnlyWhenRoutingThroughLocalStandby() throws Exception {
        CacheMatchQuery pendingQuery = new CacheMatchQuery(
                "request-2",
                List.of(11L, 22L),
                2192,
                null,
                4096,
                RoleType.PREFILL,
                "default");
        CompletableFuture<LocalStandbyHashResult> pendingHash = new CompletableFuture<>();
        when(failoverManager.activeSource())
                .thenReturn(CacheMatchSource.LOCAL_STANDBY);
        when(localStandbyHashService.getHashResult("request-2", null, 4096)).thenReturn(pendingHash);
        when(localStandbyProvider.findMatchingEngines(
                "request-2", List.of(101L), 4096, RoleType.PREFILL, "default"))
                .thenReturn(Map.of("10.0.0.2:8080", HostCacheMatch.local(1)));

        CompletableFuture<CacheMatchResult> routingResult =
                CompletableFuture.supplyAsync(() -> orchestrator().findMatchingEngines(pendingQuery));

        verify(localStandbyHashService, timeout(1_000)).getHashResult("request-2", null, 4096);
        assertFalse(routingResult.isDone());

        pendingHash.complete(new LocalStandbyHashResult(List.of(101L), 4096));
        CacheMatchResult result = routingResult.get(1, TimeUnit.SECONDS);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        assertEquals(4096, result.blockSize());
        assertEquals(1, result.hostMatch("10.0.0.2:8080").localMatchBlocks());
    }

    @Test
    void recordsWaitTimeWhenInFlightLocalStandbyHashCompletesEmpty() throws Exception {
        CacheMatchQuery pendingQuery = new CacheMatchQuery(
                "request-3",
                List.of(11L),
                2192,
                null,
                4096,
                RoleType.PREFILL,
                "default");
        CompletableFuture<LocalStandbyHashResult> pendingHash = new CompletableFuture<>();
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        when(localStandbyHashService.getHashResult("request-3", null, 4096)).thenReturn(pendingHash);

        CompletableFuture<CacheMatchResult> routingResult =
                CompletableFuture.supplyAsync(() -> orchestrator().findMatchingEngines(pendingQuery));

        verify(localStandbyHashService, timeout(1_000)).getHashResult("request-3", null, 4096);
        TimeUnit.MILLISECONDS.sleep(10);
        pendingHash.complete(LocalStandbyHashResult.empty());

        CacheMatchResult result = routingResult.get(1, TimeUnit.SECONDS);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        assertEquals(Map.of(), result.hostMatches());
        assertTrue(result.queryTimeUs() > 0);
    }

    @Test
    void resolvesEffectiveSourceFromFailoverState() {
        CacheMatchQueryOrchestrator localSyncOrchestrator =
                orchestrator(false, false);
        assertEquals(CacheMatchSource.LOCAL_SYNC, localSyncOrchestrator.effectiveSource());

        CacheMatchQueryOrchestrator orchestrator = orchestrator(true, false);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        assertEquals(CacheMatchSource.KVCM, orchestrator.effectiveSource());

        orchestrator.applyFailoverAction(CacheMatchFailoverAction.ACTIVATE_FALLBACK);
        verify(failoverManager).activateFallbackManually();
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        assertEquals(CacheMatchSource.LOCAL_STANDBY, orchestrator.effectiveSource());

        orchestrator.applyFailoverAction(CacheMatchFailoverAction.RECOVER_PRIMARY);
        verify(failoverManager).recoverPrimaryManually();
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        assertEquals(CacheMatchSource.KVCM, orchestrator.effectiveSource());
    }

    @Test
    void validatesManualFailoverActions() {
        CacheMatchQueryOrchestrator kvcmDisabled =
                orchestrator(false, false);
        assertThrows(IllegalStateException.class,
                () -> kvcmDisabled.applyFailoverAction(
                        CacheMatchFailoverAction.ACTIVATE_FALLBACK));

        assertThrows(IllegalArgumentException.class,
                () -> orchestrator(true, false).applyFailoverAction(null));
    }

    private CacheMatchQueryOrchestrator orchestrator() {
        return orchestrator(true, true);
    }

    private CacheMatchQueryOrchestrator orchestrator(boolean kvcmEnabled, boolean autoSwitch) {
        return new CacheMatchQueryOrchestrator(
                localSyncProvider,
                kvcmProvider,
                localStandbyProvider,
                localStandbyCacheManager,
                failoverManager,
                comparisonService,
                localStandbyHashService,
                cacheMetricsReporter,
                new CacheMatchConfiguration(
                        modelMetaConfig(kvcmEnabled, autoSwitch)));
    }

    private ModelMetaConfig modelMetaConfig(boolean kvcmEnabled, boolean autoSwitch) {
        LocalStandbyConfig standby = new LocalStandbyConfig();
        standby.setAutoSwitch(autoSwitch);

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(kvcmEnabled);
        kvcm.setLocalStandby(standby);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);

        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return config;
    }
}
