package org.flexlb.cache.match.localstandby;

import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.CacheHitFeedback;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.flexlb.cache.CacheMatchTestConfigurations.kvcm;
import static org.flexlb.cache.WorkerStatusTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class LocalStandbyComparisonServiceTest {

    @Test
    void waitsForLocalStandbyPredictionBeforeCompletingComparison() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider, mock(CacheMetricsReporter.class));
        CacheMatchQuery query = new CacheMatchQuery(
                "request-pending", List.of(11L), 2192, RoleType.PREFILL, "default");
        CompletableFuture<CacheMatchResult> pendingMatch = new CompletableFuture<>();
        when(provider.asyncLocalStandbyMatch(query)).thenReturn(pendingMatch);
        comparisonService.trackLocalStandbyPrediction(query);

        CacheHitFeedback feedback = new CacheHitFeedback(
                "cache_hit_comparison", "request-pending", "KVCM", "PREFILL", "default",
                "10.0.0.1", 8080, "running", 8000, 2192, 4384,
                true, 4000, 10000, 6000, 1616);
        CompletableFuture<CacheHitComparisonResult> comparison =
                comparisonService.captureComparison(feedback.requestId(), RoleType.PREFILL).apply(feedback);

        assertFalse(comparison.isDone());
        pendingMatch.complete(new CacheMatchResult(
                Map.of("10.0.0.1:8080@0", HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY, 10, 2192));

        assertEquals(2192, comparison.get(1, TimeUnit.SECONDS).localStandbyPrediction().predictedHitTokens());
    }

    @Test
    void buildsUnifiedComparisonWithLocalStandbyPrediction() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider, mock(CacheMetricsReporter.class));
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1",
                List.of(11L),
                2192,
                RoleType.PREFILL,
                "default");
        CompletableFuture<CacheMatchResult> pendingMatch = new CompletableFuture<>();
        when(provider.asyncLocalStandbyMatch(query)).thenReturn(pendingMatch);

        comparisonService.trackLocalStandbyPrediction(query);

        verify(provider).asyncLocalStandbyMatch(query);
        pendingMatch.complete(new CacheMatchResult(
                Map.of("10.0.0.1:8080@0", HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                2192));

        CacheHitFeedback feedback = new CacheHitFeedback(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                "10.0.0.1", 8080, "running", 8000, 2192, 4384,
                true, 4000, 10000,
                6000, 1616);
        CacheHitComparisonResult result =
                comparisonService.captureComparison(feedback.requestId(), RoleType.valueOf(feedback.role())).apply(feedback).get(1, TimeUnit.SECONDS);

        assertEquals(4384, result.kvcmPrediction().predictedHitTokens());
        assertEquals(6000, result.actualHitTokens());
        assertEquals(1616, result.actualHitTokens() - result.kvcmPrediction().predictedHitTokens());
        assertNotNull(result.localStandbyPrediction());
        assertEquals(2192, result.localStandbyPrediction().predictedHitTokens());
        assertEquals(3808, result.actualHitTokens() - result.localStandbyPrediction().predictedHitTokens());
    }

    @Test
    void buildsUnifiedComparisonFromResolvedFallbackPrediction() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider, mock(CacheMetricsReporter.class));
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1",
                List.of(11L),
                2192,
                RoleType.PREFILL,
                "default");
        comparisonService.trackResolvedLocalStandbyPrediction(query, new CacheMatchResult(
                Map.of("10.0.0.1:8080@0", HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                2192));
        verify(provider, never()).asyncLocalStandbyMatch(query);

        CacheHitFeedback feedback = new CacheHitFeedback(
                "cache_hit_comparison", "request-1", "LOCAL_STANDBY", "PREFILL", "default",
                "10.0.0.1", 8080, "running", 8000, 2192, 2192,
                false, 0, 0,
                6000, 3808);
        CacheHitComparisonResult result =
                comparisonService.captureComparison(feedback.requestId(), RoleType.valueOf(feedback.role())).apply(feedback).get(1, TimeUnit.SECONDS);

        assertEquals(6000, result.actualHitTokens());
        assertNotNull(result.localStandbyPrediction());
        assertEquals(2192, result.localStandbyPrediction().predictedHitTokens());
        assertEquals(3808, result.actualHitTokens() - result.localStandbyPrediction().predictedHitTokens());
    }

    @Test
    void matchesLocalStandbyPredictionByEngineIndex() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider, mock(CacheMetricsReporter.class));
        CacheMatchQuery query = new CacheMatchQuery(
                "request-index-1",
                List.of(11L),
                12000,
                RoleType.PREFILL,
                "default");
        comparisonService.trackResolvedLocalStandbyPrediction(query, new CacheMatchResult(
                Map.of(
                        "10.0.0.1:8080@0", HostCacheMatch.local(1),
                        "10.0.0.1:8080@1", HostCacheMatch.local(2)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                1024));

        CacheHitFeedback feedback = new CacheHitFeedback(
                "cache_hit_comparison", "request-index-1", "KVCM", "PREFILL", "default",
                "10.0.0.1", 8080, 1, "running", 12000, 1024, 1024,
                false, 0, 0,
                9000, 7976);

        CacheHitComparisonResult result =
                comparisonService.captureComparison(feedback.requestId(), RoleType.valueOf(feedback.role())).apply(feedback).get(1, TimeUnit.SECONDS);

        assertEquals("10.0.0.1:8080@1", result.worker());
        assertEquals(2048, result.localStandbyPrediction().predictedHitTokens());
        assertEquals(6952, result.actualHitTokens() - result.localStandbyPrediction().predictedHitTokens());
    }

    @Test
    void degradesToZeroStandbyHitWhenPredictionMissesLogicalWorker() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider, mock(CacheMetricsReporter.class));
        CacheMatchQuery query = new CacheMatchQuery(
                "request-miss-1",
                List.of(11L),
                1024,
                RoleType.PREFILL,
                "default");
        comparisonService.trackResolvedLocalStandbyPrediction(query, new CacheMatchResult(
                Map.of("10.0.0.1:8080@0", HostCacheMatch.local(2)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                1024));

        CacheHitFeedback feedback = new CacheHitFeedback(
                "cache_hit_comparison", "request-miss-1", "KVCM", "PREFILL", "default",
                "10.0.0.1", 8080, 1, "running", 12000, 1024, 1024,
                false, 0, 0,
                9000, 7976);

        CacheHitComparisonResult result =
                comparisonService.captureComparison(feedback.requestId(), RoleType.valueOf(feedback.role())).apply(feedback).get(1, TimeUnit.SECONDS);

        assertNotNull(result.localStandbyPrediction());
        assertEquals(0, result.localStandbyPrediction().predictedHitTokens());
        assertEquals(9000, result.actualHitTokens() - result.localStandbyPrediction().predictedHitTokens());
    }

    @Test
    void selectedPredictionIsAvailableBeforeEngineFeedback() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider, metricsReporter);
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default");
        CompletableFuture<CacheMatchResult> pendingMatch = new CompletableFuture<>();
        when(provider.asyncLocalStandbyMatch(query)).thenReturn(pendingMatch);
        WorkerStatus worker = workerStatus("10.0.0.1", 8080, RoleType.PREFILL);

        comparisonService.trackLocalStandbyPrediction(query);
        comparisonService.recordSelectedWorker("request-1", RoleType.PREFILL, worker, 8000);
        verifyNoInteractions(metricsReporter);
        pendingMatch.complete(new CacheMatchResult(
                Map.of(worker.getLogicalIpPort(), HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY, 10, 2192));

        verify(metricsReporter).reportLocalStandbyPrediction(
                RoleType.PREFILL, worker.getMetricIpPort(), 2192, 8000);
    }

    @Test
    void completedPredictionIsReportedWhenWorkerIsSelected() {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider, metricsReporter);
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default");
        WorkerStatus worker = workerStatus("10.0.0.1", 8080, RoleType.PREFILL);
        when(provider.asyncLocalStandbyMatch(query)).thenReturn(CompletableFuture.completedFuture(
                new CacheMatchResult(Map.of(worker.getLogicalIpPort(), HostCacheMatch.local(1)),
                        CacheMatchSource.LOCAL_STANDBY, 10, 2192)));

        comparisonService.trackLocalStandbyPrediction(query);
        verifyNoInteractions(metricsReporter);
        comparisonService.recordSelectedWorker("request-1", RoleType.PREFILL, worker, 8000);

        verify(metricsReporter).reportLocalStandbyPrediction(
                RoleType.PREFILL, worker.getMetricIpPort(), 2192, 8000);
    }

    @Test
    void emptyLocalStandbyKeysProduceZeroPrediction() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider, metricsReporter);
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1", List.of(), 2192, RoleType.PREFILL, "default");
        WorkerStatus worker = workerStatus("10.0.0.1", 8080, RoleType.PREFILL);

        comparisonService.trackLocalStandbyPrediction(query);
        comparisonService.recordSelectedWorker("request-1", RoleType.PREFILL, worker, 100);

        verify(metricsReporter).reportLocalStandbyPrediction(
                RoleType.PREFILL, worker.getMetricIpPort(), 0, 100);
        verify(provider, never()).asyncLocalStandbyMatch(query);
    }

    private ModelMetaConfig modelMetaConfig() {
        KvcmConfig kvcm = new KvcmConfig();

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);

        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return config;
    }
}
