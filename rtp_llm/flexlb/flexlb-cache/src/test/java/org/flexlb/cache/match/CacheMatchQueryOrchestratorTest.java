package org.flexlb.cache.match;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
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
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
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
            "request-1", List.of(11L, 22L), 2192L, RoleType.PREFILL, "default");

    @Test
    void invalidLocalStandbyBlockSizeReturnsFailedQueryWithoutInvokingProvider() {
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        for (long blockSize : new long[]{0L, -1L}) {
            CacheMatchQuery invalid = new CacheMatchQuery(
                    "request-invalid", List.of(11L), blockSize, RoleType.PREFILL, "default");

            CacheMatchResult result = orchestrator().findMatchingEngines(invalid);

            assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
            assertFalse(result.querySucceeded());
            assertEquals(Map.of(), result.hostMatches());
            verify(localStandbyProvider, never()).findMatchingEngines(
                    org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any(),
                    org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.any(),
                    org.mockito.ArgumentMatchers.any());
        }
    }

    @Test
    void localSyncStatusDoesNotReadUnconfiguredKvcmHealth() {
        when(configuration.isKvcmEnabled()).thenReturn(false);

        var status = orchestrator().status();

        assertEquals(CacheMatchSource.LOCAL_SYNC, status.effectiveSource());
        assertFalse(status.kvcmEnabled());
        assertNull(status.kvcmHealthState());
        assertEquals(0, status.consecutiveQueryFailures());
        assertEquals(0, status.consecutiveHeartbeatFailures());
        verify(failoverManager, never()).healthSnapshot();
    }

    @Test
    void rateLimitsKvcmFailureWarningsWithoutDroppingFallbackMetrics() {
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        when(kvcmProvider.findMatchingEngines(
                query.requestId(), query.blockCacheKeys(), query.blockSize(), query.roleType(), query.group()))
                .thenThrow(new IllegalStateException("KVCM unavailable"));
        CacheMatchQueryOrchestrator orchestrator = orchestrator();
        var logger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(CacheMatchQueryOrchestrator.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
        try {
            assertEquals(CacheMatchSource.LOCAL_STANDBY, orchestrator.findMatchingEngines(query).source());
            assertEquals(CacheMatchSource.LOCAL_STANDBY, orchestrator.findMatchingEngines(query).source());
            assertEquals(1, appender.list.stream()
                    .filter(event -> event.getFormattedMessage().contains("KVCM cache query failed"))
                    .count());
            verify(cacheMetricsReporter, times(2)).reportStandbyFallback("kvcm_query_failure");
        } finally {
            logger.detachAppender(appender);
            appender.stop();
        }
    }

    @Test
    void usesLocalSyncWhenKvcmIsDisabled() {
        when(configuration.isKvcmEnabled()).thenReturn(false);
        when(localSyncProvider.findMatchingEngines(
                "request-1", List.of(11L, 22L), 2192L,
                RoleType.PREFILL, "default"))
                .thenReturn(Map.of("10.0.0.1:8080@0", HostCacheMatch.local(2)));

        CacheMatchResult result = orchestrator().findMatchingEngines(query);

        assertEquals(CacheMatchSource.LOCAL_SYNC, result.source());
        assertEquals(2, result.exactHostMatch("10.0.0.1:8080@0").localMatchBlocks());
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
                standbyQuery.blockSize());
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        when(localStandbyProvider.findMatchingEngines(
                standbyQuery.requestId(), standbyQuery.blockCacheKeys(), standbyQuery.blockSize(),
                standbyQuery.roleType(), standbyQuery.group()))
                .thenReturn(standbyResult.hostMatches());

        CacheMatchResult result = orchestrator().findMatchingEngines(standbyQuery);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        verify(comparisonService)
                .trackResolvedLocalStandbyPrediction(standbyQuery, result);
        verify(cacheMetricsReporter).reportStandbyFallback("active_source");
        verify(localStandbyProvider, never()).asyncLocalStandbyMatch(standbyQuery);
        assertEquals(standbyQuery.blockSize(), result.blockSize());
    }

    @Test
    void fallsBackCurrentRequestAndTracksResolvedPredictionOnKvcmFailure() {
        CacheMatchQuery standbyQuery = standbyQuery();
        CacheMatchResult standbyResult = new CacheMatchResult(
                Map.of("10.0.0.3:8080", HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                standbyQuery.blockSize());
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        when(kvcmProvider.findMatchingEngines(
                standbyQuery.requestId(),
                standbyQuery.blockCacheKeys(),
                standbyQuery.blockSize(),
                standbyQuery.roleType(),
                standbyQuery.group()))
                .thenThrow(new IllegalStateException("KVCM unavailable"));
        when(localStandbyProvider.findMatchingEngines(
                standbyQuery.requestId(), standbyQuery.blockCacheKeys(), standbyQuery.blockSize(),
                standbyQuery.roleType(), standbyQuery.group()))
                .thenReturn(standbyResult.hostMatches());

        CacheMatchResult result = orchestrator().findMatchingEngines(standbyQuery);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        verify(comparisonService)
                .trackResolvedLocalStandbyPrediction(standbyQuery, result);
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
                standbyQuery.blockSize());
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        when(localStandbyProvider.findMatchingEngines(
                standbyQuery.requestId(), standbyQuery.blockCacheKeys(), standbyQuery.blockSize(),
                standbyQuery.roleType(), standbyQuery.group()))
                .thenReturn(standbyResult.hostMatches());
        doThrow(new IllegalStateException("comparison unavailable"))
                .when(comparisonService)
                .trackResolvedLocalStandbyPrediction(org.mockito.ArgumentMatchers.eq(standbyQuery),
                        org.mockito.ArgumentMatchers.any());

        CacheMatchResult result = orchestrator().findMatchingEngines(standbyQuery);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        assertEquals(1, result.exactHostMatch("10.0.0.3:8080").localMatchBlocks());
    }

    @Test
    void skipsProviderForEmptyKeys() {
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.KVCM);
        CacheMatchQuery empty = new CacheMatchQuery(
                "request-2", List.of(), 2192L, RoleType.PREFILL, "default");

        CacheMatchResult result = orchestrator().findMatchingEngines(empty);

        assertEquals(CacheMatchSource.KVCM, result.source());
        assertEquals(Map.of(), result.hostMatches());
        assertEquals(empty.blockSize(), result.blockSize());
        verify(kvcmProvider, never()).findMatchingEngines(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any());
    }

    @Test
    void retainsBlockSizeForEmptyLocalSyncQuery() {
        CacheMatchQuery empty = new CacheMatchQuery(
                "request-empty", List.of(), 2192L, RoleType.PREFILL, "default");

        CacheMatchResult result = orchestrator().findMatchingEngines(empty);

        assertEquals(CacheMatchSource.LOCAL_SYNC, result.source());
        assertEquals(Map.of(), result.hostMatches());
        assertEquals(empty.blockSize(), result.blockSize());
    }

    @Test
    void retainsBlockSizeForEmptyLocalStandbyQuery() {
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        CacheMatchQuery empty = new CacheMatchQuery(
                "request-empty", List.of(), 2192L, RoleType.PREFILL, "default");

        CacheMatchResult result = orchestrator().findMatchingEngines(empty);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        assertEquals(Map.of(), result.hostMatches());
        assertEquals(empty.blockSize(), result.blockSize());
        verify(localStandbyProvider, never()).asyncLocalStandbyMatch(empty);
    }

    @Test
    void noCacheKeysWithZeroBlockSizeSucceedsOnLocalStandby() {
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(failoverManager.activeSource()).thenReturn(CacheMatchSource.LOCAL_STANDBY);
        CacheMatchQuery empty = new CacheMatchQuery(
                "request-no-cache", List.of(), 0L, RoleType.PREFILL, "default");

        CacheMatchResult result = orchestrator().findMatchingEngines(empty);

        assertEquals(CacheMatchSource.LOCAL_STANDBY, result.source());
        assertTrue(result.querySucceeded());
        assertEquals(Map.of(), result.hostMatches());
        assertEquals(0L, result.blockSize());
        verify(localStandbyProvider, never()).findMatchingEngines(
                org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.any(),
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
                RoleType.PREFILL,
                "default");
    }
}
