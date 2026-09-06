package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.kvcm.KvcmCacheMatchProvider;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheMatchProvider;
import org.flexlb.cache.match.localsync.LocalSyncCacheMatchProvider;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
        assertEquals(2, result.hostMatch("10.0.0.1:8080").localMatchBlocks());
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
        assertEquals(1, result.hostMatch("10.0.0.2:8080").localMatchBlocks());
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
                cacheMetricsReporter,
                configuration);
    }
}
