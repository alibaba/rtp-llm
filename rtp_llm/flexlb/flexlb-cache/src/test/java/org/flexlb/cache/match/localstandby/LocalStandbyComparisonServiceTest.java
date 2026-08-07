package org.flexlb.cache.match.localstandby;

import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.CacheHitFeedback;
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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class LocalStandbyComparisonServiceTest {

    @Test
    void buildsUnifiedComparisonWithLocalStandbyPrediction() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                new CacheMatchConfiguration(modelMetaConfig()), provider);
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1",
                List.of(11L),
                2192,
                null,
                4096,
                RoleType.PREFILL,
                "default");
        CompletableFuture<CacheMatchResult> pendingMatch = new CompletableFuture<>();
        when(provider.asyncLocalStandbyMatch(query)).thenReturn(pendingMatch);

        comparisonService.trackLocalStandbyPrediction(query);

        verify(provider).asyncLocalStandbyMatch(query);
        pendingMatch.complete(new CacheMatchResult(
                Map.of("10.0.0.1:8080", HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                4096));

        CacheHitFeedback feedback = new CacheHitFeedback(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                "10.0.0.1", 8080, "running", 8000, 2192, 4384,
                true, 4000, 8000, 10000,
                6000, 1616);
        CacheHitComparisonResult result =
                comparisonService.buildCacheHitComparison(feedback).get(1, TimeUnit.SECONDS);

        assertEquals(4384, result.routing().hit());
        assertEquals(6000, result.actual().hit());
        assertEquals(1616, result.routing().delta());
        assertEquals(4000, result.kvcmDetails().local().hit());
        assertEquals(2000, result.kvcmDetails().local().delta());
        assertEquals(10000, result.kvcmDetails().p2pTotal().hit());
        assertEquals(-4000, result.kvcmDetails().p2pTotal().delta());
        assertNotNull(result.localStandby());
        assertEquals(4096, result.localStandby().hit());
        assertEquals(1904, result.localStandby().delta());
    }

    private ModelMetaConfig modelMetaConfig() {
        LocalStandbyConfig standby = new LocalStandbyConfig();

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(true);
        kvcm.setLocalStandby(standby);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);

        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return config;
    }
}
