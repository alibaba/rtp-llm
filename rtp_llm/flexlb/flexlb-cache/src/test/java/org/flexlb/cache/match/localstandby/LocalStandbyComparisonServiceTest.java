package org.flexlb.cache.match.localstandby;

import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.config.LocalStandbyConfig;
import org.flexlb.config.LocalStandbyRuntimeSettings;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.CacheHitFeedback;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;

import static org.flexlb.cache.CacheMatchTestConfigurations.kvcm;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class LocalStandbyComparisonServiceTest {

    @Test
    @DisplayName("Local Standby 热更新 TTL 应用于已有比较缓存")
    void expiresPendingPredictionUsingUpdatedRuntimeTtl() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        ConfigService configService = mock(ConfigService.class);
        AtomicReference<Function<FlexlbConfig, Optional<LocalStandbyRuntimeSettings>>> projection = new AtomicReference<>();
        AtomicReference<Consumer<Optional<LocalStandbyRuntimeSettings>>> listener = new AtomicReference<>();
        doAnswer(invocation -> {
            projection.set(invocation.getArgument(0));
            listener.set(invocation.getArgument(1));
            listener.get().accept(projection.get().apply(flexlbConfig(1_000, 1_000)));
            return null;
        }).when(configService).addUpdateListener(any(), any());
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider, configService);
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1", List.of(11L), 2_192, List.of(101L), 4_096, RoleType.PREFILL, "default");
        comparisonService.trackResolvedLocalStandbyPrediction(query, localStandbyMatch());

        FlexlbConfig updatedConfig = flexlbConfig(1, 1);
        listener.get().accept(projection.get().apply(updatedConfig));
        Thread.sleep(20);

        assertNull(comparisonService.buildCacheHitComparison(feedback()).get(1, TimeUnit.SECONDS).localStandby());
    }

    @Test
    void retainsPendingPredictionUsingExtendedRuntimeTtl() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        ConfigService configService = mock(ConfigService.class);
        AtomicReference<Function<FlexlbConfig, Optional<LocalStandbyRuntimeSettings>>> projection = new AtomicReference<>();
        AtomicReference<Consumer<Optional<LocalStandbyRuntimeSettings>>> listener = new AtomicReference<>();
        doAnswer(invocation -> {
            projection.set(invocation.getArgument(0));
            listener.set(invocation.getArgument(1));
            listener.get().accept(projection.get().apply(flexlbConfig(1, 1)));
            return null;
        }).when(configService).addUpdateListener(any(), any());
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider, configService);
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1", List.of(11L), 2_192, List.of(101L), 4_096, RoleType.PREFILL, "default");

        listener.get().accept(projection.get().apply(flexlbConfig(1_000, 1_000)));
        comparisonService.trackResolvedLocalStandbyPrediction(query, localStandbyMatch());
        Thread.sleep(20);

        assertNotNull(comparisonService.buildCacheHitComparison(feedback()).get(1, TimeUnit.SECONDS).localStandby());
    }

    @Test
    void buildsUnifiedComparisonWithLocalStandbyPrediction() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider);
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

    @Test
    void buildsUnifiedComparisonFromResolvedFallbackPrediction() throws Exception {
        LocalStandbyCacheMatchProvider provider = mock(LocalStandbyCacheMatchProvider.class);
        LocalStandbyComparisonService comparisonService = new LocalStandbyComparisonService(
                kvcm(modelMetaConfig()), provider);
        CacheMatchQuery query = new CacheMatchQuery(
                "request-1",
                List.of(11L),
                2192,
                List.of(101L),
                4096,
                RoleType.PREFILL,
                "default");
        comparisonService.trackResolvedLocalStandbyPrediction(query, new CacheMatchResult(
                Map.of("10.0.0.1:8080", HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                4096));
        verify(provider, never()).asyncLocalStandbyMatch(query);

        CacheHitFeedback feedback = new CacheHitFeedback(
                "cache_hit_comparison", "request-1", "LOCAL_STANDBY", "PREFILL", "default",
                "10.0.0.1", 8080, "running", 8000, 4096, 4096,
                false, 0, 0, 0,
                6000, 1904);
        CacheHitComparisonResult result =
                comparisonService.buildCacheHitComparison(feedback).get(1, TimeUnit.SECONDS);

        assertEquals(6000, result.actual().hit());
        assertNotNull(result.localStandby());
        assertEquals(4096, result.localStandby().hit());
        assertEquals(1904, result.localStandby().delta());
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

    private FlexlbConfig flexlbConfig(long ttlMs, long minimumTtlMs) {
        LocalStandbyConfig standby = new LocalStandbyConfig();
        standby.setTtlMs(ttlMs);
        standby.setMinimumTtlMs(minimumTtlMs);

        KvcmCacheMatchingConfig kvcm = new KvcmCacheMatchingConfig();
        kvcm.setLocalStandby(standby);

        FlexlbConfig config = new FlexlbConfig();
        config.setCacheMatching(kvcm);
        return config;
    }

    private CacheMatchResult localStandbyMatch() {
        return new CacheMatchResult(
                Map.of("10.0.0.1:8080", HostCacheMatch.local(1)),
                CacheMatchSource.LOCAL_STANDBY,
                10,
                4_096);
    }

    private CacheHitFeedback feedback() {
        return new CacheHitFeedback(
                "cache_hit_comparison", "request-1", "LOCAL_STANDBY", "PREFILL", "default",
                "10.0.0.1", 8080, "running", 8_000, 4_096, 4_096,
                false, 0, 0, 0,
                6_000, 1_904);
    }
}
