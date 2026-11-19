package org.flexlb.httpserver;

import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.impl.DefaultCacheAwareService;
import org.flexlb.engine.grpc.KvcmGrpcClient;
import org.flexlb.engine.grpc.KvcmGrpcClient.HealthSnapshot;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.test.web.reactive.server.WebTestClient;

import static org.mockito.Mockito.ignoreStubs;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

/**
 * Verifies the HTTP snapshot distinguishes the effective source from request-scoped fallback.
 */
class CacheMatchStatusServerTest {
    @ParameterizedTest
    @CsvSource({
            "false,false,false,0,LOCAL_SYNC",
            "true,true,false,1,KVCM",
            "true,true,true,3,LOCAL_STANDBY",
            "true,false,true,3,KVCM"
    })
    void cacheMatchStatusReportsTheLocalInstanceWithoutIssuingAQuery(
            boolean enabled, boolean autoSwitch, boolean standbyRecommended, int failures, String source) {
        var localCache = mock(KvCacheManager.class);
        var metrics = mock(CacheMetricsReporter.class);
        var kvcm = mock(KvcmGrpcClient.class);
        var standby = mock(LocalStandbyCacheManager.class);
        when(kvcm.enabled()).thenReturn(enabled);
        if (enabled) {
            when(kvcm.healthSnapshot()).thenReturn(new HealthSnapshot(true, standbyRecommended, failures, 0, 0));
            when(standby.autoSwitch()).thenReturn(autoSwitch);
            when(standby.mappingCount()).thenReturn(12L);
            when(standby.maximumEntryCount()).thenReturn(100L);
        }
        var server = new CacheMatchStatusServer(new DefaultCacheAwareService(localCache, metrics, kvcm, standby));
        var client = WebTestClient.bindToRouterFunction(server.cacheMatchStatusRoutes()).build();

        var response = client.get().uri("/flexlb/cache_match/status").exchange()
                .expectStatus().isOk()
                .expectHeader().valueEquals("Cache-Control", "no-store")
                .expectBody()
                .jsonPath("$.kvcmEnabled").isEqualTo(enabled)
                .jsonPath("$.localStandbyEnabled").isEqualTo(enabled)
                .jsonPath("$.configuredMode").isEqualTo(enabled ? "KVCM" : "LOCAL_SYNC")
                .jsonPath("$.autoSwitchEnabled").isEqualTo(autoSwitch)
                .jsonPath("$.effectiveSource").isEqualTo(source)
                .jsonPath("$.kvcmStandbyRecommended").isEqualTo(standbyRecommended)
                .jsonPath("$.consecutiveQueryFailures").isEqualTo(failures)
                .jsonPath("$.localStandbyEntries").isEqualTo(enabled ? 12 : 0)
                .jsonPath("$.localStandbyMaximumEntries").isEqualTo(enabled ? 100 : 0);
        if ("KVCM".equals(source)) {
            response.jsonPath("$.queryFailureFallbackSource").isEqualTo("LOCAL_STANDBY");
        } else {
            response.jsonPath("$.queryFailureFallbackSource").doesNotExist();
        }
        client.post().uri("/flexlb/cache_match/failover").exchange().expectStatus().isNotFound();
        client.post().uri("/flexlb/update_log_level").exchange().expectStatus().isNotFound();
        verifyNoInteractions(localCache, metrics);
        verifyNoMoreInteractions(ignoreStubs(kvcm, standby));
    }
}
