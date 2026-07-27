package org.flexlb.httpserver;

import org.flexlb.cache.domain.CacheMatchFailoverAction;
import org.flexlb.cache.domain.CacheMatchFailoverRequest;
import org.flexlb.cache.domain.CacheMatchStatus;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.config.CacheMatchMode;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.enums.LogLevel;
import org.flexlb.service.monitor.FlexlbLogLevelManager;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class FlexlbControlServerTest {

    @Mock
    private GeneralHttpNettyService generalHttpNettyService;
    @Mock
    private LBStatusConsistencyService lbStatusConsistencyService;
    @Mock
    private CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator;
    @Mock
    private FlexlbLogLevelManager logLevelManager;

    private WebTestClient webTestClient;

    @BeforeEach
    void setUp() {
        FlexlbControlServer server = new FlexlbControlServer(
                generalHttpNettyService,
                lbStatusConsistencyService,
                cacheMatchQueryOrchestrator,
                logLevelManager);
        webTestClient = WebTestClient.bindToRouterFunction(
                server.flexlbControlRoutes()).build();
    }

    @Test
    void updatesFlexlbLogGroupThroughFlexlbEndpoint() {
        when(logLevelManager.setLogLevel(LogLevel.DEBUG)).thenReturn(LogLevel.DEBUG);

        webTestClient.post()
                .uri("/flexlb/update_log_level")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("log_level", "debug"))
                .exchange()
                .expectStatus().isOk()
                .expectBody(String.class).isEqualTo("Success! logLevel=DEBUG");

        verify(logLevelManager).setLogLevel(LogLevel.DEBUG);
    }

    @Test
    void returnsCacheMatchStatus() {
        when(cacheMatchQueryOrchestrator.status()).thenReturn(cacheMatchStatus());

        webTestClient.get()
                .uri("/flexlb/cache_match/status")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.configuredMode").isEqualTo("KVCM")
                .jsonPath("$.autoSwitchEnabled").isEqualTo(true)
                .jsonPath("$.effectiveSource").isEqualTo("KVCM")
                .jsonPath("$.kvcmHealthState").isEqualTo("HEALTHY")
                .jsonPath("$.failoverState").doesNotExist()
                .jsonPath("$.localStandbyEntries").isEqualTo(123)
                .jsonPath("$.localStandbyMaximumEntries").isEqualTo(456);
    }

    @Test
    void activatesCacheMatchFailover() {
        when(cacheMatchQueryOrchestrator.status()).thenReturn(cacheMatchStatus());

        webTestClient.post()
                .uri("/flexlb/cache_match/failover")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "action",
                        CacheMatchFailoverAction.ACTIVATE_FALLBACK.name()))
                .exchange()
                .expectStatus().isOk();

        verify(cacheMatchQueryOrchestrator)
                .applyFailoverAction(CacheMatchFailoverAction.ACTIVATE_FALLBACK);
    }

    @Test
    void recoversKvcmPrimary() {
        when(cacheMatchQueryOrchestrator.status()).thenReturn(cacheMatchStatus());

        webTestClient.post()
                .uri("/flexlb/cache_match/failover")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "action",
                        CacheMatchFailoverAction.RECOVER_PRIMARY.name()))
                .exchange()
                .expectStatus().isOk();

        verify(cacheMatchQueryOrchestrator)
                .applyFailoverAction(CacheMatchFailoverAction.RECOVER_PRIMARY);
    }

    @Test
    void rejectsUnknownFailoverAction() {
        webTestClient.post()
                .uri("/flexlb/cache_match/failover")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("action", "UNKNOWN"))
                .exchange()
                .expectStatus().isBadRequest();

        verify(cacheMatchQueryOrchestrator, never())
                .applyFailoverAction(any());
    }

    @Test
    void forwardsCacheMatchFailoverOnFollower() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
        when(lbStatusConsistencyService.getMasterHostIpPort())
                .thenReturn("10.0.0.1:7001");
        when(generalHttpNettyService.request(
                any(CacheMatchFailoverRequest.class),
                any(URI.class),
                eq("/flexlb/cache_match/failover"),
                eq(CacheMatchStatus.class)))
                .thenReturn(Mono.just(cacheMatchStatus()));

        webTestClient.post()
                .uri("/flexlb/cache_match/failover")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "action",
                        CacheMatchFailoverAction.ACTIVATE_FALLBACK.name()))
                .exchange()
                .expectStatus().isOk();

        verify(generalHttpNettyService).request(
                any(CacheMatchFailoverRequest.class),
                any(URI.class),
                eq("/flexlb/cache_match/failover"),
                eq(CacheMatchStatus.class));
        verify(cacheMatchQueryOrchestrator, never())
                .applyFailoverAction(any());
    }

    private CacheMatchStatus cacheMatchStatus() {
        return new CacheMatchStatus(
                true,
                true,
                CacheMatchMode.KVCM,
                true,
                CacheMatchSource.KVCM,
                KvcmHealthState.HEALTHY,
                0,
                0,
                3,
                100,
                0,
                100,
                "initial",
                123,
                456);
    }
}
