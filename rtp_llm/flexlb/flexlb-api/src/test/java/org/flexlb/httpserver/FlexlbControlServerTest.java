package org.flexlb.httpserver;

import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.CacheMatchStatus;
import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.config.CacheMatchMode;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.enums.LogLevel;
import org.flexlb.service.monitor.FlexlbLogManager;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;

import java.util.Map;

import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class FlexlbControlServerTest {

    @Mock
    private CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator;
    @Mock
    private FlexlbLogManager flexlbLogManager;
    private WebTestClient webTestClient;

    @BeforeEach
    void setUp() {
        FlexlbControlServer server = new FlexlbControlServer(cacheMatchQueryOrchestrator, flexlbLogManager);
        webTestClient = WebTestClient.bindToRouterFunction(
                server.flexlbControlRoutes()).build();
    }

    @Test
    void updatesFlexlbLogGroupThroughFlexlbEndpoint() {
        webTestClient.post()
                .uri("/flexlb/update_log_level")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("log_level", "debug"))
                .exchange()
                .expectStatus().isOk()
                .expectBody(String.class).isEqualTo("Success! logLevel=DEBUG");

        verify(flexlbLogManager).setLogLevel(LogLevel.DEBUG);
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
                .jsonPath("$.localStandbyEntries").isEqualTo(123)
                .jsonPath("$.localStandbyMaximumEntries").isEqualTo(456);
    }

    @ParameterizedTest
    @ValueSource(strings = {"ACTIVATE_FALLBACK", "RECOVER_PRIMARY"})
    void cacheSourceCannotBeChangedThroughHttp(String action) {
        webTestClient.post()
                .uri("/flexlb/cache_match/failover")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("action", action))
                .exchange()
                .expectStatus()
                .isNotFound();

        verifyNoInteractions(cacheMatchQueryOrchestrator);
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
