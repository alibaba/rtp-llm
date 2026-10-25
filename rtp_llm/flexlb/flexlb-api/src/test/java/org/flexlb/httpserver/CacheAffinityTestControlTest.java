package org.flexlb.httpserver;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CacheAffinityTestControlTest {
    @Test
    void routesAreAbsentWithoutExplicitOptIn() {
        new ApplicationContextRunner()
                .withBean(ConfigService.class, () -> mock(ConfigService.class))
                .withPropertyValues("flexlb.test.control.token=property-is-not-an-environment-opt-in")
                .withUserConfiguration(CacheAffinityTestControl.class)
                .run(context -> assertThat(context).doesNotHaveBean(CacheAffinityTestControl.class));
    }

    @Test
    void interventionRequiresTokenAndExpectedPreviousValue() {
        var config = new FlexlbConfig();
        var affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(1000000000L);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var client = WebTestClient.bindToRouterFunction(
                new CacheAffinityTestControl(service, "test-secret").cacheAffinityTestRoutes()).build();
        String uri = "/rtp_llm/test/cache_affinity";
        client.post().uri(uri).bodyValue("{}").exchange().expectStatus().isNotFound();
        for (String invalid : new String[]{"{}", "[]", "{bad", "{\"maxExtraTtftMs\":-1,\"expectedMaxExtraTtftMs\":1000000000}",
                "{\"maxExtraTtftMs\":0.5,\"expectedMaxExtraTtftMs\":1000000000}"}) {
            client.post().uri(uri).header("X-Flexlb-Test-Token", "test-secret")
                    .contentType(MediaType.APPLICATION_JSON).bodyValue(invalid)
                    .exchange().expectStatus().isBadRequest();
        }
        assertThat(affinity.getMaxExtraTtftMs()).isEqualTo(1000000000L);
        client.post().uri(uri).header("X-Flexlb-Test-Token", "test-secret")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{\"expectedMaxExtraTtftMs\":1000000000,\"maxExtraTtftMs\":0}")
                .exchange().expectStatus().isOk().expectBody().jsonPath("$.before").isEqualTo(1000000000);
        client.get().uri(uri).header("X-Flexlb-Test-Token", "test-secret")
                .exchange().expectStatus().isOk().expectBody().jsonPath("$.maxExtraTtftMs").isEqualTo(0);
        client.post().uri(uri).header("X-Flexlb-Test-Token", "test-secret")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{\"expectedMaxExtraTtftMs\":1000000000,\"maxExtraTtftMs\":1}")
                .exchange().expectStatus().isEqualTo(409);
        assertThat(affinity.getMaxExtraTtftMs()).isZero();
    }
}
