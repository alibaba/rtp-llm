package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

class SamePortPrecedenceTest {

    @Configuration
    static class DispatcherCatchAllConfig {
        @Bean
        @Order(Ordered.LOWEST_PRECEDENCE)
        RouterFunction<ServerResponse> dispatcherRoutes() {
            return route()
                    .POST("/batch_infer", req -> ServerResponse.ok().bodyValue("BATCH"))
                    .route(RequestPredicates.all(), req -> ServerResponse.ok().bodyValue("PASS"))
                    .build();
        }
    }

    @Configuration
    static class MasterRoutesConfig {
        @Bean
        @Order(0)
        RouterFunction<ServerResponse> masterRoutes() {
            return route()
                    .POST("/rtp_llm/schedule", req -> ServerResponse.ok().bodyValue("MASTER"))
                    .build();
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void orderedBeansMakeMasterWinAndCatchAllForwardsRest() {
        new ApplicationContextRunner()
                // dispatcher registered BEFORE master -> precedence must come from @Order, not bean order
                .withUserConfiguration(DispatcherCatchAllConfig.class, MasterRoutesConfig.class)
                .run(context -> {
                    RouterFunction<ServerResponse> combined = context
                            .getBeanProvider(RouterFunction.class)
                            .orderedStream()
                            .map(rf -> (RouterFunction<ServerResponse>) rf)
                            .reduce((a, b) -> (RouterFunction<ServerResponse>) a.andOther(b))
                            .orElseThrow();

                    WebTestClient client = WebTestClient.bindToRouterFunction(combined).build();
                    client.post().uri("/rtp_llm/schedule").bodyValue("{}")
                            .exchange().expectBody(String.class).isEqualTo("MASTER");
                    client.post().uri("/batch_infer").bodyValue("{}")
                            .exchange().expectBody(String.class).isEqualTo("BATCH");
                    client.get().uri("/worker_status")
                            .exchange().expectBody(String.class).isEqualTo("PASS");
                });
    }
}
