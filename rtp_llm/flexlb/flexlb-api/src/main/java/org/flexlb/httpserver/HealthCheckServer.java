package org.flexlb.httpserver;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.ApplicationWarmupState;
import org.flexlb.service.grace.strategy.HealthCheckHooker;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Component
@Slf4j
public class HealthCheckServer {

    private final ApplicationWarmupState applicationWarmupState;

    public HealthCheckServer(ApplicationWarmupState applicationWarmupState) {
        this.applicationWarmupState = applicationWarmupState;
    }

    /**
     * Health check
     */
    @Bean
    public RouterFunction<ServerResponse> healthCheck() {
        return route()
                .path("/health", b -> b.GET(accept(MediaType.APPLICATION_JSON, MediaType.TEXT_PLAIN),
                        serverRequest -> this.healthHandler()))
                .build();
    }

    public Mono<ServerResponse> healthHandler() {
        // Return 404 if shutdown signal received
        if (HealthCheckHooker.isShutDownSignalReceived) {
            log.info("health check failed, because shutdown signal received");
            return ServerResponse.status(404).body(Mono.just("shutdown received"), String.class);
        }
        // Return 404 if warmup not completed
        if (!applicationWarmupState.isWarmupFinished()) {
            return ServerResponse.status(404).body(Mono.just("warm not finish"), String.class);
        }
        return ServerResponse.ok().body(Mono.just("success"), String.class);
    }

}
