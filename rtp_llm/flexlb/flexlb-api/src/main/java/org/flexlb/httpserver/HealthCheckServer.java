package org.flexlb.httpserver;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.service.grace.strategy.HealthCheckUpdater;
import org.flexlb.service.grace.strategy.QueryWarmer;
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
@Data
public class HealthCheckServer {

    /**
     * 健康检查
     */
    @Bean
    public RouterFunction<ServerResponse> healthCheck() {
        return route()
                .path("/health", b -> b.GET(accept(MediaType.APPLICATION_JSON, MediaType.TEXT_PLAIN),
                        serverRequest -> this.healthHandler()))
                .build();
    }

    public Mono<ServerResponse> healthHandler() {
        // 如果收到关闭信号，则返回404
        if (HealthCheckUpdater.isShutDownSignalReceived) {
            log.info("health check failed, because shutdown signal received");
            return ServerResponse.status(404).body(Mono.just("shutdown received"), String.class);
        }
        // 如果未完成预热，则返回404
        if (!QueryWarmer.warmUpFinished) {
            return ServerResponse.status(404).body(Mono.just("warm not finish"), String.class);
        }
        return ServerResponse.ok().body(Mono.just("success"), String.class);
    }

}
