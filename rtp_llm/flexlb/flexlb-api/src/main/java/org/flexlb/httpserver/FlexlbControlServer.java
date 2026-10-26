package org.flexlb.httpserver;

import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.dao.loadbalance.LogLevelUpdateRequest;
import org.flexlb.service.monitor.FlexlbLogManager;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

/**
 * Exposes FlexLB runtime control-plane endpoints.
 */
@Component
public class FlexlbControlServer {

    private static final String UPDATE_LOG_LEVEL_PATH = "/flexlb/update_log_level";
    private static final String CACHE_MATCH_STATUS_PATH = "/flexlb/cache_match/status";

    private final CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator;
    private final FlexlbLogManager flexlbLogManager;

    public FlexlbControlServer(CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator,
                               FlexlbLogManager flexlbLogManager) {
        this.cacheMatchQueryOrchestrator = cacheMatchQueryOrchestrator;
        this.flexlbLogManager = flexlbLogManager;
    }

    @Bean
    public RouterFunction<ServerResponse> flexlbControlRoutes() {
        return route()
                .POST(UPDATE_LOG_LEVEL_PATH, accept(MediaType.APPLICATION_JSON),
                        this::updateLogLevel)
                .GET(CACHE_MATCH_STATUS_PATH, accept(MediaType.APPLICATION_JSON),
                        this::cacheMatchStatus)
                .build();
    }

    private Mono<ServerResponse> updateLogLevel(ServerRequest request) {
        return request.bodyToMono(LogLevelUpdateRequest.class)
                .flatMap(updateRequest -> {
                    flexlbLogManager.setLogLevel(updateRequest.getLogLevel());
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(
                                    Mono.just("Success! logLevel=" + updateRequest.getLogLevel()),
                                    String.class);
                })
                .onErrorResume(e -> {
                    Logger.error("update logLevel error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    private Mono<ServerResponse> cacheMatchStatus(ServerRequest request) {
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(cacheMatchQueryOrchestrator.status());
    }

}
