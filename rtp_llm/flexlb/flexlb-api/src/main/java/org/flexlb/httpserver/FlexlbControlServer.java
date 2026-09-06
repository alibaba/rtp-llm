package org.flexlb.httpserver;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.cache.domain.CacheMatchFailoverRequest;
import org.flexlb.cache.domain.CacheMatchStatus;
import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.LogLevelUpdateRequest;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.net.URI;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

/**
 * Exposes FlexLB runtime control-plane endpoints.
 */
@Component
public class FlexlbControlServer {

    private static final String UPDATE_LOG_LEVEL_PATH = "/flexlb/update_log_level";
    private static final String CACHE_MATCH_STATUS_PATH = "/flexlb/cache_match/status";
    private static final String CACHE_MATCH_FAILOVER_PATH = "/flexlb/cache_match/failover";

    private final GeneralHttpNettyService generalHttpNettyService;
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator;

    public FlexlbControlServer(
            GeneralHttpNettyService generalHttpNettyService,
            LBStatusConsistencyService lbStatusConsistencyService,
            CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator) {
        this.generalHttpNettyService = generalHttpNettyService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.cacheMatchQueryOrchestrator = cacheMatchQueryOrchestrator;
    }

    @Bean
    public RouterFunction<ServerResponse> flexlbControlRoutes() {
        return route()
                .POST(UPDATE_LOG_LEVEL_PATH, accept(MediaType.APPLICATION_JSON),
                        this::updateLogLevel)
                .GET(CACHE_MATCH_STATUS_PATH, accept(MediaType.APPLICATION_JSON),
                        this::cacheMatchStatus)
                .POST(CACHE_MATCH_FAILOVER_PATH, accept(MediaType.APPLICATION_JSON),
                        this::updateCacheMatchFailover)
                .build();
    }

    private Mono<ServerResponse> updateLogLevel(ServerRequest request) {
        return request.bodyToMono(LogLevelUpdateRequest.class)
                .flatMap(updateRequest -> {
                    Logger.setLevel(updateRequest.getLogLevel());
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(
                                    Mono.just("Success! logLevel=" + Logger.getLevel()),
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

    private Mono<ServerResponse> updateCacheMatchFailover(ServerRequest request) {
        return request.bodyToMono(CacheMatchFailoverRequest.class)
                .flatMap(updateRequest -> {
                    if (lbStatusConsistencyService.isNeedConsistency()
                            && !lbStatusConsistencyService.isMaster()) {
                        return forwardCacheMatchFailoverToMaster(updateRequest);
                    }
                    return applyCacheMatchFailover(updateRequest);
                })
                .onErrorResume(e -> ServerResponse.badRequest()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(e.getMessage()));
    }

    private Mono<ServerResponse> applyCacheMatchFailover(CacheMatchFailoverRequest updateRequest) {
        try {
            cacheMatchQueryOrchestrator.applyFailoverAction(updateRequest.action());
            return ServerResponse.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(cacheMatchQueryOrchestrator.status());
        } catch (RuntimeException e) {
            return ServerResponse.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(e.getMessage());
        }
    }

    private Mono<ServerResponse> forwardCacheMatchFailoverToMaster(CacheMatchFailoverRequest updateRequest) {
        String master = lbStatusConsistencyService.getMasterHostIpPort();
        if (StringUtils.isBlank(master)) {
            Logger.error("Cannot update cache failover state: real master is unavailable");
            return ServerResponse.status(503)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue("real master is unavailable");
        }

        URI uri = URI.create("http://" + master);
        return generalHttpNettyService.request(
                        updateRequest,
                        uri,
                        CACHE_MATCH_FAILOVER_PATH,
                        CacheMatchStatus.class)
                .flatMap(status -> ServerResponse.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(status))
                .onErrorResume(e -> {
                    Logger.error("Failed to forward cache failover update to real master: {}",
                            master, e);
                    return ServerResponse.status(503)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue("failed to update cache failover state on real master");
                });
    }
}
