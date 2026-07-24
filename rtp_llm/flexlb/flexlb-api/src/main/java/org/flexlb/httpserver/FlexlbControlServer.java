package org.flexlb.httpserver;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.cache.domain.CacheMatchModeUpdateRequest;
import org.flexlb.cache.domain.CacheMatchStatus;
import org.flexlb.cache.domain.CacheMatchMode;
import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.LogLevelUpdateRequest;
import org.flexlb.enums.LogLevel;
import org.flexlb.service.monitor.FlexlbLogLevelManager;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Bean;
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
    private static final String CACHE_MATCH_MODE_PATH = "/flexlb/cache_match/mode";

    private final GeneralHttpNettyService generalHttpNettyService;
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator;
    private final FlexlbLogLevelManager logLevelManager;

    public FlexlbControlServer(
            GeneralHttpNettyService generalHttpNettyService,
            LBStatusConsistencyService lbStatusConsistencyService,
            CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator,
            FlexlbLogLevelManager logLevelManager) {
        this.generalHttpNettyService = generalHttpNettyService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.cacheMatchQueryOrchestrator = cacheMatchQueryOrchestrator;
        this.logLevelManager = logLevelManager;
    }

    @Bean
    public RouterFunction<ServerResponse> flexlbControlRoutes() {
        return route()
                .POST(UPDATE_LOG_LEVEL_PATH, accept(MediaType.APPLICATION_JSON),
                        this::updateLogLevel)
                .GET(CACHE_MATCH_STATUS_PATH, accept(MediaType.APPLICATION_JSON),
                        this::cacheMatchStatus)
                .POST(CACHE_MATCH_MODE_PATH, accept(MediaType.APPLICATION_JSON),
                        this::updateCacheMatchMode)
                .build();
    }

    private Mono<ServerResponse> updateLogLevel(ServerRequest request) {
        return request.bodyToMono(LogLevelUpdateRequest.class)
                .flatMap(updateRequest -> {
                    LogLevel updatedLogLevel =
                            logLevelManager.setLogLevel(updateRequest.getLogLevel());
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(
                                    Mono.just("Success! logLevel=" + updatedLogLevel),
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

    private Mono<ServerResponse> updateCacheMatchMode(ServerRequest request) {
        return request.bodyToMono(CacheMatchModeUpdateRequest.class)
                .flatMap(updateRequest -> {
                    if (lbStatusConsistencyService.isNeedConsistency()
                            && !lbStatusConsistencyService.isMaster()) {
                        return forwardCacheMatchModeToMaster(updateRequest);
                    }
                    return applyCacheMatchMode(updateRequest);
                });
    }

    private Mono<ServerResponse> applyCacheMatchMode(
            CacheMatchModeUpdateRequest updateRequest) {
        try {
            CacheMatchMode mode = CacheMatchMode.valueOf(updateRequest.mode());
            cacheMatchQueryOrchestrator.setMode(mode);
            return ServerResponse.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(cacheMatchQueryOrchestrator.status());
        } catch (RuntimeException e) {
            return ServerResponse.badRequest()
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(e.getMessage());
        }
    }

    private Mono<ServerResponse> forwardCacheMatchModeToMaster(
            CacheMatchModeUpdateRequest updateRequest) {
        String master = lbStatusConsistencyService.getMasterHostIpPort();
        if (StringUtils.isBlank(master)) {
            Logger.error("Cannot update cache match mode: real master is unavailable");
            return ServerResponse.status(503)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue("real master is unavailable");
        }

        URI uri = URI.create("http://" + master);
        return generalHttpNettyService.request(
                        updateRequest,
                        uri,
                        CACHE_MATCH_MODE_PATH,
                        CacheMatchStatus.class)
                .flatMap(status -> ServerResponse.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(status))
                .onErrorResume(e -> {
                    Logger.error("Failed to forward cache match mode update to real master: {}",
                            master, e);
                    return ServerResponse.status(503)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue("failed to update cache match mode on real master");
                });
    }
}
