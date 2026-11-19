package org.flexlb.httpserver;

import org.flexlb.cache.service.impl.DefaultCacheAwareService;
import org.springframework.context.annotation.Bean;
import org.springframework.http.CacheControl;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

/**
 * Serves a read-only snapshot of this FlexLB instance's cache matching state.
 */
@Component
public class CacheMatchStatusServer {
    private static final String CACHE_MATCH_STATUS_PATH = "/flexlb/cache_match/status";
    private final DefaultCacheAwareService cacheAwareService;

    public CacheMatchStatusServer(DefaultCacheAwareService cacheAwareService) {
        this.cacheAwareService = cacheAwareService;
    }

    @Bean
    public RouterFunction<ServerResponse> cacheMatchStatusRoutes() {
        return route().GET(CACHE_MATCH_STATUS_PATH, accept(MediaType.APPLICATION_JSON),
                request -> ServerResponse.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .cacheControl(CacheControl.noStore())
                        .bodyValue(cacheAwareService.status()))
                .build();
    }
}
