package org.flexlb.httpserver;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.config.ConfigService;
import org.flexlb.config.RoutingConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Condition;
import org.springframework.context.annotation.ConditionContext;
import org.springframework.context.annotation.Conditional;
import org.springframework.core.type.AnnotatedTypeMetadata;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.Map;

/** Opt-in control for isolated tests that intervene after a shared warmup. */
@Component
@Conditional(CacheAffinityTestControl.Enabled.class)
public final class CacheAffinityTestControl {
    private final ConfigService config;
    private final String token;

    @Autowired
    public CacheAffinityTestControl(ConfigService config) {
        this(config, System.getenv("FLEXLB_TEST_CONTROL_TOKEN"));
    }

    public CacheAffinityTestControl(ConfigService config, String token) {
        this.config = config;
        this.token = token;
    }

    // Application deliberately disables Spring getenv property lookup.
    public static final class Enabled implements Condition {
        @Override
        public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
            String value = System.getenv("FLEXLB_TEST_CONTROL_TOKEN");
            return value != null && !value.isBlank();
        }
    }

    @Bean
    public RouterFunction<ServerResponse> cacheAffinityTestRoutes() {
        return RouterFunctions.route()
                .GET("/rtp_llm/test/cache_affinity", this::read)
                .POST("/rtp_llm/test/cache_affinity", this::update)
                .build();
    }

    private boolean authorized(ServerRequest request) {
        return token != null && !token.isBlank() && token.equals(request.headers().firstHeader("X-Flexlb-Test-Token"));
    }

    private RoutingConfig.CacheAffinityConfig affinity() {
        return config.loadBalanceConfig().getRouter().getRoles().getPrefill().getCacheAffinity();
    }

    private Mono<ServerResponse> read(ServerRequest request) {
        if (!authorized(request)) return ServerResponse.notFound().build();
        var affinity = affinity();
        if (affinity == null) return ServerResponse.status(409).build();
        return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("maxExtraTtftMs", affinity.getMaxExtraTtftMs()));
    }

    private Mono<ServerResponse> update(ServerRequest request) {
        if (!authorized(request)) return ServerResponse.notFound().build();
        var affinity = affinity();
        if (affinity == null) return ServerResponse.status(409).build();
        return request.bodyToMono(JsonNode.class).flatMap(body -> {
            var value = body.get("maxExtraTtftMs");
            var expected = body.get("expectedMaxExtraTtftMs");
            if (!body.isObject() || body.size() != 2 || !nonnegativeLong(value) || !nonnegativeLong(expected)) {
                return ServerResponse.badRequest().build();
            }
            synchronized (affinity) {
                long before = affinity.getMaxExtraTtftMs();
                if (before != expected.longValue()) return ServerResponse.status(409).build();
                affinity.setMaxExtraTtftMs(value.longValue());
                return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(Map.of("before", before, "maxExtraTtftMs", value.longValue()));
            }
        }).switchIfEmpty(ServerResponse.badRequest().build())
                .onErrorResume(org.springframework.web.server.ServerWebInputException.class,
                        error -> ServerResponse.badRequest().build());
    }

    private static boolean nonnegativeLong(JsonNode value) {
        return value != null && value.isIntegralNumber() && value.canConvertToLong() && value.longValue() >= 0;
    }
}
