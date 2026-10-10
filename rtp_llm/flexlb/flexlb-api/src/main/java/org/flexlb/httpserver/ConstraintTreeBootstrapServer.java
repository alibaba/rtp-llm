package org.flexlb.httpserver;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.constraint.ConstraintTreeBootstrapRegistry;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.IdUtils;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.server.ServerWebInputException;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.Map;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Component
public class ConstraintTreeBootstrapServer {
    public static final String PATH = "/rtp_llm/constraint_tree/register";
    private final ConstraintTreeBootstrapRegistry registry;
    private final ModelMetaConfig models;
    private final LBStatusConsistencyService consistency;

    public record Registration(@JsonProperty("service_id") String serviceId,
                               @JsonProperty("http_port") int httpPort, RoleType role) { }

    public ConstraintTreeBootstrapServer(ConstraintTreeBootstrapRegistry registry, ModelMetaConfig models,
                                        LBStatusConsistencyService consistency) {
        this.registry = registry;
        this.models = models;
        this.consistency = consistency;
    }

    @Bean
    public RouterFunction<ServerResponse> constraintTreeBootstrapRoutes() {
        return route().POST(PATH, this::register).build();
    }

    private Mono<ServerResponse> register(ServerRequest request) {
        if (!consistency.isMaster()) {
            String leader = consistency.getMasterHostIpPort();
            return leader == null || leader.isBlank() ? ServerResponse.status(503).build()
                    : ServerResponse.temporaryRedirect(URI.create("http://" + leader + PATH)).build();
        }
        return request.bodyToMono(Registration.class).flatMap(value -> {
            if (value.httpPort() < 1 || value.httpPort() > 65535
                    || (value.role() != RoleType.DECODE && value.role() != RoleType.PDFUSION)) {
                throw new IllegalArgumentException("inference role and native HTTP port are required");
            }
            String serviceId = resolveServiceId(value.serviceId());
            var route = models.getServiceRoute(serviceId);
            if (route == null || route.getRoleEndpoints(value.role()).isEmpty()) {
                throw new IllegalArgumentException("service/role is not configured on this Master");
            }
            // Direct internal HTTP only: never accept a caller-supplied callback URL or forwarded IP.
            var peer = request.exchange().getRequest().getRemoteAddress();
            if (peer == null || peer.getAddress() == null) {
                throw new IllegalArgumentException("Worker peer address is unavailable");
            }
            String ip = peer.getAddress().getHostAddress();
            String host = ip.contains(":") ? "[" + ip + "]" : ip;
            boolean discovered = registry.register(IdUtils.getModelNameByServiceId(serviceId),
                    URI.create("http://" + host + ":" + value.httpPort()));
            return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(Map.of("discovered", discovered));
        }).switchIfEmpty(ServerResponse.badRequest().build())
                .onErrorResume(IllegalArgumentException.class, e -> ServerResponse.badRequest()
                        .bodyValue(Map.of("error", e.getMessage())))
                .onErrorResume(ServerWebInputException.class, e -> ServerResponse.badRequest().build())
                .onErrorResume(IllegalStateException.class, e -> ServerResponse.status(503)
                        .bodyValue(Map.of("error", e.getMessage())));
    }

    private String resolveServiceId(String serviceId) {
        if (serviceId != null && !serviceId.isBlank()) {
            return serviceId;
        }
        var services = models.getServiceIds();
        if (services.isEmpty()) {
            throw new IllegalStateException("Master service configuration is not ready; retry registration");
        }
        if (services.size() != 1) {
            throw new IllegalArgumentException("service_id is required when Master has multiple configured services");
        }
        return services.iterator().next();
    }
}
