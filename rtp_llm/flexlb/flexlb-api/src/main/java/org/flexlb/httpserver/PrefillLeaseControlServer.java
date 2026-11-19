package org.flexlb.httpserver;

import org.flexlb.service.lease.PrefillLeaseManager;
import org.flexlb.service.lease.PrefillTtftLeaseController;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.Map;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

/**
 * Manual lease controls for bounded tests; requests never change node health or discovery.
 */
@Component
public final class PrefillLeaseControlServer {
    private final PrefillLeaseManager leases;
    private final PrefillTtftLeaseController controller;

    public PrefillLeaseControlServer(PrefillLeaseManager leases, PrefillTtftLeaseController controller) {
        this.leases = leases;
        this.controller = controller;
    }

    public record Change(String group, String address, Boolean enabled) {
    }

    @Bean
    public RouterFunction<ServerResponse> prefillLeaseRoutes() {
        return route().GET("/flexlb/prefill_lease/status", request -> {
            String group = request.queryParam("group").orElse("");
            return ServerResponse.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(Map.of("workers", leases.status(group), "controller", controller.status(group)));
        })
                .POST("/flexlb/prefill_lease/change",
                        request -> request.bodyToMono(Change.class)
                                .switchIfEmpty(Mono.error(new IllegalArgumentException("JSON body is required")))
                                .flatMap(change -> {
                                    if (change.group() == null || change.address() == null
                                            || change.enabled() == null) {
                                        return Mono.error(new IllegalArgumentException(
                                                "group, address and enabled are required"));
                                    }
                                    return ServerResponse.ok()
                                            .contentType(MediaType.APPLICATION_JSON)
                                            .bodyValue(
                                                    leases.change(change.group(), change.address(), change.enabled()));
                                })
                                .onErrorResume(IllegalArgumentException.class,
                                        failure -> ServerResponse.badRequest()
                                                .contentType(MediaType.APPLICATION_JSON)
                                                .bodyValue(Map.of("error", failure.getMessage()))))
                .build();
    }
}
