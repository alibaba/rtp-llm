package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.springframework.context.annotation.DependsOn;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Objects;

/** Drives request expiration without adding registry access to its lifecycle owner. */
@Component
@DependsOn("requestShutdownOrchestrator")
final class RequestExpirationOrchestrator {

    private final RequestLifecycleCoordinator lifecycle;
    private final EndpointRegistry registry;

    RequestExpirationOrchestrator(
            RequestLifecycleCoordinator lifecycle,
            EndpointRegistry registry) {
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.registry = Objects.requireNonNull(registry, "registry");
    }

    @Scheduled(fixedRate = 60000L)
    void maintainExpiration() {
        lifecycle.maintainExpiration(registry::evictExpiredOrphans);
    }
}
