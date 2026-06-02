package org.flexlb.service.grace;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.service.grace.strategy.ActiveRequestShutdownHooker;
import org.flexlb.service.grace.strategy.HealthCheckHooker;
import org.flexlb.service.grace.strategy.LbConsistencyHooker;
import org.springframework.stereotype.Component;

/**
 * Graceful shutdown orchestrator.
 * <p>
 * Executes shutdown hooks in a fixed order:
 * 1. HealthCheck offline  — stop accepting new requests
 * 2. ZK / LB consistency offline — deregister from service discovery
 * 3. Active request drain — wait for in-flight requests to complete
 */
@Slf4j
@Component
public class GracefulShutdownService {

    private final HealthCheckHooker healthCheckHooker;
    private final LbConsistencyHooker lbConsistencyHooker;
    private final ActiveRequestShutdownHooker activeRequestShutdownHooker;

    public GracefulShutdownService(HealthCheckHooker healthCheckHooker,
                                   LbConsistencyHooker lbConsistencyHooker,
                                   ActiveRequestShutdownHooker activeRequestShutdownHooker) {
        this.healthCheckHooker = healthCheckHooker;
        this.lbConsistencyHooker = lbConsistencyHooker;
        this.activeRequestShutdownHooker = activeRequestShutdownHooker;
    }

    public void offline() {
        log.info("Graceful shutdown: step 1 — mark unhealthy");
        healthCheckHooker.beforeShutdown();

        log.info("Graceful shutdown: step 2 — deregister from service discovery");
        lbConsistencyHooker.beforeShutdown();

        log.info("Graceful shutdown: step 3 — drain active requests");
        activeRequestShutdownHooker.beforeShutdown();
    }
}
