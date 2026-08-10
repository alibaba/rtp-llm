package org.flexlb.service.grace;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.service.grace.strategy.ActiveRequestShutdownHooker;
import org.flexlb.service.grace.strategy.HealthCheckHooker;
import org.flexlb.service.grace.strategy.LbConsistencyHooker;
import org.flexlb.service.optimizer.OnlineOptimizerHooker;
import org.springframework.stereotype.Component;

/**
 * Graceful shutdown orchestrator.
 * <p>
 * Executes shutdown hooks in a fixed order:
 * 1. HealthCheck offline  — stop accepting new requests
 * 2. ZK / LB consistency offline — deregister from service discovery
 * 3. Active request drain — wait for in-flight requests to complete
 * 4. OnlineOptimizer shutdown — stop optimizer refresh and retry threads
 */
@Slf4j
@Component
public class GracefulShutdownService {

    private final HealthCheckHooker healthCheckHooker;
    private final LbConsistencyHooker lbConsistencyHooker;
    private final ActiveRequestShutdownHooker activeRequestShutdownHooker;
    private final OnlineOptimizerHooker onlineOptimizerHooker;

    public GracefulShutdownService(HealthCheckHooker healthCheckHooker,
                                   LbConsistencyHooker lbConsistencyHooker,
                                   ActiveRequestShutdownHooker activeRequestShutdownHooker,
                                   OnlineOptimizerHooker onlineOptimizerHooker) {
        this.healthCheckHooker = healthCheckHooker;
        this.lbConsistencyHooker = lbConsistencyHooker;
        this.activeRequestShutdownHooker = activeRequestShutdownHooker;
        this.onlineOptimizerHooker = onlineOptimizerHooker;
    }

    public void offline() {
        log.info("Graceful shutdown: step 1 — mark unhealthy");
        healthCheckHooker.beforeShutdown();

        log.info("Graceful shutdown: step 2 — deregister from service discovery");
        lbConsistencyHooker.beforeShutdown();

        log.info("Graceful shutdown: step 3 — drain active requests");
        activeRequestShutdownHooker.beforeShutdown();

        log.info("Graceful shutdown: step 4 — shutdown OnlineOptimizer");
        onlineOptimizerHooker.beforeShutdown();
    }
}
