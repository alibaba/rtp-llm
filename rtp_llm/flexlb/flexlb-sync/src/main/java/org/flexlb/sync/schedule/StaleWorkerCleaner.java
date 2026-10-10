package org.flexlb.sync.schedule;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Periodically evicts workers that have stopped sending WorkerStatus reports
 * (crash, network partition, OOM kill, etc.) from the routing tables.
 *
 * <p>Once the last successful status report is older than
 * {@code workerRegistry.health.statusStaleAfterMs}, the cleaner retires the
 * exact WorkerStatus generation and its endpoint together. Removing both
 * owners prevents routing from retaining a worker which service discovery no
 * longer observes.
 *
 * <p>Runs at {@code workerRegistry.health.cleanupIntervalMs} (default 3 seconds)
 * via Spring {@link Scheduled}. The configured stale timeout is deliberately longer than
 * one status RPC timeout so a single delayed poll does not evict a live worker.
 */
@Component
public class StaleWorkerCleaner {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final long workerTimeoutUs;
    private final CacheAwareService cacheAwareService;
    private final EndpointRegistry endpointRegistry;

    @Autowired
    public StaleWorkerCleaner(
            ConfigService configService,
            CacheAwareService cacheAwareService,
            EndpointRegistry endpointRegistry) {
        this.cacheAwareService = Objects.requireNonNull(
                cacheAwareService, "cacheAwareService");
        this.endpointRegistry = Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        this.workerTimeoutUs = configService.loadBalanceConfig().getWorkerRegistry()
                .getHealth().getStatusStaleAfterMs() * 1000L;
    }

    @Scheduled(fixedRateString = "#{@configService.loadBalanceConfig().workerRegistry.health.cleanupIntervalMs}")
    public void cleanExpiredWorkers() {
        List<EndpointRegistry.Retirement> retirements = new ArrayList<>();
        try {
            // Close every expired routing gate before waiting for any endpoint drain.
            for (RoleType role : RoleType.values()) {
                for (var item : endpointRegistry.statusSnapshot(role).entrySet()) {
                    WorkerStatus status = item.getValue();
                    var retirement = endpointRegistry.beginRetirementIfStale(role, item.getKey(), status, workerTimeoutUs);
                    if (retirement != null) { retirements.add(retirement); }
                }
            }
        } finally {
            for (EndpointRegistry.Retirement retirement : retirements) {
                retirement.complete(cacheAwareService, logger);
                WorkerStatus status = retirement.status();
                logger.warn("Retiring expired worker: {}, role: {}, generation={}",
                        status.getIpPort(), status.getRole(), status.getGenerationId());
            }
        }
    }
}
