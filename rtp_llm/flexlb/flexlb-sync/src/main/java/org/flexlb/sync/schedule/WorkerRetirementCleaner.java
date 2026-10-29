package org.flexlb.sync.schedule;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Failures;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Retries retirement of confirmed discovery removals and worker-reported failures.
 */
@Component
public class WorkerRetirementCleaner {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final CacheAwareService cacheAwareService;
    private final EndpointRegistry endpointRegistry;

    @Autowired
    public WorkerRetirementCleaner(
            CacheAwareService cacheAwareService, EndpointRegistry endpointRegistry) {
        this.cacheAwareService = Objects.requireNonNull(cacheAwareService, "cacheAwareService");
        this.endpointRegistry = Objects.requireNonNull(endpointRegistry, "endpointRegistry");
    }

    @Scheduled(
            fixedRateString =
                    "#{@configService.loadBalanceConfig().workerRegistry.health.cleanupIntervalMs}")
    public void retireUnavailableWorkers() {
        List<EndpointRegistry.Retirement> retirements = new ArrayList<>();
        Throwable failure = null;
        try {
            // Close every unavailable routing gate before scheduling cleanup.
            for (RoleType role : RoleType.values()) {
                for (var item : endpointRegistry.statusSnapshot(role).entrySet()) {
                    WorkerStatus status = item.getValue();
                    var retirement =
                            endpointRegistry.beginRequestedRetirement(role, item.getKey(), status);
                    if (retirement != null) {
                        retirements.add(retirement);
                    }
                }
            }
        } catch (Throwable detachFailure) {
            failure = detachFailure;
        }
        for (EndpointRegistry.Retirement retirement : retirements) {
            failure =
                    Failures.run(
                            failure, () -> retirement.completeAsync(cacheAwareService, logger));
            WorkerStatus status = retirement.status();
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    logger.warn(
                                            "Retiring unavailable worker: {}, role: {}, generation={}, trace={}",
                                            status.getLogicalIpPort(),
                                            status.getRole(),
                                            status.getGenerationId(),
                                            status.trace().snapshot().compact()));
        }
        Failures.rethrow(failure, "Worker retirement scan failed");
    }
}
