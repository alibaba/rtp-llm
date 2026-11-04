package org.flexlb.sync.schedule;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Completes pending retirement after confirmed discovery removal or three valid not-alive
 * responses. RPC failures and elapsed time alone never evict workers.
 *
 * <p>Runs at {@code workerRegistry.health.cleanupIntervalMs} (default 3 seconds)
 * via Spring {@link Scheduled}.
 */
@Component
public class ExpirationCleaner {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final CacheAwareService cacheAwareService;
    private final WorkerDirectory workerDirectory;

    @Autowired
    public ExpirationCleaner(CacheAwareService cacheAwareService, WorkerDirectory workerDirectory) {
        this.cacheAwareService = Objects.requireNonNull(cacheAwareService, "cacheAwareService");
        this.workerDirectory = Objects.requireNonNull(workerDirectory, "workerDirectory");
    }

    @Scheduled(
            fixedRateString =
                    "#{@configService.loadBalanceConfig().workerRegistry.health.cleanupIntervalMs}")
    public void retireUnavailableWorkers() {
        if (logger.isInfoEnabled()) {
            logger.info(
                    "event=worker_counts phase=CLEANUP_START "
                            + "prefill_discovered={} prefill_published={} "
                            + "decode_discovered={} decode_published={} "
                            + "vit_discovered={} vit_published={} "
                            + "pdfusion_discovered={} pdfusion_published={}",
                    workerDirectory.discoveredCount(RoleType.PREFILL),
                    workerDirectory.routingCapacity(RoleType.PREFILL),
                    workerDirectory.discoveredCount(RoleType.DECODE),
                    workerDirectory.routingCapacity(RoleType.DECODE),
                    workerDirectory.discoveredCount(RoleType.VIT),
                    workerDirectory.routingCapacity(RoleType.VIT),
                    workerDirectory.discoveredCount(RoleType.PDFUSION),
                    workerDirectory.routingCapacity(RoleType.PDFUSION));
        }
        List<PendingRetirement> retirements = new ArrayList<>();
        for (RoleType role : RoleType.values()) {
            retirements.addAll(beginPendingRetirements(workerDirectory.statusSnapshot(role), role));
        }
        completeRetirements(retirements);
    }

    /** Phase one: close unhealthy routing gates without waiting on drains. */
    private List<PendingRetirement> beginPendingRetirements(
            Map<String, WorkerStatus> workerStatusMap, RoleType role) {
        if (workerStatusMap.isEmpty()) {
            return List.of();
        }

        List<PendingRetirement> retirements = new ArrayList<>();
        for (Map.Entry<String, WorkerStatus> item : workerStatusMap.entrySet()) {
            WorkerStatus workerStatus = item.getValue();
            WorkerStatus.RetirementReason reason = workerStatus.retirementReason();
            if (reason == null) {
                continue;
            }

            EndpointRegistry.DetachedGeneration endpointToRetire;
            workerStatus.trace().record(Stage.RETIREMENT_CHECK_LOCK_ATTEMPT);
            if (!workerStatus.lock.tryLock()) {
                continue;
            }
            try {
                workerStatus.trace().record(Stage.RETIREMENT_CHECK_LOCK_ACQUIRED);
                if (!workerDirectory.isCurrentStatus(role, item.getKey(), workerStatus)) {
                    continue;
                }
                if (!workerStatus.isActiveGeneration()) {
                    continue;
                }
                endpointToRetire =
                        workerDirectory.beginRetirement(
                                role, item.getKey(), workerStatus, reason.name());
            } finally {
                workerStatus.lock.unlock();
                workerStatus.trace().record(Stage.RETIREMENT_CHECK_LOCK_RELEASED);
            }
            retirements.add(
                    new PendingRetirement(workerStatus, role, item.getKey(), endpointToRetire));
        }
        return retirements;
    }

    /** Phase two: hand off drains and exact-generation cleanup without waiting. */
    private void completeRetirements(List<PendingRetirement> retirements) {
        for (PendingRetirement retirement : retirements) {
            workerDirectory.completeRetirementAsync(
                    retirement.role(),
                    retirement.ipPort(),
                    retirement.workerStatus(),
                    retirement.endpointToRetire(),
                    cacheAwareService,
                    logger);
            logger.warn(
                    "Retiring unavailable worker: {}, role: {}, generation={}, reason={}",
                    retirement.ipPort(),
                    retirement.role(),
                    retirement.workerStatus().getGenerationId(),
                    retirement.workerStatus().retirementReason());
        }
    }

    private record PendingRetirement(
            WorkerStatus workerStatus,
            RoleType role,
            String ipPort,
            EndpointRegistry.DetachedGeneration endpointToRetire) {}
}
