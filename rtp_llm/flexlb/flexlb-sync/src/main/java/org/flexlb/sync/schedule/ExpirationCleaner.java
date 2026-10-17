package org.flexlb.sync.schedule;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
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
public class ExpirationCleaner {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final long workerTimeoutUs;
    private final long statusCommitGraceUs;
    private final CacheAwareService cacheAwareService;
    private final WorkerDirectory workerDirectory;

    @Autowired
    public ExpirationCleaner(
            ConfigService configService,
            CacheAwareService cacheAwareService,
            WorkerDirectory workerDirectory) {
        this.cacheAwareService = Objects.requireNonNull(
                cacheAwareService, "cacheAwareService");
        this.workerDirectory = Objects.requireNonNull(
                workerDirectory, "workerDirectory");
        this.workerTimeoutUs = resolveWorkerTimeoutUs(configService);
        this.statusCommitGraceUs = configService.loadBalanceConfig().getWorkerRegistry()
                .getHealth().getStatusCommitGraceMs() * 1000L;
    }

    /**
     * Resolve the worker expiration timeout in microseconds.
     *
     * <p>The worker registry owns this timeout. It is configured in milliseconds
     * and converted to the monotonic microsecond clock used by WorkerStatus.
     * The default 10 s is twice the default status RPC timeout, avoiding
     * retirement on one transiently delayed poll.
     */
    private static long resolveWorkerTimeoutUs(ConfigService configService) {
        long configMs = configService.loadBalanceConfig().getWorkerRegistry()
                .getHealth().getStatusStaleAfterMs();
        return configMs * 1000L;
    }

    @Scheduled(fixedRateString = "#{@configService.loadBalanceConfig().workerRegistry.health.cleanupIntervalMs}")
    public void cleanExpiredWorkers() {
        if (logger.isInfoEnabled()) {
            logger.info("event=worker_counts phase=CLEANUP_START "
                            + "prefill_discovered={} prefill_published={} "
                            + "decode_discovered={} decode_published={} "
                            + "vit_discovered={} vit_published={} "
                            + "pdfusion_discovered={} pdfusion_published={}",
                    workerDirectory.discoveredCount(RoleType.PREFILL), workerDirectory.routingCapacity(RoleType.PREFILL),
                    workerDirectory.discoveredCount(RoleType.DECODE), workerDirectory.routingCapacity(RoleType.DECODE),
                    workerDirectory.discoveredCount(RoleType.VIT), workerDirectory.routingCapacity(RoleType.VIT),
                    workerDirectory.discoveredCount(RoleType.PDFUSION), workerDirectory.routingCapacity(RoleType.PDFUSION));
        }
        List<PendingRetirement> retirements = new ArrayList<>();
        for (RoleType role : RoleType.values()) {
            retirements.addAll(beginExpiredRetirements(
                    workerDirectory.statusSnapshot(role), role));
        }
        completeRetirements(retirements);
    }

    /** Phase one: close every expired routing gate without waiting on drains. */
    private List<PendingRetirement> beginExpiredRetirements(
            Map<String, WorkerStatus> workerStatusMap,
            RoleType role) {
        if (MapUtils.isEmpty(workerStatusMap)) {
            return List.of();
        }

        List<PendingRetirement> retirements = new ArrayList<>();
        for (Map.Entry<String, WorkerStatus> item
                : workerStatusMap.entrySet()) {
            WorkerStatus workerStatus = item.getValue();

            EndpointRegistry.DetachedGeneration endpointToRetire = null;
            boolean retirementStarted = false;
            workerStatus.trace().record(Stage.EXPIRATION_LOCK_ATTEMPT);
            if (!workerStatus.lock.tryLock()) { continue; }
            try {
                workerStatus.trace().record(Stage.EXPIRATION_LOCK_ACQUIRED);
                if (workerStatusMap.get(item.getKey()) != workerStatus) {
                    continue;
                }
                if (!workerStatus.isActiveGeneration()) {
                    continue;
                }
                WorkerStatus.HealthRetirementReason reason = workerStatus.tryClaimHealthRetirement(
                        System.nanoTime() / 1000, workerTimeoutUs, statusCommitGraceUs);
                if (reason != null) {
                    endpointToRetire = workerDirectory.beginRetirement(
                            role, item.getKey(), workerStatus, reason.name(), workerTimeoutUs);
                    retirementStarted = true;
                }
            } finally {
                workerStatus.lock.unlock();
                workerStatus.trace().record(Stage.EXPIRATION_LOCK_RELEASED);
            }
            if (retirementStarted) {
                retirements.add(new PendingRetirement(
                        workerStatus,
                        role,
                        item.getKey(),
                        endpointToRetire));
            }
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
                    "Retiring expired worker: {}, role: {}, generation={}",
                    retirement.ipPort(), retirement.role(),
                    retirement.workerStatus().getGenerationId());
        }
    }

    private record PendingRetirement(
            WorkerStatus workerStatus,
            RoleType role,
            String ipPort,
            EndpointRegistry.DetachedGeneration endpointToRetire) {
    }
}
