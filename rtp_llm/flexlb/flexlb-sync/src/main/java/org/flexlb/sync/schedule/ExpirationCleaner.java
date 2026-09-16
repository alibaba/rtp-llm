package org.flexlb.sync.schedule;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
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
 * <p>Runs every {@code WORKER_CLEAN_INTERVAL_MS} (default 3 s) via Spring
 * {@link Scheduled}. The configured stale timeout is deliberately longer than
 * one status RPC timeout so a single delayed poll does not evict a live worker.
 */
@Component
public class ExpirationCleaner {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final long workerTimeoutUs;
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

    @Scheduled(fixedRateString = "${WORKER_CLEAN_INTERVAL_MS:3000}")
    public void cleanExpiredWorkers() {
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
            workerStatus.lock.lock();
            try {
                if (workerStatusMap.get(item.getKey()) != workerStatus) {
                    continue;
                }
                if (!workerStatus.isActiveGeneration()) {
                    continue;
                }
                WorkerStatus.PollHealth health = workerStatus.pollHealth();
                long expirationTime = health.lastSuccessfulPollUs()
                        + workerTimeoutUs;
                long currentTime = System.nanoTime() / 1000;
                if (currentTime > expirationTime) {
                    endpointToRetire = workerDirectory.beginRetirement(
                            role, item.getKey(), workerStatus);
                    retirementStarted = true;
                }
            } finally {
                workerStatus.lock.unlock();
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

    /** Phase two: await already-detached generations and finalize identities. */
    private void completeRetirements(List<PendingRetirement> retirements) {
        for (PendingRetirement retirement : retirements) {
            workerDirectory.completeRetirement(
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
