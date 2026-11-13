package org.flexlb.sync.schedule;

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
import java.util.function.LongSupplier;

/**
 * Completes pending retirement after confirmed discovery removal, three valid
 * not-alive responses, or — only when {@code workerRegistry.health.statusStaleAfterMs}
 * is configured — a last successful status poll older than that opt-in threshold.
 * RPC failures alone never evict workers, and elapsed time alone never evicts
 * them while the threshold is unset.
 *
 * <p>Runs at {@code workerRegistry.health.cleanupIntervalMs} (default 3 seconds)
 * via Spring {@link Scheduled}.
 */
@Component
public class ExpirationCleaner {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final CacheAwareService cacheAwareService;
    private final WorkerDirectory workerDirectory;
    /** Null keeps elapsed time alone from ever evicting a worker. */
    private final Long statusStaleAfterUs;
    private final LongSupplier statusNowUs;

    @Autowired
    public ExpirationCleaner(
            CacheAwareService cacheAwareService,
            WorkerDirectory workerDirectory,
            ConfigService configService) {
        this(cacheAwareService, workerDirectory,
                configService.loadBalanceConfig().getWorkerRegistry()
                        .getHealth().getStatusStaleAfterMs(),
                () -> System.nanoTime() / 1000L);
    }

    ExpirationCleaner(
            CacheAwareService cacheAwareService,
            WorkerDirectory workerDirectory,
            Long statusStaleAfterMs,
            LongSupplier statusNowUs) {
        this.cacheAwareService = Objects.requireNonNull(
                cacheAwareService, "cacheAwareService");
        this.workerDirectory = Objects.requireNonNull(
                workerDirectory, "workerDirectory");
        this.statusStaleAfterUs =
                statusStaleAfterMs == null ? null : statusStaleAfterMs * 1000L;
        this.statusNowUs = Objects.requireNonNull(statusNowUs, "statusNowUs");
    }

    @Scheduled(fixedRateString = "#{@configService.loadBalanceConfig().workerRegistry.health.cleanupIntervalMs}")
    public void retireUnavailableWorkers() {
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
            retirements.addAll(beginPendingRetirements(
                    workerDirectory.statusSnapshot(role), role));
        }
        completeRetirements(retirements);
    }

    /** Phase one: close unhealthy routing gates without waiting on drains. */
    private List<PendingRetirement> beginPendingRetirements(
            Map<String, WorkerStatus> workerStatusMap,
            RoleType role) {
        if (workerStatusMap.isEmpty()) {
            return List.of();
        }

        List<PendingRetirement> retirements = new ArrayList<>();
        for (Map.Entry<String, WorkerStatus> item
                : workerStatusMap.entrySet()) {
            WorkerStatus workerStatus = item.getValue();
            WorkerStatus.RetirementReason reason = workerStatus.retirementReason();
            if (reason == null && statusStaleAfterUs == null) { continue; }

            EndpointRegistry.DetachedGeneration endpointToRetire;
            workerStatus.trace().record(Stage.RETIREMENT_CHECK_LOCK_ATTEMPT);
            if (!workerStatus.lock.tryLock()) { continue; }
            try {
                workerStatus.trace().record(Stage.RETIREMENT_CHECK_LOCK_ACQUIRED);
                if (!workerDirectory.isCurrentStatus(role, item.getKey(), workerStatus)) {
                    continue;
                }
                if (!workerStatus.isActiveGeneration()) {
                    continue;
                }
                if (reason == null) {
                    reason = staleReasonFor(workerStatus);
                }
                if (reason == null) {
                    continue;
                }
                endpointToRetire = workerDirectory.beginRetirement(
                        role, item.getKey(), workerStatus, reason.name());
            } finally {
                workerStatus.lock.unlock();
                workerStatus.trace().record(Stage.RETIREMENT_CHECK_LOCK_RELEASED);
            }
            retirements.add(new PendingRetirement(
                    workerStatus,
                    role,
                    item.getKey(),
                    endpointToRetire));
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
                    retirement.ipPort(), retirement.role(),
                    retirement.workerStatus().getGenerationId(), retirement.workerStatus().retirementReason());
        }
    }

    /**
     * Decide the opt-in stale reason for one still-undecided active generation;
     * returns {@code null} while it stays fresh.
     */
    private WorkerStatus.RetirementReason staleReasonFor(WorkerStatus workerStatus) {
        if (workerStatus.markStatusStale(statusStaleAfterUs, statusNowUs.getAsLong())) {
            return WorkerStatus.RetirementReason.STATUS_STALE;
        }
        return null;
    }

    private record PendingRetirement(
            WorkerStatus workerStatus,
            RoleType role,
            String ipPort,
            EndpointRegistry.DetachedGeneration endpointToRetire) {
    }
}
