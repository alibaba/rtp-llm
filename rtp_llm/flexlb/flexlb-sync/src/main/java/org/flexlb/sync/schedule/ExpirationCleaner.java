package org.flexlb.sync.schedule;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointRetireCause;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerLifecycleState;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Iterator;
import java.util.Map;

/**
 * Periodically evicts workers that have stopped sending WorkerStatus reports
 * (crash, network partition, OOM kill, etc.) from the routing tables.
 *
 * <p>Without this cleaner, the Master would keep routing requests to dead
 * workers, producing a flood of 8400 / 8513 errors (the "decode death spiral"):
 * every dispatched request times out on a dead endpoint, the request is
 * retried onto another stale entry, and the cycle amplifies until the entire
 * decode fleet appears saturated. This component is the backstop that breaks
 * the cycle — once a worker's last report is older than {@code workerTimeoutMs}
 * (default 15 s = 3× the 5 s gRPC sync timeout), the entry is removed from both
 * {@link EngineWorkerStatus} and {@link EndpointRegistry}, forcing the
 * scheduler to rediscover live workers on the next sync round.
 *
 * <p>Runs every {@code WORKER_CLEAN_INTERVAL_MS} (default 3 s) via Spring
 * {@link Scheduled}. The timeout is intentionally generous (3× gRPC timeout)
 * to avoid racing a transient gRPC delay and evicting a still-alive endpoint.
 */
@Component
public class ExpirationCleaner {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final long workerTimeoutUs;
    private final EndpointRegistry endpointRegistry;

    @Autowired
    public ExpirationCleaner(EndpointRegistry endpointRegistry, ConfigService configService) {
        this(endpointRegistry, resolveWorkerTimeoutUs(configService));
    }

    /**
     * Resolve the worker expiration timeout in microseconds.
     *
     * <p>Priority:
     * <ol>
     *   <li>Legacy env var {@code WORKER_TIMEOUT_US} (microseconds) — backward compat,
     *       honored if explicitly set to a valid value.</li>
     *   <li>FlexlbConfig {@code workerTimeoutMs} (milliseconds, default 15000) —
     *       converted to microseconds. Overridable via env {@code WORKER_TIMEOUT_MS}.</li>
     * </ol>
     * The default 15 s is 3× the gRPC sync timeout (5 s), eliminating the race
     * where a transient gRPC delay causes the cleaner to evict a still-alive endpoint.
     */
    private static long resolveWorkerTimeoutUs(ConfigService configService) {
        long configMs = configService.loadBalanceConfig().getWorkerTimeoutMs();
        String legacy = System.getenv("WORKER_TIMEOUT_US");
        if (legacy != null && !legacy.trim().isEmpty()) {
            try {
                long legacyUs = Long.parseLong(legacy.trim());
                logger.warn("Using legacy WORKER_TIMEOUT_US={}us (override config workerTimeoutMs={}ms)",
                        legacyUs, configMs);
                return legacyUs;
            } catch (NumberFormatException ignored) {
                logger.warn("Invalid WORKER_TIMEOUT_US='{}', falling back to workerTimeoutMs={}ms", legacy, configMs);
            }
        }
        return configMs * 1000L;
    }

    ExpirationCleaner(EndpointRegistry endpointRegistry, long workerTimeoutUs) {
        this.endpointRegistry = endpointRegistry;
        this.workerTimeoutUs = workerTimeoutUs;
    }

    @Scheduled(fixedRateString = "${WORKER_CLEAN_INTERVAL_MS:3000}")
    public void cleanExpiredWorkers() {
        ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        this.doClean(modelWorkerStatus.getPrefillStatusMap(), RoleType.PREFILL);
        this.doClean(modelWorkerStatus.getDecodeStatusMap(), RoleType.DECODE);
        this.doClean(modelWorkerStatus.getPdFusionStatusMap(), RoleType.PDFUSION);
        this.doClean(modelWorkerStatus.getVitStatusMap(), RoleType.VIT);
    }

    public void doClean(Map<String, WorkerStatus> workerStatusMap, RoleType role) {
        if (MapUtils.isEmpty(workerStatusMap)) {
            return;
        }

        for (Iterator<Map.Entry<String, WorkerStatus>> it = workerStatusMap.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<String, WorkerStatus> item = it.next();
            WorkerStatus workerStatus = item.getValue();

            long currentTime = System.nanoTime() / 1000;
            WorkerLifecycleState lifecycle = workerStatus.getLifecycleState();
            long freshnessTime;
            if (lifecycle == WorkerLifecycleState.READY) {
                freshnessTime = workerStatus.getStatusLastUpdateTime().get();
            } else if (lifecycle == WorkerLifecycleState.PROBING) {
                // A discovered-but-not-yet-healthy worker has no successful status
                // heartbeat by definition. Keep it while discovery remains fresh.
                freshnessTime = workerStatus.getDiscoveryLastSeenTime().get();
            } else {
                continue;
            }

            if (freshnessTime > 0L && currentTime > freshnessTime + workerTimeoutUs) {
                retireExpiredGeneration(workerStatusMap, role, item.getKey(), workerStatus);
            }
        }
    }

    private void retireExpiredGeneration(Map<String, WorkerStatus> workerStatusMap,
                                         RoleType role,
                                         String ipPort,
                                         WorkerStatus expected) {
        WorkerLifecycleState expiredState;
        expected.lock.lock();
        try {
            if (workerStatusMap.get(ipPort) != expected) {
                return;
            }
            expiredState = expected.getLifecycleState();
            long freshnessTime;
            if (expiredState == WorkerLifecycleState.READY) {
                freshnessTime = expected.getStatusLastUpdateTime().get();
            } else if (expiredState == WorkerLifecycleState.PROBING) {
                freshnessTime = expected.getDiscoveryLastSeenTime().get();
            } else {
                return;
            }
            long currentTime = System.nanoTime() / 1000;
            if (freshnessTime <= 0L
                    || currentTime <= freshnessTime + workerTimeoutUs
                    || !expected.tryBeginRetirement()) {
                return;
            }
        } finally {
            expected.lock.unlock();
        }

        logger.info("Retiring expired worker generation: {}, role: {}, previousState: {}",
                ipPort, role, expiredState);
        boolean endpointRemoved = false;
        try {
            EndpointRetireCause cause = expiredState == WorkerLifecycleState.PROBING
                    ? EndpointRetireCause.DISCOVERY_REMOVED
                    : EndpointRetireCause.STATUS_EXPIRED;
            endpointRemoved = endpointRegistry.retire(role, ipPort, expected, cause);
        } finally {
            workerStatusMap.remove(ipPort, expected);
            expected.markClosed();
        }
        logger.info("Retired expired worker generation: {}, role: {}, endpointRemoved: {}",
                ipPort, role, endpointRemoved);
    }
}
