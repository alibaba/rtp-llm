package org.flexlb.sync.schedule;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.sync.status.WorkerGenerationManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.concurrent.ConcurrentMap;

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
    private final WorkerGenerationManager generationManager;

    @Autowired
    public ExpirationCleaner(WorkerGenerationManager generationManager, ConfigService configService) {
        this(generationManager, resolveWorkerTimeoutUs(configService));
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

    ExpirationCleaner(WorkerGenerationManager generationManager, long workerTimeoutUs) {
        this.generationManager = generationManager;
        this.workerTimeoutUs = workerTimeoutUs;
    }

    @Scheduled(fixedRateString = "${WORKER_CLEAN_INTERVAL_MS:3000}")
    public void cleanExpiredWorkers() {
        ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        this.doClean(modelWorkerStatus.getPrefillStatusMap(), RoleType.PREFILL);
        this.doClean(modelWorkerStatus.getDecodeStatusMap(), RoleType.DECODE);
        this.doClean(modelWorkerStatus.getPdFusionStatusMap(), RoleType.PDFUSION);
        this.doClean(modelWorkerStatus.getVitStatusMap(), RoleType.VIT);
        this.doClean(modelWorkerStatus.getFrontendStatusMap(), RoleType.FRONTEND);
    }

    public void doClean(ConcurrentMap<String, WorkerStatus> workerStatusMap, RoleType role) {
        if (MapUtils.isEmpty(workerStatusMap)) {
            return;
        }

        for (var item : workerStatusMap.entrySet()) {
            WorkerStatus expected = item.getValue();
            boolean removed = generationManager.retireIf(
                    workerStatusMap, role, item.getKey(), expected,
                    current -> System.nanoTime() / 1000
                            > current.getStatusLastUpdateTime().get() + workerTimeoutUs);
            if (removed) {
                logger.warn("Removed expired worker: {}, role: {}", item.getKey(), role);
            }
        }
    }
}
