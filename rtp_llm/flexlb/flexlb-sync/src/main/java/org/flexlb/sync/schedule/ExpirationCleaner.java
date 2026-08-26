package org.flexlb.sync.schedule;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.sync.lifecycle.WorkerGenerationRetirement;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Iterator;
import java.util.Map;
import java.util.Objects;

/**
 * Periodically evicts workers that have stopped sending WorkerStatus reports
 * (crash, network partition, OOM kill, etc.) from the routing tables.
 *
 * <p>Without this cleaner, the Master would keep routing requests to dead
 * workers, producing a flood of 8400 / 8513 errors (the "decode death spiral"):
 * every dispatched request times out on a dead endpoint, the request is
 * retried onto another stale entry, and the cycle amplifies until the entire
 * decode fleet appears saturated. This component is the backstop that breaks
 * the cycle — once a worker's last report is older than
 * {@code workerRegistry.health.statusStaleAfterMs}, the entry is removed from both
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
    private final CacheAwareService cacheAwareService;

    @Autowired
    public ExpirationCleaner(
            EndpointRegistry endpointRegistry,
            ConfigService configService,
            CacheAwareService cacheAwareService) {
        this(endpointRegistry, cacheAwareService,
                resolveWorkerTimeoutUs(configService));
    }

    /**
     * Resolve the worker expiration timeout in microseconds.
     *
     * <p>The worker registry owns this timeout. It is configured in milliseconds
     * and converted to the monotonic microsecond clock used by WorkerStatus.
     * The default 10 s is 2× the gRPC sync timeout (5 s), eliminating the race
     * where a transient gRPC delay causes the cleaner to evict a still-alive endpoint.
     */
    private static long resolveWorkerTimeoutUs(ConfigService configService) {
        long configMs = configService.loadBalanceConfig().getWorkerRegistry()
                .getHealth().getStatusStaleAfterMs();
        return configMs * 1000L;
    }

    ExpirationCleaner(
            EndpointRegistry endpointRegistry,
            CacheAwareService cacheAwareService,
            long workerTimeoutUs) {
        this.endpointRegistry = endpointRegistry;
        this.cacheAwareService = Objects.requireNonNull(
                cacheAwareService, "cacheAwareService");
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
                    endpointToRetire = WorkerGenerationRetirement.begin(
                            workerStatus, endpointRegistry, role,
                            item.getKey());
                    retirementStarted = true;
                }
            } finally {
                workerStatus.lock.unlock();
            }
            if (retirementStarted) {
                WorkerGenerationRetirement.complete(
                        workerStatus, workerStatusMap, cacheAwareService,
                        role, item.getKey(), endpointToRetire, logger);
                logger.warn(
                        "Retiring expired worker: {}, role: {}, generation={}",
                        item.getKey(), role, workerStatus.getGenerationId());
            }
        }
    }

}
