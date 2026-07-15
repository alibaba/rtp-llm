package org.flexlb.httpserver;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Resolves the cache block hash configuration reported by healthy prefill workers.
 */
@Slf4j
@Component
public class WorkerBlockHashConfigResolver {

    private static final long REFRESH_INTERVAL_MINUTES = 1L;
    private static final long PROBLEM_LOG_INTERVAL_NANOS = TimeUnit.MINUTES.toNanos(1);

    private final AtomicReference<BlockHashConfig> cachedConfig = new AtomicReference<>();
    private final AtomicLong nextUnavailableWarningNanos = new AtomicLong();
    private final AtomicLong nextInconsistentErrorNanos = new AtomicLong();
    private final ScheduledExecutorService refreshExecutor;

    public WorkerBlockHashConfigResolver() {
        refreshExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "worker-block-hash-config-refresher");
            thread.setDaemon(true);
            return thread;
        });
        refreshExecutor.scheduleWithFixedDelay(
                this::refresh,
                0,
                REFRESH_INTERVAL_MINUTES,
                TimeUnit.MINUTES);
    }

    public BlockHashConfig resolve() {
        BlockHashConfig config = cachedConfig.get();
        if (config == null) {
            refresh();
            config = cachedConfig.get();
        }
        if (config == null) {
            throw new IllegalStateException(
                    "block hash configuration is unavailable from healthy prefill workers");
        }
        return config;
    }

    void refresh() {
        try {
            refreshConfig();
        } catch (Exception e) {
            log.error("Failed to refresh block hash configuration from worker status", e);
        }
    }

    private synchronized void refreshConfig() {
        ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        Set<BlockHashConfig> detectedConfigs = new HashSet<>();
        collectConfigs(modelWorkerStatus.getPrefillStatusMap(), detectedConfigs);
        if (detectedConfigs.isEmpty()) {
            collectConfigs(modelWorkerStatus.getPdFusionStatusMap(), detectedConfigs);
        }

        if (detectedConfigs.isEmpty()) {
            BlockHashConfig currentConfig = cachedConfig.get();
            if (currentConfig == null && shouldLog(nextUnavailableWarningNanos)) {
                log.warn("No block hash configuration available from healthy prefill workers yet");
            } else {
                log.debug("No block hash configuration available; keeping cached value: {}",
                        currentConfig);
            }
            return;
        }
        if (detectedConfigs.size() > 1) {
            if (shouldLog(nextInconsistentErrorNanos)) {
                log.error("Inconsistent block hash configurations from healthy prefill workers: {}; "
                                + "keeping cached value: {}",
                        detectedConfigs, cachedConfig.get());
            } else {
                log.debug("Inconsistent block hash configurations from healthy prefill workers: {}; "
                                + "keeping cached value: {}",
                        detectedConfigs, cachedConfig.get());
            }
            return;
        }

        BlockHashConfig detectedConfig = detectedConfigs.iterator().next();
        BlockHashConfig previousConfig = cachedConfig.getAndSet(detectedConfig);
        nextUnavailableWarningNanos.set(0L);
        nextInconsistentErrorNanos.set(0L);
        if (previousConfig == null) {
            log.info("Resolved worker block hash configuration: {}", detectedConfig);
        } else if (!previousConfig.equals(detectedConfig)) {
            log.warn("Worker block hash configuration changed from {} to {}",
                    previousConfig, detectedConfig);
        }
    }

    private boolean shouldLog(AtomicLong nextLogTimeNanos) {
        long now = System.nanoTime();
        long next = nextLogTimeNanos.get();
        return now >= next
                && nextLogTimeNanos.compareAndSet(next, now + PROBLEM_LOG_INTERVAL_NANOS);
    }

    private void collectConfigs(
            Map<String, WorkerStatus> workerStatusMap,
            Set<BlockHashConfig> configs) {
        for (WorkerStatus workerStatus : workerStatusMap.values()) {
            if (workerStatus == null || !workerStatus.isAlive()) {
                continue;
            }
            CacheStatus cacheStatus = workerStatus.getCacheStatus();
            if (cacheStatus != null && cacheStatus.getBlockSize() > 0) {
                configs.add(new BlockHashConfig(
                        cacheStatus.getBlockSize(),
                        workerStatus.getBlockHashLookaheadTokens()));
            }
        }
    }

    @PreDestroy
    public void shutdown() {
        refreshExecutor.shutdown();
    }

    public record BlockHashConfig(long blockSize, int lookaheadTokens) {
        public BlockHashConfig {
            if (blockSize <= 0) {
                throw new IllegalArgumentException("block_size must be greater than 0");
            }
            if (lookaheadTokens < 0) {
                throw new IllegalArgumentException(
                        "block_hash_lookahead_tokens must not be negative");
            }
        }
    }
}
