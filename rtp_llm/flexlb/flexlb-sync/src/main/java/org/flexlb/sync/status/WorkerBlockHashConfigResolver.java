package org.flexlb.sync.status;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.BlockHashConfig;
import org.flexlb.cache.hash.BlockHashConfigResolver;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Resolves the block hash configuration reported by alive Prefill workers.
 *
 * <p>PD Fusion workers are used only when no alive Prefill worker reports a valid
 * configuration. The last valid configuration remains available while workers are
 * temporarily unavailable or report inconsistent values.
 */
@Slf4j
@Component
public class WorkerBlockHashConfigResolver implements BlockHashConfigResolver {

    private static final long REFRESH_INTERVAL_MINUTES = 1L;
    private static final long PROBLEM_LOG_INTERVAL_NANOS = TimeUnit.MINUTES.toNanos(1);

    private final WorkerStatusProvider workerStatusProvider;
    private final ScheduledExecutorService refreshExecutor;

    private volatile BlockHashConfig cachedConfig;
    private long nextUnavailableWarningNanos;
    private long nextInconsistentErrorNanos;

    public WorkerBlockHashConfigResolver(WorkerStatusProvider workerStatusProvider) {
        this.workerStatusProvider = workerStatusProvider;
        this.refreshExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "worker-block-hash-config-refresher");
            thread.setDaemon(true);
            return thread;
        });
    }

    @PostConstruct
    public void startRefreshing() {
        refreshExecutor.scheduleWithFixedDelay(this::refresh, 0, REFRESH_INTERVAL_MINUTES, TimeUnit.MINUTES);
    }

    @Override
    public BlockHashConfig resolve() {
        BlockHashConfig config = cachedConfig;
        if (config == null) {
            refresh();
            config = cachedConfig;
        }
        if (config == null) {
            throw new IllegalStateException("block hash configuration is unavailable from alive engine workers");
        }
        return config;
    }

    void refresh() {
        try {
            refreshCachedConfig();
        } catch (Exception e) {
            log.error("Failed to refresh block hash configuration from worker status", e);
        }
    }

    private synchronized void refreshCachedConfig() {
        Set<BlockHashConfig> detectedConfigs = findPreferredBlockHashConfigs();

        if (detectedConfigs.isEmpty()) {
            BlockHashConfig currentConfig = cachedConfig;
            if (currentConfig == null && shouldLogUnavailableWarning()) {
                log.warn("No block hash configuration available from alive Prefill or PD Fusion workers yet");
            } else {
                log.debug("No block hash configuration available; keeping cached value: {}", currentConfig);
            }
            return;
        }
        if (detectedConfigs.size() > 1) {
            if (shouldLogInconsistentError()) {
                log.error("Inconsistent block hash configurations from alive workers: {}; keeping cached value: {}",
                        detectedConfigs, cachedConfig);
            } else {
                log.debug("Inconsistent block hash configurations from alive workers: {}; keeping cached value: {}",
                        detectedConfigs, cachedConfig);
            }
            return;
        }

        updateCachedConfig(detectedConfigs.iterator().next());
    }

    private Set<BlockHashConfig> findPreferredBlockHashConfigs() {
        Set<BlockHashConfig> prefillConfigs = findBlockHashConfigsFromAliveWorkers(RoleType.PREFILL);
        return prefillConfigs.isEmpty()
                ? findBlockHashConfigsFromAliveWorkers(RoleType.PDFUSION)
                : prefillConfigs;
    }

    private Set<BlockHashConfig> findBlockHashConfigsFromAliveWorkers(RoleType roleType) {
        Set<BlockHashConfig> configs = new HashSet<>();
        for (WorkerStatus workerStatus : workerStatusProvider.getWorkerStatuses(roleType, null)) {
            if (workerStatus == null || !workerStatus.isAlive()) {
                continue;
            }
            CacheStatus cacheStatus = workerStatus.getCacheStatus();
            if (cacheStatus == null || cacheStatus.getBlockSize() <= 0) {
                continue;
            }
            configs.add(new BlockHashConfig(cacheStatus.getBlockSize(), workerStatus.getBlockHashLookaheadTokens()));
        }
        return configs;
    }

    private void updateCachedConfig(BlockHashConfig detectedConfig) {
        BlockHashConfig previousConfig = cachedConfig;
        cachedConfig = detectedConfig;
        nextUnavailableWarningNanos = 0L;
        nextInconsistentErrorNanos = 0L;
        if (previousConfig == null) {
            log.info("Resolved worker block hash configuration: {}", detectedConfig);
        } else if (!previousConfig.equals(detectedConfig)) {
            log.warn("Worker block hash configuration changed from {} to {}", previousConfig, detectedConfig);
        }
    }

    private boolean shouldLogUnavailableWarning() {
        long now = System.nanoTime();
        if (now < nextUnavailableWarningNanos) {
            return false;
        }
        nextUnavailableWarningNanos = now + PROBLEM_LOG_INTERVAL_NANOS;
        return true;
    }

    private boolean shouldLogInconsistentError() {
        long now = System.nanoTime();
        if (now < nextInconsistentErrorNanos) {
            return false;
        }
        nextInconsistentErrorNanos = now + PROBLEM_LOG_INTERVAL_NANOS;
        return true;
    }

    @PreDestroy
    public void shutdown() {
        refreshExecutor.shutdown();
    }
}
