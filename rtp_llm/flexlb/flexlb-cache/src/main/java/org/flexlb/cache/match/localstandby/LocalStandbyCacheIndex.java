package org.flexlb.cache.match.localstandby;

import lombok.extern.slf4j.Slf4j;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Expiring reverse index from a block hash to the workers expected to cache that block.
 *
 * <p>The index is an approximate standby for KVCM. Updates only refresh timestamps and never
 * maintain an exact LRU order. As the index approaches its estimated metadata capacity, the
 * effective TTL is reduced and background cleanup runs more frequently.
 */
@Slf4j
class LocalStandbyCacheIndex {

    private static final long CLEANUP_CHECK_INTERVAL_MS = 10_000;
    private static final int NORMAL_CLEANUP_BATCH_DIVISOR = 10;
    private static final int PRESSURE_CLEANUP_BATCH_DIVISOR = 5;
    private static final int NORMAL_CHECKS_BEFORE_CLEANUP = 3;
    private static final int PRESSURE_CHECKS_BEFORE_CLEANUP = 2;
    private static final double FULL_SCAN_TRIGGER_RATIO = 0.9;

    private volatile ExpirationSettings expirationSettings;
    private final boolean automaticCleanupEnabled;
    private final ConcurrentHashMap<Long, ConcurrentHashMap<String, Long>> blockToEnginesMap = new ConcurrentHashMap<>();
    // Prevent duplicate request-triggered full scans from being queued or run concurrently.
    private final AtomicBoolean highWatermarkCleanupTriggered = new AtomicBoolean();
    private final AtomicLong mappingCount = new AtomicLong();
    private final ScheduledExecutorService cleanupExecutor;
    private volatile long maximumEntries;
    private Iterator<Long> cleanupIterator;
    private int checksSinceLastCleanup;

    LocalStandbyCacheIndex(long ttlMs,
                           long minimumTtlMs,
                           double ttlReductionStartRatio,
                           long maximumEntries,
                           boolean enabled) {
        this.expirationSettings = expirationSettings(ttlMs, minimumTtlMs, ttlReductionStartRatio);
        this.maximumEntries = maximumEntries;
        this.automaticCleanupEnabled = enabled;
        this.cleanupExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "local-standby-cache-cleaner");
            thread.setDaemon(true);
            return thread;
        });
        if (enabled) {
            cleanupExecutor.scheduleWithFixedDelay(
                    this::runCleanupCheck,
                    CLEANUP_CHECK_INTERVAL_MS,
                    CLEANUP_CHECK_INTERVAL_MS,
                    TimeUnit.MILLISECONDS);
        }
    }

    int addWorkerBlockMappings(String workerIpPort, List<Long> blockCacheKeys) {
        if (workerIpPort == null || workerIpPort.isEmpty() || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return 0;
        }

        long lastUpdatedNanos = System.nanoTime();
        int[] rejectedMappings = new int[1];
        for (Long blockCacheKey : blockCacheKeys) {
            if (blockCacheKey == null) {
                continue;
            }

            // Refreshing a popular mapping avoids outer-map compute(), which serializes updates
            // for the same block hash.
            ConcurrentHashMap<String, Long> existingWorkers = blockToEnginesMap.get(blockCacheKey);
            if (existingWorkers != null && existingWorkers.replace(workerIpPort, lastUpdatedNanos) != null) {
                continue;
            }

            blockToEnginesMap.compute(blockCacheKey, (blockHash, mappedWorkers) -> {
                ConcurrentHashMap<String, Long> currentWorkers = mappedWorkers;
                if (currentWorkers == null) {
                    currentWorkers = new ConcurrentHashMap<>();
                }

                if (currentWorkers.replace(workerIpPort, lastUpdatedNanos) != null) {
                    return currentWorkers;
                }
                if (!incrementMappingCountIfBelowLimit()) {
                    rejectedMappings[0]++;
                    return currentWorkers.isEmpty() ? null : currentWorkers;
                }
                currentWorkers.put(workerIpPort, lastUpdatedNanos);
                return currentWorkers;
            });
        }
        requestHighWatermarkCleanupIfNeeded();
        return rejectedMappings[0];
    }

    Map<String, Long> getUnexpiredEnginesForBlock(Long blockCacheKey, long queryTimeNanos) {
        if (blockCacheKey == null) {
            return null;
        }
        ConcurrentHashMap<String, Long> workers = blockToEnginesMap.get(blockCacheKey);
        if (workers == null) {
            return null;
        }

        // Keep the common query path read-only unless an expired mapping is observed.
        long effectiveTtlNanos = effectiveTtlNanos();
        for (Long lastUpdatedNanos : workers.values()) {
            if (isExpired(lastUpdatedNanos, queryTimeNanos, effectiveTtlNanos)) {
                return removeExpiredWorkerMappings(blockCacheKey, queryTimeNanos, effectiveTtlNanos);
            }
        }
        return workers;
    }

    void updateMaximumEntries(long newMaximumEntries) {
        maximumEntries = newMaximumEntries;
        requestHighWatermarkCleanupIfNeeded();
    }

    void updateExpirationSettings(long ttlMs, long minimumTtlMs, double ttlReductionStartRatio) {
        expirationSettings = expirationSettings(ttlMs, minimumTtlMs, ttlReductionStartRatio);
    }

    long maximumEntryCount() {
        return maximumEntries;
    }

    long mappingCount() {
        return mappingCount.get();
    }

    void shutdown() {
        cleanupExecutor.shutdown();
    }

    void runCleanupCheck() {
        try {
            if (capacityUsageRatio() >= FULL_SCAN_TRIGGER_RATIO) {
                checksSinceLastCleanup = 0;
                runHighWatermarkFullScan();
                return;
            }

            checksSinceLastCleanup++;
            if (checksSinceLastCleanup < checksBeforeCleanup()) {
                return;
            }
            checksSinceLastCleanup = 0;
            removeExpiredMappingsBatch();
        } catch (RuntimeException e) {
            log.warn("Failed to run Local Standby cache cleanup", e);
        }
    }

    void runHighWatermarkFullScan() {
        try {
            long mappingsBeforeCleanup = mappingCount.get();
            removeExpiredMappingsFullScan();
            cleanupIterator = null;
            long mappingsAfterCleanup = mappingCount.get();
            log.info("Completed high-watermark Local Standby cache full scan, "
                            + "before={}, after={}, expiredRemoved={}",
                    mappingsBeforeCleanup,
                    mappingsAfterCleanup,
                    Math.max(0, mappingsBeforeCleanup - mappingsAfterCleanup));
        } catch (RuntimeException e) {
            log.warn("Failed to run high-watermark Local Standby cache full scan", e);
        }
    }

    int checksBeforeCleanup() {
        if (capacityUsageRatio() >= expirationSettings.ttlReductionStartRatio()) {
            return PRESSURE_CHECKS_BEFORE_CLEANUP;
        }
        return NORMAL_CHECKS_BEFORE_CLEANUP;
    }

    long effectiveTtlNanos() {
        ExpirationSettings settings = expirationSettings;
        if (maximumEntries <= 0 || mappingCount.get() <= 0) {
            return settings.ttlNanos();
        }

        double usageRatio = capacityUsageRatio();
        if (usageRatio <= settings.ttlReductionStartRatio()) {
            return settings.ttlNanos();
        }
        if (usageRatio >= 1.0) {
            return settings.minimumTtlNanos();
        }

        double reductionProgress =
                (usageRatio - settings.ttlReductionStartRatio()) / (1.0 - settings.ttlReductionStartRatio());
        long ttlRange = settings.ttlNanos() - settings.minimumTtlNanos();
        return settings.ttlNanos() - (long) (ttlRange * reductionProgress);
    }

    void removeExpiredMappingsBatch() {
        try {
            /*
             * Normal cleanup scans about 10% of block hashes. After the configured pressure
             * threshold, each pass scans about 20%.
             */
            int batchDivisor = capacityUsageRatio() >= expirationSettings.ttlReductionStartRatio()
                    ? PRESSURE_CLEANUP_BATCH_DIVISOR
                    : NORMAL_CLEANUP_BATCH_DIVISOR;
            int blockBatchSize = Math.max(1, (blockToEnginesMap.size() + batchDivisor - 1)
                    / batchDivisor);
            if (cleanupIterator == null || !cleanupIterator.hasNext()) {
                cleanupIterator = blockToEnginesMap.keySet().iterator();
            }

            long cleanupTimeNanos = System.nanoTime();
            long effectiveTtlNanos = effectiveTtlNanos();
            int scannedBlocks = 0;
            while (cleanupIterator.hasNext() && scannedBlocks < blockBatchSize) {
                removeExpiredWorkerMappings(cleanupIterator.next(), cleanupTimeNanos, effectiveTtlNanos);
                scannedBlocks++;
            }
            if (!cleanupIterator.hasNext()) {
                cleanupIterator = null;
            }
        } catch (RuntimeException e) {
            log.warn("Failed to clean up expired Local Standby cache mappings", e);
        }
    }

    private Map<String, Long> removeExpiredWorkerMappings(Long blockCacheKey, long cleanupTimeNanos,
                                                          long effectiveTtlNanos) {
        return blockToEnginesMap.computeIfPresent(blockCacheKey, (blockHash, workers) -> {
            for (Map.Entry<String, Long> workerEntry : workers.entrySet()) {
                Long lastUpdatedNanos = workerEntry.getValue();
                if (isExpired(lastUpdatedNanos, cleanupTimeNanos, effectiveTtlNanos)
                        && workers.remove(workerEntry.getKey(), lastUpdatedNanos)) {
                    mappingCount.decrementAndGet();
                }
            }
            return workers.isEmpty() ? null : workers;
        });
    }

    private void removeExpiredMappingsFullScan() {
        long cleanupTimeNanos = System.nanoTime();
        long effectiveTtlNanos = effectiveTtlNanos();
        for (Long blockCacheKey : blockToEnginesMap.keySet()) {
            removeExpiredWorkerMappings(blockCacheKey, cleanupTimeNanos, effectiveTtlNanos);
        }
    }

    private boolean incrementMappingCountIfBelowLimit() {
        if (mappingCount.get() >= maximumEntries) {
            return false;
        }
        mappingCount.incrementAndGet();
        return true;
    }

    private void requestHighWatermarkCleanupIfNeeded() {
        if (!automaticCleanupEnabled || capacityUsageRatio() < FULL_SCAN_TRIGGER_RATIO) {
            return;
        }
        if (!highWatermarkCleanupTriggered.compareAndSet(false, true)) {
            return;
        }
        try {
            cleanupExecutor.execute(() -> {
                try {
                    runCleanupCheck();
                } finally {
                    highWatermarkCleanupTriggered.set(false);
                }
            });
        } catch (RejectedExecutionException e) {
            highWatermarkCleanupTriggered.set(false);
            if (!cleanupExecutor.isShutdown()) {
                log.warn("Failed to schedule immediate Local Standby cache cleanup", e);
            }
        }
    }

    private boolean isExpired(long lastUpdatedNanos, long currentTimeNanos, long effectiveTtlNanos) {
        return currentTimeNanos - lastUpdatedNanos >= effectiveTtlNanos;
    }

    private ExpirationSettings expirationSettings(long ttlMs, long minimumTtlMs, double ttlReductionStartRatio) {
        return new ExpirationSettings(
                TimeUnit.MILLISECONDS.toNanos(ttlMs),
                TimeUnit.MILLISECONDS.toNanos(minimumTtlMs),
                ttlReductionStartRatio);
    }

    private double capacityUsageRatio() {
        return maximumEntries <= 0 ? 0 : (double) mappingCount.get() / maximumEntries;
    }

    private record ExpirationSettings(long ttlNanos, long minimumTtlNanos, double ttlReductionStartRatio) {
    }
}
