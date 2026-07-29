package org.flexlb.cache.match.localstandby;

import lombok.extern.slf4j.Slf4j;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
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
    private static final int CLEANUP_BATCH_DIVISOR = 10;
    private static final int NORMAL_CHECKS_BEFORE_CLEANUP = 3;
    private static final int PRESSURE_CHECKS_BEFORE_CLEANUP = 2;
    private static final int CRITICAL_CHECKS_BEFORE_CLEANUP = 1;

    private final long entryTtlNanos;
    private final long minimumEntryTtlNanos;
    private final double ttlReductionStartRatio;
    private final ConcurrentHashMap<Long, ConcurrentHashMap<String, Long>> blockToEnginesMap = new ConcurrentHashMap<>();
    private final AtomicLong mappingCount = new AtomicLong();
    private final ScheduledExecutorService cleanupExecutor;
    private volatile long maximumEntries;
    private Iterator<Long> cleanupIterator;
    private int checksSinceLastCleanup;

    LocalStandbyCacheIndex(long entryTtlMs,
                           long minimumEntryTtlMs,
                           double ttlReductionStartRatio,
                           long maximumEntries,
                           boolean enabled) {
        this.entryTtlNanos = TimeUnit.MILLISECONDS.toNanos(entryTtlMs);
        this.minimumEntryTtlNanos = TimeUnit.MILLISECONDS.toNanos(minimumEntryTtlMs);
        this.ttlReductionStartRatio = ttlReductionStartRatio;
        this.maximumEntries = maximumEntries;
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

    void addWorkerBlockMappings(String workerIpPort, List<Long> blockCacheKeys) {
        if (workerIpPort == null || workerIpPort.isEmpty() || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return;
        }

        long lastUpdatedNanos = System.nanoTime();
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

                Long previousUpdatedAt = currentWorkers.put(workerIpPort, lastUpdatedNanos);
                if (previousUpdatedAt == null) {
                    mappingCount.incrementAndGet();
                }
                return currentWorkers;
            });
        }
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
        long effectiveEntryTtlNanos = effectiveEntryTtlNanos();
        for (Long lastUpdatedNanos : workers.values()) {
            if (isExpired(lastUpdatedNanos, queryTimeNanos, effectiveEntryTtlNanos)) {
                return removeExpiredWorkerMappings(blockCacheKey, queryTimeNanos, effectiveEntryTtlNanos);
            }
        }
        return workers;
    }

    void updateMaximumEntries(long newMaximumEntries) {
        maximumEntries = newMaximumEntries;
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

    int checksBeforeCleanup() {
        double usageRatio = capacityUsageRatio();
        if (usageRatio >= 1.0) {
            return CRITICAL_CHECKS_BEFORE_CLEANUP;
        }
        if (usageRatio >= ttlReductionStartRatio) {
            return PRESSURE_CHECKS_BEFORE_CLEANUP;
        }
        return NORMAL_CHECKS_BEFORE_CLEANUP;
    }

    long effectiveEntryTtlNanos() {
        if (maximumEntries <= 0 || mappingCount.get() <= 0) {
            return entryTtlNanos;
        }

        double usageRatio = capacityUsageRatio();
        if (usageRatio <= ttlReductionStartRatio) {
            return entryTtlNanos;
        }
        if (usageRatio >= 1.0) {
            return minimumEntryTtlNanos;
        }

        double reductionProgress =
                (usageRatio - ttlReductionStartRatio) / (1.0 - ttlReductionStartRatio);
        long ttlRange = entryTtlNanos - minimumEntryTtlNanos;
        return entryTtlNanos - (long) (ttlRange * reductionProgress);
    }

    void removeExpiredMappingsBatch() {
        try {
            /*
             * Each pass scans roughly 10% of block hashes. Together with the 30/20/10-second
             * cleanup cadence, a complete scan takes about 300/200/100 seconds.
             */
            int blockBatchSize = Math.max(1, (blockToEnginesMap.size() + CLEANUP_BATCH_DIVISOR - 1)
                    / CLEANUP_BATCH_DIVISOR);
            if (cleanupIterator == null || !cleanupIterator.hasNext()) {
                cleanupIterator = blockToEnginesMap.keySet().iterator();
            }

            long cleanupTimeNanos = System.nanoTime();
            long effectiveEntryTtlNanos = effectiveEntryTtlNanos();
            int scannedBlocks = 0;
            while (cleanupIterator.hasNext() && scannedBlocks < blockBatchSize) {
                removeExpiredWorkerMappings(cleanupIterator.next(), cleanupTimeNanos, effectiveEntryTtlNanos);
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
                                                          long effectiveEntryTtlNanos) {
        return blockToEnginesMap.computeIfPresent(blockCacheKey, (blockHash, workers) -> {
            for (Map.Entry<String, Long> workerEntry : workers.entrySet()) {
                Long lastUpdatedNanos = workerEntry.getValue();
                if (isExpired(lastUpdatedNanos, cleanupTimeNanos, effectiveEntryTtlNanos)
                        && workers.remove(workerEntry.getKey(), lastUpdatedNanos)) {
                    mappingCount.decrementAndGet();
                }
            }
            return workers.isEmpty() ? null : workers;
        });
    }

    private boolean isExpired(long lastUpdatedNanos, long currentTimeNanos, long effectiveEntryTtlNanos) {
        return currentTimeNanos - lastUpdatedNanos >= effectiveEntryTtlNanos;
    }

    private double capacityUsageRatio() {
        return maximumEntries <= 0 ? 0 : (double) mappingCount.get() / maximumEntries;
    }
}
