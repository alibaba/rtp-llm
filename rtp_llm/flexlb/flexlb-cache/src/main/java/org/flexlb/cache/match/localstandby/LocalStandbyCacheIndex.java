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
 * <p>{@code blockToEnginesMap} is the only metadata store. Expired mappings are removed lazily
 * during queries and incrementally by the background cleaner.
 */
@Slf4j
class LocalStandbyCacheIndex {

    private static final long MAX_CLEANUP_INTERVAL_MS = 30_000;
    private static final int CLEANUP_BATCH_DIVISOR = 10;

    private final long entryTtlNanos;
    private final ConcurrentHashMap<Long, ConcurrentHashMap<String, Long>> blockToEnginesMap = new ConcurrentHashMap<>();
    private final AtomicLong mappingCount = new AtomicLong();
    private final AtomicLong maximumEntries;
    private final ScheduledExecutorService cleanupExecutor;
    private Iterator<Long> cleanupIterator;

    LocalStandbyCacheIndex(long entryTtlMs, long maximumEntries, boolean enabled) {
        this.entryTtlNanos = TimeUnit.MILLISECONDS.toNanos(entryTtlMs);
        this.maximumEntries = new AtomicLong(maximumEntries);
        this.cleanupExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "local-standby-cache-cleaner");
            thread.setDaemon(true);
            return thread;
        });
        if (enabled) {
            long cleanupIntervalMs = Math.max(1_000, Math.min(MAX_CLEANUP_INTERVAL_MS, entryTtlMs / 10));
            cleanupExecutor.scheduleWithFixedDelay(
                    this::removeExpiredMappingsBatch, cleanupIntervalMs, cleanupIntervalMs, TimeUnit.MILLISECONDS);
        }
    }

    int addWorkerBlockMappings(String workerIpPort, List<Long> blockCacheKeys) {
        long expiresAtNanos = System.nanoTime() + entryTtlNanos;
        int[] rejectedMappings = new int[1];
        for (Long blockCacheKey : blockCacheKeys) {
            if (blockCacheKey == null) {
                continue;
            }

            // Most updates only extend the TTL of an existing mapping. Keep this path outside
            // outer-map compute(), which serializes all updates for the same popular block.
            ConcurrentHashMap<String, Long> existingWorkers = blockToEnginesMap.get(blockCacheKey);
            if (existingWorkers != null && existingWorkers.replace(workerIpPort, expiresAtNanos) != null) {
                continue;
            }

            blockToEnginesMap.compute(blockCacheKey, (blockHash, mappedWorkers) -> {
                ConcurrentHashMap<String, Long> currentWorkers = mappedWorkers;
                if (currentWorkers == null) {
                    currentWorkers = new ConcurrentHashMap<>();
                }
                // The mapping may have changed after the fast-path lookup. Recheck it while
                // performing the structural update so capacity accounting remains correct.
                if (currentWorkers.replace(workerIpPort, expiresAtNanos) != null) {
                    return currentWorkers;
                }
                // The capacity limit is approximate; avoid synchronization on the write path.
                if (mappingCount.get() >= maximumEntries.get()) {
                    rejectedMappings[0]++;
                    return currentWorkers.isEmpty() ? null : currentWorkers;
                }
                mappingCount.incrementAndGet();
                currentWorkers.put(workerIpPort, expiresAtNanos);
                return currentWorkers;
            });
        }
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
        // Keep the common query path read-only. Acquire the block-level compute lock only when
        // an expired mapping is actually observed.
        for (Long expiresAtNanos : workers.values()) {
            if (expiresAtNanos <= queryTimeNanos) {
                return removeExpiredWorkerMappings(blockCacheKey, queryTimeNanos);
            }
        }
        return workers;
    }

    void updateMaximumEntries(long maximumEntries) {
        this.maximumEntries.set(maximumEntries);
    }

    long maximumEntryCount() {
        return maximumEntries.get();
    }

    long mappingCount() {
        return mappingCount.get();
    }

    void shutdown() {
        cleanupExecutor.shutdown();
    }

    void removeExpiredMappingsBatch() {
        try {
            /*
             * CLEANUP_BATCH_DIVISOR=10 scans roughly 10% of the block index per run.
             * Adding divisor - 1 rounds the division up, while Math.max guarantees at least one:
             * 1000 blocks -> 100, 103 blocks -> 11, 3 blocks -> 1.
             */
            int blockBatchSize = Math.max(1, (blockToEnginesMap.size() + CLEANUP_BATCH_DIVISOR - 1)
                    / CLEANUP_BATCH_DIVISOR);
            if (cleanupIterator == null || !cleanupIterator.hasNext()) {
                cleanupIterator = blockToEnginesMap.keySet().iterator();
            }

            long cleanupTimeNanos = System.nanoTime();
            int scannedBlocks = 0;
            while (cleanupIterator.hasNext() && scannedBlocks < blockBatchSize) {
                removeExpiredWorkerMappings(cleanupIterator.next(), cleanupTimeNanos);
                scannedBlocks++;
            }
            if (!cleanupIterator.hasNext()) {
                cleanupIterator = null;
            }
        } catch (RuntimeException e) {
            log.warn("Failed to clean up expired Local Standby cache mappings", e);
        }
    }

    private Map<String, Long> removeExpiredWorkerMappings(Long blockCacheKey, long cleanupTimeNanos) {
        return blockToEnginesMap.computeIfPresent(blockCacheKey, (blockHash, workers) -> {
            for (Map.Entry<String, Long> workerEntry : workers.entrySet()) {
                Long expiresAtNanos = workerEntry.getValue();
                if (expiresAtNanos <= cleanupTimeNanos
                        && workers.remove(workerEntry.getKey(), expiresAtNanos)) {
                    mappingCount.decrementAndGet();
                }
            }
            return workers.isEmpty() ? null : workers;
        });
    }
}
