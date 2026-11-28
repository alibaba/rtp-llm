package org.flexlb.cache.match.localsync;

import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSets;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.config.ConfigService;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RejectedExecutionException;

/**
 * Holds primitive cache keys per logical worker. The cache manager serializes each worker's
 * updates and retirement, so a diff can read both key sets concurrently without copying them.
 */
@Slf4j
@Component
public class EngineLocalView {
    private final ConcurrentHashMap<String, LongSet> engineViews = new ConcurrentHashMap<>();

    private final int diffParallelism;
    private final DynamicCacheIntervalService dynamicIntervalManager;

    private volatile ForkJoinPool customPool;

    private volatile boolean closed;

    public EngineLocalView(
            ConfigService configService, DynamicCacheIntervalService dynamicIntervalManager) {
        int configuredParallelism =
                configService
                        .loadBalanceConfig()
                        .getWorkerRegistry()
                        .getCacheStatus()
                        .getDiffParallelism();
        this.diffParallelism =
                Math.min(Runtime.getRuntime().availableProcessors(), configuredParallelism);
        this.dynamicIntervalManager = dynamicIntervalManager;
    }

    @PreDestroy
    public synchronized void shutdown() {
        closed = true;
        if (customPool != null) {
            customPool.shutdown();
        }
    }

    /**
     * Calculate diff result
     *
     * @param engineIPort    Engine IP
     * @param newCacheBlocks New cache block set
     * @return Diff calculation result
     */
    public DiffResult calculateDiff(String engineIPort, Set<Long> newCacheBlocks) {
        if (engineIPort == null || newCacheBlocks == null) {
            return DiffResult.empty(engineIPort);
        }

        LongSet oldKeys = engineViews.getOrDefault(engineIPort, LongSets.EMPTY_SET);
        LongSet newKeys =
                newCacheBlocks instanceof LongSet primitive
                        ? primitive
                        : new LongOpenHashSet(newCacheBlocks);
        LongSet addedBlocks;
        LongSet removedBlocks;
        ForkJoinPool pool = diffPool();
        ForkJoinTask<LongSet> addedTask = null;
        try {
            addedTask = pool.submit(() -> difference(newKeys, oldKeys));
            ForkJoinTask<LongSet> removedTask = pool.submit(() -> difference(oldKeys, newKeys));
            addedBlocks = addedTask.join();
            removedBlocks = removedTask.join();
        } catch (RejectedExecutionException error) {
            addedBlocks = addedTask == null ? difference(newKeys, oldKeys) : addedTask.join();
            removedBlocks = difference(oldKeys, newKeys);
        }

        // Update statistics in dynamic sync interval manager
        int diffSize = addedBlocks.size() + removedBlocks.size();
        dynamicIntervalManager.updateDiffStatistics(diffSize);

        return DiffResult.builder()
                .engineIp(engineIPort)
                .addedBlocks(addedBlocks)
                .removedBlocks(removedBlocks)
                .version("1.0.0")
                .build();
    }

    /**
     * Add or update engine cache block
     *
     * @param engineIPort     Engine IP
     * @param blockCacheKey   Cache block hash value
     */
    public void addOrUpdateCacheBlock(String engineIPort, long blockCacheKey) {
        if (engineIPort == null) {
            log.warn(
                    "Invalid parameters: engineIPort={}, blockCacheKey={}",
                    engineIPort,
                    blockCacheKey);
            return;
        }
        LongSet engineCache =
                engineViews.computeIfAbsent(engineIPort, key -> new LongOpenHashSet());

        engineCache.add(blockCacheKey);
    }

    /**
     * Remove engine cache block
     *
     * @param engineIPort   Engine IP
     * @param blockCacheKey Cache block hash value
     */
    public void removeCacheBlock(String engineIPort, long blockCacheKey) {
        if (engineIPort == null) {
            return;
        }

        LongSet engineCache = engineViews.get(engineIPort);
        if (engineCache == null) {
            return;
        }

        boolean isRemoveSuccess = engineCache.remove(blockCacheKey);

        if (isRemoveSuccess) {

            // If engine cache is empty, optionally remove the entire engine entry
            if (engineCache.isEmpty()) {
                engineViews.remove(engineIPort);
            }
        }
    }

    /**
     * Detach an engine's cached keys for removal from the global index.
     *
     * @param engineIPort Engine IP
     */
    public Set<Long> removeAllCacheBlockOfEngine(String engineIPort) {
        if (engineIPort == null) {
            return Set.of();
        }

        LongSet removed = engineViews.remove(engineIPort);
        return removed == null ? Set.of() : removed;
    }

    /**
     * Clear all data
     */
    public void clear() {

        engineViews.clear();
        log.info("Cleared engine local view");
    }

    public int size(String engineIpPort) {
        LongSet engineCache = engineViews.get(engineIpPort);
        return engineCache == null ? 0 : engineCache.size();
    }

    /**
     * Get engine view map size (number of current engines)
     *
     * @return engineViews map size
     */
    public int getEngineViewsMapSize() {
        return engineViews.size();
    }

    /**
     * Get all engine IP:Port set.
     *
     * @return All engine IP:Port set
     */
    public Set<String> getAllEngineIpPorts() {
        return engineViews.keySet();
    }

    private static LongSet difference(LongSet keys, LongSet other) {
        LongSet result = new LongOpenHashSet();
        for (var iterator = keys.iterator(); iterator.hasNext(); ) {
            long key = iterator.nextLong();
            if (!other.contains(key)) {
                result.add(key);
            }
        }
        return result;
    }

    private ForkJoinPool diffPool() {
        if (closed) {
            throw new IllegalStateException("Engine local view is closed");
        }
        ForkJoinPool pool = customPool;
        if (pool != null) {
            return pool;
        }
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException("Engine local view is closed");
            }
            if (customPool == null) {
                customPool = new ForkJoinPool(diffParallelism);
            }
            return customPool;
        }
    }
}
