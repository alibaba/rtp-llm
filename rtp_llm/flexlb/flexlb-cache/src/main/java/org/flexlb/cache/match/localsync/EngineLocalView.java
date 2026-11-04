package org.flexlb.cache.match.localsync;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.primitive.LongKeySet;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WorkerRegistryConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RejectedExecutionException;

/**
 * Engine local view (small hash table)
 * Manages local cache state and metadata for each engine
 * Storage structure: EngineIpPort -> HashMap<Long>
 *
 * @author FlexLB
 */
@Slf4j
@Component
public class EngineLocalView {

    /**
     * Core storage structure: EngineIpPort -> Set<Long>
     */
    private final ConcurrentHashMap<String, Set<Long>> engineViews = new ConcurrentHashMap<>();

    /**
     * Custom ForkJoin thread pool for parallel computation
     */
    private volatile ForkJoinPool customPool;

    private final int diffParallelism;

    public EngineLocalView() {
        this(new WorkerRegistryConfig.ExecutorsConfig());
    }

    @Autowired
    public EngineLocalView(ConfigService configService) {
        this(configService.loadBalanceConfig().getWorkerRegistry().getExecutors());
    }

    private EngineLocalView(WorkerRegistryConfig.ExecutorsConfig executors) {
        diffParallelism =
                Math.min(
                        Runtime.getRuntime().availableProcessors(),
                        executors.getCacheDiffParallelism());
    }

    private volatile boolean closed;

    /**
     * Dynamic sync interval manager
     */
    @Autowired private DynamicCacheIntervalService dynamicIntervalManager;

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

        Set<Long> oldCacheBlocks = getEngineCacheBlocks(engineIPort);

        ForkJoinPool pool = diffPool();
        ForkJoinTask<LongKeySet> addedTask = null;
        ForkJoinTask<LongKeySet> removedTask = null;
        Set<Long> addedBlocks;
        Set<Long> removedBlocks;
        try {
            addedTask = pool.submit(() -> LongKeySet.difference(newCacheBlocks, oldCacheBlocks));
            removedTask = pool.submit(() -> LongKeySet.difference(oldCacheBlocks, newCacheBlocks));
            addedBlocks = addedTask.join();
            removedBlocks = removedTask.join();
        } catch (RejectedExecutionException error) {
            addedBlocks =
                    addedTask == null
                            ? LongKeySet.difference(newCacheBlocks, oldCacheBlocks)
                            : addedTask.join();
            removedBlocks = LongKeySet.difference(oldCacheBlocks, newCacheBlocks);
        } finally {
            // Both tasks borrow the mutable local view under the caller's cache lock.
            // Even if one task fails, the other must finish before that lock is released.
            if (addedTask != null) {
                addedTask.quietlyJoin();
            }
            if (removedTask != null) {
                removedTask.quietlyJoin();
            }
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
    public void addOrUpdateCacheBlock(String engineIPort, Long blockCacheKey) {
        if (engineIPort == null || blockCacheKey == null) {
            log.warn(
                    "Invalid parameters: engineIPort={}, blockCacheKey={}",
                    engineIPort,
                    blockCacheKey);
            return;
        }
        Set<Long> engineCache = engineViews.computeIfAbsent(engineIPort, k -> new LongKeySet());

        engineCache.add(blockCacheKey);
    }

    /**
     * Remove engine cache block
     *
     * @param engineIPort   Engine IP
     * @param blockCacheKey Cache block hash value
     */
    public void removeCacheBlock(String engineIPort, Long blockCacheKey) {
        if (engineIPort == null || blockCacheKey == null) {
            return;
        }

        Set<Long> engineCache = engineViews.get(engineIPort);
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
     * Detach all cache blocks of an engine
     *
     * @param engineIPort Engine IP
     */
    public void removeAllCacheBlockOfEngine(String engineIPort) {
        if (engineIPort == null) {
            return;
        }

        // Production retirement removes this view only after global cleanup
        // succeeds, while holding this generation's cacheMutationLock.
        Set<Long> removed = engineViews.remove(engineIPort);
        // Warn if removal fails
        if (removed == null) {
            log.warn("Remove failed, the engine: {} not exist.", engineIPort);
        }
    }

    /**
     * Get all cache block IDs of an engine
     *
     * @param engineIPort Engine IP
     * <p>Returns a borrowed mutable set, not a snapshot. Callers iterating it
     * must hold the owning generation's cacheMutationLock for the entire use.
     * The global index lock does not protect writes to this local view.
     *
     * @return Cache block ID set
     */
    public Set<Long> getEngineCacheBlocks(String engineIPort) {
        if (engineIPort == null) {
            return Collections.emptySet();
        }
        Set<Long> engineCache = engineViews.get(engineIPort);
        return engineCache == null ? Collections.emptySet() : engineCache;
    }

    /**
     * Clear all data
     */
    public void clear() {

        engineViews.clear();
        log.info("Cleared engine local view");
    }

    public int size(String engineIpPort) {
        Set<Long> engineCache = engineViews.get(engineIpPort);
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
}
