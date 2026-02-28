package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

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
    private final ForkJoinPool customPool = new ForkJoinPool(Math.min(Runtime.getRuntime().availableProcessors(), 8));

    /**
     * Cache metrics reporter
     */
    @Autowired
    private CacheMetricsReporter cacheMetricsReporter;
    
    /**
     * Dynamic sync interval manager
     */
    @Autowired
    private DynamicCacheIntervalService dynamicIntervalManager;

    /**
     * Calculate diff result
     *
     * @param engineIPort       Engine IP
     * @param newCacheBlocks New cache block set
     * @param role           Engine role
     * @return Diff calculation result
     */
    public DiffResult calculateDiff(String engineIPort, Set<Long> newCacheBlocks, String role) {
        if (engineIPort == null || newCacheBlocks == null) {
            return DiffResult.empty(engineIPort);
        }

        Set<Long> oldCacheBlocks = getEngineCacheBlocks(engineIPort);

        // Efficiently calculate differences
        Set<Long> addedBlocks = ConcurrentHashMap.newKeySet(128);
        Set<Long> removedBlocks = ConcurrentHashMap.newKeySet(128);

        // Use custom ForkJoin thread pool to parallel compute added and removed cache blocks
        ForkJoinTask<?> addedTask = customPool.submit(() ->
            newCacheBlocks.parallelStream()
                 .filter(blockCacheKey -> !oldCacheBlocks.contains(blockCacheKey))
                 .forEach(addedBlocks::add)
        );

        ForkJoinTask<?> removedTask = customPool.submit(() ->
            oldCacheBlocks.parallelStream()
                .filter(blockCacheKey -> !newCacheBlocks.contains(blockCacheKey))
                .forEach(removedBlocks::add)
        );

        addedTask.join();
        removedTask.join();

        cacheMetricsReporter.reportCacheDiffMetrics(engineIPort, role, addedBlocks.size(), removedBlocks.size());

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
            log.warn("Invalid parameters: engineIPort={}, blockCacheKey={}", engineIPort, blockCacheKey);
            return;
        }
        Set<Long> engineCache = engineViews.computeIfAbsent(engineIPort, k -> ConcurrentHashMap.newKeySet());

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
     * Remove all cache blocks of an engine
     *
     * @param engineIPort Engine IP
     */
    public void removeAllCacheBlockOfEngine(String engineIPort) {
        if (engineIPort == null) {
            return;
        }

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
     * Get all engine IP:Port set
     *
     * @return All engine IP:Port set
     */
    public Set<String> getAllEngineIpPorts() {
        return engineViews.keySet();
    }
}