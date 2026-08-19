package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

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
     * Cache metrics reporter
     */
    @Autowired
    private CacheMetricsReporter cacheMetricsReporter;

    /**
     * Dynamic sync interval manager
     */
    @Autowired
    private DynamicCacheIntervalService dynamicIntervalManager;

    /** Commit a snapshot after the global index update succeeds. */
    void commitSnapshot(String engineIpPort, Set<Long> newCacheBlocks) {
        Set<Long> local = engineViews.computeIfAbsent(
                engineIpPort, ignored -> ConcurrentHashMap.newKeySet());
        local.removeIf(block -> !newCacheBlocks.contains(block));
        // ConcurrentHashMap key-set add is already idempotent; avoid a second
        // hash lookup for every stable key in a large snapshot.
        local.addAll(newCacheBlocks);
        if (local.isEmpty()) {
            engineViews.remove(engineIpPort, local);
        }
    }

    void reportDiff(String role, int added, int removed) {
        try {
            cacheMetricsReporter.reportCacheDiffMetrics(role, added, removed);
            dynamicIntervalManager.updateDiffStatistics(added + removed);
        } catch (RuntimeException metricFailure) {
            log.warn("Failed to report cache diff metrics for role={}", role, metricFailure);
        }
    }

    /**
     * Remove all cache blocks of an engine
     *
     * @param engineIPort Engine IP
     */
    void removeAllCacheBlockOfEngine(String engineIPort) {
        if (engineIPort == null) {
            return;
        }

        engineViews.remove(engineIPort);
    }

    /**
     * Get all cache block IDs of an engine
     *
     * @param engineIPort Engine IP
     * @return Cache block ID set
     */
    Set<Long> getEngineCacheBlocks(String engineIPort) {
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

    int size(String engineIpPort) {
        Set<Long> engineCache = engineViews.get(engineIpPort);
        return engineCache == null ? 0 : engineCache.size();
    }

    /**
     * Get engine view map size (number of current engines)
     *
     * @return engineViews map size
     */
    int getEngineViewsMapSize() {
        return engineViews.size();
    }
}
