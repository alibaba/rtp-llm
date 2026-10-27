package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.metric.SafeFlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * KV cache manager
 * Core functions:
 * 1. Unified management of two-level hash table
 * 2. Provide advanced cache query and matching services
 *
 * @author FlexLB
 */
@Slf4j
@Component
public class KvCacheManager {

    @Autowired private GlobalCacheIndex globalCacheIndex;

    @Autowired private EngineLocalView engineLocalView;

    /**
     * Cache metrics reporter
     */
    @Autowired private CacheMetricsReporter cacheMetricsReporter;

    /** The owning generation's cache lock serializes each address and its retry. */
    private final Map<String, Long> pendingAdditions = new ConcurrentHashMap<>();

    private final AtomicInteger indexReports = new AtomicInteger();

    @PostConstruct
    public void init() {
        log.info("KvCacheManager initialized successfully");
    }

    @PreDestroy
    public void destroy() {
        log.info("KvCacheManager shutting down...");
        clear();
    }

    /**
     * Query engine cache matching status
     *
     * @param blockCacheKeys List of cache block hash values to query
     * @param candidateEngineIpPorts exact live candidate addresses
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength
     */
    public Map<String /*engineIpPort*/, Integer /*prefixMatchLength*/> findMatchingEngines(
            List<Long> blockCacheKeys, List<String> candidateEngineIpPorts) {

        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }

        // Batch calculate prefix match length
        return globalCacheIndex.batchCalculatePrefixMatchLength(
                candidateEngineIpPorts, blockCacheKeys);
    }

    /**
     * Update engine cache status
     *
     * @param engineIPort    Engine IP:Port
     * @param role           Engine role
     * @param newCacheBlocks New cache block set (blockCacheKeys)
     */
    public void updateEngineCache(String engineIPort, String role, Set<Long> newCacheBlocks) {
        if (engineIPort == null || newCacheBlocks == null) {
            return;
        }

        Long pending = pendingAdditions.get(engineIPort);
        if (pending != null) {
            addCacheBlock(engineIPort, pending);
            pendingAdditions.remove(engineIPort);
        }

        // Calculate diff
        DiffResult diffResult = engineLocalView.calculateDiff(engineIPort, newCacheBlocks, role);
        if (!diffResult.hasChanges()) {
            return;
        }

        for (Long key : diffResult.getAddedBlocks()) {
            addCacheBlock(engineIPort, key);
        }
        for (Long key : diffResult.getRemovedBlocks()) {
            globalCacheIndex.removeCacheBlock(engineIPort, key);
            engineLocalView.removeCacheBlock(engineIPort, key);
        }

        // Report metrics
        SafeFlexMonitor.run(
                "cache.local",
                () ->
                        cacheMetricsReporter.reportEngineLocalMetrics(
                                engineIPort, role, engineLocalView.size(engineIPort)));
        reportIndexMetrics();
    }

    private void reportIndexMetrics() {
        // Coalesce concurrent changes; an older size report cannot overwrite retirement forever.
        if (indexReports.getAndIncrement() != 0) {
            return;
        }
        int pending = 1;
        do {
            SafeFlexMonitor.run(
                    "cache.index",
                    () ->
                            cacheMetricsReporter.reportIndexMetrics(
                                    globalCacheIndex.totalBlocks(),
                                    globalCacheIndex.totalMappings(),
                                    engineLocalView.getEngineViewsMapSize()));
            pending = indexReports.addAndGet(-pending);
        } while (pending != 0);
    }

    private void addCacheBlock(String address, Long key) {
        try {
            engineLocalView.addOrUpdateCacheBlock(address, key);
            globalCacheIndex.addCacheBlock(key, address);
        } catch (RuntimeException | Error failure) {
            pendingAdditions.put(address, key);
            throw failure;
        }
    }

    /** Remove both local and global cache-index state for one engine. */
    public void removeEngineCache(String engineIpPort) {
        if (engineIpPort == null) {
            return;
        }
        Set<Long> ownedKeys = engineLocalView.getEngineCacheBlocks(engineIpPort);
        globalCacheIndex.removeAllCacheBlockOfEngine(engineIpPort, ownedKeys);
        engineLocalView.removeAllCacheBlockOfEngine(engineIpPort);
        pendingAdditions.remove(engineIpPort);
        SafeFlexMonitor.run(
                "cache.local.remove",
                () -> cacheMetricsReporter.removeEngineLocalMetrics(engineIpPort));
        reportIndexMetrics();
    }

    /**
     * Clear all data
     */
    public void clear() {

        globalCacheIndex.clear();
        engineLocalView.clear();
        pendingAdditions.clear();

        // Report
        SafeFlexMonitor.run("cache.local.clear", cacheMetricsReporter::clearEngineLocalMetrics);
        reportIndexMetrics();

        log.info("Cleared all cache data");
    }
}
