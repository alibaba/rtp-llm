package org.flexlb.cache.core;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;

/**
 * KV cache manager
 * Core functions:
 * 1. Unified management of two-level hash table
 * 2. Provide advanced cache query and matching services
 *
 * @author FlexLB
 */
@Slf4j
@Getter
@Component
public class KvCacheManager {
    
    @Autowired
    private GlobalCacheIndex globalCacheIndex;
    
    @Autowired
    private EngineLocalView engineLocalView;
    
    @Autowired
    private WorkerStatusProvider workerStatusProvider;
    
    /**
     * Cache metrics reporter
     */
    @Autowired
    private CacheMetricsReporter cacheMetricsReporter;

    /**
     * Performance statistics
     */
    private final LongAdder totalUpdates = new LongAdder();

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
     * @param roleType       Engine role to query
     * @param group          Engine group to query
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength
     */
    public Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(List<Long> blockCacheKeys,
        RoleType roleType, String group) {

        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }

        // Use candidate engine list
        List<String> enginesIpPorts = workerStatusProvider.getWorkerIpPorts(roleType, group);

        // Batch calculate prefix match length
        return globalCacheIndex.batchCalculatePrefixMatchLength(enginesIpPorts, blockCacheKeys);
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
            DiffResult.empty(engineIPort);
            return;
        }

        // Calculate diff
        DiffResult diffResult = engineLocalView.calculateDiff(engineIPort, newCacheBlocks, role);
        if (!diffResult.hasChanges()) {
            return;
        }

        // Apply added cache blocks
        for (Long addedBlock : diffResult.getAddedBlocks()) {
            boolean contains = newCacheBlocks.contains(addedBlock);
            if (contains) {
                // Update local view
                engineLocalView.addOrUpdateCacheBlock(engineIPort, addedBlock);
                // Update global index
                globalCacheIndex.addCacheBlock(addedBlock, engineIPort);
            }
        }

        // Apply removed cache blocks
        for (Long removedBlock : diffResult.getRemovedBlocks()) {
            // Remove from local view
            engineLocalView.removeCacheBlock(engineIPort, removedBlock);
            // Remove from global index
            globalCacheIndex.removeCacheBlock(engineIPort, removedBlock);
        }

        totalUpdates.increment();
        // Report metrics
        cacheMetricsReporter.reportEngineLocalMetrics(engineIPort, role, engineLocalView.size(engineIPort));
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
        cacheMetricsReporter.reportEngineViewsMapSize(engineLocalView.getEngineViewsMapSize());
    }
    
    /**
     * Clear all data
     */
    public void clear() {

        globalCacheIndex.clear();
        engineLocalView.clear();

        totalUpdates.reset();
        // Report
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());

        log.info("Cleared all cache data");
    }
}