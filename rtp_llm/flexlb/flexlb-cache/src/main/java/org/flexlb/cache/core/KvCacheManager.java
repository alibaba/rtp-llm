package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
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
import java.util.concurrent.ConcurrentHashMap;

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

    /**
     * Addresses whose last authoritative clear did not finish. The next
     * publication retries that clear before applying a new snapshot, so a
     * partial global removal cannot be mistaken for committed local state.
     */
    private final Set<String> pendingClear = ConcurrentHashMap.newKeySet();

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
            return;
        }

        if (pendingClear.contains(engineIPort)) {
            clearEngineCache(engineIPort);
        }

        Set<Long> oldCacheBlocks = engineLocalView.getEngineCacheBlocks(engineIPort);
        GlobalCacheIndex.CacheDiffStats diff = globalCacheIndex.applyEngineCacheSnapshot(
                engineIPort, oldCacheBlocks, newCacheBlocks);
        // The local view is the commit marker. Keep it unchanged until the
        // global index update succeeds so the same version can be retried.
        engineLocalView.commitSnapshot(engineIPort, newCacheBlocks);
        engineLocalView.reportDiff(role, diff.added(), diff.removed());

        // Report metrics
        reportIndexMetrics(engineIPort, role);
    }

    public void clearEngineCache(String engineIpPort) {
        if (engineIpPort == null) {
            return;
        }
        try {
            Set<Long> oldCacheBlocks = engineLocalView.getEngineCacheBlocks(engineIpPort);
            globalCacheIndex.removeEngineCacheBlocks(engineIpPort, oldCacheBlocks);
            engineLocalView.removeAllCacheBlockOfEngine(engineIpPort);
            pendingClear.remove(engineIpPort);
            reportIndexMetrics(engineIpPort, "unknown");
        } catch (RuntimeException cleanupFailure) {
            pendingClear.add(engineIpPort);
            throw cleanupFailure;
        }
    }

    private void reportIndexMetrics(String engineIpPort, String role) {
        try {
            cacheMetricsReporter.reportEngineLocalMetrics(
                    engineIpPort.split(":")[0], role, engineLocalView.size(engineIpPort));
            cacheMetricsReporter.reportGlobalCacheMetrics(
                    globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
            cacheMetricsReporter.reportEngineViewsMapSize(engineLocalView.getEngineViewsMapSize());
        } catch (RuntimeException metricFailure) {
            log.warn("Failed to report cache index metrics for engine={}",
                    engineIpPort, metricFailure);
        }
    }

    /**
     * Clear all data
     */
    public void clear() {

        globalCacheIndex.clear();
        engineLocalView.clear();
        pendingClear.clear();

        // Report
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());

        log.info("Cleared all cache data");
    }
}
