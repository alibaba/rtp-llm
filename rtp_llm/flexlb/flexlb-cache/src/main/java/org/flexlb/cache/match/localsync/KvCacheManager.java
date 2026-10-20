package org.flexlb.cache.match.localsync;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.WorkerIdentity;
import org.flexlb.dao.master.WorkerStatus;
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

    // An interrupted add may have reached only the local view. Retain just that
    // key for replay; the next diff recovers the remaining changes from the views.
    // Callers serialize each address through its generation's cache lock.
    private final Map<String, Long> pendingAdditions = new ConcurrentHashMap<>();

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
    public Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(
            List<Long> blockCacheKeys,
            List<String> candidateEngineIpPorts) {

        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }

        // Batch calculate prefix match length
        return globalCacheIndex.batchCalculatePrefixMatchLength(
                candidateEngineIpPorts, blockCacheKeys);
    }

    /**
     * Query engine cache matching status for live workers in a role/group.
     *
     * @param blockCacheKeys List of cache block hash values to query
     * @param roleType       Engine role to query
     * @param group          Engine group to query
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength
     */
    public Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(
            List<Long> blockCacheKeys,
            RoleType roleType,
            String group) {

        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }

        List<String> engineIpPorts = workerStatusProvider.getWorkerStatuses(roleType, group)
                .stream()
                .map(WorkerStatus::getLogicalIpPort)
                .toList();

        return globalCacheIndex.batchCalculatePrefixMatchLength(
                engineIpPorts, blockCacheKeys);
    }

    public void updateEngineCache(WorkerStatus workerStatus, String role, Set<Long> newCacheBlocks) {
        updateEngineCache(workerStatus, role, newCacheBlocks, null);
    }

    public void updateEngineCache(WorkerStatus workerStatus, String role, Set<Long> newCacheBlocks,
                                  EndpointTrace.Poll pollTrace) {
        if (workerStatus == null) {
            return;
        }
        updateEngineCache(
                workerStatus.getWorkerIdentity(),
                workerStatus.getMetricIpPort(),
                role,
                newCacheBlocks, pollTrace);
    }

    private void updateEngineCache(
            WorkerIdentity identity,
            String metricIpPort,
            String role,
            Set<Long> newCacheBlocks, EndpointTrace.Poll pollTrace) {
        String engineIPort = identity == null ? null : identity.getLogicalIpPort();
        if (engineIPort == null || newCacheBlocks == null) {
            return;
        }

        Long pending = pendingAdditions.get(engineIPort);
        if (pending != null) {
            addCacheBlock(engineIPort, pending, pollTrace);
            pendingAdditions.remove(engineIPort);
        }

        // Calculate diff
        DiffResult diffResult = engineLocalView.calculateDiff(engineIPort, newCacheBlocks);
        cacheMetricsReporter.reportCacheDiffMetrics(
                metricIpPort, role,
                diffResult.getAddedBlocks().size(), diffResult.getRemovedBlocks().size());
        if (!diffResult.hasChanges()) {
            return;
        }

        for (Long key : diffResult.getAddedBlocks()) {
            addCacheBlock(engineIPort, key, pollTrace);
        }
        for (Long key : diffResult.getRemovedBlocks()) {
            // Retain ownership until global removal succeeds for cleanup/replay.
            globalCacheIndex.removeCacheBlock(engineIPort, key, pollTrace);
            engineLocalView.removeCacheBlock(engineIPort, key);
        }

        // Report metrics
        cacheMetricsReporter.reportEngineLocalMetrics(
                metricIpPort, role, engineLocalView.size(engineIPort));
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
        cacheMetricsReporter.reportEngineViewsMapSize(engineLocalView.getEngineViewsMapSize());
    }

    private void addCacheBlock(String address, Long key, EndpointTrace.Poll pollTrace) {
        try {
            // Track ownership first, including a global write that fails midway.
            engineLocalView.addOrUpdateCacheBlock(address, key);
            globalCacheIndex.addCacheBlock(key, address, pollTrace);
        } catch (RuntimeException | Error failure) {
            pendingAdditions.put(address, key);
            throw failure;
        }
    }

    /** Remove both local and global cache-index state for one engine. */
    public void removeEngineCache(String engineIpPort, EndpointTrace trace) {
        if (engineIpPort == null) {
            return;
        }
        // The cache lock excludes writers. Retain ownership on failure so a
        // retry can safely repeat removal of any already-removed mappings.
        Set<Long> removedBlocks = engineLocalView.getEngineCacheBlocks(engineIpPort);
        globalCacheIndex.removeAllCacheBlockOfEngine(engineIpPort, removedBlocks, trace);
        engineLocalView.removeAllCacheBlockOfEngine(engineIpPort);
        pendingAdditions.remove(engineIpPort);
        cacheMetricsReporter.reportGlobalCacheMetrics(
                globalCacheIndex.totalBlocks(),
                globalCacheIndex.totalMappings());
    }

    /**
     * Clear all data
     */
    public void clear() {

        globalCacheIndex.clear();
        engineLocalView.clear();
        pendingAdditions.clear();

        // Report
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());

        log.info("Cleared all cache data");
    }
}
