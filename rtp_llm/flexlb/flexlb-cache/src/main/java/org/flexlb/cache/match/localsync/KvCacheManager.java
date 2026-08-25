package org.flexlb.cache.match.localsync;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
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
    private final Map<String, String> physicalIpPortByLogicalIpPort =
            new ConcurrentHashMap<>();

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
     * @return prefix match lengths keyed by logical {@code ip:port@engineIndex} identity
     */
    public Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(List<Long> blockCacheKeys,
        RoleType roleType, String group) {

        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }

        // Use candidate engine list
        List<String> enginesIpPorts = workerStatusProvider.getWorkerStatuses(roleType, group).stream()
                .map(WorkerStatus::getLogicalIpPort)
                .toList();

        // Batch calculate prefix match length
        return globalCacheIndex.batchCalculatePrefixMatchLength(enginesIpPorts, blockCacheKeys);
    }

    /**
     * Update engine cache status
     *
     * @param engineIPort    logical worker identity in {@code ip:port@engineIndex} format; the
     *                       index identifies one independently routable engine behind the physical
     *                       frontend
     * @param physicalIpPort shared frontend identity in {@code ip:port} format
     * @param ipIndex        metrics identity in {@code ip@engineIndex} format
     * @param role           Engine role
     * @param newCacheBlocks New cache block set (blockCacheKeys)
     */
    public void updateEngineCache(
            String engineIPort,
            String physicalIpPort,
            String ipIndex,
            String role,
            Set<Long> newCacheBlocks) {
        if (engineIPort == null || newCacheBlocks == null) {
            DiffResult.empty(engineIPort);
            return;
        }
        if (physicalIpPort != null) {
            physicalIpPortByLogicalIpPort.put(engineIPort, physicalIpPort);
        }

        // Calculate diff
        DiffResult diffResult = engineLocalView.calculateDiff(
                engineIPort, ipIndex, newCacheBlocks, role);
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
        cacheMetricsReporter.reportEngineLocalMetrics(
                ipIndex, role, engineLocalView.size(engineIPort));
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
        cacheMetricsReporter.reportEngineViewsMapSize(engineLocalView.getEngineViewsMapSize());
    }

    /**
     * Remove cache metadata for engines that are no longer present in service discovery.
     *
     * @param activeEngineIpPorts active physical engine addresses in {@code ip:port} format
     */
    public void removeStaleEngineCaches(Collection<String> activeEngineIpPorts) {
        if (activeEngineIpPorts == null) {
            return;
        }
        Set<String> activePhysicalIpPorts = new HashSet<>(activeEngineIpPorts);
        Set<String> staleEngineIpPorts = new HashSet<>(engineLocalView.getAllEngineIpPorts());
        staleEngineIpPorts.removeIf(engineIpPort ->
                activePhysicalIpPorts.contains(
                        physicalIpPortByLogicalIpPort.getOrDefault(engineIpPort, engineIpPort)));
        for (String staleEngineIpPort : staleEngineIpPorts) {
            long startTime = System.nanoTime() / 1000;
            engineLocalView.removeAllCacheBlockOfEngine(staleEngineIpPort);
            globalCacheIndex.removeAllCacheBlockOfEngine(staleEngineIpPort);
            physicalIpPortByLogicalIpPort.remove(staleEngineIpPort);
            log.info("Removed stale engine cache: {}, cost={}us",
                    staleEngineIpPort, System.nanoTime() / 1000 - startTime);
        }
    }

    /**
     * Clear all data
     */
    public void clear() {

        globalCacheIndex.clear();
        engineLocalView.clear();
        physicalIpPortByLogicalIpPort.clear();

        totalUpdates.reset();
        // Report
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());

        log.info("Cleared all cache data");
    }
}
