package org.flexlb.cache.match.localsync;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerIdentity;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.cache.EngineCacheInvalidator;
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
public class KvCacheManager implements EngineCacheInvalidator {

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
     * @param identity       worker identity providing the logical {@code ip:port@engineIndex},
     *                       physical {@code ip:port}, and metrics {@code ip@engineIndex}
     *                       representations of one engine behind the shared frontend
     * @param role           Engine role
     * @param newCacheBlocks New cache block set (blockCacheKeys)
     */
    public void updateEngineCache(
            WorkerIdentity identity, String role, Set<Long> newCacheBlocks) {
        String engineIPort = identity == null ? null : identity.getLogicalIpPort();
        if (engineIPort == null || newCacheBlocks == null) {
            return;
        }
        String physicalIpPort = identity.getPhysicalIpPort();
        if (physicalIpPort != null) {
            physicalIpPortByLogicalIpPort.put(engineIPort, physicalIpPort);
        }

        // Calculate diff
        DiffResult diffResult = engineLocalView.calculateDiff(engineIPort, newCacheBlocks);
        cacheMetricsReporter.reportCacheDiffMetrics(
                identity.getIpIndex(),
                role,
                diffResult.getAddedBlocks().size(),
                diffResult.getRemovedBlocks().size());
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

        // Report metrics
        cacheMetricsReporter.reportEngineLocalMetrics(
                identity.getIpIndex(), role, engineLocalView.size(engineIPort));
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
        cacheMetricsReporter.reportEngineViewsMapSize(engineLocalView.getEngineViewsMapSize());
    }

    @Override
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

        // Report
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());

        log.info("Cleared all cache data");
    }
}
