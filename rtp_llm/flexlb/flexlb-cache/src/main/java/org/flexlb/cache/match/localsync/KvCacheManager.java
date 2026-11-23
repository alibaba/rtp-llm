package org.flexlb.cache.match.localsync;

import com.google.common.util.concurrent.Striped;
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
import java.util.concurrent.locks.Lock;

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
    @Autowired private GlobalCacheIndex globalCacheIndex;
    @Autowired private EngineLocalView engineLocalView;
    @Autowired private WorkerStatusProvider workerStatusProvider;

    /**
     * Cache metrics reporter
     */
    @Autowired private CacheMetricsReporter cacheMetricsReporter;

    private final Map<String, String> physicalIpPortByLogicalIpPort = new ConcurrentHashMap<>();
    // Discovery cleanup and status application may run concurrently for the same logical worker.
    private final Striped<Lock> workerCacheLocks = Striped.lock(256);

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
     * Clear all data
     */
    public void clear() {

        globalCacheIndex.clear();
        engineLocalView.clear();
        physicalIpPortByLogicalIpPort.clear();

        // Report
        cacheMetricsReporter.reportGlobalCacheMetrics(
                globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());

        log.info("Cleared all cache data");
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
     * Query engine cache matching status for live workers in a role/group.
     *
     * @param blockCacheKeys List of cache block hash values to query
     * @param roleType       Engine role to query
     * @param group          Engine group to query
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength
     */
    public Map<String /*engineIpPort*/, Integer /*prefixMatchLength*/> findMatchingEngines(
            List<Long> blockCacheKeys, RoleType roleType, String group) {

        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }

        List<String> engineIpPorts =
                workerStatusProvider.getWorkerStatuses(roleType, group).stream()
                        .map(WorkerStatus::getLogicalIpPort)
                        .toList();

        return globalCacheIndex.batchCalculatePrefixMatchLength(engineIpPorts, blockCacheKeys);
    }

    public void updateEngineCache(
            WorkerStatus workerStatus, String role, Set<Long> newCacheBlocks) {
        if (workerStatus == null) {
            return;
        }
        WorkerIdentity identity = workerStatus.getWorkerIdentity();
        if (identity == null || newCacheBlocks == null) {
            return;
        }
        Lock lock = workerCacheLocks.get(identity.getLogicalIpPort());
        lock.lock();
        try {
            updateEngineCache(identity, workerStatus.getMetricIpPort(), role, newCacheBlocks);
        } finally {
            lock.unlock();
        }
    }

    /** Remove both local and global cache-index state for one engine. */
    public void removeEngineCache(String engineIpPort) {
        if (engineIpPort == null) {
            return;
        }
        removeCachedWorkerBlocks(engineIpPort, Set.of());
        cacheMetricsReporter.reportGlobalCacheMetrics(
                globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
    }

    /**
     * Remove cache metadata for engines that are no longer present in service discovery.
     */
    @Override
    public void removeStaleEngineCaches(Collection<String> activeEngineIpPorts) {
        if (activeEngineIpPorts == null) {
            return;
        }
        Set<String> activePhysicalIpPorts = new HashSet<>(activeEngineIpPorts);
        for (String engineIpPort : engineLocalView.getAllEngineIpPorts()) {
            if (activePhysicalIpPorts.contains(
                    physicalIpPortByLogicalIpPort.getOrDefault(engineIpPort, engineIpPort))) {
                continue;
            }
            long startTime = System.nanoTime() / 1000;
            if (removeCachedWorkerBlocks(engineIpPort, activePhysicalIpPorts)) {
                log.info(
                        "Removed stale engine cache: {}, cost={}us",
                        engineIpPort,
                        System.nanoTime() / 1000 - startTime);
            }
        }
    }

    private boolean removeCachedWorkerBlocks(
            String engineIpPort, Set<String> activePhysicalIpPorts) {
        Lock lock = workerCacheLocks.get(engineIpPort);
        lock.lock();
        try {
            if (activePhysicalIpPorts.contains(
                    physicalIpPortByLogicalIpPort.getOrDefault(engineIpPort, engineIpPort))) {
                return false;
            }
            Set<Long> removedBlocks = engineLocalView.removeAllCacheBlockOfEngine(engineIpPort);
            globalCacheIndex.removeAllCacheBlockOfEngine(engineIpPort, removedBlocks);
            physicalIpPortByLogicalIpPort.remove(engineIpPort);
            return true;
        } finally {
            lock.unlock();
        }
    }

    private void updateEngineCache(
            WorkerIdentity identity, String metricIpPort, String role, Set<Long> newCacheBlocks) {
        String engineIPort = identity.getLogicalIpPort();
        physicalIpPortByLogicalIpPort.put(engineIPort, identity.getPhysicalIpPort());

        // Calculate diff
        DiffResult diffResult = engineLocalView.calculateDiff(engineIPort, newCacheBlocks);
        cacheMetricsReporter.reportCacheDiffMetrics(
                metricIpPort,
                role,
                diffResult.getAddedBlocks().size(),
                diffResult.getRemovedBlocks().size());
        if (!diffResult.hasChanges()) {
            return;
        }

        // Apply added cache blocks
        for (var added = diffResult.getAddedBlocks().iterator(); added.hasNext(); ) {
            long block = added.nextLong();
            engineLocalView.addOrUpdateCacheBlock(engineIPort, block);
            globalCacheIndex.addCacheBlock(block, engineIPort);
        }

        // Apply removed cache blocks
        for (var removed = diffResult.getRemovedBlocks().iterator(); removed.hasNext(); ) {
            long removedBlock = removed.nextLong();
            // Remove from local view
            engineLocalView.removeCacheBlock(engineIPort, removedBlock);
            // Remove from global index
            globalCacheIndex.removeCacheBlock(engineIPort, removedBlock);
        }

        // Report metrics
        cacheMetricsReporter.reportEngineLocalMetrics(
                metricIpPort, role, engineLocalView.size(engineIPort));
        cacheMetricsReporter.reportGlobalCacheMetrics(
                globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
        cacheMetricsReporter.reportEngineViewsMapSize(engineLocalView.getEngineViewsMapSize());
    }
}
