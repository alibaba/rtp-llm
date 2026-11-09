package org.flexlb.cache.match.localsync;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.EndpointTrace;
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
    @Autowired private GlobalCacheIndex globalCacheIndex;
    @Autowired private EngineLocalView engineLocalView;
    @Autowired private WorkerStatusProvider workerStatusProvider;

    /**
     * Cache metrics reporter
     */
    @Autowired private CacheMetricsReporter cacheMetricsReporter;

    // The owning worker supplies the same lock used by cache application and retirement.
    private final Map<String, WorkerStatus> cacheOwners = new ConcurrentHashMap<>();
    // Only an interrupted addition needs replay: its local ownership may already be recorded.
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
     * Clear all data
     */
    public void clear() {

        globalCacheIndex.clear();
        engineLocalView.clear();
        cacheOwners.clear();
        pendingAdditions.clear();

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

    public void updateEngineCache(WorkerStatus worker, String role, Set<Long> keys) {
        updateEngineCache(worker, role, keys, null);
    }

    public void updateEngineCache(
            WorkerStatus workerStatus,
            String role,
            Set<Long> newCacheBlocks,
            EndpointTrace.Poll pollTrace) {
        if (workerStatus == null
                || workerStatus.getLogicalIpPort() == null
                || newCacheBlocks == null) {
            return;
        }
        workerStatus.cacheMutationLock.lock();
        try {
            String address = workerStatus.getLogicalIpPort();
            WorkerStatus previous = cacheOwners.putIfAbsent(address, workerStatus);
            if (previous != null && previous != workerStatus) {
                throw new IllegalStateException(
                        "Previous cache generation has not been retired: " + address);
            }
            updateEngineCache(
                    workerStatus.getWorkerIdentity(),
                    workerStatus.getMetricIpPort(),
                    role,
                    newCacheBlocks,
                    pollTrace);
        } finally {
            workerStatus.cacheMutationLock.unlock();
        }
    }

    /**
     * Removes snapshots absent from discovery while excluding updates to each owning generation.
     */
    @Override
    public void removeStaleEngineCaches(Collection<String> activeEngineIpPorts) {
        if (activeEngineIpPorts == null) {
            return;
        }
        Set<String> active = new HashSet<>(activeEngineIpPorts);
        for (Map.Entry<String, WorkerStatus> entry : cacheOwners.entrySet()) {
            WorkerStatus owner = entry.getValue();
            if (!active.contains(owner.getWorkerIdentity().getPhysicalIpPort())) {
                removeEngineCache(entry.getKey(), owner, owner.trace());
            }
        }
    }

    /**
     * Remove both local and global cache-index state for one engine.
     */
    public void removeEngineCache(String engineIpPort) {
        removeEngineCache(engineIpPort, null);
    }

    public void removeEngineCache(String engineIpPort, EndpointTrace trace) {
        if (engineIpPort == null) {
            return;
        }
        WorkerStatus owner = cacheOwners.get(engineIpPort);
        if (owner != null) {
            removeEngineCache(engineIpPort, owner, trace);
        }
    }

    /**
     * Removes only the snapshot owned by the captured worker generation.
     */
    public void removeEngineCache(WorkerStatus owner, EndpointTrace trace) {
        if (owner != null) {
            removeEngineCache(owner.getLogicalIpPort(), owner, trace);
        }
    }

    private void updateEngineCache(
            WorkerIdentity identity,
            String metricIpPort,
            String role,
            Set<Long> newCacheBlocks,
            EndpointTrace.Poll pollTrace) {
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
                metricIpPort,
                role,
                diffResult.getAddedBlocks().size(),
                diffResult.getRemovedBlocks().size());
        if (!diffResult.hasChanges()) {
            return;
        }

        for (Long key : diffResult.getAddedBlocks()) {
            addCacheBlock(engineIPort, key, pollTrace);
        }
        for (Long key : diffResult.getRemovedBlocks()) {
            globalCacheIndex.removeCacheBlock(engineIPort, key, pollTrace);
            engineLocalView.removeCacheBlock(engineIPort, key);
        }

        // Report metrics
        cacheMetricsReporter.reportEngineLocalMetrics(
                metricIpPort, role, engineLocalView.size(engineIPort));
        cacheMetricsReporter.reportGlobalCacheMetrics(
                globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
        cacheMetricsReporter.reportEngineViewsMapSize(engineLocalView.getEngineViewsMapSize());
    }

    private void addCacheBlock(String address, Long key, EndpointTrace.Poll pollTrace) {
        try {
            engineLocalView.addOrUpdateCacheBlock(address, key);
            globalCacheIndex.addCacheBlock(key, address, pollTrace);
        } catch (RuntimeException | Error failure) {
            pendingAdditions.put(address, key);
            throw failure;
        }
    }

    private void removeEngineCache(String address, WorkerStatus owner, EndpointTrace trace) {
        owner.cacheMutationLock.lock();
        try {
            // A delayed discovery callback must not remove a replacement generation.
            if (cacheOwners.get(address) != owner) {
                return;
            }
            Set<Long> removedBlocks = engineLocalView.getEngineCacheBlocks(address);
            globalCacheIndex.removeAllCacheBlockOfEngine(address, removedBlocks, trace);
            engineLocalView.removeAllCacheBlockOfEngine(address);
            pendingAdditions.remove(address);
            owner.invalidateCacheIndex();
            cacheOwners.remove(address, owner);
            cacheMetricsReporter.reportGlobalCacheMetrics(
                    globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
        } finally {
            owner.cacheMutationLock.unlock();
        }
    }
}
