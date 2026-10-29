package org.flexlb.cache.match.localsync;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerIdentity;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.cache.EngineCacheInvalidator;
import org.flexlb.metric.SafeFlexMonitor;
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

    private final Map<String, String> physicalIpPortByLogicalIpPort = new ConcurrentHashMap<>();

    /**
     * The owning generation's cache lock serializes each address and its retry.
     */
    private final com.google.common.util.concurrent.Striped<java.util.concurrent.locks.Lock>
            cacheIndexLocks = com.google.common.util.concurrent.Striped.lock(1024);

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
        updateEngineCache(
                workerStatus.getWorkerIdentity(),
                workerStatus.getMetricIpPort(),
                role,
                newCacheBlocks);
    }

    private void updateEngineCache(
            WorkerIdentity identity, String metricIpPort, String role, Set<Long> newCacheBlocks) {
        String engineIPort = identity == null ? null : identity.getLogicalIpPort();
        if (engineIPort == null || newCacheBlocks == null) {
            return;
        }
        var cacheLock = cacheIndexLocks.get(engineIPort);
        cacheLock.lock();
        try {
            applyCacheUpdate(identity, metricIpPort, role, newCacheBlocks);
        } finally {
            cacheLock.unlock();
        }
    }

    private void applyCacheUpdate(
            WorkerIdentity identity, String metricIpPort, String role, Set<Long> newCacheBlocks) {
        String engineIPort = identity.getLogicalIpPort();
        physicalIpPortByLogicalIpPort.put(engineIPort, identity.getPhysicalIpPort());

        Long pending = pendingAdditions.get(engineIPort);
        if (pending != null) {
            addCacheBlock(engineIPort, pending);
            pendingAdditions.remove(engineIPort);
        }

        // Calculate diff
        DiffResult diffResult = engineLocalView.calculateDiff(engineIPort, newCacheBlocks);
        SafeFlexMonitor.run(
                "cache.diff",
                () ->
                        cacheMetricsReporter.reportCacheDiffMetrics(
                                metricIpPort,
                                role,
                                diffResult.getAddedBlocks().size(),
                                diffResult.getRemovedBlocks().size()));
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
                                metricIpPort, role, engineLocalView.size(engineIPort)));
        SafeFlexMonitor.run(
                "cache.global",
                () ->
                        cacheMetricsReporter.reportGlobalCacheMetrics(
                                globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings()));
        SafeFlexMonitor.run(
                "cache.views",
                () ->
                        cacheMetricsReporter.reportEngineViewsMapSize(
                                engineLocalView.getEngineViewsMapSize()));
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

    /**
     * Remove both local and global cache-index state for one engine.
     */
    public void removeEngineCache(String engineIpPort) {
        if (engineIpPort == null) {
            return;
        }
        var cacheLock = cacheIndexLocks.get(engineIpPort);
        cacheLock.lock();
        try {
            Set<Long> ownedKeys = engineLocalView.getEngineCacheBlocks(engineIpPort);
            globalCacheIndex.removeAllCacheBlockOfEngine(engineIpPort, ownedKeys);
            engineLocalView.removeAllCacheBlockOfEngine(engineIpPort);
            physicalIpPortByLogicalIpPort.remove(engineIpPort);
            pendingAdditions.remove(engineIpPort);
            SafeFlexMonitor.run(
                    "cache.global",
                    () ->
                            cacheMetricsReporter.reportGlobalCacheMetrics(
                                    globalCacheIndex.totalBlocks(),
                                    globalCacheIndex.totalMappings()));
        } finally {
            cacheLock.unlock();
        }
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
        Set<String> staleEngineIpPorts = new HashSet<>(engineLocalView.getAllEngineIpPorts());
        staleEngineIpPorts.removeIf(
                engineIpPort ->
                        activePhysicalIpPorts.contains(
                                physicalIpPortByLogicalIpPort.getOrDefault(
                                        engineIpPort, engineIpPort)));
        for (String staleEngineIpPort : staleEngineIpPorts) {
            long startTime = System.nanoTime() / 1000;
            removeEngineCache(staleEngineIpPort);
            log.info(
                    "Removed stale engine cache: {}, cost={}us",
                    staleEngineIpPort,
                    System.nanoTime() / 1000 - startTime);
        }
    }

    /**
     * Clear all data
     */
    public void clear() {

        globalCacheIndex.clear();
        engineLocalView.clear();
        physicalIpPortByLogicalIpPort.clear();
        pendingAdditions.clear();

        // Report
        SafeFlexMonitor.run(
                "cache.global",
                () ->
                        cacheMetricsReporter.reportGlobalCacheMetrics(
                                globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings()));

        log.info("Cleared all cache data");
    }
}
