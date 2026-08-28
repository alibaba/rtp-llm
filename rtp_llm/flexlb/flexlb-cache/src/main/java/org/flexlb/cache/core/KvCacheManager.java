package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatch;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.domain.EngineGeneration;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** Thin cache API over the single generation-fenced cache index. */
@Slf4j
@Component
public class KvCacheManager {

    private final GlobalCacheIndex cacheIndex;
    private final CacheMetricsReporter cacheMetricsReporter;
    private final DynamicCacheIntervalService dynamicIntervalManager;

    @Autowired
    public KvCacheManager(
            GlobalCacheIndex cacheIndex,
            CacheMetricsReporter cacheMetricsReporter,
            DynamicCacheIntervalService dynamicIntervalManager) {
        this.cacheIndex = cacheIndex;
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.dynamicIntervalManager = dynamicIntervalManager;
    }

    @PostConstruct
    public void init() {
        log.info("KvCacheManager initialized successfully");
    }

    @PreDestroy
    public void destroy() {
        log.info("KvCacheManager shutting down...");
        clear();
    }

    public Map<EngineGeneration, CacheMatch> findMatchingEngines(
            List<Long> blockCacheKeys,
            List<EngineGeneration> candidates) {
        if (blockCacheKeys == null || blockCacheKeys.isEmpty()
                || candidates == null || candidates.isEmpty()) {
            return Map.of();
        }
        return cacheIndex.batchCalculatePrefixMatches(
                candidates, blockCacheKeys);
    }

    public boolean activateEngineGeneration(
            String engineIpPort, long generationId) {
        boolean activated = cacheIndex.activateEngineGeneration(
                engineIpPort, generationId);
        if (activated) {
            reportIndexMetricsSafely();
        }
        return activated;
    }

    /** Replace one exact generation's full cache snapshot. */
    public boolean updateEngineCache(
            String engineIpPort,
            long generationId,
            RoleType roleType,
            Set<Long> newCacheBlocks) {
        if (newCacheBlocks == null) {
            throw new IllegalArgumentException("newCacheBlocks must not be null");
        }
        Set<Long> immutableSnapshot = Set.copyOf(newCacheBlocks);
        Optional<DiffResult> committed = cacheIndex.replaceEngineCache(
                engineIpPort, generationId, immutableSnapshot);
        if (committed.isEmpty()) {
            return false;
        }

        DiffResult diff = committed.get();
        String role = roleType == null ? "unknown" : roleType.getCode();
        reportCommittedUpdateSafely(
                engineIpPort, role, immutableSnapshot.size(), diff);
        return true;
    }

    public boolean retireEngineGeneration(
            String engineIpPort, long generationId) {
        boolean retired = cacheIndex.retireEngineGeneration(
                engineIpPort, generationId);
        if (retired) {
            reportIndexMetricsSafely();
        }
        return retired;
    }

    public void clear() {
        cacheIndex.clear();
        reportIndexMetricsSafely();
        log.info("Cleared all cache data");
    }

    private void reportCommittedUpdateSafely(
            String engineIpPort,
            String role,
            int cacheBlockCount,
            DiffResult diff) {
        try {
            cacheMetricsReporter.reportCacheDiffMetrics(
                    role, diff.addedBlocks().size(), diff.removedBlocks().size());
            dynamicIntervalManager.updateDiffStatistics(
                    diff.addedBlocks().size() + diff.removedBlocks().size());
            cacheMetricsReporter.reportEngineLocalMetrics(
                    engineHost(engineIpPort), role, cacheBlockCount);
        } catch (RuntimeException telemetryFailure) {
            log.warn("Cache update telemetry failed for {}: {}",
                    engineIpPort, telemetryFailure.getMessage());
        }
        reportIndexMetricsSafely();
    }

    private void reportIndexMetricsSafely() {
        try {
            GlobalCacheIndex.IndexMetrics snapshot = cacheIndex.metricsSnapshot();
            cacheMetricsReporter.reportGlobalCacheMetrics(
                    snapshot.totalBlocks(), snapshot.totalMappings());
            cacheMetricsReporter.reportEngineViewsMapSize(snapshot.engineCount());
        } catch (RuntimeException telemetryFailure) {
            log.warn("Cache index telemetry failed: {}",
                    telemetryFailure.getMessage());
        }
    }

    private static String engineHost(String engineIpPort) {
        int separator = engineIpPort.lastIndexOf(':');
        return separator > 0
                ? engineIpPort.substring(0, separator)
                : engineIpPort;
    }
}
