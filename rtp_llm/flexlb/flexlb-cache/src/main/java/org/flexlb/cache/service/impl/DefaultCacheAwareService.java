package org.flexlb.cache.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.domain.CacheMatch;
import org.flexlb.cache.domain.EngineGeneration;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Default implementation of cache-aware service
 * Provides unified cache management service, encapsulating underlying KvCacheManager
 *
 * @author FlexLB
 */
@Slf4j
@Service
public class DefaultCacheAwareService implements CacheAwareService {

    private final KvCacheManager kvCacheManager;
    private final CacheMetricsReporter cacheMetricsReporter;

    @Autowired
    public DefaultCacheAwareService(
            KvCacheManager kvCacheManager,
            CacheMetricsReporter cacheMetricsReporter) {
        this.kvCacheManager = kvCacheManager;
        this.cacheMetricsReporter = cacheMetricsReporter;
    }

    @Override
    public Map<EngineGeneration, CacheMatch> findMatchingEngines(
            List<Long> blockCacheKeys,
            RoleType roleType,
            List<EngineGeneration> candidates) {

        long startTime = System.nanoTime() / 1000;

        try {
            if (blockCacheKeys == null || blockCacheKeys.isEmpty()
                    || candidates == null || candidates.isEmpty()) {
                return Collections.emptyMap();
            }

            Map<EngineGeneration, CacheMatch> resultMap =
                    kvCacheManager.findMatchingEngines(
                            blockCacheKeys, candidates);

            reportFindMetric(roleType, startTime, "0");

            return resultMap;
        } catch (Exception e) {
            reportFindMetric(roleType, startTime, "1");
            log.error("Error finding matching engines for role: {}", roleType, e);
            return Collections.emptyMap();
        }
    }

    /** Query telemetry must never erase a successfully computed match set. */
    private void reportFindMetric(RoleType roleType, long startTime, String outcome) {
        try {
            cacheMetricsReporter.reportFindMatchingEnginesRT(
                    roleType, startTime, outcome);
        } catch (RuntimeException metricFailure) {
            log.warn("Failed to report cache lookup metric for role: {}", roleType,
                    metricFailure);
        }
    }

    @Override
    public boolean activateEngineGeneration(
            String engineIpPort,
            RoleType roleType,
            long generationId) {
        if (!ownsCache(roleType)) {
            return true;
        }
        try {
            return kvCacheManager.activateEngineGeneration(
                    engineIpPort, generationId);
        } catch (RuntimeException e) {
            log.error("Error activating cache generation for {}#{}",
                    engineIpPort, generationId, e);
            return false;
        }
    }

    @Override
    public WorkerCacheUpdateResult updateEngineBlockCache(
            String engineIpPort,
            RoleType roleType,
            long generationId,
            CacheStatus cacheStatus) {
        long startTime = System.nanoTime() / 1000;
        String role = roleType == null ? "unknown" : roleType.getCode();

        try {
            if (engineIpPort == null || engineIpPort.isBlank()
                    || generationId <= 0L || roleType == null
                    || cacheStatus == null) {
                WorkerCacheUpdateResult result = buildResult(
                        WorkerCacheUpdateResult.Outcome.INVALID_INPUT,
                        engineIpPort,
                        cacheStatus,
                        "Engine identity, role, generation and cache status are required");
                reportUpdateMetric(role, startTime, "0");
                return result;
            }
            if (cacheStatus.getCachedKeys() == null) {
                WorkerCacheUpdateResult result = buildResult(
                        WorkerCacheUpdateResult.Outcome.INVALID_INPUT,
                        engineIpPort,
                        cacheStatus,
                        "Worker cached keys are null");
                reportUpdateMetric(role, startTime, "0");
                return result;
            }

            Set<Long> cachedKeys = cacheStatus.getCachedKeys();
            boolean applied = kvCacheManager.updateEngineCache(
                    engineIpPort,
                    generationId,
                    roleType,
                    cachedKeys);
            if (!applied) {
                reportUpdateMetric(role, startTime, "stale");
                return buildResult(
                        WorkerCacheUpdateResult.Outcome.STALE_GENERATION,
                        engineIpPort,
                        cacheStatus,
                        "Engine generation is no longer active");
            }

            reportUpdateMetric(role, startTime, "1");
            return buildResult(
                    WorkerCacheUpdateResult.Outcome.APPLIED,
                    engineIpPort,
                    cacheStatus,
                    null);

        } catch (Throwable e) {
            log.error("Error updating worker cache for: {}", engineIpPort, e);
            reportUpdateMetric(role, startTime, "0");
            return buildResult(
                    WorkerCacheUpdateResult.Outcome.FAILED,
                    engineIpPort,
                    cacheStatus,
                    e.getMessage());
        }
    }

    /** Metrics are observational and must not change an already committed result. */
    private void reportUpdateMetric(String role, long startTime, String outcome) {
        try {
            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(
                    role, startTime, outcome);
        } catch (RuntimeException metricFailure) {
            log.warn("Failed to report cache update metric for role: {}", role,
                    metricFailure);
        }
    }

    @Override
    public boolean retireEngineGeneration(
            String engineIpPort,
            RoleType roleType,
            long generationId) {
        if (!ownsCache(roleType)) {
            return true;
        }
        try {
            return kvCacheManager.retireEngineGeneration(
                    engineIpPort, generationId);
        } catch (RuntimeException e) {
            log.error("Error retiring cache generation for {}#{}",
                    engineIpPort, generationId, e);
            return false;
        }
    }

    private static boolean ownsCache(RoleType roleType) {
        return roleType == RoleType.PREFILL || roleType == RoleType.PDFUSION;
    }

    private WorkerCacheUpdateResult buildResult(
            WorkerCacheUpdateResult.Outcome outcome,
            String engineIpPort,
            CacheStatus cacheStatus,
            String errorMessage) {
        return WorkerCacheUpdateResult.builder()
            .outcome(outcome)
            .engineIpPort(engineIpPort)
            .cacheBlockCount(cacheStatus == null || cacheStatus.getCachedKeys() == null
                    ? 0 : cacheStatus.getCachedKeys().size())
            .availableKvCache(cacheStatus == null
                    ? 0 : cacheStatus.getAvailableKvCache())
            .totalKvCache(cacheStatus == null
                    ? 0 : cacheStatus.getTotalKvCache())
            .cacheVersion(cacheStatus == null ? -1 : cacheStatus.getVersion())
            .errorMessage(errorMessage)
            .build();
    }
}
